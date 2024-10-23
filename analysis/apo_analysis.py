# based on https://github.com/bjing2016/EigenFold/blob/master/ensemble_analysis.ipynb

import argparse
import tempfile
from pathlib import Path
import torch, os, warnings, io, subprocess
from Bio.PDB import PDBIO, Chain, Residue, Polypeptide, Atom, PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.SeqUtils import seq1, seq3


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from Bio import pairwise2
from scipy.spatial.transform import Rotation as R
from scipy.stats import pearsonr, spearmanr, kendalltau, linregress

from slm.utils.eval_utils import split_pdbfile, load_apo_targets, load_codnas_targets
from slm.utils.plot_utils import scatterplot_apo
from slm.utils.protein_residues import normal as RESIDUES
from slm.utils.tm_utils import tmscore


# OMP_NUM_THREADS
os.environ['OMP_NUM_THREADS'] = '8'

def PROCESS_RESIDUES(d):
    d['HIS'] = d['HIP']
    d = {key: val for key, val in d.items() if seq1(key) != 'X'}
    for key in d:
        atoms = d[key]['atoms']
        d[key] = {'CA': 'C'} | {key: val['symbol'] for key, val in atoms.items() if val['symbol'] != 'H' and key != 'CA'}
    return d
    
RESIDUES = PROCESS_RESIDUES(RESIDUES)
parser = PDBParser()
my_dir = f"/tmp/{os.getpid()}"
if not os.path.isdir(my_dir):
    os.mkdir(my_dir)

# load with alignment to seqres
def pdb_to_npy(pdb_path, model_num=0, chain_id=None, seqres=None):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        try:
            model = parser.get_structure('', pdb_path)[model_num]
            if chain_id is not None:
                chain = model.child_dict[chain_id]
            else:
                chain = model.child_list[0]
        except: print(f'Error opening PDB file {pdb_path}'); return

    coords = []
    seq = ''
    try:
        for resi in list(chain):
            if (resi.id[0] != ' ') or ('CA' not in resi.child_dict) or (resi.resname not in RESIDUES): continue
            co = np.zeros((14, 3)) * np.nan
            atoms = RESIDUES[resi.resname]
            seq += resi.resname
            for i, at in enumerate(atoms):
                try: co[i] = resi.child_dict[at].coord
                except: pass
            coords.append(co)
        coords = np.stack(coords)
        seq = seq1(seq)
    except: print(f'Error parsing chain {pdb_path}'); return
    if not seqres: return coords, seq

    coords_= np.zeros((len(seqres), 14, 3)) * np.nan
    alignment = pairwise2.align.globalxx(seq, seqres)[0]
    
    if '-' in alignment.seqB: 
        print(f'Alignment gaps {pdb_path}'); return
    mask = np.array([a == b for a, b in zip(alignment.seqA, alignment.seqB)])
    coords_[mask] = coords
    return coords_, mask


class PDBFile:
    def __init__(self, molseq):
        self.molseq = molseq
        self.blocks = []
        self.chain = chain = Chain.Chain('A')
        j = 1
        for i, aa in enumerate(molseq):
            aa = Residue.Residue(
                id=(' ', i+1, ' '), 
                resname=seq3(aa).upper(), 
                segid='    '
            )
            for atom in RESIDUES[aa.resname.upper()]:
                at = Atom.Atom(
                    name=atom, coord=None, bfactor=0, occupancy=1, altloc=' ',
                    fullname=f' {atom} ', serial_number=j, element=RESIDUES[aa.resname][atom]
                )
                j += 1
                aa.add(at)
            chain.add(aa)
       
          
    def add(self, coords):
        if type(coords) is np.ndarray:
            coords = coords.astype(np.float64)
        elif type(coords) is torch.Tensor:
            coords = coords.cpu().double().numpy()
        
        for i, resi in enumerate(self.chain):
            atoms = RESIDUES[resi.resname]
            resi['CA'].coord = coords[i]

        k = len(self.chain)
        for i, resi in enumerate(self.chain):
            atoms = RESIDUES[resi.resname]
            for j, at in enumerate(atoms):
                if at != 'CA': 
                    try: resi[at].coord = coords[k] + resi['CA'].coord
                    except: resi[at].coord = (np.nan, np.nan, np.nan)
                    k+=1

        pdbio = PDBIO()
        pdbio.set_structure(self.chain)

        stringio = io.StringIO()
        stringio.close, close = lambda: None, stringio.close
        pdbio.save(stringio)
        block = stringio.getvalue().split('\n')[:-2]
        stringio.close = close; stringio.close()

        self.blocks.append(block)
        return self
        
    def clear(self):
        self.blocks = []
        return self
    
    def write(self, path=None, idx=None, reverse=False):
        is_first = True
        str_ = ''
        blocks = self.blocks[::-1] if reverse else self.blocks
        for block in ([blocks[idx]] if idx is not None else blocks):
            block = [line for line in block if 'nan' not in line]
            if not is_first:
                block = [line for line in block if 'CONECT' not in line]
            else:
                is_first = False
            str_ += 'MODEL\n'
            str_ += '\n'.join(block)
            str_ += '\nENDMDL\n'
        if not path:
            return str_
        with open(path, 'w') as f:
            f.write(str_)
            
def renumber_pdb(molseq, X_path, X_renum, start=1):
    assert type(molseq) == str
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        model = parser.get_structure('', X_path)[0]
    chain = model.child_list[0]
    polypeptide = Polypeptide.Polypeptide(chain.get_residues())
    seq = polypeptide.get_sequence()
    
    alignment = pairwise2.align.globalxx(seq, molseq)[0]
    assert '-' not in alignment.seqB
    numbering = [i+start for i, c in enumerate(alignment.seqA) if c != '-']
    
    for n, resi in enumerate(chain):
        resi.id = (' ', 10000+n, ' ')
    
    for n, resi in zip(numbering, chain):
        resi.id = (' ', n, ' ')
        
    pdbio = PDBIO()
    pdbio.set_structure(chain)
    pdbio.save(X_renum)


# laod structure with alignment 
def get_structures(path1, path2, seq):
    struct1, seq1 = pdb_to_npy(path1)
    align1 = pairwise2.align.globalxx(seq, seq1)[0]
    struct1_fixed = np.ones((len(seq), 3)) * np.nan
    i, j = 0, 0
    for c, d in zip(align1.seqA, align1.seqB):
        if c != '-' and d != '-': struct1_fixed[i] = struct1[j, 0]
        if c != '-': i+=1
        if d != '-': j+=1

    struct2, seq2 = pdb_to_npy(path2)
    align2 = pairwise2.align.globalxx(seq, seq2)[0]
    struct2_fixed = np.ones((len(seq), 3)) * np.nan
    i, j = 0, 0
    for c, d in zip(align2.seqA, align2.seqB):
        if c != '-' and d != '-': struct2_fixed[i] = struct2[j, 0]
        if c != '-': i+=1
        if d != '-': j+=1
    
    struct1 = struct1_fixed - np.nanmean(struct1_fixed, 0)
    struct2 = struct2_fixed - np.nanmean(struct2_fixed, 0)
    mask = ~np.isnan(struct1[:,0]) & ~np.isnan(struct2[:,0])
    
    rot = R.align_vectors(struct1[mask], struct2[mask])[0].as_matrix()
    struct2 = struct2 @ rot.T
    
    return struct1, struct2


def analyze(pairs_df, samples_dir, reference_dir, output_dir):
    samples_dir = Path(samples_dir)
    tmpair = []
    tm1max_list = []
    tm2max_list = []
    ensvar_list = []
    tm_ens_list = []

    rmsd_dict = {}
    rmsf_dict = {}

    for i, (name1, row) in tqdm.tqdm(enumerate(pairs_df.iterrows()), total=len(pairs_df)):
        tm1, tm2 = [], []
        seq = row.seqres
        try: name2 = row.holo
        except: name2 = row.other
        # load ground truth pairs
        path1 = reference_dir / "structures" / f'{name1[:2]}/{name1}'
        path2 = reference_dir / "structures" / f'{name2[:2]}/{name2}'

        # subdf = samples_csv[samples_csv.index == name1]
        ensvar = []

        struct1, struct2 = get_structures(path1, path2, seq)
        rmsd_dict[name1] = np.square(struct1-struct2).sum(-1)**0.5

        rmsf_dict[name1] = []

        # split each pdb_files
        with tempfile.TemporaryDirectory() as tmpdirname:
            sample = samples_dir / f"{name1}"
            split_pdbfile(sample, Path(tmpdirname), sep='.', verbose=False)
            samples = os.listdir(tmpdirname)

            for j in range(len(samples)):
                path3 = f"{tmpdirname}/{name1.replace('.pdb', '')}.{j}.pdb"
                res = tmscore(path1, path3, return_dict=True)
                tm1.append(res['tm'])
                res = tmscore(path2, path3, return_dict=True)
                tm2.append(res['tm'])

                for k in range(j+1, len(samples)):
                    path4 = f"{tmpdirname}/{name1.replace('.pdb', '')}.{k}.pdb"
                    res = tmscore(path4, path3, return_dict=True)
                    ensvar.append(res['tm'])
                    struct1, struct2 = get_structures(path3, path4, seq)
                    rmsf_dict[name1].append(np.square(struct1-struct2).sum(-1)**0.5)
        _rmsf = np.stack(rmsf_dict[name1])

        rmsf_dict[name1] = np.mean(_rmsf**2, 0)**0.5
        
        ensvar_list.append(np.mean(ensvar))

        tm1 = np.array(tm1); tm2 = np.array(tm2)
        tm1max_list.append(tm1.max()); tm2max_list.append(tm2.max())
        tm_ens_list.append((tm1.max() + tm2.max()) / 2)
        res = tmscore(path1, path2, return_dict=True)
        vline = res['tm']

        res = tmscore(path2, path1, return_dict=True)
        hline = res['tm']
        tmpair.append((hline+vline)/2)

    if len(tm1max_list) != len(pairs_df):
        print("skip add to ")
    else:
        pairs_df['tm1max'] = tm1max_list
        pairs_df['tm2max'] = tm2max_list
        pairs_df['ensvar'] = ensvar_list
        pairs_df['tmpair'] = tmpair

    scatterplot_apo(x=tmpair, y=tm_ens_list, save_to=output_dir / f"ens_{samples_dir.stem}.png")
    scatterplot_apo(x=tmpair, y=ensvar_list, save_to=output_dir / f"var_{samples_dir.stem}.png", ylabel="TM diversity", regplot=True)
    
    tmens_mean_median = f"{round(np.mean(tm_ens_list), 3)} / {round(np.median(tm_ens_list), 3)}"
    
    # print("\n", tm_ens_list, "\n")
    return rmsd_dict, rmsf_dict, tmens_mean_median


def main(samples_dir, reference_dir, output_dir, task='apo'):
    assert task in ['apo', 'codnas'], f"Invalid task {task}"
    print("\n###################")
    
    print(f"Evaluation Task: {task}")
    print(samples_dir.stem)
    
    if task == 'apo':
        data_df = pd.read_csv(reference_dir / 'splits' / 'apo.csv', index_col='name').sort_index()
    else:
        data_df = pd.read_csv(reference_dir / 'splits' / 'codnas.csv', index_col='name').sort_index()
    rmsd_dict, rmsf_dict, tmens_mean_median = analyze(data_df, samples_dir, reference_dir, output_dir)
    
    def getmask(name):
        return ~np.isnan(rmsd_dict[name]) & ~np.isnan(rmsf_dict[name])

    correls = {name: {
        'pearson': pearsonr(rmsd_dict[name][getmask(name)], rmsf_dict[name][getmask(name)])[0],
        'spearman': spearmanr(rmsd_dict[name][getmask(name)], rmsf_dict[name][getmask(name)]).correlation,
        'kendall': kendalltau(rmsd_dict[name][getmask(name)], rmsf_dict[name][getmask(name)]).correlation
    } for name in data_df.index}
    data_df = data_df.join(pd.DataFrame(correls).T)

    global_rmsd = np.concatenate(list(rmsd_dict.values()))
    global_rmsf = np.concatenate(list(rmsf_dict.values()))
    mask = ~np.isnan(global_rmsd) & ~np.isnan(global_rmsf)  # fix bug
    
    metric_tm = round(pearsonr(data_df.ensvar, data_df.tmpair)[0], 3)
    rmsd_global = round(pearsonr(global_rmsd[mask], global_rmsf[mask])[0], 3)
    rmsd_pt_mean = round(data_df.pearson.mean(), 3)
    rmsd_pt_median = round(data_df.pearson.median(), 3)

    print('TM correlation', metric_tm)
    print(f"TM-ens (mean/median): {tmens_mean_median}")
    print('RMSD global correlation', rmsd_global)
    print('RMSD per-target correlation mean/median', f"{rmsd_pt_mean} / {rmsd_pt_median}")
    print("###################")

    return metric_tm, rmsd_global, rmsd_pt_mean, rmsd_pt_median, tmens_mean_median



def get_argparser():
    parser = argparse.ArgumentParser(description="Evaluate the ensemble of protein structures.")
    parser.add_argument("samples", type=str, help="Path to the data directory.")
    parser.add_argument("--reference", type=str, default="data/ground_truth/apo", help="Path to reference data.")
    parser.add_argument("--batch_samples", action="store_true", help="Batch evaluation of samples.")
    parser.add_argument("--task", type=str, default="apo", choices=["apo", "codnas"])
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_argparser()
    working_dir = Path(args.samples)
    reference_dir = Path(args.reference)
    assert working_dir.is_dir(), f"Invalid directory {working_dir}"
    assert reference_dir.is_dir(), f"Invalid directory {reference_dir}"

    if args.batch_samples:
        # batch evaluation
        runs = [f for f in working_dir.iterdir() if f.is_dir() and not f.name.startswith('asset')]
    else:
        runs = [working_dir]
        working_dir = working_dir.parent
    print(f"Found {len(runs)} runs in {working_dir}: {[run.stem for run in runs]}")

    output_dir = working_dir / 'asset'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    run_vs_metric = {
        'name': [],
        'rmsd_global': [],
        'rmsd_per_target_mean/median': [],
        'tm_ens_mean/median': [],
    }
    
    # loop over runs
    for run in tqdm.tqdm(runs):
        res = main(run, reference_dir=reference_dir, task=args.task, output_dir=output_dir)
        run_vs_metric['name'].append(run.stem)
        # run_vs_metric['metric_tm'].append(res[0])
        run_vs_metric['rmsd_global'].append(res[1])
        run_vs_metric['rmsd_per_target_mean/median'].append(f"{res[2]} / {res[3]}")
        run_vs_metric['tm_ens_mean/median'].append(res[4])
        # plot scatter plot
        per_target_tm_ens = []
        per_target_tm_var = []  # average TM score diversity
        
    pd.DataFrame(run_vs_metric).to_csv(output_dir / f'metrics_{args.task}.csv', index=False)
    
    