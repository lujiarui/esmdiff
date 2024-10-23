from typing import *
from pathlib import Path
from glob import glob
import time

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

from scipy.spatial import distance
from scipy.special import kl_div
from scipy import stats
from deeptime.decomposition import TICA

from slm.utils.residue_constants import restypes_with_x
from slm.utils import protein
from slm.models.utils import protseq_to_data
####################
EPS = 1e-12
PSEUDO_C = 1e-6
####################

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        if result is None:
            return None
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time ({func.__name__}): {elapsed_time:.2f} sec")
        return (*result, float(f'{elapsed_time:.2f}'))
    return wrapper


def position_specific_entropy(tokens):
    # tokens (N_frames, L) LongTensors
    # Calculate the position-specific entropy of a sequence.

    # Calculate the frequency of each token at each position.
    entropies = torch.zeros(tokens.size(1), device=tokens.device)
    N = tokens.size(0)
    for col in range(tokens.size(1)):
        freqs = torch.bincount(tokens[:, col]) / N
        freqs = freqs[freqs > 0]
        entropy = -torch.sum(freqs * torch.log2(freqs)) 
        entropies[col] = entropy
    return entropies

def rmsf(coords):
    """Calculate root mean square fluctuation of a trajectory."""
    assert len(coords.shape) == 3, f"coords shape should be 3D, got {len(coords.shape)}"
    return np.sqrt(np.mean(np.var(coords, axis=0), axis=-1))    # FIXME: not sure if this is correct

def correlations(x, y):
    res_sp = stats.spearmanr(x, y)
    res_pr = stats.pearsonr(x, y)
    print(f"Spearman: {res_sp.statistic:.3f}, p-value: {res_sp.pvalue:.6f}")
    print(f"Pearson: {res_pr.statistic:.3f}, p-value: {res_pr.pvalue:.6f}")
    return res_sp.statistic, res_pr.statistic


def adjacent_ca_distance(coords):
    """Calculate distance array for a single chain of CA atoms. Only k=1 neighbors.
    Args:
        coords: (..., L, 3)
    return 
        dist: (..., L-1)
    """    
    assert len(coords.shape) in (2, 3), f"CA coords should be 2D or 3D, got {coords.shape}" # (B, L, 3)
    dX = coords[..., :-1, :] - coords[..., 1:, :] # (..., L-1, 3)
    dist = np.sqrt(np.sum(dX**2, axis=-1))
    return dist # (..., L-1)


def distance_matrix_ca(coords):
    """Calculate distance matrix for a single chain of CA atoms. W/o exclude neighbors.
    Args:
        coords: (..., L, 3)
    Return:
        dist: (..., L, L)
    """
    assert len(coords.shape) in (2, 3), f"CA coords should be 2D or 3D, got {coords.shape}" # (B, L, 3)
    dX = coords[..., None, :, :] - coords[..., None, :] # (..., L, L, 3)
    dist = np.sqrt(np.sum(dX**2, axis=-1))
    return dist # (..., L, L)


def pairwise_distance_ca(coords, k=1):
    """Calculate pairwise distance vector for a single chain of CA atoms. W/o exclude neighbors.
    Args:
        coords: (..., L, 3)
    Return:
        dist: (..., D) (D=L * (L - 1) // 2) when k=1)
    """
    assert len(coords.shape) in (2, 3), f"CA coords should be 2D or 3D, got {coords.shape}" # (B, L, 3)
    dist = distance_matrix_ca(coords)
    L = dist.shape[-1]
    row, col = np.triu_indices(L, k=k)
    triu = dist[..., row, col]  # unified (but unclear) order
    return triu # (..., D)


def radius_of_gyration(coords, masses=None):
    """Compute the radius of gyration for every frame.
    
    Args:
        coords: (..., num_atoms, 3)
        masses: (num_atoms,)
        
    Returns:
        Rg: (..., )
        
    If masses are none, assumes equal masses.
    """
    assert len(coords.shape) in (2, 3), f"CA coords should be 2D or 3D, got {coords.shape}" # (B, L, 3)
    
    if masses is None:
        masses = np.ones(coords.shape[-2])
    else:
        assert len(masses.shape) == 1, f"masses should be 1D, got {masses.shape}"
        assert masses.shape[0] == coords.shape[-2], f"masses {masses.shape} != number of particles {coords.shape[-2]}"

    weights = masses / masses.sum()
    centered = coords - coords.mean(-2, keepdims=True) 
    squared_dists = (centered ** 2).sum(-1)
    Rg = (squared_dists * weights).sum(-1) ** 0.5
    return Rg


def _steric_clash(coords, ca_vdw_radius=1.7, allowable_overlap=0.4, k_exclusion=0):
    """ https://www.schrodinger.com/sites/default/files/s3/public/python_api/2022-3/_modules/schrodinger/structutils/interactions/steric_clash.html#clash_iterator
    Calculate the number of clashes in a single chain of CA atoms.
    
    Usage: 
        n_clash = calc_clash(coords)
    
    Args:
        coords: (n_atoms, 3), CA coordinates, coords should from one protein chain.
        ca_vdw_radius: float, default 1.7.
        allowable_overlap: float, default 0.4.
        k_exclusion: int, default 0. Exclude neighbors within [i-k-1, i+k+1].
        
    """
    assert np.isnan(coords).sum() == 0, "coords should not contain nan"
    assert len(coords.shape) in (2, 3), f"CA coords should be 2D or 3D, got {coords.shape}" # (B, L, 3)
    assert k_exclusion >= 0, "k_exclusion should be non-negative"
    bar = 2 * ca_vdw_radius - allowable_overlap
    # L = len(coords)
    # dist = np.sqrt(np.sum((coords[:L-k_exclusion, None, :] - coords[None, k_exclusion:, :])**2, axis=-1))   
    pwd = pairwise_distance_ca(coords, k=k_exclusion+1) # by default, only excluding self (k=1)
    assert len(pwd.shape) == 2, f"pwd should be 2D, got {pwd.shape}"
    n_clash = np.sum(pwd < bar, axis=-1)
    return n_clash.astype(int) #(..., )  #np.prod(dist.shape)


def validity(ca_coords_dict, **clash_kwargs):
    """Calculate clash validity of ensembles. 
    Args:
        ca_coords_dict: {k: (B, L, 3)}
    Return:
        valid: {k: validity in [0,1]}
    """
    n_clash = {
        k: _steric_clash(v, **clash_kwargs)
            for k, v in ca_coords_dict.items()
    }
    results = {
        k: 1.0 - (v>0).mean() for k, v in n_clash.items()
    }
    results = {k: np.around(v, decimals=4) for k, v in results.items()}
    return results


def bonding_validity(ca_coords_dict, ref_key='target', eps=1e-6):
    """Calculate bonding dissociation validity of ensembles."""
    adj_dist = {k: adjacent_ca_distance(v)
            for k, v in ca_coords_dict.items()
    }
    thres = adj_dist[ref_key].max()+ 1e-6
    
    results = {
        k: (v < thres).all(-1).sum().item() / len(v) 
            for k, v in adj_dist.items()
    }
    results = {k: np.around(v, decimals=4) for k, v in results.items()}
    return results


def idp_metrics(ca_coords_dict, ref_key='target', pwd_offset=3):
    # calculate pwd, rg, and contact probability
    # n_bins = 50 follows idpGAN
    pseudo_c = 0.01
    mse_pwd = {}
    mae_pwd = {}

    mse_contact = {}
    mae_contact = {}
    
    mse_rg = {}
    mae_rg = {}
    
    L  = len(ca_coords_dict[ref_key])
    ref_pwd = pairwise_distance_ca(ca_coords_dict[ref_key], k=pwd_offset)
    ref_pwd_mean = ref_pwd.mean(axis=0)   # (D)
    ref_rg_mean = radius_of_gyration(ca_coords_dict[ref_key]).mean(axis=0)    # (1, )
    ref_contacts = np.log( (ref_pwd < 8.0).mean(axis=0) + pseudo_c)   # (D, )

    for name, ca_coords in ca_coords_dict.items():
        ca_pwd = pairwise_distance_ca(ca_coords, k=pwd_offset)
        ca_rg_mean = radius_of_gyration(ca_coords).mean(axis=0)
        ca_contacts = np.log( (ca_pwd < 8.0).mean(axis=0) + pseudo_c) # (D, )
        
        mse_pwd[name] = np.mean((ca_pwd.mean(axis=0) - ref_pwd_mean) ** 2) 
        mse_rg[name] = np.mean((ca_rg_mean - ref_rg_mean) ** 2)
        mse_contact[name] = np.mean((ca_contacts - ref_contacts) ** 2)
        
        mae_pwd[name] = np.mean(np.abs(ca_pwd.mean(axis=0) - ref_pwd_mean))
        mae_rg[name] = np.mean(np.abs(ca_rg_mean - ref_rg_mean))
        mae_contact[name] = np.mean(np.abs(ca_contacts - ref_contacts))


    return mse_pwd, mse_rg, mse_contact, mae_pwd, mae_rg, mae_contact


def js_pwd(ca_coords_dict, ref_key='target', n_bins=50, pwd_offset=3, weights=None, kl=False):
    # n_bins = 50 follows idpGAN
    # k=3 follows 2for1
    
    ca_pwd = {
        k: pairwise_distance_ca(v, k=pwd_offset) for k, v in ca_coords_dict.items()
    }   # (B, D)
    
    if weights is None:
        weights = {}
    weights.update({k: np.ones(len(v)) for k,v in ca_coords_dict.items() if k not in weights})
        
    d_min = ca_pwd[ref_key].min(axis=0) # (D, )
    d_max = ca_pwd[ref_key].max(axis=0)
    ca_pwd_binned = {
        k: np.apply_along_axis(lambda a: np.histogram(a[:-2], bins=n_bins, weights=weights[k], range=(a[-2], a[-1]))[0]+PSEUDO_C, 0, 
                            np.concatenate([v, d_min[None], d_max[None]], axis=0))
        for k, v in ca_pwd.items()
    }   # (N_bins, D)-> (N_bins * D, )
    # js divergence per channel and average
    if kl:
        results = {k: kl_div(v, ca_pwd_binned[ref_key]).mean() 
                for k, v in ca_pwd_binned.items() if k != ref_key}
    else:
        results = {k: distance.jensenshannon(v, ca_pwd_binned[ref_key], axis=0).mean() 
                    for k, v in ca_pwd_binned.items() if k != ref_key}
    results[ref_key] = 0.0
    results = {k: np.around(v, decimals=4) for k, v in results.items()}
    return results


def js_tica(ca_coords_dict, ref_key='target', n_bins=50, lagtime=20, return_tic=True, weights=None):
    """Coordinate -> pairwise distance (PwD) -> TIC x2 -> JS
    """
    # n_bins = 50 follows idpGAN
    ca_pwd = {
        k: pairwise_distance_ca(v) for k, v in ca_coords_dict.items()
    }   # (B, D)
    estimator = TICA(dim=2, lagtime=lagtime).fit(ca_pwd[ref_key])
    tica = estimator.fetch_model()
    # dimension reduction into 2D
    ca_dr2d = {  
        k: tica.transform(v) for k, v in ca_pwd.items()
    }
    if weights is None: weights = {}
    weights.update({k: np.ones(len(v)) for k,v in ca_coords_dict.items() if k not in weights})
    
    d_min = ca_dr2d[ref_key].min(axis=0) # (D, )
    d_max = ca_dr2d[ref_key].max(axis=0)
    ca_dr2d_binned = {
        k: np.apply_along_axis(lambda a: np.histogram(a[:-2], bins=n_bins, weights=weights[k], range=(a[-2], a[-1]))[0]+PSEUDO_C, 0, 
                            np.concatenate([v, d_min[None], d_max[None]], axis=0))
                for k, v in ca_dr2d.items()
    }   # (N_bins, 2) 
    results = {k: distance.jensenshannon(v, ca_dr2d_binned[ref_key], axis=0).mean() 
                for k, v in ca_dr2d_binned.items() if k != ref_key}
    results[ref_key] = 0.0
    results = {k: np.around(v, decimals=4) for k, v in results.items()}
    if return_tic:
        return results, ca_dr2d
    return results


def js_rg(ca_coords_dict, ref_key='target', n_bins=50, weights=None, return_rg=False, kl=False):
    ca_rg = {
        k: radius_of_gyration(v) for k, v in ca_coords_dict.items()
    }   # (B, )
    if weights is None:
        weights = {}
    weights.update({k: np.ones(len(v)) for k,v in ca_coords_dict.items() if k not in weights})
        
    d_min = ca_rg[ref_key].min() # (1, )
    d_max = ca_rg[ref_key].max()
    ca_rg_binned = {
        k: np.histogram(v, bins=n_bins, weights=weights[k], range=(d_min, d_max))[0]+PSEUDO_C 
            for k, v in ca_rg.items()
    }   # (N_bins, )
    # print("ca_rg_binned shape", {k: v.shape for k, v in ca_rg_binned.items()})
    if kl:
        results = {k: kl_div(v, ca_rg_binned[ref_key]).mean() 
                for k, v in ca_rg_binned.items() if k != ref_key}
    else:
        results = {k: distance.jensenshannon(v, ca_rg_binned[ref_key], axis=0).mean() 
                for k, v in ca_rg_binned.items() if k != ref_key}
    
    results[ref_key] = 0.0
    results = {k: np.around(v, decimals=4) for k, v in results.items()}
    if return_rg:
        return results, ca_rg
    return results


def load_apo_targets(apo_base: Path, apo=True):
    if not isinstance(apo_base, Path): apo_base = Path(apo_base)
    df = pd.read_csv(apo_base/ "splits" / "apo.csv")
    structure_dir = apo_base / "structures"
    if apo:
        names = df["name"].to_list()
    else:
        names = df["holo"].to_list()
    targets = []
    for name in names:
        prefix2 = name[:2]
        targets.append(structure_dir / prefix2 / name)
    return targets

def load_codnas_targets(apo_base: Path, apo=True):
    if not isinstance(apo_base, Path): apo_base = Path(apo_base)
    df = pd.read_csv(apo_base/ "splits" / "codnas.csv")
    structure_dir = apo_base / "structures"
    if apo:
        names = df["name"].to_list()
    else:
        names = df["other"].to_list()
    targets = []
    for name in names:
        prefix2 = name[:2]
        targets.append(structure_dir / prefix2 / name)
    return targets

def load_atlas_targets(atlas_base: Path, split='test', return_names=False):
    assert split in ['train', 'val', 'test', 'all'], f"split should be in ['train', 'val', 'test'], but got {split}"
    if split == 'all':
        df = pd.read_csv(atlas_base / "splits" / "atlas.csv")
    else:
        df = pd.read_csv(atlas_base / "splits" / f"atlas_{split}.csv")
    names = df["name"].to_list()
    if return_names:
        return names
        
    targets = []
    for name in names:
        targets.append(atlas_base / "processed" / f"{name}.npz")
    return targets

def load_atlas_processed(path: Path):
    if not isinstance(path, Path): 
        path = Path(path)
    data = {}
    data_dict = dict(np.load(path, allow_pickle=True))
    data.update({
        # str
        'accession_name': path.stem,
        'sequence': data_dict['sequence'][0].decode("utf-8"),
        # ndarray
        'trajectory': data_dict['all_atom_positions'],  # (301, L, 37, 3)
        'trajectory_mask': data_dict['all_atom_mask'],  # (301, L, 37),
        'residue_index': data_dict['residue_index'],    # (L, )
    })
    return data
    
def load_mdcath_processed(path: Path, n_models_per_traj=100):
    if not isinstance(path, Path): 
        path = Path(path)
    data = {}
    data_dict = dict(np.load(path, allow_pickle=True))
    bb_traj = data_dict['backbone_positions']   # (T, L, 4, 3)
    traj_lens = data_dict['traj_lens']
    tl_cumsum = traj_lens.cumsum()
    tmp_end = bb_traj[tl_cumsum - 1]
    tl_cumsum[-1] = 0
    tmp_start = bb_traj[tl_cumsum]
    start_end_pos = np.concatenate(
        [tmp_end, tmp_start], axis=0
    )
    assert start_end_pos.shape[0] == 2 * len(traj_lens)
    traj = np.zeros((start_end_pos.shape[0], start_end_pos.shape[1], 37, 3))
    traj[:, :, :4] = start_end_pos

    data.update({
        # str
        'accession_name': path.stem,
        'aatype': data_dict['aatype'],  # (L, )
        'trajectory_lens': data_dict['traj_lens'],  # (5, ) 
        # ndarray
        'trajectory': traj,  # (2T, L, 37, 3)
        'residue_mask': data_dict['mask'],  # (L, ),
        'residue_index': data_dict['residue_index'],    # (L, )
    })
    return data

def load_ped_targets(ped_base: Path, load_ensemble_dir=False, return_df=False):
    if not isinstance(ped_base, Path): ped_base = Path(ped_base)
    df = pd.read_csv(ped_base / "ped_114.csv")
    names = df["entry_id"].to_list()
    if load_ensemble_dir:
        _paths = (ped_base / "bb").iterdir()
    else:
        _paths = (ped_base / "singleton").iterdir()
    targets = [f for f in _paths if f.stem in names]
    if return_df:
        return targets, df
    return targets


def write_pdb_string(pdb_string: str, save_to: str):
    """Write pdb string to file"""
    with open(save_to, 'w') as f:
        f.write(pdb_string)
        
def read_pdb_to_string(pdb_file):
    """Read PDB file as pdb string. Convenient API"""
    with open(pdb_file, 'r') as fi:
        pdb_string = ''
        for line in fi:
            if line.startswith('END') or line.startswith('TER') \
                    or line.startswith('MODEL') or line.startswith('ATOM'):
                pdb_string += line
        return pdb_string

def merge_pdbfiles(input: Path, save_to: Path, verbose=True):
    """ordered merging process of pdbs"""
    if isinstance(input, Path):
        pdb_files = [f for f in input.iterdir() if f.suffix == '.pdb']
    elif isinstance(input, list):
        pdb_files = input
    else:
        raise ValueError(f"Unrecognized input type: {type(input)}")

    assert len(pdb_files) > 0
        
    save_to.parent.mkdir(parents=True, exist_ok=True)
    
    model_number = 0
    pdb_lines = []
    if verbose: 
        _iter = tqdm(pdb_files, desc='Merging PDBs')
    else:
        _iter = pdb_files
    
    for pdb_file in _iter:
        with open(pdb_file, 'r') as pdb:
            lines = pdb.readlines()
        single_model = True
        
        for line in lines: 
            if line.startswith('MODEL') or line.startswith('ENDMDL'):
                single_model = False
                break
        
        if single_model: # single model
            model_number += 1
            pdb_lines.append(f"MODEL     {model_number}")
            for line in lines: 
                if line.startswith('TER') or line.startswith('ATOM'): 
                    pdb_lines.append(line.strip())
            pdb_lines.append("ENDMDL")
        else:        # multiple models
            for line in lines:
                if line.startswith('MODEL'):
                    model_number += 1
                    if model_number > 1:
                        pdb_lines.append("ENDMDL")
                    pdb_lines.append(f"MODEL     {model_number}")
                elif line.startswith('END'):
                    continue
                elif line.startswith('TER') or line.startswith('ATOM'): 
                    pdb_lines.append(line.strip())
    pdb_lines.append('ENDMDL')
    pdb_lines.append('END')
    pdb_lines = [_line.ljust(80) for _line in pdb_lines]
    pdb_str = '\n'.join(pdb_lines) + '\n'
    with open(save_to, 'w') as fo:
        fo.write(pdb_str)
    if verbose:
        print(f"Merged {len(pdb_files)} PDB files into {save_to} with {model_number} models.")


def split_pdbfile(input, output_dir=None, sep='_', verbose=True):
    """Split a PDB file into multiple PDB files in output_dir.
    Preassume that each model is wrapped by 'MODEL' and 'ENDMDL'.
    """
    assert input.exists() and input.suffix == '.pdb', f"File {input} does not exist or not a .pdb file."

    
    if output_dir is not None:  # also dump to output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    
    i = 0
    pdb_strs = []
    pdb_string = ''
    with open(input, 'r') as fi:
        # pdb_string = ''
        for line in fi:
            if line.startswith('MODEL'):
                pdb_string = ''
            elif line.startswith('ATOM') or line.startswith('TER'):
                pdb_string += line
            elif line.startswith('ENDMDL') or line.startswith('END'):
                if pdb_string == '': continue
                pdb_string += 'END\n'
                if output_dir is not None:
                    _save_to = output_dir / f'{input.stem}{sep}{i}.pdb'
                    with open(_save_to, 'w') as fo:
                        fo.write(pdb_string)
                pdb_strs.append(pdb_string)
                pdb_string = ''
                i += 1
            else:
                if verbose:
                    print(f"Warning: line '{line}' is not recognized. Skip.")
    if verbose:
        print(f">>> Split pdb {input} into {i}/{len(pdb_strs)} structures.")
    return pdb_strs


def merge_all_targets_from_dir(input: Path, target_list, **kwargs):
    for target in tqdm(target_list, desc="Merging all the targets..."):
        tmp = list(input.glob(f"{target}*.pdb"))
        merge_pdbfiles(tmp, save_to=input / f"{target}.pdb", **kwargs)
        print(f">>> Unlink/remove the split files ({len(tmp)})")
        for p in tmp: p.unlink()

def load_seq_from_pdb(pdb_path: Path):
    prot = protein.from_pdb_file(pdb_path)
    return ''.join([restypes_with_x[i] for i in prot.aatype])

def desc_seq_to_fasta(desc_seq: dict, save_to: Path):
    with open(save_to, 'w') as f:
        for k, v in desc_seq.items():
            f.write(f">{k}\n{v}\n")
