import os
from typing import *
import tempfile
import argparse
from pathlib import Path
from glob import glob 

from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import mdtraj
from scipy.spatial import distance
from deeptime.decomposition import TICA
 
from slm.models.utils import load_coords
from slm.utils.plot_utils import scatterplot_2d
from slm.utils.eval_utils import (
    js_tica, 
    js_pwd,
    js_rg,
    validity,
    bonding_validity,
    split_pdbfile,
)
from slm.utils.geo_utils import squared_deviation
from slm.utils.tm_utils import tm_n_ensemble, tm_diversity


_device = "cuda" if torch.cuda.is_available() else "cpu"


def bpti_evaluation(
    preds: list, 
    target: Path, 
    output_dir: Path=None, 
    special=None, 
    special_name=None, 
    save_plot=True,
):
    """Evaluate the ensemble of protein structures.
    """
    d_coords = dict()
    for pred in preds:
        tmp = load_coords(pred)
        if len(tmp) > 100:
            tmp = tmp[:100]
        d_coords[pred.stem] = tmp
    
    assert len(d_coords) > 0, f"No prediction found from {preds}. Exit."
    d_coords['target'] = load_coords(target, max_n_model=-1)
    
    if special is not None:
        special_name = special_name if special_name else  "MISC"
        d_coords[special_name] = load_coords(special)
    
    for k, v in d_coords.items():
        print(f"Loaded {v.shape[0]} structures from {k}...")

    js, d_tica = js_tica(d_coords, ref_key='target', lagtime=500, return_tic=True)    # lagtime
    js_pairwise_dist = js_pwd(d_coords, ref_key='target')
    js_rg_dist = js_rg(d_coords, ref_key='target')
    val_clash = validity(d_coords)
    val_bond = bonding_validity(d_coords)

    # pop out the target and kinetic clusters
    for d in [d_coords, js, js_pairwise_dist, js_rg_dist, val_clash, val_bond]:
        d.pop('target')
        if special_name in d:
            d.pop(special_name)
    print("JS divergence of pairwise distance:", js_pairwise_dist)
    print("JS divergence of TICA:", js)
    
    print("JS divergence of RG:", js_rg_dist)
    print("Validity of clash:", val_clash)
    print("Validity of bonding:", val_bond)
    # save to csv
    names = list(js.keys())
    results = {
        "name": names,    # methods
        "js_pwd": [js_pairwise_dist[k] for k in names],
        "js_tica": [js[k] for k in names],
        "js_rg": [js_rg_dist[k] for k in names],
        "val_clash": [val_clash[k] for k in names],
        "val_bond": [val_bond[k] for k in names],
    }
    if output_dir is not None:
        df = pd.DataFrame(results)
        df.to_csv(output_dir / "js_metrics.csv", index=False)
    
    if save_plot:
        assert output_dir is not None, "Please specify the output directory."
        output_dir.mkdir(parents=True, exist_ok=True)
        save_to = output_dir / f"tica2d_all.png"
        scatterplot_2d(
            d_tica,
            save_to=save_to,  
            ref_key='target',
            n_max_point=1000,
            pop_ref=False,
            xylim_key=special_name,
            # remarks=f"JS={js['predict']:.3f}",
        )
    return results


def bpti_rmsd_clusters(
    preds: list[Path],
    path_to_clusters: Path,
    output_dir: Path,
    use_tm=True,
):
    """Calculate the best RMSD of the predicted structures to the target structures.
    """

    if use_tm:
        best_tm_rmsd_div = {}
        for pred in preds:
            print("\n>>> evaluation run:", pred.stem, '\n')
            with tempfile.TemporaryDirectory() as tmpdir:
                split_pdbfile(pred / 'bpti.pdb', Path(tmpdir))
                best_tm_list, best_rmsd_list = tm_n_ensemble(Path(tmpdir), path_to_clusters)
                tm_div = tm_diversity(Path(tmpdir))
                best_tm_rmsd_div[pred.name] = [np.mean(best_tm_list), np.mean(best_rmsd_list), tm_div]
            # row as methods, column as clusters
        print(best_tm_rmsd_div)
        df = pd.DataFrame(best_tm_rmsd_div, index=['TM-ens', 'RMSD-ens', 'TM-div']).T
        print(df)
        df.to_csv(output_dir / "bpti_tm_rmsd_div.csv")

    else:
        clusters = []
        for i in range(1, 6):
            clusters.append(load_coords(path_to_clusters / f"bpti_{i}.pdb"))
        clusters = np.stack(clusters)  # (5, L, 3)
        L = clusters.shape[1]
        N = clusters.shape[1] * clusters.shape[2]
        d_coords = dict()
        for pred in preds:
            tmp = load_coords(pred, max_n_model=100)
            d_coords[pred.stem] = tmp
            print(f"{pred.stem}: {tmp.shape[0]} loaded...")
        
        map_to_tensor = lambda x: torch.tensor(x, device=_device)
        clusters = map_to_tensor(clusters).view(clusters.shape[0], -1, 3)  # flatten to (B, N, 3)
        for k, v in d_coords.items():
            d_coords[k] = map_to_tensor(v).view(v.shape[0], -1, 3)  # flatten to (B, N, 3)

        # each d_coords[key]: (N, 3)
        # clusters: (K, 3)
        best_rmsd = {}
        for k, v in d_coords.items():
            v_repeat = v.repeat(clusters.shape[0], 1, 1) # (1000 *> 5, L, 3)
            clusters_repeat = clusters.repeat_interleave(v.shape[0], dim=0) # (1000 *> 5, L, 3)
            rmsd = _squared_deviation(v_repeat, clusters_repeat, reduction='rmsd')
            rmsd = rmsd.view(v.shape[0], clusters.shape[0])

            best_rmsd[k] = list(rmsd.min(dim=0).values.detach().cpu().numpy())  # (K, )
            best_rmsd_id = list(rmsd.argmin(dim=0).detach().cpu().numpy())  # (K, )
            print(f"Best RMSD for {k}: {','.join(best_rmsd[k])}")
            print(f"Best RMSD ID for {k}: {','.join(best_rmsd_id)}")
        
        # row as methods, column as clusters
        df = pd.DataFrame(best_rmsd).T
        print(df)
        df.to_csv(output_dir / "bpti_rmsd_clusters.csv", index=False)


def get_argparser():
    parser = argparse.ArgumentParser(description="Evaluate the ensemble of protein structures.")
    parser.add_argument("samples", type=str, help="Path to the data directory.")
    parser.add_argument("--reference", type=str, default="data/ground_truth/apo", help="Path to reference data.")
    parser.add_argument("--batch_samples", action="store_true", help="Batch evaluation of samples.")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_argparser()
    working_dir = Path(args.samples)
    reference_dir = Path(args.reference)
    assert working_dir.is_dir(), f"Invalid directory {working_dir}"
    assert reference_dir.is_dir(), f"Invalid directory {reference_dir}"

    # pre-specified paths
    bpti_target = reference_dir / "bpti.npy"    # ca-only array parsed from desres trajectory
    bpti_clus = reference_dir / "bpti_5"    # 5 pdb clusters of bpti downloaded from science 2010 paper

    if args.batch_samples:
        # batch evaluation
        runs = [f for f in working_dir.iterdir() if f.is_dir() and not f.name.startswith('asset')]
    else:
        runs = [working_dir]
        working_dir = working_dir.parent
    print(f"Found {len(runs)} runs in {working_dir}: {[run.stem for run in runs]}")

    output_dir = working_dir / 'asset'
    output_dir.mkdir(parents=True, exist_ok=True)
        
    ##############################################################################
    bpti_rmsd_clusters(runs, bpti_clus, output_dir)
    bpti_evaluation(
        preds=runs,
        target=bpti_target,
        output_dir=output_dir,
        special=bpti_clus,
        special_name="kinetic_clusters",
        save_plot=True,
    )
    
