import os
from typing import *
import argparse
from pathlib import Path
from glob import glob 

from tqdm import tqdm
import numpy as np
import pandas as pd

from slm.models.utils import load_coords
from slm.utils.eval_utils import (
    idp_metrics,
    load_ped_targets,
)

def idp_evaluation(
    preds: Path, 
    target: Path, 
    output_dir: Path, 
    target_name: str,
):
    """ Evaluate the ensemble of protein structures.
    Metrics:
        TM-score: like apo/holo
        PCA: like bpti
    """
    output_dir = output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_to = output_dir / f"{target_name}.png"
    d_coords = dict()
    targets = target.iterdir()
    _ens_id = 1
    _all_target = []
    
    for ensemble in targets:
        _all_target.append(load_coords(ensemble, verbose=False))
    
    # concat of targets
    all_target = np.concatenate([v for v in _all_target], axis=0)    # (N_all, )
    d_coords['target'] = all_target

    for name, pred in preds.items():
        d_coords[name] = load_coords(pred, verbose=False)

    return idp_metrics(d_coords, ref_key='target')


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
    target_ensembles = load_ped_targets(reference_dir, load_ensemble_dir=True)
    all_target_mae = {
        "pwd": [],
        "rg": [],
        "contact": [],
    }

    for target_name in tqdm(target_ensembles, desc="Loop over targets..."):
        preds = {
            run.stem: run / f"{target_name.stem}.pdb" for run in runs 
                if (run / f"{target_name.stem}.pdb").exists()
        }
            
        _, _, _, mae_pwd, mae_rg, mae_contact = idp_evaluation(
            preds=preds,
            target=target_name,
            output_dir=output_dir,
            target_name=target_name.stem,
        )

        all_target_mae["pwd"].append(mae_pwd)
        all_target_mae["rg"].append(mae_rg)
        all_target_mae["contact"].append(mae_contact)

    # avg across all targets
    names = list(all_target_mae["pwd"][0].keys())

    out_dict = {
        "name": names,

        "mae_pwd_mean": [np.mean([d[k] for d in all_target_mae["pwd"]]) for k in names],
        "mae_pwd_median": [np.median([d[k] for d in all_target_mae["pwd"]]) for k in names],
        "mae_rg_mean": [np.mean([d[k] for d in all_target_mae["rg"]]) for k in names],
        "mae_rg_median": [np.median([d[k] for d in all_target_mae["rg"]]) for k in names],
        "mae_contact_mean": [np.mean([d[k] for d in all_target_mae["contact"]]) for k in names],
        "mae_contact_median": [np.median([d[k] for d in all_target_mae["contact"]]) for k in names],
    }

    # save to csv, row as methods, column as rg mae 
    df = pd.DataFrame(out_dict)
    print(df)
    df.to_csv(output_dir / "idp_metrics.csv", index=False)

