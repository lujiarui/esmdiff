import os
from pathlib import Path
import subprocess
import logging
from glob import glob

import numpy as np

def exec_subproc(cmd, timeout: int = 100) -> str:
    """Execute the external docking-related command.
    """
    if not isinstance(cmd, str):
        cmd = ' '.join([str(entry) for entry in cmd])
    try:
        rtn = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, timeout=timeout)
        out, err, return_code = str(rtn.stdout, 'utf-8'), str(rtn.stderr, 'utf-8'), rtn.returncode
        if return_code != 0:
            logging.error('[ERROR] Execuete failed with command "' + cmd + '", stderr: ' + err)
            raise ValueError('Return code is not 0, check input files')
        return out
    except subprocess.TimeoutExpired:
        logging.error('[ERROR] Execuete failed with command ' + cmd)
        raise ValueError(f'Timeout({timeout})')


def parse_tmscore_output(out: str) -> dict:
    """Parse TM-score output.
    Args:
        out: str in utf-8 encoding from stdout of TM-score.
    """
    start = out.find('RMSD')
    end = out.find('rotation')
    out = out[start:end]
    try:
        rmsd, _, tm, _, gdt_ts, gdt_ha, _, _ = out.split('\n')
    except:
        raise ValueError('TM-score output format is not correct:\n{out}')
    
    rmsd = float(rmsd.split('=')[-1])
    tm = float(tm.split('=')[1].split()[0])
    gdt_ts = float(gdt_ts.split('=')[1].split()[0])
    gdt_ha = float(gdt_ha.split('=')[1].split()[0])
    return {'rmsd': rmsd, 'tm': tm, 'gdt_ts': gdt_ts, 'gdt_ha': gdt_ha}


def tmscore(model_pdb_path, native_pdb_path, return_dict=False):
    """Calculate TM-score between two PDB files, each containing 
        only one chain and only CA atoms.
    Args:
        model_pdb_path (str): path to PDB file 1 (model)
        native_pdb_path (str): path to PDB file 2 (native)
    Returns:
        dict of scores('rmsd': float, 'tm': float, 'gdt_ts': float, 'gdt_ha': float)
    """
    out = exec_subproc(['TMscore', '-seq', model_pdb_path, native_pdb_path])
    res = parse_tmscore_output(out)
    if return_dict:
        return res
    return res['tm']
    

def tm_ensemble(predict_dir: Path, t1: Path, t2: Path):
    """Calculate diversity among pairs of modeled protein structures for each score.
    
    Returns:
        dict of scores('rmsd': float, 'tm': float, 'gdt_ts': float, 'gdt_ha': float, 'lddt': float)
    """
    if not isinstance(predict_dir, Path):
        predict_dir, t1, t2 = map(Path, [predict_dir, t1, t2])
    tm_list_1 = []
    tm_list_2 = []

    candidate_list = []
    for path in predict_dir.iterdir():
        accession_name = path.stem.replace('_', '.')    # historical reason
        if accession_name.startswith(t1.stem) or accession_name.startswith(t2.stem):
            candidate_list.append(path)
    assert len(candidate_list) > 0, f"No candidate files found for {t1.stem} or {t2.stem} under {predict_dir}"
    for path in candidate_list:
        tm_list_1.append(tmscore(path, t1))
        tm_list_2.append(tmscore(path, t2))

    tm_score = 0.5 * max(tm_list_1) + 0.5 * max(tm_list_2)
    # rmsd_score = 0.5 * min(rmsd_list_1) + 0.5 * min(rmsd_list_2)
    
    return tm_score

def tm_n_ensemble(predict_dir: Path, ensemble_paths: list[Path], max_n_model: int = 100):
    """Calculate diversity among pairs of modeled protein structures for each score.
    
    Returns:
        dict of scores('rmsd': float, 'tm': float, 'gdt_ts': float, 'gdt_ha': float, 'lddt': float)
    """
    if isinstance(ensemble_paths, Path):
        ensemble_paths = list(ensemble_paths.iterdir())
        ensemble_paths = sorted(ensemble_paths, key=lambda x: int(x.stem[-1]))
    if not isinstance(predict_dir, Path):
        predict_dir, ensemble_paths = map(Path, [predict_dir] + ensemble_paths)
    best_tm_list = []
    best_rmsd_list = []
    best_tm_ids = []
    best_rmsd_ids = []
    N = len(ensemble_paths)

    candidate_list = []
    for path in predict_dir.iterdir():
        candidate_list.append(path)
    assert len(candidate_list) > 0, f"No candidate files found under {predict_dir}"
    if len(candidate_list) > max_n_model:
        print(f"Downsample {len(candidate_list)} models to {max_n_model} models.")
        candidate_list = np.random.choice(candidate_list, max_n_model, replace=False)

    for i in range(N):
        tmp = []
        tmp_rmsd = []
        for path in candidate_list:
            tm = tmscore(path, ensemble_paths[i], return_dict=True)
            tmp.append(tm['tm'])
            tmp_rmsd.append(tm['rmsd'])
        best_tm_list.append(max(tmp))
        best_rmsd_list.append(min(tmp_rmsd))
        best_tm_ids.append(np.argmax(tmp))
        best_rmsd_ids.append(np.argmin(tmp_rmsd))
    
    print(f"\nReport for {predict_dir.parent.stem}:\n")
    print("Best-TM-score", ','.join(map(str, best_tm_list)))
    print("Best-RMSD", ','.join(map(str, best_rmsd_list)))
    print("Best-TM-ids", ','.join(map(str, best_tm_ids)))
    print("Best-RMSD-ids", ','.join(map(str, best_rmsd_ids)))
    
    print("\n")
    print("TM-ensemble", np.mean(best_tm_list))
    print("RMSD-ensemble", np.mean(best_rmsd_list))
    
    return best_tm_list, best_rmsd_list

def tm_diversity(predict_dir: Path):
    """Calculate diversity among pairs of modeled protein structures for each score.
    
    Returns:
        dict of scores('rmsd': float, 'tm': float, 'gdt_ts': float, 'gdt_ha': float, 'lddt': float)
    """
    if not isinstance(predict_dir, Path):
        predict_dir = Path(predict_dir)

    candidate_list = list(predict_dir.iterdir())

    tm_list = []
    for i in range(len(candidate_list)-1):
        for j in range(i+1, len(candidate_list)):
            tm_list.append(tmscore(candidate_list[i], candidate_list[j]))

    tm_score = np.mean(tm_list)
    return tm_score
