"""
Dump protein encodings from ESM3 model.
"""
from pathlib import Path
import os, sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from functools import partial
import time
import shutil
import pickle

import torch

# from esm repo quickstart
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein
from esm.utils.constants import esm3 as C

from slm.models.utils import protseq_to_data, pdb_to_data
from slm.utils.eval_utils import load_atlas_processed, load_mdcath_processed
from slm.utils.residue_constants import restypes_with_x
 

def decode_seq(aatype):
    return ''.join([restypes_with_x[i] for i in aatype])

def load_pickled_protein_dict(path: Path, model: ESM3):
    # Load pickled protein
    with open(path, 'rb') as f:
        data_dict = pickle.load(f)
    
    data_dict = protseq_to_data(
        sequence=decode_seq(data_dict['aatype']),
        model=model,
        coordinates=torch.tensor(data_dict['atom_positions']),
    )# -> L+2
    # data_dict['mask'] = data_dict['atom_mask']    # L
    return data_dict    # L, *


def parse_clusters(tsv_file):
    # two column, the first column is the cluster name, second columns
    # contains the records in that cluster
    # out: a dictionary, key is the cluster name, value is a list of records
    with open(tsv_file, 'r') as f:
        lines = f.readlines()
    clusters = {}
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) != 2: continue
        cluster, record = parts
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(record)
    return clusters


if __name__ == '__main__':
    # This will download the model weights and instantiate the model on your machine.
    print("Instantiate esm3 model...")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda").eval()
    # to float
    model = model.float()
    
    tmpdir = Path("/tmp")
    # timestamp 
    this_tmpdir = tmpdir / str(int(time.time()))
    this_tmpdir.mkdir(exist_ok=True)
    print(f"Created temporary directory {this_tmpdir}")
    
    load_fn_pkl = partial(load_pickled_protein_dict, model=model)
    load_fn_pdb = partial(pdb_to_data, model=model, encode_only=True)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    suffix = sys.argv[3]
    assert input_dir.is_dir(), f"Invalid input directory: {input_dir}"
    assert not output_dir.exists(), f"Existing output directory: {output_dir}"
    assert suffix in ['pdb', 'pkl'], f"Invalid suffix: {suffix}"
    print(f"Output directory: {output_dir}")

    cnt = 0

    paths = list(input_dir.glob(f'**/*.{suffix}'))
    print(f"Found {len(paths)} *.{suffix} files in {input_dir}")
    cnt = 0
    for p in tqdm(paths, desc="Get ESM3 embedding from pdb files..."):
        base = p.name
        save_to = this_tmpdir / base.replace(f'.{suffix}', '.pth')
        if os.path.exists(save_to): 
            a = torch.load(save_to)
            if 'coordinates' in a: 
                continue
        if suffix == 'pkl':
            d = load_fn_pkl(p)
        elif suffix == 'pdb':
            d = load_fn_pdb(p)
        else:
            raise ValueError(f"Invalid suffix: {suffix}")

        torch.save(d, save_to)
        cnt += 1
        if cnt == 1:
            print({k:v.shape if torch.is_tensor(v) else v for k,v in d.items()})
    print(f"Finished written {cnt} .pth files")
    # copy to the output directory
    shutil.copytree(this_tmpdir, output_dir)