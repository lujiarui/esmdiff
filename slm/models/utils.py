import os
from glob import glob
from pathlib import Path
from typing import Optional
from collections import OrderedDict

import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import biotite.structure as struct
from biotite.structure.io.pdb import PDBFile

import esm

from esm.models.vqvae import (
    StructureTokenDecoder,
    StructureTokenEncoder,
)
from esm.tokenization import StructureTokenizer
from esm.utils.decoding import decode_structure
from esm.utils import encoding, decoding, structure
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, GenerationConfig


def round_embeddings_to_clusters(z, cluster_centers):
    """
    Args:
        z: condition a la latent embedding [B, L, D]
        cluster_centers: [K, D]
    """
    z_flat = z.view(-1, z.size(-1))
    dist = torch.sum(z_flat ** 2, dim=1, keepdim=True) + \
            torch.sum(cluster_centers ** 2, dim=1) - \
            2 * torch.matmul(z_flat, cluster_centers.t())    # (BL, K)
    # Get the encoding that has the min distance
    encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1) # (BL, 1)
    inds = encoding_inds.detach().clone().view(z.shape[:-1]) # (B, L)
    # Negative distance may be used for cross entropy loss
    dist = dist.view(z.shape[:-1] + cluster_centers.shape[0]) # (B, L, K)
    return inds, dist # dist keep grad


def protstr_tokens_to_coords(
    structure_tokens: torch.Tensor,
    structure_decoder: StructureTokenDecoder,
    structure_tokenizer: StructureTokenizer,
    sequence: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    # https://github.com/evolutionaryscale/esm/blob/main/esm/utils/decoding.py#L139
    is_singleton = len(structure_tokens.size()) == 1
    if is_singleton:
        structure_tokens = structure_tokens.unsqueeze(0)
    else:
        raise ValueError(
            f"Only one structure can be decoded at a time, got structure tokens of shape {structure_tokens.size()}"
        )
    decoding._bos_eos_warn("Structure", structure_tokens[0], structure_tokenizer)

    decoder_output = structure_decoder.decode(structure_tokens)
    bb_coords: torch.Tensor = decoder_output["bb_pred"][
        0, 1:-1, ...
    ]  # Remove BOS and EOS tokens
    bb_coords = bb_coords.detach().cpu()

    if "plddt" in decoder_output:
        plddt = decoder_output["plddt"][0, 1:-1]
        plddt = plddt.detach().cpu()
    else:
        plddt = None

    if "ptm" in decoder_output:
        ptm = decoder_output["ptm"]
    else:
        ptm = None

    chain = ProteinChain.from_backbone_atom_coordinates(bb_coords, sequence=sequence)
    chain = chain.infer_oxygen()
    return torch.tensor(chain.atom37_positions), plddt, ptm
  
def prot_tensor_to_dict(prot):
    data_dict = {
        "sequence_tokens": prot.sequence,
        "structure_tokens": prot.structure,
        "ss8_tokens": prot.secondary_structure,
        "sasa_tokens": prot.sasa,
        "function_tokens": prot.function,
        "residue_annotation_tokens": prot.residue_annotations,
        "structure_coords": prot.coordinates,    
    }
    # Create batch dimension
    data_dict = {
        k: v.unsqueeze(0) for k,v in data_dict.items() if v is not None
    }
    return data_dict

@torch.no_grad()
def pdb_to_data(pdb_file, **kwargs):
    prot = ESMProtein.from_pdb(pdb_file)
    return protseq_to_data(sequence=prot.sequence, coordinates=prot.coordinates, **kwargs)
        

@torch.no_grad()
def protseq_to_data(
    sequence: str, 
    model: ESM3, 
    # device: torch.device = torch.device('cpu'), 
    coordinates: torch.Tensor | None = None,
    encode_only: bool = False,
    mask_ids: Optional[list] = None,
    filled_ids: Optional[list] = None,
    total_size: Optional[int] = None,
):
    # in: sequence
    # out: z, structure tokens, mask, ...
    if mask_ids is not None:
        sequence = list(sequence)
        for idx in mask_ids:
            assert 0 <= idx < len(sequence), f"Invalid mask index {idx} for sequence of length {len(sequence)}"
            sequence[idx] = '_'
            coordinates[idx] = float("Inf")
        sequence = ''.join(sequence)
    elif filled_ids is not None:
        # sanity check
        assert total_size is not None, "total_size must be provided when fill_ids is not None"
        assert all(0 <= idx < total_size for idx in fill_ids), f"Invalid fill index {fill_ids} for sequence of length {total_size}"
        _seq = ['_'] * total_size
        _coord = coordinates.new_ones(total_size, 37, 3) * float("Inf")
        for idx in filled_ids:
            _seq[idx] = sequence[idx]
            _coord[idx] = coordinates[idx]
        sequence = ''.join(_seq)
        coordinates = _coord

    prot = ESMProtein(sequence=sequence, coordinates=coordinates)
    gt_tokens = model.encode(prot)
    if encode_only:
        return {
            # [L, ]
            "sequence_tokens": gt_tokens.sequence,
            "structure_tokens": gt_tokens.structure, # None if no coordinates input, is tensor during training 
            # [L, ]
            "sequence": sequence, 
            "coordinates": coordinates,
        }

    protseq = ESMProtein(sequence=sequence)
    input_tokens = model.encode(protseq)

    kw_tokens = prot_tensor_to_dict(input_tokens)
    outs = model(**kw_tokens)
    data_dict = {
        # [L+2, *]
        "embeddings": outs.embeddings.squeeze(0),
        "structure_logits": outs.structure_logits.squeeze(0),
        "sequence_tokens": input_tokens.sequence,
        "sequence": sequence,   # [L, ]
        "coordinates": coordinates, # [L, 37, 3]  
        # label
        "structure_tokens": gt_tokens.structure, # None if no coordinates input, the label during training 
       
    }
    return data_dict    

def encode_decode(
    model: ESM3, 
    pdb: Path | str,   # single model
):
    from esm.utils.structure.protein_chain import ProteinChain

    if isinstance(pdb, str) or isinstance(pdb, Path):
        pdb_file = Path(pdb)
        assert pdb_file.exists(), f"File {pdb_file} does not exist."
        prot = ESMProtein.from_pdb(pdb_file)
    elif isinstance(pdb, ESMProtein):
        prot = pdb
    elif isinstance(pdb, ProteinChain):
        prot = ESMProtein.from_protein_chain(pdb)
    else:
        raise ValueError(f"Invalid input type: {type(pdb)}: {pdb}")

    coords = prot.coordinates
    tokens = model.encode(prot)
    coords_pred, plddt, ptm = decode_structure(
        structure_tokens=tokens.structure,
        structure_decoder=model.get_structure_decoder(), 
        structure_tokenizer=model.tokenizers.structure,
        sequence=prot.sequence,
    )
    # print("coords", coords.shape)
    # print("coords_pred", coords_pred.shape)
    return coords, coords_pred  # (L, 37, 3)



def cross_entropy(input, target, reduction='none', ignore_index=-100):
    # input = logits [B, L, V]
    # target = categories [B, L]
    logits_first = input.transpose(1, 2) # [B, V, L]
    return F.cross_entropy(logits_first, target, reduction=reduction, ignore_index=ignore_index) # [B, L]


def forward_and_get_loss(
    model: ESM3, 
    sequence_tokens: torch.Tensor,
    structure_tokens: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    *,
    chain_id: torch.Tensor | None = None,
    sequence_id: torch.Tensor | None = None,
):
    """Forward pass and get loss.
    Tailored for conformation generation task.
    """
    # print("sequence_tokens", sequence_tokens.shape, sequence_tokens.dtype)
    # print("structure_tokens", structure_tokens.shape, structure_tokens.dtype)
    # print("labels", labels.shape, labels.dtype)
    # print("mask", mask.shape, mask.dtype)
    # print("model", list(model.parameters())[0].dtype)
    
    forward_output = model(
        sequence_tokens=sequence_tokens,
        structure_tokens=structure_tokens,
        chain_id=chain_id,
        sequence_id=sequence_id,
    )   # ESMOutput
    unreduced_loss = cross_entropy(forward_output.structure_logits, labels)
    loss = (unreduced_loss * mask).sum() / mask.sum()
    
    return {
        "embeddings": forward_output.embeddings,
        "structure_logits": forward_output.structure_logits,
        "sequence_logits": forward_output.sequence_logits,
        "loss": loss,
    }


def _backbone_coords_from_pdb(pdb_path: str, target_atoms: Optional[list] = ["N", "CA", "C"]):
    structure = PDBFile.read(pdb_path)
    structure_list = structure.get_structure()
    
    coords_list = []
    for b_idx in range(structure.get_model_count()):
        chain = structure_list[b_idx]

        backbone_atoms = chain[struct.filter_backbone(chain)]   # This includes the “N”, “CA” and “C” atoms of amino acids.
        ret_coords = OrderedDict()
        # init dict
        for k in target_atoms:
            ret_coords[k] = []
            
        for c in backbone_atoms:
            if c.atom_name in ret_coords:
                ret_coords[c.atom_name].append(c.coord)
                
        ret_coords = [np.vstack(v) for k,v in ret_coords.items()]
        if len(target_atoms) == 1:
            ret_coords = ret_coords[0]  # L, 3
        else:
            ret_coords = np.stack(ret_coords, axis=1)   # L, na, 3
        
        coords_list.append(ret_coords)
    
    coords_list = np.stack(coords_list, axis=0) # B, L, na, 3 or B, L, 3 (ca only)
    return coords_list


def _backbone_coords_from_npy(npy_path: str, coordinate_scale: float = 0.1):
    return np.load(npy_path) / coordinate_scale # nm -> angstrom


def load_coords(
    input_path: Path, 
    max_n_model: Optional[int] = 10000,
    uniform_sample: bool = True,
    ca_only: bool = True,
    verbose: bool = True,
):
    """Extract backbone coordinates from PDB file. Only CA atoms are extracted.
    
    Args:
        input_path (str): The path to the PDB file.
        max_n_model (int): The maximum number of models to extract.
    """
    assert os.path.exists(input_path), f"File {input_path} does not exist."
    if isinstance(input_path, str):
        input_path = Path(input_path)
    # print(f"Trying to load protein coords from {input_path}")
    target_atoms = ["CA"] if ca_only else ["N", "CA", "C"]

    if input_path.name.endswith('.pdb'):
        coords = _backbone_coords_from_pdb(input_path, target_atoms)
    elif input_path.name.endswith('.npy'):
        coords = _backbone_coords_from_npy(input_path)
    elif input_path.is_dir():
        coords = np.concatenate([
            _backbone_coords_from_pdb(f, target_atoms)
                for f in input_path.iterdir() if f.name.endswith('.pdb')
        ], axis=0) 
    else:
        print("[Warning] Unrecoginzed path, infer as glob pattern")
        coords = np.concatenate([
            _backbone_coords_from_pdb(f, target_atoms)
                for f in glob(str(input_path)) if f.endswith('.pdb')
        ], axis=0) 

    if max_n_model is not None and len(coords) > max_n_model > 0:
        if uniform_sample:
            # sample sequence with uniform stride
            stride = len(coords) // max_n_model
            coords = coords[::stride]
        else:
            coords = coords[:max_n_model]
    if verbose:
        print(f"Loaded {len(coords)} models from input: {input_path} (shape={coords.shape})")
    return coords
