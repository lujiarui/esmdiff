# function utilities to generate output sequence based on the context sequence
# by default, the output sequence has the same length as the context
#
import os
from time import strftime, time
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from torch import Tensor
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from esm.utils.constants import esm3 as C
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, ESMProteinTensor
from slm.models.utils import protseq_to_data

from slm.utils import (
    RankedLogger,
    checkpoint_utils,
    eval_utils,
)

log = RankedLogger(__name__, rank_zero_only=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MyDataset(torch.utils.data.Dataset):
    _root = Path("data/targets")
    
    def __init__(self, task = None, pdb_dir: Path = None):
        self.task = task
        if task in ("apo", "codnas", "ped", "bpti"):
            pdb_dir = self._root / task

        if pdb_dir: # bpti 
            print("Loading pdb files from the given directory...", pdb_dir)
            if isinstance(pdb_dir, str):
                pdb_dir = Path(pdb_dir)
            assert pdb_dir.exists(), f"{pdb_dir} does not exist."
            target_paths = list(pdb_dir.iterdir())
            # filter out folders
            target_paths = [p for p in target_paths if p.is_file() and p.suffix == ".pdb"]
            self.data = [
                ESMProtein.from_pdb(p).sequence for p in target_paths
            ]
        else:
            raise ValueError("Please provide either pdb_dir or atlas_base.")

        self._target_paths = target_paths
        self.model = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda").float()
        self.model.eval()
    
    @property 
    def accession_codes(self):
        return [p.stem for p in self._target_paths]

    def __len__(self):
        return len(self.data)
    
    @torch.no_grad()
    def __getitem__(self, idx):
        data_dict = protseq_to_data(
            sequence=self.data[idx],
            model=self.model,
            # coordinates=prot.coordinates,

        )   # -> L+2
        for k, v in data_dict.items():
            if not torch.is_tensor(v):
                continue
            data_dict[k] = v[1:-1]
            assert len(data_dict[k]) == len(data_dict['sequence'])

        data_dict["mask"] = torch.ones_like(data_dict["embeddings"][..., 0])
        data_dict["accession_code"] = self.accession_codes[idx]
        return data_dict

    @torch.no_grad()
    def decode(self, structure_tokens, sequence_tokens, save_to=None):
        # per-sample input!!
        assert len(structure_tokens) == len(sequence_tokens), f"{len(structure_tokens)} != {len(sequence_tokens)}"
        # add BOS and EOS to tensors
        sequence_tokens = torch.cat(
            [torch.LongTensor([C.SEQUENCE_BOS_TOKEN]), 
            sequence_tokens.cpu(), 
            torch.LongTensor([C.SEQUENCE_EOS_TOKEN])]
        )
        structure_tokens = torch.cat(
            [torch.LongTensor([C.STRUCTURE_BOS_TOKEN]), 
            structure_tokens.cpu(), 
            torch.LongTensor([C.STRUCTURE_EOS_TOKEN])]
        )
        
        prot = ESMProteinTensor(sequence=sequence_tokens, structure=structure_tokens)
        prot = prot.to("cuda")
        
        raw_protein = self.model.decode(prot)
        if save_to is not None:
            raw_protein.to_pdb(save_to)


def __generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


# https://github.com/meta-llama/llama/blob/main/llama/generation.py#L130
# https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/generation_utils.py#L741 
# adapt the function to acommodate customed model input
# @ 0905: runnable with T5 decoder
@torch.no_grad()
def generate(
    model: nn.Module, 
    context: torch.Tensor, 
    context_mask: torch.Tensor = None, 
    pad_token_id: int = C.STRUCTURE_PAD_TOKEN, 
    do_sample: bool = True,
    greedy: bool = False,
    temperature: float = 1.0,
    top_p: float = 0.95,
    logprobs: bool = False,
    model_type: str = "clm",
):
    B, L = context.shape[:2]    # no BOS/EOS
    device = context.device

    # initialize the output tensor including BOS == pad_token_id
    tokens = torch.full((B, L+1), pad_token_id, dtype=torch.long, device=device) 
    
    if context_mask is None:
        context_mask = torch.ones_like(context[..., 0])

    prev_pos = 0
    last_hidden_state = None
    past_key_values = None
    for cur_pos in tqdm(range(1, L+1), desc="Sequential Decoding..."):
        if model_type == "clm":          
            out = model(
                inputs_embeds=context,    # [B, L, D]
                decoder_input_ids=tokens[:, cur_pos-1:cur_pos],
                attention_mask=context_mask,   # for encoder
                return_dict=True,
                encoder_outputs=last_hidden_state,
                past_key_values=past_key_values,
            )
            past_key_values = out["past_key_values"]
            
            logits = out["logits"]    # [B, L, V]
            if last_hidden_state is None:
                last_hidden_state = [out["encoder_last_hidden_state"]]    # [B, L, D]
            last_logits = logits[:, -1]
        
        elif model_type == "jlm":
            out = model(
                structure_tokens=tokens[:, cur_pos-1:cur_pos],   # B, L>=1
                past_key_values=past_key_values,
                sequence_embeddings=context, # B,L
            )
            last_logits = out["structure_logits"][:, -1]  # B, V_struc
            past_key_values = out["past_key_values"]
        
        # Disable generate special tokens 
        for spec_id in range(C.VQVAE_CODEBOOK_SIZE, last_logits.size(-1)):
            last_logits[..., spec_id] = -np.inf
        
        # Sampling or search
        if do_sample:
            probs = torch.softmax(last_logits / temperature, dim=-1)  # [B, V]
            if top_p is not None and 0 < top_p < 1:
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.multinomial(probs, num_samples=1)
            next_token = next_token.reshape(-1) # [B]
            tokens[:, cur_pos] = next_token
        else:
            raise NotImplementedError("Only support random sampling.")
        
        prev_pos = cur_pos

    if logprobs:
        logits = model(
            inputs_embeds=context,
            decoder_input_ids=tokens[:, :-1],
            attention_mask=context_mask,
            return_dict=True,
        )["logits"]    # [B, L, V]
        
        token_logprobs = -F.cross_entropy(
            input=logits.transpose(1, 2),
            target=tokens[:, 1:],
            reduction="none",
            ignore_index=pad_token_id,
        )
    
    return (tokens[:, 1:], token_logprobs if logprobs else None)


def make_output_name(**kwargs):
    # turn dict into string
    return "_".join([f"{k}={v}" for k, v in kwargs.items()]) + strftime("-%m-%d-%H")


@hydra.main(version_base="1.3", config_path="../configs", config_name="predict.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    assert cfg.ckpt_path is not None, "Please provide the path to the model checkpoint."
    if 'ConditionalLanguageModeling' in cfg.ckpt_path:
        _model_type = 'clm'
    elif 'JointLanguageModeling' in cfg.ckpt_path:
        _model_type = 'jlm'
    else:
        raise ValueError("Unrecognized model type.")

    # infer the configure file path from ckpt 
    exp_cfg = Path(cfg.ckpt_path).parent.parent / ".hydra" / "config.yaml"
    if exp_cfg.exists():
        log.info(f"Loading experiment configuration from {exp_cfg}. Override model configuration.")
        cfg.model.net.update(OmegaConf.load(exp_cfg).model.net)

    
    log.info(f"Instantiating model <{cfg.model.net._target_}>")
    net = hydra.utils.instantiate(cfg.model.net)

    log.info("Loading model checkpoint...")
    model, _ = checkpoint_utils.load_hf_network_checkpoint(net, cfg.ckpt_path)
    model = model.to("cuda")

    inf_cfg = cfg.inference
    log.info("Loading sampling dataset...")
    if inf_cfg.target and inf_cfg.target in ("bpti", "apo", "codnas", "ped"):
        data = MyDataset(inf_cfg.target)
    else:
        data = MyDataset(pdb_dir=Path(inf_cfg.input))
        
    print(f"Final number of predicting target: {len(data)}")
    n_samples, bsz = inf_cfg.n_samples, inf_cfg.batch_size
    ########################

    log.info("Starting predictions.")
    
    output_dir = Path(inf_cfg.output)
    n_outer_loop = n_samples // bsz
    residual_bsz = n_samples % bsz
    
    def predict_fn(context, batch_size=1, model_type=_model_type, **kwargs):
        context = context.unsqueeze(0).repeat(batch_size, 1, 1)
        tokens, _token_logprobs = generate(model, context, logprobs=False, model_type=model_type,**kwargs)
        tokens = tokens.detach().clone()
        return tokens
    
    params_grid = {
        "temperature":  [1.0,],#[ 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        "top_p": [0.95],
        "greedy": [False],
        "do_sample": [True],
        "batch_size": [bsz for _ in range(n_outer_loop)],
    }
    if residual_bsz > 0:
        params_grid["batch_size"].append(residual_bsz)

    for greedy in params_grid["greedy"]:
        for do_sample in params_grid["do_sample"]:
            for temperature in params_grid["temperature"]:
                for top_p in params_grid["top_p"]:
                    tmp_output_dir = output_dir / make_output_name(T=temperature, top_p=top_p, sample=do_sample, N=n_samples)
                    tmp_output_dir.mkdir(parents=True, exist_ok=True)
                    sampled_targets = []
                    for idx in tqdm(range(len(data)), desc=f"Running loop for samples w/ params [T={temperature}, top_p={top_p}]..."):
                        data_dict = data[idx]
                        base = data_dict["accession_code"]
                        # print(f"Sampling {base}...")
                        start = time()
                        context = data_dict["embeddings"].to("cuda")
                        all_tokens = []
                        
                        for bsz in params_grid["batch_size"]:
                            all_tokens.append(predict_fn(context, bsz, temperature=temperature, top_p=top_p, do_sample=do_sample, greedy=greedy))

                        gen_end = time()
                        # Decode structures 
                        tokens = torch.cat(all_tokens, dim=0)
                        for k in tqdm(range(len(tokens)), desc="Decoding generated proteins..."):
                            data.decode(tokens[k], data_dict["sequence_tokens"], save_to=tmp_output_dir / f"{base}_{k}.pdb")   # even loop, it is fast
                        decode_end = time()

                        sampled_targets.append(base)
                    eval_utils.merge_all_targets_from_dir(tmp_output_dir, sampled_targets, verbose=True)

if __name__ == "__main__":
    main()

