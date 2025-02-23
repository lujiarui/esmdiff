import torch
from pathlib import Path
from omegaconf import OmegaConf
import hydra


def load_hf_network_checkpoint(model, ckpt_path):
    """Load state dict from checkpoint file.

    :param model: The model to load the state dict into.
    :param ckpt_path: The path to the checkpoint file.
    """
    if ckpt_path is None:
        print("No checkpoint file provided, using random initialization.")
        return model, None
    
    # The ckpt_path ending with .ckpt is a checkpoint file saved by pytorch-lightning.
    # If the ckpt_path is a .pth file, it is viewed as a checkpoint file saved by pytorch.
    # In both case, only net parameters are loaded. 
    # (This may avoid the ambiguity of loading #epochs/lr/earlystop state for finetuning)
    if ckpt_path.endswith(".pth") or ckpt_path.endswith(".ckpt"):  
        net_params = torch.load(ckpt_path, map_location=torch.device('cpu'))['state_dict']
        # print(list(net_params.keys())[:100])
        net_params = {k.replace('net.', ''): v for k, v in net_params.items()}
        model.load_state_dict(net_params, strict=False)
        ckpt_path = None
    elif ckpt_path.endswith(".pt") and "mp_rank_" in ckpt_path:
        net_params = torch.load(ckpt_path, map_location=torch.device('cpu'))['module']
        net_params = {k.replace('net.', ''): v for k, v in net_params.items()}
        model.load_state_dict(net_params)
        ckpt_path = None
    else:
        # suffix check
        raise ValueError(f"ckpt_path {ckpt_path} is not a valid checkpoint file.")
    
    return model, ckpt_path



# accommondate the ckpt from lightning and deepspeed
def load_state_dict_from_lightning_ckpt(ckpt_path, device="cuda"):
    print(f"Loading ESMDiff ckpt from {ckpt_path}")
    ckpt_path = Path(ckpt_path)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
    assert ckpt_path.suffix in [".ckpt", ".pt"], f"Unsupported ckpt format: {ckpt_path}"
    if ckpt_path.is_dir():  # load deepspeed ckpt
        ckpt_path = ckpt_path / "checkpoint/mp_rank_00_model_states.pt"
        exp_cfg_path = ckpt_path.parent.parent.parent.parent / ".hydra/config.yaml"
    else:
        exp_cfg_path = ckpt_path.parent.parent / ".hydra/config.yaml"
    if exp_cfg_path.exists():
        cfg = OmegaConf.load(exp_cfg_path)
    else:
        print(f"Config file not found: {exp_cfg_path}. Use default config.")
        exp_cfg_path = "configs/experiment/mdlm.yaml"
        cfg = OmegaConf.load(exp_cfg_path)
    print(f"Loaded experiment config: {exp_cfg_path}...")

    model = hydra.utils.instantiate(cfg.model)
    print(f"Sucessfully instantiated model ...")

    if ckpt_path.suffix == ".pt":
        all_params = torch.load(ckpt_path, map_location=torch.device(device))['module']
        model.load_state_dict(all_params)
    else:
        raise ValueError(f"Unsupported ckpt format: {ckpt_path}")

    print(f"Sucessfully loaded model from {ckpt_path}...")
    
    # necessary when decoding from tokens
    model.noise_removal = True
    model.to(device)
    model.eval()
    
    return model