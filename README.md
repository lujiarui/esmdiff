# esmdiff

## Installation
```bash
conda create -n py310 python=3.10 -y
conda activate py310
pip install -r requirements.txt
pip install -e .
```

The first time running ESM3:

> In order to download the weights, we require users to accept our non-commercial license.
> The weights are stored on HuggingFace Hub under [HuggingFace/EvolutionaryScale/esm3](https://huggingface.co/EvolutionaryScale/esm3).
> Please create an account and accept the license.

```py
from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig

# Will instruct you how to get an API key from huggingface hub, make one with "Read" permission.
login()
```


## Process data
(Optionally) Download the whole pdb database and extract/process structures to pickle (.pkl) files:
```bash
# to download the whole pdb database
python scripts/download_pdb_mmcif.sh ./pdb_mmcif 

# ad hoc: biopython=1.79 is required for mmcif_parsing
pip install biopython==1.79
python scripts/pdb/preprocess.py --mmcif_dir pdb_data/pdb_mmcif --output_dir pdb_data/processed_chains --per_chain --strip_array

# turn back to this version for main usage
pip install biopython==1.84 
```

For training purpose, the VQ-VAE encoding should be pre-computed:
```bash
# turn processed pickle (above) or pdb files into 
python scripts/dump.py pdb_data/processed_chains pdb_data/processed_chains_encoding pkl
# if you have some dataset of pdb files at hand
python scripts/dump.py pdb_data/raw_pdb pdb_data/raw_pdb_encoding pdb
```

## Training
```bash
sbatch train.sh experiment=jlm paths.data_dir=pdb_data/raw_pdb_encoding data.batch_size=16 logger=csv 
sbatch train.sh experiment=clm paths.data_dir=pdb_data/raw_pdb_encoding data.batch_size=16 logger=csv 
sbatch train.sh experiment=mdlm paths.data_dir=pdb_data/raw_pdb_encoding data.batch_size=16 logger=csv 
```

##  Inference
Sample from HuggingFace-based models (T5, GPT2), for example:
```bash
python slm/sample_hf.py ckpt_path=logs/ConditionalLanguageModeling/runs/dev_exp_name/checkpoints/epoch_999.ckpt inference.input=data/targets/bpti inference.output=outputs/inference inference.batch_size=32 inference.n_samples=100
# or 
python slm/sample_hf.py ckpt_path=logs/ConditionalLanguageModeling/runs/dev_exp_name/checkpoints/epoch_999.ckpt inference.target=bpti inference.output=outputs/inference inference.batch_size=32 inference.n_samples=100
```

Sample from ESMDiff (masked diffusion fine-tuned ESM3):
```bash
python slm/sample_esmdiff.py --input data/targets/bpti --output outputs/inference_esmdiff --num_steps 25 --num_samples 100 --ckpt logs/MaskedDiffusionLanguageModeling/runs/dev_exp_name/checkpoints/epoch_999.ckpt --mode ddpm
# inpainting 
python slm/sample_esmdiff.py --input data/targets/bpti --output outputs/inference_esmdiff --num_steps 25 --num_samples 100 --mask_ids 1,2,3,4,5
```


## Evaluation of samples
See `./analysis`.



## LICENSE
The source code and model can be used for non-commerical purpose. For any parts related to ESM3, please strictly follow the EvolutionaryScale Community License Agreement <https://www.evolutionaryscale.ai/policies/community-license-agreement>. 


## Remarks
*This repo is still work-in-progress and more features will be added in the short future.*