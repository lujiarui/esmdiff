# Analysis


## Acquire reference data
By default, you can put the reference data in `data/ground_truth`
```bash
mkdir -p data/ground_truth
```

The BPTI MD simulation reference data is from paper [science.1187409](https://www.science.org/doi/abs/10.1126/science.1187409), where the MD trajectory data is obtained by sending request to [DEShaw Research](https://www.deshawresearch.com/index.html) (protected under DESRES's License) and the PDB structures of the kinetic clusters are obtained from the supplementary data in [science.1187409](https://www.science.org/doi/abs/10.1126/science.1187409).

The apo/holo targets and fold-switchers can be found in the [EigenFold](https://github.com/bjing2016/EigenFold) repository. You can download as follows:
```bash
git clone https://github.com/bjing2016/EigenFold.git
mv EigenFold data/ground_truth/apo
```

The curated experimental IDP targets mentioned in this study can be downloaded from this [drive](https://drive.google.com/file/d/1Li4pHVuqxdZJaFKG3iMwqSevNpDewpnO/view?usp=sharing). Download and unzip and put the directory into `data/ground_truth/`.


## TMscore
To evaluate the relevant metrics, [TMscore](https://zhanggroup.org/TM-score/) binary is required to be installed and set: `export PATH=$PATH:/path/to/TMscore/bin-directory`.

## Run

To evaluate the generated samples, provide input as follows:
```bash
python analysis/apo_analysis.py logs/toy_sampling/apo/samples_seed42 --reference data/ground_truth/apo --task apo
python analysis/bpti_analysis.py logs/toy_sampling/bpti/samples_seed42 --reference data/ground_truth/bpti/
python analysis/ped_analysis.py logs/toy_sampling/bpti/samples_seed42 --reference data/ground_truth/ped/

```
