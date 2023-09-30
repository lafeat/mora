# MORA

This repository contains code
for reproducing the results in our NeurIPS 2022 paper
["MORA: Improving Ensemble Robustness Evaluation with Model-Reweighing Attack"](https://openreview.net/pdf?id%253Dd_m7OKOmPiM).


## Dependencies

Create the conda environment called `mora`
containing all the dependencies by running:
```shell
conda env create -f environment.yml
```
We used PyTorch 1.4.0 for all the experiments,
and the code were tested on an NVIDIA TITAN Xp GPU.


## Pretrained models

The pretrained models
for the ensemble defense strategies (ADP, DVERGE, GAL)
can be accessed
via [this link](https://drive.google.com/drive/folders/1i96Bk_bCWXhb7afSNp1t3woNjO1kAMDH?usp=sharing).
The pre-trained models are located
in the folder named `checkpoints`.
Download and place the checkpoints
into a `checkpoints/` folder under this repo
before running evaluation scripts. 


## Usage

Examples of evaluation scripts can be found in `scripts/evaluation.sh`.
