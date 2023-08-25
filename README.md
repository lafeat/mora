# MORA
This repository contains code for reproducing our NeurIPS 2022 submitted paper ["MORA: Improving Ensemble Robustness Evaluation
with Model-Reweighing Attack
"].

# Dependencies
Create the conda environment called `MORA` containing all the dependencies by running
```
conda env create -f environment.yml
```
We were using PyTorch 1.4.0 for all the experiments. You may want to install other versions of PyTorch according to the cuda version of your computer/server.
The code is run and tested on a single TITAN Xp GPU. Running on multiple GPUs with parallelism may need adjustments.

# Data and pre-trained models
The ensemble  ADP DVERGE GAL defense strategies pre-trained models can be accessed via [this link](https://drive.google.com/drive/folders/1i96Bk_bCWXhb7afSNp1t3woNjO1kAMDH?usp=sharing), The pre-trained models are stored in the folder named `checkpoints`. Download and put `checkpoints` under this repo. 

# Usage
Examples of evaluation scripts can be found in `scripts/evaluation.sh`.
