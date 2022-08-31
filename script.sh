#!/bin/bash

#SBATCH --partition=unkillable                           # Ask for unkillable job
#SBATCH --cpus-per-task=2                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=32G                                        # Ask for 32 GB of RAM
#SBATCH --time=10:00:00                                   # The job will run for 3 hours
#SBATCH --error=job_error.txt

module load anaconda/3

# conda activate envPytorch # torch, torchvision and python installed

python main_simCLR.py --train_type pretrain --val_pc .05  > ./pretrain/out05.txt
python main_simCLR.py --train_type pretrain --val_pc .025  > ./pretrain/out025.txt
python main_simCLR.py --train_type pretrain --val_pc .015  > ./pretrain/out015.txt
python main_simCLR.py --train_type pretrain --val_pc .01  > ./pretrain/out01.txt
python main_simCLR.py --train_type pretrain --val_pc .005  > ./pretrain/out005.txt

python main_simCLR.py --train_type persistent_finetune --val_pc .05  > ./persistent_finetune/out05.txt
python main_simCLR.py --train_type persistent_finetune --val_pc .025  > ./persistent_finetune/out025.txt
python main_simCLR.py --train_type persistent_finetune --val_pc .015  > ./persistent_finetune/out015.txt
python main_simCLR.py --train_type persistent_finetune --val_pc .01  > ./persistent_finetune/out01.txt
python main_simCLR.py --train_type persistent_finetune --val_pc .005  > ./persistent_finetune/out005.txt

