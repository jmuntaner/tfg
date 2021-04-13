#!/bin/bash

#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/joan.muntaner/experiments/outputs/deen_rl2.log

DATA_DIR="/home/usuaris/veu/joan.muntaner/experiments/data/iwslt14.tokenized.de-en"

source ~/.bashrc
conda activate myenv

python continue_train.py --load_fn ./model.deen.pth --model_fn ./model.2.rl.pth \
--train $DATA_DIR/train --valid $DATA_DIR/valid --lang deen \
--init_epoch 31 --iteration_per_update 1 --max_grad_norm 5 --rl_n_epochs 40
