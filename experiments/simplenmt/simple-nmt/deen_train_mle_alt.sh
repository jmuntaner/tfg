#!/bin/bash

#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/joan.muntaner/experiments/outputs/deen_mle_alt.log

DATA_DIR="/home/usuaris/veu/joan.muntaner/experiments/data/iwslt14.tokenized.de-en"

source ~/.bashrc
conda activate myenv

python train.py --train $DATA_DIR/train --valid $DATA_DIR/valid --lang deen \
--gpu_id 0 --batch_size 128 --n_epochs 30 --max_length 50 --dropout 0.2 \
--word_vec_size 512 --hidden_size 1024 --n_layers 4 --max_grad_norm 5 --iteration_per_update 2 \
--lr 1e-3 --lr_step 0 --use_adam --rl_n_epochs 0 --lr_decay_start 12 \
--model_fn ./model_deen_alt/model.deen.pth
