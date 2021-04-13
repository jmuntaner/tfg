#!/bin/bash

#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/joan.muntaner/experiments/outputs/deen_transformer_mrt.log

DATA_DIR="/home/usuaris/veu/joan.muntaner/experiments/data/iwslt14.tokenized.de-en"
MODEL_DIR="/home/usuaris/veu/joan.muntaner/experiments/simplenmt/simple-nmt/model.deen.transformer.18.1.25-3.48.1.42-4.13.pth"

source ~/.bashrc
conda activate myenv

python continue_train.py --load_fn $MODEL_DIR --model_fn ./model.rl.transformer.pth \
--train $DATA_DIR/train --valid $DATA_DIR/valid --lang deen \
--use_transformer --init_epoch 41 --max_grad_norm 5 --iteration_per_update 1 --rl_n_epochs 30 --batch_size 8
