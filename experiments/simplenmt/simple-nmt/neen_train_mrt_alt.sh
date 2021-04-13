#!/bin/bash

#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/joan.muntaner/experiments/outputs/neen_rl_alt.log

DATA_DIR="/home/usuaris/veu/joan.muntaner/experiments/data/wiki_ne_en_bpe5000"
BASE_MODEL="/home/usuaris/veu/joan.muntaner/experiments/simplenmt/simple-nmt/model_neen_alt/model.neen.05.0.63-1.87.4.53-92.53.pth"

source ~/.bashrc
conda activate myenv

python continue_train.py --load_fn $BASE_MODEL --model_fn ./model_neen_alt/model.neen.rl.pth \
--train $DATA_DIR/train.bpe --valid $DATA_DIR/valid.bpe --lang neen \
--init_epoch 31 --iteration_per_update 1 --max_grad_norm 5 --rl_n_epochs 40