#!/bin/bash

#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/joan.muntaner/experiments/outputs/generate_neen_rl.log

DATA_DIR="/home/usuaris/veu/joan.muntaner/experiments/data/wiki_ne_en_bpe5000"
MODEL="/home/usuaris/veu/joan.muntaner/experiments/simplenmt/simple-nmt/model/model.neen.rl.mrt.15.55.94-6.74.pth"

source ~/.bashrc
conda activate myenv

python translate.py --model_fn $MODEL --gpu_id 0 --lang neen < $DATA_DIR/test.bpe.ne > test.results.rl.neen.txt
