#!/bin/bash

#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/joan.muntaner/experiments/outputs/generate_neen_mle.log

DATA_DIR="/home/usuaris/veu/joan.muntaner/experiments/data/wiki_ne_en_bpe5000"
MODEL="/home/usuaris/veu/joan.muntaner/experiments/simplenmt/simple-nmt/model/model.neen.02.1.11-3.02.4.69-109.15.pth"

source ~/.bashrc
conda activate myenv

python translate.py --model_fn $MODEL --gpu_id 0 --lang neen < $DATA_DIR/test.bpe.ne > test.results.neen.txt
