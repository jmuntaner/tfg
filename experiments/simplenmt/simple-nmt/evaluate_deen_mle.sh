#!/bin/bash

#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/joan.muntaner/experiments/outputs/generate_deen_mle.log

DATA_DIR="/home/usuaris/veu/joan.muntaner/experiments/data/iwslt14.tokenized.de-en"
MODEL="/home/usuaris/veu/joan.muntaner/experiments/simplenmt/simple-nmt/model_deen/model.deen.12.1.44-4.23.1.48-4.38.pth"

source ~/.bashrc
conda activate myenv

python translate.py --model_fn $MODEL --gpu_id 0 --lang deen < $DATA_DIR/test.de > test.results.deen2.txt
