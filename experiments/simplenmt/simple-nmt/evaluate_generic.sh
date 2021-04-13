#!/bin/bash

#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/joan.muntaner/experiments/outputs/generate_generic.log

DATA_DIR="/home/usuaris/veu/joan.muntaner/experiments/data/iwslt14.tokenized.de-en"
MODEL="/home/usuaris/veu/joan.muntaner/experiments/simplenmt/simple-nmt/model.2.rl.mrt.31.42.39-45.97.pth"
OUTPUT_FILE="test.mrt.deen.2.40epchs.txt"

source ~/.bashrc
conda activate myenv

python translate.py --model_fn $MODEL --gpu_id 0 --lang deen < $DATA_DIR/test.de > $OUTPUT_FILE
