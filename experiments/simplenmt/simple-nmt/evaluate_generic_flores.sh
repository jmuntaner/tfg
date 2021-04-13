#!/bin/bash

#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/joan.muntaner/experiments/outputs/generate_generic.log

DATA_DIR="/home/usuaris/veu/joan.muntaner/experiments/data/wiki_ne_en_bpe5000"
MODEL="/home/usuaris/veu/joan.muntaner/experiments/simplenmt/simple-nmt/model_neen_alt/model.neen.rl.mrt.37.68.87-8.16.pth"
OUTPUT_FILE="test.mrt.neen.best.v2"

source ~/.bashrc
conda activate myenv

python translate.py --model_fn $MODEL --gpu_id 0 --lang neen < $DATA_DIR/test.bpe.ne > $OUTPUT_FILE
