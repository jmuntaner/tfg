#!/bin/bash

#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/joan.muntaner/experiments/outputs/generate_deen_rl.log

DATA_DIR="/home/usuaris/veu/joan.muntaner/experiments/data/iwslt14.tokenized.de-en"

source ~/.bashrc
conda activate myenv

python translate.py --model_fn ./model.rl.mrt.10.39.83-45.52.pth --gpu_id 0 --lang deen < /home/usuaris/veu/joan.muntaner/experiments/data/iwslt14.tokenized.de-en/test.de > test.results.rl.txt
