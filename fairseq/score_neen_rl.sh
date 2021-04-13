#!/bin/bash

#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/joan.muntaner/experiments/outputs/score_neen_rl.log

FAIRSEQ_DIR="/home/usuaris/veu/joan.muntaner/tfg/fairseq/fairseq"

source ~/.bashrc
conda activate tfg2

fairseq-score -s /home/usuaris/veu/joan.muntaner/experiments/simplenmt/simple-nmt/test.results.rl.neen.detok.txt -r /home/usuaris/veu/joan.muntaner/experiments/data/wiki_ne_en_bpe5000/test.en