#!/bin/bash

#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/joan.muntaner/experiments/outputs/score_deen_mle.log

FAIRSEQ_DIR="/home/usuaris/veu/joan.muntaner/tfg/fairseq/fairseq"

source ~/.bashrc
conda activate tfg2

python $FAIRSEQ_DIR/fairseq-score.py test.results.txt -ref /home/usuaris/veu/joan.muntaner/experiments/data/iwslt14.tokenized.de-en/test.en