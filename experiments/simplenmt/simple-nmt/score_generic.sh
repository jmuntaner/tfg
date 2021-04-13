#!/bin/bash

#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/joan.muntaner/experiments/outputs/score_deen_mrt_40epochs.log


FAIRSEQ_DIR="/home/usuaris/veu/joan.muntaner/tfg/fairseq"
HYPS_DIR="/home/usuaris/veu/joan.muntaner/experiments/simplenmt/simple-nmt/test.mrt.deen.2.40epchs.txt"
REFS_DIR="/home/usuaris/veu/joan.muntaner/experiments/data/iwslt14.tokenized.de-en/test.en"

source ~/.bashrc
conda activate tfg2

cd $FAIRSEQ_DIR

fairseq-score -s $HYPS_DIR -r $REFS_DIR