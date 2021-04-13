#!/bin/bash

#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/joan.muntaner/tfg/new/logs/baseline/generate_baseline_transformer.log
#SBATCH --mail-type=END
#SBATCH --mail-user=joan.francesc.muntaner@estudiantat.upc.edu


FAIRSEQ_DIR="/home/usuaris/veu/joan.muntaner/tfg/fairseq"
CP_DIR="/home/usuaris/veu/joan.muntaner/tfg/fairseq/checkpoints/baseline"


source ~/.bashrc
conda activate tfg2

fairseq-generate $FAIRSEQ_DIR/data-bin/iwslt14.tokenized.de-en \
    --path $FAIRSEQ_DIR/checkpoints/baseline/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe
