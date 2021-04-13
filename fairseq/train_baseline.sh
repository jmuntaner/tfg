#!/bin/bash

#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/joan.muntaner/tfg/new/logs/baseline/baseline_transformer.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=joan.francesc.muntaner@estudiantat.upc.edu


FAIRSEQ_DIR="/home/usuaris/veu/joan.muntaner/tfg/fairseq"
CP_DIR="/home/usuaris/veu/joan.muntaner/tfg/fairseq/checkpoints/baseline"


source ~/.bashrc
conda activate tfg2

python $FAIRSEQ_DIR/train.py $FAIRSEQ_DIR/data-bin/iwslt14.tokenized.de-en \
 --arch  transformer_iwslt_de_en --optimizer adam --adam-betas '(0.9, 0.98)' \
 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.0005 --min-lr 1e-09 --dropout 0.3 \
 --weight-decay 0.0 --criterion cross_entropy \
 --max-tokens 4000  --save-dir $CP_DIR  --max-update 50000 --keep-last-epochs 2
