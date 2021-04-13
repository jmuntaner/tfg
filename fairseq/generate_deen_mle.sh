#!/bin/bash

#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/joan.muntaner/experiments/outputs/generate_deen_mle.log
#SBATCH --mail-type=END
#SBATCH --mail-user=joan.francesc.muntaner@estudiantat.upc.edu


FAIRSEQ_DIR="/home/usuaris/veu/joan.muntaner/tfg/fairseq"


source ~/.bashrc
conda activate tfg2

fairseq-generate $FAIRSEQ_DIR/data-bin/iwslt14.tokenized.de-en \
    --path /home/usuaris/veu/joan.muntaner/experiments/simplenmt/simple-nmt/model.deen.12.1.37-3.95.1.48-4.39.pth \
    --batch-size 128 --beam 5 --remove-bpe