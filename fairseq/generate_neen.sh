#!/bin/bash

#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=20G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/joan.muntaner/tfg/new/logs/flores/generate_baseline_neen.log
#SBATCH --mail-type=END
#SBATCH --mail-user=joan.francesc.muntaner@estudiantat.upc.edu


FAIRSEQ_DIR="/home/usuaris/veu/joan.muntaner/tfg/fairseq"
FLORES_DIR="/home/usuaris/veu/joan.muntaner/tfg/fairseq/flores"
CP_DIR="/home/usuaris/veu/joan.muntaner/tfg/fairseq/flores/checkpoints/baseline"


source ~/.bashrc
conda activate tfg2

fairseq-generate $FLORES_DIR/data-bin/wiki_ne_en_bpe5000/ \
    --path $CP_DIR/checkpoint_best.pt \
    --beam 5 --lenpen 1.2 --source-lang ne --target-lang en --task translation --remove-bpe=sentencepiece
