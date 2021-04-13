#!/bin/bash

#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=50G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/joan.muntaner/tfg/new/logs/v2/v2_transformer.log
#SBATCH --mail-type=END
#SBATCH --mail-user=joan.francesc.muntaner@estudiantat.upc.edu


FAIRSEQ_DIR="/home/usuaris/veu/joan.muntaner/tfg/fairseq"
CP_DIR="/home/usuaris/veu/joan.muntaner/tfg/fairseq/checkpoints/v2"


source ~/.bashrc
conda activate tfg2

python $FAIRSEQ_DIR/train.py $FAIRSEQ_DIR/data-bin/iwslt14.tokenized.de-en \
 --arch  transformer_iwslt_de_en --restore-file /home/usuaris/veu/joan.muntaner/tfg/fairseq/checkpoints/baseline/checkpoint_best.pt	\
 --reset-optimizer --reset-lr-scheduler	--reset-meters \
 --optimizer adam --adam-betas '(0.9, 0.98)' \
 --lr-scheduler inverse_sqrt --lr 1e-5 --dropout 0.3 \
 --weight-decay 0.0 --criterion v2 --mle-weight 0.1 --rl-weight 0.9 \
 --max-tokens 200  --save-dir $CP_DIR  --max-update 500000 --max-epoch 20 --keep-last-epochs 2
