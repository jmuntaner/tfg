#!/bin/bash

#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=50G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/joan.muntaner/tfg/new/logs/flores/train_v2_neen.log
#SBATCH --mail-type=END
#SBATCH --mail-user=joan.francesc.muntaner@estudiantat.upc.edu


FAIRSEQ_DIR="/home/usuaris/veu/joan.muntaner/tfg/fairseq"
FLORES_DIR="/home/usuaris/veu/joan.muntaner/tfg/fairseq/flores"
CP_DIR="/home/usuaris/veu/joan.muntaner/tfg/fairseq/flores/checkpoints/v2"


source ~/.bashrc
conda activate tfg2

python $FAIRSEQ_DIR/train.py $FLORES_DIR/data-bin/wiki_ne_en_bpe5000/ \
    --source-lang ne --target-lang en \
    --arch transformer --share-all-embeddings \
    --encoder-layers 5 --decoder-layers 5 \
    --encoder-embed-dim 512 --decoder-embed-dim 512 \
    --encoder-ffn-embed-dim 2048 --decoder-ffn-embed-dim 2048 \
    --encoder-attention-heads 2 --decoder-attention-heads 2 \
    --encoder-normalize-before --decoder-normalize-before \
    --dropout 0.4 --attention-dropout 0.2 --relu-dropout 0.2 \
    --weight-decay 0.0001 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-7 \
    --lr 1e-3 --min-lr 1e-9 \
    --max-tokens 4000 \
    --update-freq 4 \
    --max-epoch 100 --criterion v2 --mle-weight 0.1 --rl-weight 0.9 --keep-last-epochs 2 --save-dir $CP_DIR
