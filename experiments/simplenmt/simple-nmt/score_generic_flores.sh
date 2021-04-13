#!/bin/bash

#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=10G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=/home/usuaris/veu/joan.muntaner/experiments/outputs/score_neen_mrt_v2.log


FAIRSEQ_DIR="/home/usuaris/veu/joan.muntaner/tfg/fairseq"

HYPS_DIR="/home/usuaris/veu/joan.muntaner/experiments/simplenmt/simple-nmt/test.mrt.neen.best.v2"
REFS_DIR="/home/usuaris/veu/joan.muntaner/experiments/data/wiki_ne_en_bpe5000/test.en"

DETOK_DIR="/home/usuaris/veu/joan.muntaner/experiments/utils_preprocessing/nlp_preprocessing"
TMP_DIR="/home/usuaris/veu/joan.muntaner/experiments/simplenmt/simple-nmt/tmp.txt"

source ~/.bashrc

conda activate myenv
python $DETOK_DIR/detokenizer.py < $HYPS_DIR > $TMP_DIR

conda activate tfg2
cd $FAIRSEQ_DIR

fairseq-score -s $TMP_DIR -r $REFS_DIR