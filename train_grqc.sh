#!/bin/bash
#SBATCH -J htiling_grqc_b128_lr5_2d
#SBATCH -o htiling_grqc_b128_lr5_2d.o%j
#SBATCH -e htiling_grqc_b128_lr5_2d.o%j
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=4000
#SBATCH -t 24:00:00
#SBATCH --partition=default_partition  --gres=gpu:0

python3 embed.py \
       -dim 2 \
       -com_n 1 \
       -dscale 1.0 \
       -lr 5.0 \
       -epochs 1000 \
       -negs 50 \
       -burnin 20 \
       -ndproc 1 \
       -manifold HTiling_rsgd \
       -dset wordnet/grqc.csv \
       -batchsize 128 \
       -eval_each 20 \
       -sparse \
       -train_threads 1