#!/bin/bash
#SBATCH -J xMCsHalfspaceMCG_b128_lr5_2d
#SBATCH -o xMCsHalfspaceMCG_b128_lr5_2d.o%j
#SBATCH -e xMCsHalfspaceMCG_b128_lr5_2d.o%j
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=6000
#SBATCH -t 48:00:00
#SBATCH --partition=gpu  --gres=gpu:1

python3 embed.py \
       -dim 2 \
       -com_n 2 \
       -dscale 1.0 \
       -lr 5.0 \
       -epochs 500 \
       -negs 50 \
       -burnin 20 \
       -ndproc 1 \
       -manifold xMCsHalfspaceMCG \
       -dset wordnet/noun_closure.csv \
       -batchsize 128 \
       -eval_each 20 \
       -sparse \
       -train_threads 1