#!/bin/bash
#SBATCH -J xMCshalfspaceMCG_mm_b128_lr5_test
#SBATCH -o xMCshalfspaceMCG_mm_b128_lr5_test.o%j
#SBATCH -e xMCshalfspaceMCG_mm_b128_lr5_test.o%j
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=2000
#SBATCH -t 24:00:00
#SBATCH --partition=gpu  --gres=gpu:1

python3 embed.py \
       -dim 2 \
       -com_n 2 \
       -dscale 1.0 \
       -lr 5.0 \
       -epochs 1000 \
       -negs 50 \
       -burnin 20 \
       -ndproc 1 \
       -manifold xMCsHalfspaceMCG \
       -dset wordnet/mammal_closure.csv \
       -batchsize 128 \
       -eval_each 20 \
       -sparse \
       -train_threads 1