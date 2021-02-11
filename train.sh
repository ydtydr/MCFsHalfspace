#!/bin/bash
#SBATCH -J mammal
#SBATCH -o mammal.o%j
#SBATCH -e mammal.o%j
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=2000
#SBATCH -t 24:00:00
#SBATCH --partition=mpi-cpus  --gres=gpu:0

python3 embed.py \
       -dim 2 \
       -com_n 1 \
       -dscale 2.0 \
       -lr 0.3 \
       -epochs 10 \
       -negs 50 \
       -burnin 20 \
       -ndproc 4 \
       -manifold xMCsHalfspace \
       -dset wordnet/mammal_closure.csv \
       -batchsize 10 \
       -eval_each 20 \
       -sparse \
       -train_threads 1