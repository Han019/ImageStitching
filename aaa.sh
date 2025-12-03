#!/usr/bin/bash

#SBATCH -J stitching
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ce_ugrad
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out 

source /data/$USER/anaconda3/etc/profile.d/conda.sh

conda activate panorama

pwd
which python

python aaa.py

exit 0
