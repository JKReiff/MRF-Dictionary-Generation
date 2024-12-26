#!/bin/bash
#SBATCH --job-name GRU_DoE                    # Custom name
#SBATCH -t 7-00:00:00                         # Max runtime of 3 hours
#SBATCH -p batch                              # Choose partition (interactive or batch)
#SBATCH -q batch                              # Choose QoS, must be same as partition
#SBATCH --cpus-per-task 8                     # Request 2 cores
#SBATCH --mem=32G                             # Request RAM (memory)
#SBATCH --gpus=1                              # Request 1 GPU
#SBATCH -o /mnt/workspace/%u/example-%j.out   # Write output (stdout) to this file
#SBATCH -e /mnt/workspace/%u/example-%j.err   # Write errors (stderr) to this file
#SBATCH --mail-type=ALL                       # Notify when it ends
#SBATCH --mail-user=slack:U05USQ8JC3U         # Notify via slack

## Load conda and activate your environment
module load conda
conda activate hamilton_nn

## Run a python script
python GRU_h5_attention.py

## Or run a notebook:
# jupyter notebook --port SOMEPORT --no-browser