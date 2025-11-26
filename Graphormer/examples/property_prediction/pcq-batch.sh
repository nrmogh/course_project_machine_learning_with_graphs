#!/bin/bash
#SBATCH --job-name=pcq_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8

# Load conda 
source ~/.bashrc
conda activate graphormer

cd ~/tina_project/Graphormer/examples/property_prediction/

bash pcqv1-small.sh