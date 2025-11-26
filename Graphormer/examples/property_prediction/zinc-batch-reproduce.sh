#!/bin/bash
#SBATCH --job-name=zinc_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

# Load conda 
source ~/.bashrc
conda activate graphormer

cd ~/tina_project/Graphormer/examples/property_prediction/

bash zinc_bs-256_wu-40k.sh