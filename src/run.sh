#!/bin/bash
#SBATCH --job-name=ge_classification
#SBATCH --output=output/output_final.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1gb
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00

uv run python3 -u main.py