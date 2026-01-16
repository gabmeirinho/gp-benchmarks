#!/bin/bash
#SBATCH --job-name=m6_parse_heart
#SBATCH --output=output/heart_parse.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1gb
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00

DATASET="heart_grammar"

uv run python3 -u parser.py --dataset "${DATASET}"
