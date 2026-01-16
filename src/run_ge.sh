#!/bin/bash
#SBATCH --job-name=ge_classification
#SBATCH --array=42-71
#SBATCH --output=output/heart_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1gb
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00

uv run python3 -u ge_benchmarks.py --dataset datasets/heart.csv --seed ${SLURM_ARRAY_TASK_ID} --target-index -1 --class-weight None --time-limit 600 --num-individuals 100 --grammar 0
