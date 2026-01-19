#!/bin/bash
#SBATCH --job-name=ge_classification
#SBATCH --array=42-71
#SBATCH --output=output/yeast_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1gb
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00

uv run python3 -u ../scripts/benchmarks.py --dataset data/uci/yeast.csv --seed ${SLURM_ARRAY_TASK_ID} --target-index -1 --class-weight None --time-limit 600 --num-individuals 100 --grammar 0
