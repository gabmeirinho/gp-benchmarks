This repository contains a framework for automated feature engineering using Genetic Programming (GP), built on top of the `geneticengine` library. The system evolves complex mathematical expressions and aggregations to augment datasets and improve classification performance.

## Overview

The core logic is implemented in benchmarks.py. It uses Genetic Programming to discover new features that maximize the weighted F1-score while minimizing the number of features and the complexity of the expression trees.

### Fitness Function
The multi-objective fitness function is defined in benchmarks.py as:
1. Maximize Weighted F1-score: The performance of a `RandomForestClassifier` on the augmented dataset.
2. Minimize Unique Features: The number of original features used in the expression.
3. Minimize Node Count: The total number of operators and terminals in the GP tree.

Mathematically, the goal is to optimize:
$$f(individual) = [F1_{weighted}, -Count_{features}, -Count_{nodes}]$$

## Project Structure

*   benchmarks.py: Benchmarking script including preprocessing, GP search, and post-run evaluation (RFE).
*   parser.py: Utility to aggregate and summarize results from multiple seeds.
*   data: Directory for datasets.
*   src: SLURM batch scripts for distributed execution.

## Getting Started

### Prerequisites

This project uses `uv` for python package management.

### Running Experiments

To run a single experiment on a dataset:

```sh
uv run python3 scripts/benchmarks.py --dataset data/uci/yeast.csv --seed 42 --target-index -1 --time-limit 600
```

### Batch Execution (SLURM)

Use the provided shell scripts in src to run experiments across multiple seeds:

```sh
sbatch src/run_ge.sh
```

### Parsing Results

After running multiple seeds, use parser.py to generate a summary CSV:

```sh
uv run python3 scripts/parser.py --dataset yeast
```

This script executes `parse_results`, which combines individual seed outputs and calculates statistics (Mean, Std, Max, etc.) for metrics such as `weighted_f1` and `augmented_corr_cv_f1`.

## Feature Grammar

The GP engine evolves expressions using the following components defined in benchmarks.py:
*   Arithmetic: `Add`, `Subtract`, `Multiply`, `Divide`.
*   Aggregations: `Mean` and `Max` using sliding windows (lookbacks).
*   Terminals: `Feature` indices from the original dataset.