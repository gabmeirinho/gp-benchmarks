import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

def parse_results(results_dir: Path, output_path: Path | None = None) -> Path:
    if not results_dir.exists() or not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    csv_files: Iterable[Path] = sorted(results_dir.glob("*.csv"))
    csv_files = [f for f in csv_files if f.is_file() and not f.name.endswith(("_combined.csv", "_summary.csv"))]
    if not csv_files:
        raise ValueError(f"No CSV files found in {results_dir}")

    frames = [pd.read_csv(csv_path) for csv_path in csv_files]
    combined = pd.concat(frames, ignore_index=True)

    # Keep only the metrics we care about and drop feature name lists to save space.
    drop_cols = [c for c in ["elapsed_seconds"] if c in combined.columns]
    if drop_cols:
        combined = combined.drop(columns=drop_cols)

    required_columns = ["seed", "weighted_f1", "num_features", "num_nodes"]
    missing = [c for c in required_columns if c not in combined.columns]
    if missing:
        raise ValueError(f"Expected columns missing from input CSVs: {missing}")
    
    metrics = ["weighted_f1", "num_features", "num_nodes", "archive_size", "augmented_set_size", "augmented_set_size_after_corr", "augmented_cv_f1", "augmented_corr_cv_f1"]

    stats_funcs = [
        ("Max", pd.Series.max),
        ("Min", pd.Series.min),
        ("Mean", pd.Series.mean),
        ("Median", pd.Series.median),
        ("Std", pd.Series.std),
    ]

    stats_df = pd.DataFrame(
        [
            {"statistic": name, **combined[metrics].agg(func).to_dict()}
            for name, func in stats_funcs
        ]
    )

    best_row = combined.loc[combined["weighted_f1"].idxmax()]
    best_df = pd.DataFrame(
        {
            "statistic": ["BestIndividual"],
            "weighted_f1": [best_row["weighted_f1"]],
            "num_features": [best_row["num_features"]],
            "num_nodes": [best_row["num_nodes"]],
            "features": [best_row.get("features")],
            "archive_size": [best_row.get("archive_size")],
            "augmented_set_size": [best_row.get("augmented_set_size")],
            "augmented_set_size_after_corr": [best_row.get("augmented_set_size_after_corr")],
            "augmented_cv_f1": [best_row.get("augmented_cv_f1")],
            "augmented_corr_cv_f1": [best_row.get("augmented_corr_cv_f1")],
        }
    )

    worst_row = combined.loc[combined["weighted_f1"].idxmin()]
    worst_df = pd.DataFrame(
        {
            "statistic": ["WorstIndividual"],
            "weighted_f1": [worst_row["weighted_f1"]],
            "num_features": [worst_row["num_features"]],
            "num_nodes": [worst_row["num_nodes"]],
            "features": [worst_row.get("features")],
            "archive_size": [worst_row.get("archive_size")],
            "augmented_set_size": [worst_row.get("augmented_set_size")],
            "augmented_set_size_after_corr": [worst_row.get("augmented_set_size_after_corr")],
            "augmented_cv_f1": [worst_row.get("augmented_cv_f1")],
            "augmented_corr_cv_f1": [worst_row.get("augmented_corr_cv_f1")],
        }
    )

    output_df = pd.concat([stats_df, best_df, worst_df], ignore_index=True)

    
    if output_path is None:
        stem = results_dir.name.removesuffix("_results")
        output_path = results_dir / f"{stem}_summary.csv"

    output_df.to_csv(output_path, index=False)
    print(f"Saved combined results to {output_path}")

    combined.to_csv(results_dir / "all_seeds_combined.csv", index=False)
    print(f"Saved all seeds combined results to {results_dir / 'all_seeds_combined.csv'}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Combine per-seed GE result CSVs into one summary CSV."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--results-dir",
        help="Path to a directory containing per-seed CSVs (e.g., ge/heart_ge_results).",
    )
    group.add_argument(
        "--dataset",
        help="Dataset stem; will look for <stem>_results under the script directory.",
    )
    parser.add_argument(
        "--output",
        help="Optional output CSV path. Defaults to <results-dir>/<stem>_summary.csv.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    results_dir = Path(args.results_dir) if args.results_dir else base_dir / f"{args.dataset}_results"
    output_path = Path(args.output) if args.output else None

    parse_results(results_dir, output_path=output_path)


if __name__ == "__main__":
    main()
