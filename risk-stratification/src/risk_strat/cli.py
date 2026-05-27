from __future__ import annotations

import argparse

from .model import format_metrics, run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a readmission risk model on the UCI diabetes 130-US hospitals dataset."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Optional CSV export of the same diabetes cohort. If omitted, the UCI dataset is fetched.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="readmitted_binary",
        help="Binary target column name in the CSV export (default: readmitted_binary).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Directory to store model and metrics artifacts.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible train/test splitting.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_training(
        input_csv=args.input_csv,
        target_column=args.target_column,
        output_dir=args.output_dir,
        random_state=args.random_state,
    )
    print(format_metrics(result.metrics))
    print(f"Model artifact: {result.model_path}")
    print(f"Metrics artifact: {result.metrics_path}")


if __name__ == "__main__":
    main()

