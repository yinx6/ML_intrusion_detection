from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from ids_ml.pipelines.train import train_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IDS-ML models.")
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--target", default="Label", help="Target column name")
    parser.add_argument("--output-dir", default="artifacts", help="Artifacts output directory")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional sample size for fast experiments")
    parser.add_argument("--tracking-uri", default=None, help="Optional MLflow tracking URI")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data to hold out for testing (default: 0.20)")
    parser.add_argument("--split-seed", type=int, default=42, help="Random seed for the train/test split (default: 42)")
    args = parser.parse_args()

    metrics = train_all(
        data_path=args.data,
        target_col=args.target,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        tracking_uri=args.tracking_uri,
        test_size=args.test_size,
        split_seed=args.split_seed,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
