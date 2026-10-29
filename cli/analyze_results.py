from __future__ import annotations

import argparse
from pathlib import Path

from src.scaling import analyze_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate runs and fit L(D)=L_inf+bD^-beta")
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()
    fit = analyze_results(args.results_dir / "summary.csv", args.results_dir)
    print(fit)


if __name__ == "__main__":
    main()
