from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.feature_cache import prepare_feature_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen ViT-Tiny dataset features")
    parser.add_argument("--config", default="configs/data_sweep_vit_tiny.yaml")
    parser.add_argument("--force", action="store_true", help="Rebuild an existing cache")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root / args.config)
    prepare_feature_cache(cfg, project_root, force=args.force)


if __name__ == "__main__":
    main()
