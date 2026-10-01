#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/data_sweep_vit_tiny.yaml}"
python check_environment.py
python prepare_features.py --config "$CONFIG"
python run_sweep.py --config "$CONFIG"
