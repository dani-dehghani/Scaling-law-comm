#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

CONFIG="${1:-configs/data_sweep_global_hpo.yaml}"
python -m cli.check_environment
python -m cli.prepare_features --config "$CONFIG"
python -m cli.run_sweep --config "$CONFIG"
