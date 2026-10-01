# Frozen ViT-Tiny data-scaling experiments

Research snapshot: 1 October 2026. This project studies how labeled dataset size affects a linear classifier trained on a fixed pretrained ViT-Tiny representation.

## Review entry points

1. [CIFAR-100 experiment report](reports/cifar-100_experiment_report.md): completed work, results, interpretation and pending experiments.
2. [Historical comparison](reports/comparison/README.md): measured before/after results and protocol differences.
3. [Reproducibility and artifact scope](REPRODUCIBILITY.md): provenance, validation and archival requirements.

CIFAR-100 main evaluation is complete: 120 runs, 12 sizes, 10 seeds, fitted beta 0.520413. Places365 global HPO and refinement are complete; the boundary extension is reported running remotely, and its completed artifacts are not present in this snapshot. Final Places365 evaluation remains pending.

## Project layout

- `src/`, root Python entry points and shell launchers: current implementation.
- `configs/`: experiment definitions, including the blocked Places365 final config.
- `tests/`: scientific and software invariant tests.
- Root protocol Markdown files: search designs, training/evaluation procedures and acceptance gates.
- `reports/`: compact trial/run tables, selections, completion records, fits and figures.
- `MANIFEST.csv`: packaged file hashes and source-relative provenance.
- `VALIDATION.json`: checks performed on this snapshot.

## Environment and execution

Install a suitable PyTorch and torchvision build for the execution machine, then:

```bash
python -m pip install -r requirements.txt
python check_environment.py
```

For developer checks:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests
```

Follow [MAIN_EXPERIMENT_PROTOCOL.md](MAIN_EXPERIMENT_PROTOCOL.md) for the CIFAR-100 main procedure. Follow [PLACES365_WD_BOUNDARY_PROTOCOL.md](PLACES365_WD_BOUNDARY_PROTOCOL.md) for the current Places365 acceptance/promotion procedure. The final Places365 config requires successful promotion before execution.

To regenerate the historical comparison from saved metrics without training or new test evaluation:

```bash
python reports/comparison/build_figures.py
```

Raw data, features, checkpoints, sampled-index arrays and complete histories remain in research storage. This is a code and result-evidence snapshot, not a complete training archive.
