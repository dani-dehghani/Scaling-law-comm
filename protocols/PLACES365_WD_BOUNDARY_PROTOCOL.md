> Historical protocol. Original scientific text and old command spelling are preserved; use the [protocol index](README.md) and [current execution guide](../docs/implementation.md) for status and relocated commands.

# Places365 Weight-Decay Boundary Extension

## Why this experiment exists

The completed 2,400-trial refinement solved the 10,000-step convergence
problem and selected a strong global recipe: LR `0.0001`, WD `0.3`, constant
schedule, and batch size 256. Its mean relative validation regret was 0.132%.

It did not pass the registered acceptance gate because WD `0.3` was the upper
tested boundary. WD `0.3` was also the per-D winner at seven of eight sizes.
At D=1,300, the per-D LR `0.0003` was the upper LR boundary. The official
labeled Places365 validation split remains locked.

## Registered grid

```text
learning rates: 2e-5, 3e-5, 1e-4, 2e-4, 3e-4, 5e-4
weight decays:  0.1, 0.3, 1.0, 3.0
schedule:       constant
batch size:     256
maximum steps:  30,000
dataset sizes:  8
seeds:          10
------------------------------------------------------------
total:          6 x 4 x 8 x 10 = 1,920 validation-only trials
```

WD 0.1 and 0.3 connect this run to the completed refinement. WD 1.0 and 3.0
extend the open boundary. LR 5e-4 extends beyond the D=1,300 boundary; the
remaining LRs preserve the useful global and small-D regions.

## Run

```bash
source .venv/bin/activate
python run_hpo.py \
  --config configs/hpo_places365_wd_boundary_8d_10seeds_30k.yaml \
  --dry-run
```

The dry run must report 24 configurations, 8 sizes, 10 seeds, and 1,920 total
trials. Then:

```bash
tmux new -s places365_wd_boundary
./run_places365_wd_boundary_hpo.sh
```

The launcher uses 24 sequential shards of 80 trials. Ordinary restart is safe;
completed `trial_metrics.json` files are skipped.

## Acceptance and promotion

Require exact completion, an interior global LR and WD, finite results, and no
selected-global training-budget warning. Then promote the selected recipe:

```bash
python promote_places365_final_config.py
```

The script refuses promotion if any gate fails. On success it changes
`configs/data_sweep_places365_final.yaml` from `blocked` to `approved`, copies
the selected recipe into the final and smoke configs, and leaves the smoke
config restricted to `--train-only`.

Next commands:

```bash
python run_sweep.py \
  --config configs/data_sweep_places365_final_smoke.yaml \
  --train-only

python run_sweep.py \
  --config configs/data_sweep_places365_final.yaml \
  --dry-run

python run_sweep.py \
  --config configs/data_sweep_places365_final.yaml \
  --train-only
```

Only after the 80-run training-only sweep is complete with no budget warnings
may the same final config be run without `--train-only` for the one-time locked
evaluation.
