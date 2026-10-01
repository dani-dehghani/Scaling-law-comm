# Places365 Focused HPO Refinement

## Why this run exists

The completed 5,280-trial Places365 HPO selected one strong global
configuration inside its tested grid:

```text
learning rate = 0.0002
weight decay  = 0.01
batch size    = 256
schedule      = constant
mean relative validation regret = 0.1717%
```

That run successfully localized the useful optimizer region, but it did not
pass the registered acceptance gate:

1. Weight decay `0.01` was the upper tested boundary and won at every `D`.
2. All ten selected-global runs at `D=250,000` and `D=500,000` reached the
   10,000-step ceiling, with mean best steps 9,698 and 9,714.
3. The per-`D` winners at `D=1,300`, `3,650`, and `10,000` used the lower LR
   boundary `0.00003`; the `D=10,000` winner was also budget-limited.

Therefore the next scientific question is not whether to add more Places365
images. The eight dataset sizes stay fixed. The question is whether the
optimizer search and training budget were wide enough to compare those sizes
fairly.

## Frozen protocol

The refinement preserves:

- the same pretrained frozen ViT-Tiny and cached 192-dimensional features;
- the same fixed internal validation split of 50 images per class;
- the same locked official labeled validation split for later final evaluation;
- the same eight nested dataset sizes and ten seeds;
- AdamW, constant schedule, batch size 256, evaluation every 20 steps,
  patience of 50 evaluations, and minimum delta `0.00001`.

Only the registered HPO grid and shared maximum step ceiling change.

## Refined grid

```text
learning rates:  1e-5, 2e-5, 3e-5, 1e-4, 2e-4, 3e-4
weight decays:   0.001, 0.01, 0.03, 0.1, 0.3
schedule:        constant
batch size:      256
maximum steps:   30,000
data sizes:      8
seeds:           10
---------------------------------------------------------
total:           6 x 5 x 8 x 10 = 2,400 trials
```

The LR grid extends below the small-data boundary and retains the competitive
global region. The WD grid keeps two anchors at or below the old winner and
extends three values above it. The common 30,000-step ceiling is three times
the original ceiling; early stopping prevents already-converged trials from
using it unnecessarily.

Results from the 10,000-step HPO are not copied into this directory. Promoted
configurations are rerun because comparisons under different maximum budgets
would not be controlled.

## Test protection

This remains a validation-only experiment. `run_hpo.py` loads the cache with
`include_test=False`. The official labeled Places365 validation split must not
be inspected, evaluated, or used to change this refinement.

## Launch

From the repository root on the GPU machine:

```bash
source .venv/bin/activate
python run_hpo.py \
  --config configs/hpo_places365_refinement_8d_10seeds_30k.yaml \
  --dry-run
```

The dry run must report:

```text
Configurations: 30
Dataset sizes: 8
Seeds: 10
Total trials: 2400
```

Then launch in tmux:

```bash
tmux new -s places365_refine
./run_places365_hpo_refinement.sh
```

Detach with `Ctrl-b`, then `d`. Reattach with:

```bash
tmux attach -t places365_refine
```

The wrapper uses 24 sequential shards of 100 trials. Re-running the same
wrapper is safe: completed `trial_metrics.json` files are skipped. Do not use
`--force` during ordinary resume.

## Acceptance gate

After aggregation, require:

1. `completion.json` reports 2,400/2,400 unique expected trials.
2. The selected global LR and WD are not at a search boundary.
3. No selected-global data size has `training_budget_warning=True`.
4. The global winner is stable across seeds and remains close to the per-`D`
   winners.
5. Histories are finite and show a validation minimum rather than continued
   improvement at step 30,000.

If WD `0.3` wins repeatedly, extend the WD boundary again. If selected runs
remain near 30,000 steps, increase the shared ceiling again. Do not open the
locked final evaluation split until these checks pass.

## What happens after acceptance

Create one fixed-recipe Places365 data-scaling configuration using the accepted
global LR, WD, schedule, batch size, and a demonstrated sufficient common step
ceiling. Run all 80 `(D, seed)` combinations in training-only mode. Only after
all 80 runs pass completeness and convergence may the official labeled
Places365 validation split be evaluated once and the final scaling curve be
fit.
