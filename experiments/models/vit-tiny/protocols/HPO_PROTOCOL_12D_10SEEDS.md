> Historical protocol. Original scientific text and old command spelling are preserved; use the [protocol index](README.md) and [current execution guide](../README.md) for status and relocated commands.

# Full HPO protocol for the ViT-Tiny data-scaling experiment

## 1. Scientific question

The final claim is not merely that one training run obtained a lower loss with
more data. The claim is that test loss follows a stable relationship with the
number of labeled training examples, after giving the frozen ViT-Tiny probe a
reasonable and reproducible opportunity to optimize at every `D`.

The HPO objective is therefore the mean validation cross-entropy loss across
seeds. Test features remain inaccessible until the hyperparameter policy has
been locked.

There is no mathematically "complete" finite hyperparameter search: learning
rate, regularization, training time, and batch size are continuous or unbounded.
Here, complete means that we search the scientifically important region at
useful resolution, diagnose boundaries and convergence, and perform a separate
sensitivity check for choices that should remain controlled.

## 2. Stage 2A: core LR and weight-decay search

Configuration:

```text
configs/hpo_core_12d_10seeds.yaml
```

The design contains:

```text
12 dataset sizes
10 seeds
11 learning rates
 6 weight decays
 1 constant schedule
 1 batch size
--------------------------------
7,920 validation-only trials
```

The dataset sizes are the same 12 points used by the earlier formal sweep:

```text
200, 500, 1,000, 1,500, 2,500, 5,000,
7,500, 10,000, 15,000, 25,000, 35,000, 45,000
```

The LR range is concentrated around the useful pilot region. `0.01` and `0.03`
were excluded because the completed pilot already showed large, consistent
validation-loss penalties. `0.00003` extends the lower boundary because the
pilot winner at `D=5,000` was the lowest LR tested.

Weight decay includes zero and spans `1e-6` through `1e-2` for nonzero
regularization. The maximum optimization budget is raised from 3,000 to 10,000 steps
because the pilot winners at `D=5,000` and `D=45,000` were still improving near
the old limit. The smaller `min_delta` prevents the checkpoint rule from
discarding improvements between `1e-5` and `1e-4`.

## 3. Before the expensive run

On the remote GPU machine, from the repository root:

```bash
python check_environment.py
python -m pytest -q
python run_hpo.py --config configs/hpo_core_12d_10seeds.yaml --dry-run
python prepare_features.py --config configs/hpo_core_12d_10seeds.yaml
```

The dry run must report exactly:

```text
Configurations: 66
Dataset sizes: 12
Seeds: 10
Total trials: 7920
```

Run a smoke shard before committing the full budget. A shard is a deterministic
fraction of the full task list, not a different experiment:

```bash
python run_hpo.py \
  --config configs/hpo_core_12d_10seeds.yaml \
  --num-shards 7920 --shard-index 0
```

This executes exactly one deliberately slow corner trial. Inspect its history,
recorded `wall_time_seconds`, and GPU utilization. If the run is healthy,
execute the full search. On one GPU, run normally or run shards sequentially.
On multiple GPUs, assign
one shard process to each GPU with `CUDA_VISIBLE_DEVICES`; all workers must use
the same shared result directory.

Every completed trial is resumable. Re-running the same command skips finished
trial metric files. Do not use `--force` unless a trial is known to be invalid.

After every shard has completed:

```bash
python run_hpo.py \
  --config configs/hpo_core_12d_10seeds.yaml \
  --aggregate-only
```

Aggregation refuses to produce final selections unless all 7,920 unique trials
are present. `completion.json` records expected, completed, missing, duplicate,
and unexpected counts.

## 4. How selection works

For each `(D, LR, weight decay)` combination, the runner computes:

- mean validation loss over 10 seeds;
- standard deviation over seeds;
- mean validation accuracy;
- mean best step;
- fraction of seeds whose best checkpoint was near the step limit.

`selected_by_D.json` chooses the lowest mean validation loss independently at
each `D`. This is the optimized envelope.

`selected_global.json` chooses one fixed configuration by mean relative regret:

```text
regret(D, config) = loss(D, config) / best_loss(D) - 1
```

Averaging relative regret gives every dataset size equal influence. This fixed
global configuration is the primary controlled scaling curve; the per-D
envelope is a sensitivity result showing whether tuning changes the conclusion.

## 5. Mandatory diagnostics before accepting Stage 2A

Do not accept a selected configuration merely because it is ranked first.
Check:

1. `completion.json` says complete with zero missing, duplicate, or unexpected trials.
2. The selected LR and weight decay are not repeatedly at a search boundary.
3. `training_budget_warning` is false at every `D`.
4. The best configuration beats nearby alternatives consistently across seeds.
5. Training histories contain finite losses and show a real validation minimum.
6. Both the fixed-global curve and per-D optimized envelope decrease with `D`.

If a winner is at a boundary, extend only that boundary and rerun the necessary
region. If a winner is near 10,000 steps, increase the budget before interpreting
its loss as converged.

## 6. Stage 2B: schedule and batch-size sensitivity

Schedule and batch size are intentionally not multiplied into Stage 2A. Doing
so would create 47,520 runs before learning which LR/regularization region is
useful. Random-search and adaptive-resource HPO literature shows why spending
equal compute on every Cartesian combination is inefficient.

After Stage 2A, take the union of:

- the three best global LR/weight-decay pairs;
- the two best pairs at each `D`.

For that promoted set, test:

```text
schedules: constant, cosine
batch sizes: 64, 128, 256
all 12 D values
all 10 seeds
```

Batch-size comparisons must use an equal-example or equal-epoch budget, not an
equal-step budget. One step with batch size 256 processes four times as many
examples as one step with batch size 64. This sensitivity stage will be created
from the Stage 2A results so that it contains only promoted LR/weight-decay
pairs and avoids rerunning thousands of poor configurations.

## 7. Locking the final experiment

Before any new test evaluation, record:

- the selected fixed global configuration;
- any per-D configuration policy used for the envelope;
- checkpoint selection by validation loss;
- the exact 12 D values and 10 seeds;
- the stopping and convergence rules;
- the scaling-law model and diagnostics to be reported.

Then run two final test sweeps:

1. Primary: one fixed global configuration for every `D`.
2. Sensitivity: validation-selected per-D configurations.

Agreement between these curves supports a data-scaling interpretation. A large
disagreement means optimization interacts with `D` and must be part of the
reported conclusion.

## 8. Methodological basis

- Bergstra and Bengio, "Random Search for Hyper-Parameter Optimization":
  https://www.jmlr.org/papers/v13/bergstra12a.html
- Li et al., "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization":
  https://www.jmlr.org/papers/v18/16-558.html
- Dosovitskiy et al., "An Image is Worth 16x16 Words":
  https://arxiv.org/abs/2010.11929
- Hestness et al., "Deep Learning Scaling is Predictable, Empirically":
  https://arxiv.org/abs/1712.00409

The important shared principles are validation-only model selection, searching
optimization settings instead of assuming one, fixed disjoint evaluation data,
multiple seeds, reproducible data subsets, and convergence checks before fitting
a scaling relationship.
