> Historical protocol. Original scientific text and old command spelling are preserved; use the [protocol index](README.md) and [current execution guide](../README.md) for status and relocated commands.

# Main Global-HPO Data-Scaling Experiment

## Scientific question

For a fixed pretrained ViT-Tiny representation and one fixed linear-probe
training recipe, how does CIFAR-100 test loss change as the number of labeled
training examples `D` increases?

The independent variable is `D`. The backbone, feature cache, train/validation
split, optimizer, learning rate, weight decay, batch size, schedule, early
stopping rule, and maximum step ceiling remain fixed.

## Frozen protocol

- Model: `vit_tiny_patch16_224.augreg_in21k_ft_in1k`
- Backbone: frozen
- Feature dimension: 192
- Classifier: one trainable 100-class linear layer
- Dataset sizes: 12 values from 200 through 45,000
- Seeds: 0 through 9
- Optimizer: AdamW
- Learning rate: `0.0001`
- Weight decay: `0.01`
- Batch size: 256
- Schedule: constant
- Maximum steps: 30,000
- Validation interval: 20 steps
- Early stopping patience: 50 evaluations, equivalent to 1,000 steps
- Minimum validation improvement: `0.00001`

Learning rate and weight decay are the global configuration selected by the
completed validation-only 7,920-trial HPO. The maximum step ceiling was raised
from 10,000 because every seed at `D=45,000` selected a checkpoint near the old
limit. This ceiling is identical for every `D`; early stopping is the common
convergence rule.

## Two-stage test protection

First train and validate every run while keeping test features inaccessible:

```bash
python run_sweep.py \
  --config configs/data_sweep_global_hpo.yaml \
  --prepare-cache \
  --train-only
```

Inspect `results/global_hpo_main_sweep/completion.json`. It must report:

- `training_complete: true`
- 120 trained runs
- an empty `budget_warning_runs` list
- `test_evaluation_complete: false`

If any budget warning remains, increase `max_steps` for every `D`, choose a new
output directory, and repeat the validation-only run. Do not evaluate test.

After the convergence gate passes, run the same configuration without
`--train-only`:

```bash
python run_sweep.py \
  --config configs/data_sweep_global_hpo.yaml
```

The script reuses all 120 checkpoints, evaluates the test split, writes the
summary, fits `L(D) = L_inf + b * D^(-beta)`, and creates loss and accuracy
figures. Test results must not be used to revise this protocol.

## Matched control

`configs/data_sweep_legacy_matched_control.yaml` uses the same step budget,
early stopping, dataset sizes, and seeds, but restores the earlier LR `0.003`
and weight decay `0.0001`. This optional control isolates the effect of those
two optimizer settings more cleanly than comparison with the historical
epoch-based experiment.

## Comparison

After downloading the new result directories, compare the main run with the
historical and matched-control runs:

```bash
python compare_sweeps.py \
  --run global_hpo=results/global_hpo_main_sweep \
  --run matched_legacy=results/legacy_hparams_matched_control \
  --run historical_lr003=reports/baselines/formal_lr_003 \
  --output-dir reports/main_sweep_comparison
```

The historical run is a useful benchmark, but it used an epoch-based cosine
schedule. It is not a controlled LR-only comparison. The matched control is the
appropriate causal comparison for LR and weight decay.
