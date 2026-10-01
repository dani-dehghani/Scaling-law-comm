# Places365 Global Hyperparameter Search

## Scientific objective

This validation-only experiment selects one global linear-probe configuration
for a frozen ImageNet-pretrained ViT-Tiny on Places365-Standard. It does not use
the public Places365 validation split during model selection.

Places365's public test labels are unavailable. Therefore the protocol makes a
fixed internal validation split of 50 images per class from the official
training split and locks the official labeled validation split as the final
evaluation set for the later data-scaling experiment.

## Registered data protocol

- Official Places365-Standard training images: 1,803,460
- Internal HPO validation: 18,250 images (50 per class)
- Remaining training pool: 1,785,210 images
- Locked final evaluation: official 36,500-image validation split
- HPO sizes: 1,300; 3,650; 10,000; 25,000; 45,000; 100,000;
  250,000; and 500,000
- HPO seeds: 0 through 9
- Subsets: deterministic, nested, and stratified-proportional

The same ImageNet-pretrained checkpoint and deterministic inference transform
used for CIFAR-100 are retained. The backbone is frozen and every feature is
computed once. Only a 192-to-365 linear classifier is trained.

## Search grid

The registered grid deliberately matches the completed CIFAR-100 core HPO:

```text
11 learning rates
 6 weight decays
 1 constant schedule
 1 batch size (256)
 8 dataset sizes
10 seeds
--------------------------------
5,280 validation-only trials
```

The global selector minimizes mean relative validation-loss regret across all
eight dataset sizes, giving each size equal influence. `D=45,000` is included
explicitly so Places365 and CIFAR-100 can later be compared at the same number
of labeled training examples under the same LR/weight-decay grid, batch size,
schedule, maximum steps, validation interval, patience, and minimum delta.

The shared ceiling is 10,000 optimizer steps, matching the CIFAR-100 core HPO.
This controls the optimization protocol at the matched-size point, but it means
the largest Places365 subsets receive fewer effective epochs. Any selected
configuration whose best checkpoint lies near the ceiling is not accepted as
converged; it triggers a separate, registered budget refinement.

## Launch procedure

The 256-pixel Places365 archive is large. Confirm that the repository's `data/`
filesystem has sufficient space before starting. From the repository root:

```bash
python run_hpo.py \
  --config configs/hpo_places365_global_8d_10seeds.yaml \
  --dry-run
```

The dry run must report 66 configurations and 5,280 total trials.

On the shared GPU workstation, launch the resumable low-priority workflow in a
named tmux session:

```bash
tmux new -s places365_hpo
./run_places365_hpo.sh
```

Detach with `Ctrl-b`, then `d`. The script first downloads/loads Places365 and
creates the feature cache. It then runs 24 sequential shards of 220 trials each
so no single normal process accumulates the entire experiment's CPU time. Both
the cache and finished trials are reused after interruption.

Monitor from another shell with:

```bash
tail -f results/hpo_places365_global_8d_10seeds/logs/prepare_features.log
tail -f results/hpo_places365_global_8d_10seeds/logs/shard_0000_of_0024.log
nvidia-smi
```

Do not start a second copy of the same shard. Do not use `--force` during normal
resume. If the wrapper stops, running `./run_places365_hpo.sh` again safely
reuses the cache, skips completed trials, and continues.

## Acceptance gate

The final `completion.json` must contain 5,280 completed trials with zero
missing, unexpected, or duplicate trials. Inspect the selected configuration's
boundary labels and the fraction of runs whose best checkpoint is near the
10,000-step ceiling. A boundary winner or training-budget warning requires a
registered refinement before the final Places365 scaling sweep.

Raw images, cached features, checkpoints, and individual trial histories are
not committed to GitHub. Commit the code, immutable config, this protocol, and
compact aggregate CSV/JSON evidence after completion.
