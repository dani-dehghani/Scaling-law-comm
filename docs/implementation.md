# Frozen ViT-Tiny data-size implementation

The program caches 192-dimensional image features from a pretrained frozen encoder, samples nested labeled subsets, trains affine classifiers and measures loss/accuracy. It does not fine-tune the encoder or implement a transmission channel. Read the [main report](../reports/thesis_progress.md) for the scientific interpretation and [script guide](script_guide.md) for simple flows.

## File roles

| Folder | Purpose |
|---|---|
| `src/` | Reusable configuration, features, sampling, training, selection, fitting and I/O |
| `cli/` | Command-line programs; invoke as `python -m cli.NAME` from the repository root |
| `scripts/` | Shell launchers combining commands; one preserved historical plot builder |
| `configs/` | YAML recipes; table below separates executed evidence from old templates |
| `protocols/` | Historical research designs and acceptance rules; old command spelling is historical |
| `tests/` | Small synthetic software/scientific regression checks |
| `artifacts/` | Compact immutable result tables and original config/selection/completion snapshots |

Run commands below from the repository root, `Scaling-law-comm/`. Install a compatible torch/torchvision pair for the intended machine before the remaining [requirements](../requirements.txt); [development requirements](../requirements-dev.txt) include pytest. Version ranges are not a locked historical environment.

## Inspect evidence without training

```bash
python -m cli.verify_artifacts
python -m cli.verify_artifacts --refit
python -m cli.build_report_figures
python -m cli.run_sweep --config configs/data_sweep_global_hpo.yaml --dry-run
python -m cli.run_hpo --config configs/hpo_places365_wd_boundary_8d_10seeds_30k.yaml --dry-run
```

The verifier's first command needs only the standard library; optional refitting needs NumPy/SciPy. Figure generation needs NumPy/pandas/Matplotlib and writes report figures only. A dry run loads the configuration and prints counts without training or final-data access. The training entry points import PyTorch even for dry runs.

From the repository root, run `python -m pytest -q`. Tests include synthetic head fitting; they do not download images or rerun the archived experiments.

## Configuration status

| Configuration | Meaning |
|---|---|
| `data_sweep_global_hpo.yaml` | Main CIFAR recipe, 12 sizes × 10 seeds; shared shell launcher's default |
| `hpo_core_12d_10seeds.yaml` | Historical core CIFAR search, 7,920 trials |
| `hpo_places365_global_8d_10seeds.yaml` | Broad Places search, 5,280 trials |
| `hpo_places365_refinement_8d_10seeds_30k.yaml` | Refinement, 2,400 trials |
| `hpo_places365_wd_boundary_8d_10seeds_30k.yaml` | Extended WD search, 1,920 trials |
| `data_sweep_places365_final.yaml`, `_smoke.yaml` | Preserved **blocked provisional templates** (.0001/.3), not the executed approved .0002/1.0 recipe |
| `data_sweep_vit_tiny.yaml` | Earlier exploratory CIFAR recipe; still the old Python-entry-point default, so supply `--config` explicitly |
| `hpo_lr_stage1.yaml` | Early 72-trial LR pilot; Python HPO default is historical |
| `data_sweep_legacy_matched_control.yaml` | Proposed control; no completed matched-control evidence imported |

For what actually ran, read [CIFAR snapshot](../artifacts/cifar100_main/config_snapshot.json) or [approved Places snapshot](../artifacts/places365_main/config_snapshot.json). JSON snapshots can also be read by the YAML loader for inspection, for example `python -m cli.run_sweep --config artifacts/places365_main/config_snapshot.json --dry-run`. Do not alter original artifact snapshots to configure a new experiment.

## Execution and output boundaries

`bash scripts/run_all.sh` performs environment reporting, feature preparation and a full CIFAR sweep: it is an expensive execution command, not a preview. It defaults to the documented main recipe and can download/build caches if required. The three Places HPO launchers prepare features, run shards **sequentially** and aggregate; they do not launch shards in parallel. Their log directories assume their named default experiment, so use the Python CLI directly for custom experiment output layouts.

Normal training writes ignored `data/`, `cache/` and `results/` directories under the repository root. Nothing trains inside `artifacts/`. Recover the historical data/cache identities and resolve [known integrity gaps](implementation_status.md) before using this snapshot for new archival results. No new training was needed to publish this report.

`cli.analyze_results` rewrites aggregate/fit/plot outputs in its supplied result directory. Use a separate working copy for new analyses, never an original artifact folder. `cli.compare_sweeps` similarly writes to an explicit derivative output directory. `cli.promote_places365_final_config` is a deliberate configuration-writing program, not a read-only evidence viewer.

[Reproducibility scope](reproducibility.md) · [artifact map](../artifacts/README.md) · [protocol index](../protocols/README.md)
