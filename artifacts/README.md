# Compact recorded evidence

Original numerical files are preserved byte for byte. Do not train or regenerate analysis in these directories. New figures are generated outside this tree from the saved tables.

| Collection | Content |
|---|---|
| `cifar100_main`, `places365_main` | 120/80 main run summaries, D aggregates, original fits, executed configs and completion |
| `cifar100_core_hpo` | 7,920 validation trials, aggregate/selection/completion evidence |
| `places365_global_hpo` | 5,280 validation trials |
| `places365_refinement_hpo` | 2,400 validation trials |
| `places365_boundary_hpo` | 1,920 validation trials; selected LR=.0002, WD=1.0 |
| `baselines/` | 35, 70 and 120 historical CIFAR run records and fits |
| `archival_package_2026-10-01/` | Previous manifest, validation and reproducibility note; historical scope |

[MANIFEST.csv](MANIFEST.csv) gives current relative paths, sizes and SHA-256 hashes (excluding itself). Older manifests refer to their original package layout, not this tree. The [register](../docs/experiment_index.csv) maps experiment IDs to these collections.

Training heads, feature tensors, ID arrays and full histories remain in the local archive. Existing PNGs were relocated to [archived report figures](../reports/figures/archived). Imported compact data are evidence for the stated measured protocols, not proof of all historical source/data integrity.
