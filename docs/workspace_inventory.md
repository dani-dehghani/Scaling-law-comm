# Workspace and evidence map

The original research archive remains outside Git as `LOCAL_ARCHIVE` (`vit_tiny_results` on the researcher's laptop). The paper folder likewise remains outside Git. This document maps roles rather than treating every duplicate directory as new scientific evidence.

| Local collection | Meaning | Repository destination |
|---|---|---|
| `global_hpo_main_sweep/` | Main 120-run CIFAR results with heads/histories/subset IDs | Compact originals in `artifacts/cifar100_main/` |
| `places365_final_sweep/` | Main 80-run Places results | Compact originals in `artifacts/places365_main/` |
| `places365_wd_boundary_analysis/` | Completed Places extended WD search | Compact originals in `artifacts/places365_boundary_hpo/` |
| `places365_hpo_results/hpo_places365_global_8d_10seeds/` | Complete broad Places search | Existing compact `artifacts/places365_global_hpo/` |
| Top-level `hpo_places365_global_8d_10seeds/` | Partial copy: 2,767 records overlap the complete collection | Not counted again |
| Nested refinement extraction | Complete 2,400-trial refinement | Existing compact `artifacts/places365_refinement_hpo/` |
| `results/places365_final_sweep/` | Training-only mirror; 160 matching run JSONs, not another final sweep | Not counted again |
| `formal_vit_tiny_cifar100_data_scaling_v1/` | Environment/configuration/feature metadata collection | Retained locally; not confused with similarly named numerical results |
| `vit_tiny_cifar100_data_scaling_formal_v1/`, LR baselines | Early numerical curves | Existing compact `artifacts/baselines/` |
| `smoke_D50/`, pilots, LR checks | Exploratory stages; smoke name does not equal recorded D | Register records scope; original detailed files remain local |
| `vit_tiny_data_scaling/` | Later working source, reports and learning material | Separate source history; not blindly overlaid onto GitHub |
| `github_ready_2026-10-01/`, `thesis_github_package_2026-09-30/` | Earlier export packages, not additional experiments | Relevant compact artifacts already represented |
| `review_2026-10-05/`, `review_2026-10-20/`, `output/` | Audits, derived analyses and educational artifacts | Relevant verification/diagnostic derivatives copied with dates |
| Compressed archives and extracted copies | Containers or duplicates | Retained locally, not counted as repeated trials |
| `Thesis Papers/` | 15 PDFs (13 unique), two planning DOCX files | Paraphrased review, file hashes and primary links only |

`artifacts/` paths above are relative to the [ViT-Tiny implementation](../experiments/models/vit-tiny/README.md). Machine-readable experiment locations are root-relative in the [register](experiment_index.csv); `LOCAL_ARCHIVE/…` entries deliberately identify evidence not imported to Git.

The untouched original archive contains additional learning exports, logs and intermediate files. They are supporting history, not automatically verified thesis results. No files were deleted from that archive or the original paper folder. Local heads and subset arrays must be backed up separately; Git here is not their backup.
