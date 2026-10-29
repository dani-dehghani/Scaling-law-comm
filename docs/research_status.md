# Research status — 28 October 2026

The completed result is a fixed-representation downstream D baseline on CIFAR-100 and Places365. The planned model-size and communication studies have not been completed. The [full report](../reports/thesis_progress.md) is the authoritative current narrative.

| Area | Current evidence | Open point |
|---|---|---|
| Main CIFAR D study | 120 final runs, 12 sizes, 10 seeds; 76.608% at D=45k | Global HPO selected upper-edge WD=.01; predictive fit and full provenance remain unresolved |
| Main Places D study | 80 final runs, 8 sizes, 10 seeds; 46.668% at D=500k | Official validation supplies final evaluation; no original feature cache locally |
| Hyperparameter selection | Four imported complete grids, 17,520 trials; global choices independently recalculated | Different search budgets/spaces; no claim of globally optimal recipes |
| Historical baselines | 225 compact records retained | Different grids/recipes; comparisons are descriptive |
| Literature | 15 PDF files, 13 unique; method passages reviewed and page referenced | Original proposal and several secondary-cited papers absent |
| Software | Original numerical source retained; entry points relocated; tests and dry runs recorded | Newer local repairs are a separate version; passing tests do not establish image-to-result reproduction |
| Fits | Recorded parameters independently reproduced; separate exploratory diagnostics retained | Model form/extrapolation and broader uncertainty remain open |

The completed Places boundary search selected LR=.0002/WD=1.0. This resolves the old README's pending status. The blocked Places YAMLs remain historical provisional templates, not the executed approved snapshot; use the artifact snapshot to understand what actually ran.

## Work recorded in this update

- Organized code, protocols, compact artifacts, reports and figures into separate roles.
- Imported completed compact Places evidence and preserved original result bytes.
- Reconciled stale summaries without rewriting original protocols as if they had always contained later knowledge.
- Added a comprehensive narrative, experiment register, paper comparison and beginner script guide.
- Verified compact numbers and refits and checked available local checkpoint/history/subset consistency.

This is the 28 October review/publication update. Existing training was not rerun and execution dates are not inferred from copied file timestamps.

## Next work, in order

1. Supervisor decision: formalize the existing D baseline, or move to the planned N sweep after reproducibility work. Recover the original thesis proposal.
2. Preserve original caches, source-data manifests, exact weights/preprocessing, environment and execution revisions; address integrity and reuse concerns before further archival experiments.
3. Prespecify alternative-fit and forward-size prediction comparisons; report conditional seed uncertainty separately from model error.
4. If needed, run a separately designed exposure-matched control and a CIFAR WD-boundary check. Do not use final benchmark scores for new recipe selection.
5. Specify pretrained lineage and N/head controls before a model-size sweep; define channel/power/noise and full-width equivalence before channel experiments.

See [register](experiment_index.csv), [implementation concerns](implementation_status.md) and [literature implications](literature/README.md).

## Follow-up — 29 October 2026

The researcher has moved the implementation and evidence into root-level folders. The moved code passed 42 tests and compact verification, using the environment recorded in the [layout check](validation/layout_checks_2026-10-29.json). Navigation/register/manifest fixes are included in the root-layout change; final checkout verification is described in the dated record. The expanded [contribution and dataset-size comparison](literature/dataset_sweep_and_novelty.md) supports a [candidate research question](research_question.md); novelty and experiment order are not yet approved or certified. No channel experiment has run.
