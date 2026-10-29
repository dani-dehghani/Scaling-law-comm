# Reproducibility and evidence scope — 28 October 2026

The current [compact verification](validation/compact_verification.json) checks five saved sweep tables (425 rows), four complete HPO grids (17,520 rows), aggregates and global selection, and independently recalculates saved point fits. The [local archive audit](validation/local_archive_verification.json) also checks 200 main heads, histories and nested ID sets. No new feature extraction, historical retraining or final-data inference was performed.

The original GitHub numerical source is preserved; relocated entry points adjust their project root and imports/callers. Newer local working-source repairs are not silently merged. Source history and unresolved gaps are described in [implementation status](implementation_status.md). The exact source/environment used for every old run remains unverified.

The former reproducibility note, manifest and validation describe the October 1 package. They are preserved in [archival package records](../artifacts/archival_package_2026-10-01/README.md). Their old test counts and “Places pending” statement are historical, not current conclusions. The new [artifact manifest](../artifacts/MANIFEST.csv) hashes files at their current paths and preserves byte identity of imported evidence; hashes establish identity, not correctness.

Images, historical feature tensors, raw execution logs, head weights and sampled ID arrays are outside this compact repository. Original image content, duplicate leakage, weight/preprocessing execution and feature-label alignment were not independently certified. Requirement ranges are not an environment lock. A completion flag and passing tests do not close these gaps.

The bootstrap and leave-one-size-out results in [dated diagnostics](../reports/tables/diagnostics_2026-10-20/README.md) are separately labeled derivatives; their bootstrap was not rerun here. Report error bars show seed sample SD, not a confidence interval. See [validation record](validation/README.md) for current executed checks and environment limitations.
