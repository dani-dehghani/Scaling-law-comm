# Reproducibility and evidence scope

## Provenance

This snapshot contains the current local code and configurations plus copies of completed result artifacts. `MANIFEST.csv` lists relative source locations in the research workspace and SHA-256 checksums. Empty source fields denote documentation or derivatives generated for this snapshot. Source references to the earlier package denote byte-identical evidence copies, not a new training run.

The current code has not been established as the exact historical revision for every run. Full historical environment and execution provenance have not been verified. HPO trial tables contain recorded environment fields, but those are not a complete independent reproduction guarantee. Dependency files specify installation requirements, not an immutable executed environment.

## Checks performed

- CIFAR-100 main: all 120 expected unique (D, seed) pairs; aggregate means and sample standard deviations agree with the run table.
- Core CIFAR-100 HPO: exact configured grid coverage for 7,920 trials, no duplicate keys, finite positive validation losses.
- Places365 global and refinement HPO: exact grid coverage for 5,280 and 2,400 trials, no duplicate keys, finite positive validation losses.
- Selected Places365 refinement recipe: 80 runs and zero budget-warning runs under the code's near-limit heuristic.
- Python source syntax: passed.
- Local tests: 36 passed, three skipped, zero failures. The skipped tests require scipy.optimize, which cannot import because of a local macOS SciPy binary loading error. Scaling-fit recovery and those fitting-input checks remain unverified in this environment.
- Places365 boundary config dry run: 24 recipes, eight sizes, ten seeds, 1,920 trials.

See `VALIDATION.json`. No GPU training, new held-out evaluation, or independent refitting was performed to prepare this snapshot. The reported fits were copied from completed artifacts.

## Archival requirements and limitations

Raw images, feature caches, linear-head checkpoints, sampled subset arrays, complete histories and original execution logs remain outside this Git snapshot. Retain them separately for reproduction. Bootstrap intervals, fit-range sensitivity, residual diagnostics and a completed matched-control analysis are not supplied here as verified final evidence. Earlier exploratory analyses are not treated as final uncertainty results in this snapshot.

Remote Places365 boundary execution is researcher-reported and has no completed local evidence. The final config is blocked pending its selection checks. CIFAR-100 and Places365 held-out metrics must not be used to revise HPO selection.
