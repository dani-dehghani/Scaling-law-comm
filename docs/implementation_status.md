# Implementation, versions and unresolved checks

Review date: 28 October 2026. The organization starts from GitHub main commit `7d153537986b2d983308e503bdf9f6f5a82457b2`. The numerical `src/` files are preserved byte for byte; the CLI files move to a package and resolve the experiment root one level above their new directory. Tests/callers and shell launchers follow those moves. The shared shell launcher now defaults to the documented main CIFAR recipe. New programs verify compact tables and regenerate report figures; they do not train models.

There are three source histories: the original GitHub snapshot, the later local working implementation, and the unknown exact source revision of each historical run. A newer helper's presence does not establish that older results used it. The October 20 local audit's 101 tests are not the test count of this repository; see [current validation](validation/README.md).

## Concerns to resolve before new archival experiments

The October 20 review of newer local code identified these failure paths. They are recorded here as follow-up work, not represented as fixes already merged into this older snapshot:

- Interrupted forced runs can pair a newer checkpoint with older metrics; bind generations/content identities before accepting reuse.
- Aggregate-only HPO reuse/completion checks can accept evidence without sufficiently binding experiment and training budget.
- Validation-cache metadata uses inconsistent size-key conventions in a validation path.
- Source-image identity, tensor content, row/label alignment and class mapping are not fully authenticated by the cache fingerprint.
- Declared class counts do not fully constrain labels and head construction.
- Unknown scientific configuration keys can be accepted without affecting execution; extraction determinism is not fully enforced by the runtime flag.
- Recipe promotion is not fully bound to the target dataset/model/budget; two output replacements are not one transaction.
- A nondefault randomly initialized encoder cannot reliably authenticate its own cache across reconstruction.

None was shown to corrupt the archived main numbers. The older repository may have additional gaps; this organization is not a complete security or scientific correctness certification. Do not run new expensive experiments merely because a YAML loads or tests pass.

The original feature caches and raw images are missing locally. Subset-ID nesting, readable heads, matching histories and table agreement cannot independently establish correct feature-label alignment, content-level uniqueness, historical checkpoint identity or absence of all data leakage.

[Source comparisons and organization record](organization_log.md) distinguish relocation from scientific changes. [The main report](../reports/thesis_progress.md) states which results can be presented now.
