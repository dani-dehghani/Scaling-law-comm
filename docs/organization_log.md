# Organization record — 28 October 2026

Base: GitHub main commit `7d153537986b2d983308e503bdf9f6f5a82457b2`. Work was first prepared in an isolated staging directory, then checked before applying it to a separate Git branch. The original local experiment archive and paper folder remain untouched.

| Previous location | Current location | Treatment |
|---|---|---|
| Root experiment Python programs | `experiments/models/vit-tiny/cli/` | Relocated; package invocation, root resolution and test callers updated |
| Root experiment `.sh` files | `experiments/models/vit-tiny/scripts/` | Relocated; run from the experiment directory and call `python -m cli.NAME` |
| `run_all.sh` default | Main CIFAR YAML | Corrected default to `data_sweep_global_hpo.yaml`; numerical recipe unchanged |
| Root experiment protocol Markdown | `experiments/models/vit-tiny/protocols/` | Original scientific text preserved; historical navigation banner added |
| Experiment `reports/` numerical collections | `experiments/models/vit-tiny/artifacts/` | Original numerical bytes preserved |
| Earlier manifest/validation/reproducibility | `artifacts/archival_package_2026-10-01/` | Preserved with historical scope |
| Earlier experiment report Markdown | `reports/experiments/archive/` | Historical banner and repaired navigation; missing comparison link gets an explicit navigation note |
| Existing result PNGs | `reports/figures/archived/` | Original images preserved |
| Historical comparison builder | `scripts/build_historical_comparison.py` | Reads relocated tables; writes root report derivatives |
| Missing local Places final/boundary evidence | `artifacts/places365_main/`, `places365_boundary_hpo/` | Compact original evidence imported; not new execution |
| Current research explanation | `reports/thesis_progress.md` | New consolidated narrative, all main tables, four regenerated figures, literature synthesis and limitations |
| Current navigation/status | Root/experiment README, `docs/` | Supersedes stale initialized/pending wording |

Numerical source modules and all ten original YAML configurations are byte-identical to the original clone. The blocked provisional Places templates are deliberately documented as historical; the approved executed snapshot is separately preserved. No later working-source repair is silently merged or attributed to old results. The new verification and report-figure programs add inspection, not training behavior. Tests follow relocated imports, and a root pytest configuration supports running them from the repository root.

The paper inventory identifies 15 files, two duplicate pairs and 13 unique reviews. Originals are not copied into Git. Planning documents are recorded by identity and paraphrased scope; the original thesis proposal is absent. Archive copies, partial result mirrors and tar files are not counted as new trials.

See [validation](validation/README.md), [source hashes/checks](validation/organization_checks.json), [artifact hashes](../experiments/models/vit-tiny/artifacts/MANIFEST.csv) and [evidence map](workspace_inventory.md). Existing Git history retains pre-organization versions of rewritten navigation documents. Raw images, feature tensors and weights are not published here.

Git attributes preserve artifact bytes during checkout; CSV whitespace checks recognize the original CRLF line endings rather than normalizing recorded evidence.
