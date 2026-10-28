# Validation record — 28 October 2026

Checks were run on the staged organization before publishing. They do not train the archived models or access the original final-evaluation features.

| Check | Result / record |
|---|---|
| Pytest | **42 passed**: 39 pre-existing tests plus three new artifact-verifier regressions |
| Compact numerical evidence | [JSON](compact_verification.json): 425 saved sweep rows, 17,520 HPO trials; exact grid/identity, aggregate arithmetic, global recipe and five point-fit recalculations passed |
| Local full archive | [JSON](local_archive_verification.json): all 200 main heads/histories readable and matching recorded metadata/history; nested unique subset IDs; independent point refits |
| Original numerical source/config identity | [Organization checks](organization_checks.json): all `src/` files and ten YAMLs unchanged against base clone |
| Commands and syntax | All ten YAML dry runs and two main snapshot dry runs; Python AST and four shell syntax checks |
| Navigation | Current local Markdown targets and imported register paths checked; original archival package prose intentionally retains its old path references |
| Figures | Four regenerated plots inspected for scales, labels, error bars, split names and clipping; historical builder rerun in a temporary copy for a smoke check |

Commands, from the implementation directory:

```bash
python -m cli.verify_artifacts --refit
python -m cli.build_report_figures
python -m cli.run_sweep --config configs/data_sweep_global_hpo.yaml --dry-run
```

From repository root: `python -m pytest -q -p no:cacheprovider`.

Actual review environment: Python 3.11.6, NumPy 1.26.4, pandas 2.1.3, Matplotlib 3.10.3, PyTorch 2.4.1 CPU, PyYAML 6.0.2, pytest 7.4.3. The installed system SciPy 1.15.1 has a binary loading failure; checks used the existing isolated SciPy 1.14.1 under the local archive's `output/places365_claim_audit_20261013/scipy_dependency` via PYTHONPATH. No dependency installation or upgrade was performed. The environment is not the historical training environment or a clean installation of the requested development dependency range (`pytest>=8`). The old October 1 note's three SciPy skips do not apply to this successful isolated refit.

The archive audit's [source snapshot](archive_audit_source.py.txt) is preserved as historical text. It checks recorded evidence and checkpoint shapes; it does not re-evaluate heads on final feature tensors. The new portable verifier covers compact files available in Git.

**Not checked:** original image contents and duplicate leakage; historical feature-label alignment; exact executed source/weight/environment identity; full retraining; historical access chronology; new final inference; rerunning the 100-replicate exploratory bootstrap. These limits remain in the report. Passing consistency checks is not a certification that all scientific assumptions are correct.
