# Scaling-law-comm

Research on scaling, visual representations and task-oriented communication. Current completed evidence concerns **labeled-data scaling with a frozen ViT-Tiny encoder**; encoder-size and channel studies remain planned.

Start with the [comprehensive thesis progress report](reports/thesis_progress.md). It explains the question, pipeline, experiment progression, results, plots, paper findings and limitations, with links to supporting artifacts.

- [Current progress and next steps](docs/research_status.md)
- [Experiment register](docs/experiment_index.csv)
- [13-paper methodology review](docs/literature/README.md) and [comparison table](docs/literature/paper_comparison.csv)
- [Simple script explanations, diagrams and pseudocode](docs/script_guide.md)
- [Implementation and execution guide](experiments/models/vit-tiny/README.md)
- [Evidence verification and reproducibility limits](experiments/models/vit-tiny/REPRODUCIBILITY.md)
- [Organization record](docs/organization_log.md)

```text
Scaling-law-comm/
├── README.md
├── docs/                  # Status, experiment register, literature, learning guides
├── reports/
│   ├── thesis_progress.md # One current comprehensive narrative
│   ├── experiments/       # Main scorecards and marked historical reports
│   ├── figures/           # Regenerated report figures and archived originals
│   └── tables/            # Separately identified derived diagnostics
└── experiments/models/vit-tiny/
    ├── README.md
    ├── src/               # Reusable scientific implementation
    ├── cli/               # Programs invoked with python -m cli.NAME
    ├── scripts/           # Shell launchers and historical figure builder
    ├── configs/           # Current and clearly documented historical templates
    ├── protocols/         # Preserved experiment designs and acceptance rules
    ├── tests/
    └── artifacts/         # Compact recorded metrics, fits and snapshots
```

**Dated update — 28 October 2026:** reconciled the local archive with GitHub, added completed Places365 boundary HPO and final-evaluation evidence, checked 425 compact sweep records and 17,520 HPO trials, reviewed all 13 unique supplied papers for dataset-size methodology, and consolidated the research narrative. No new scientific training or final evaluation was performed for this update. Experiment execution is not being dated to this week.

Main endpoints: CIFAR-100 76.608% at D=45,000; Places365 46.668% at D=500,000. These are ten-seed means under different task-specific fixed recipes; see the report for SDs and split definitions. Fitted exponents (.5204 and .4655) describe the observed protocols and do not establish a universal law.

Next: agree the D-baseline/N-sweep scope with the supervisor; recover source/cache provenance and resolve execution-integrity gaps; specify predictive-fit and compute controls before new experiments. The original archive and paper folder remain external. Raw images, feature tensors, weights and copyrighted PDFs are not stored here.
