# Scaling-law-comm

Research on scaling, visual representations and task-oriented communication. Current completed evidence concerns **labeled-data scaling with a frozen ViT-Tiny encoder**; encoder-size and channel studies remain planned.

Start with the [comprehensive thesis progress report](reports/thesis_progress.md). It explains the question, pipeline, experiment progression, results, plots, paper findings and limitations, with links to supporting artifacts.

- [Current progress and next steps](docs/research_status.md)
- [Experiment register](docs/experiment_index.csv)
- [13-paper methodology review](docs/literature/README.md) and [comparison table](docs/literature/paper_comparison.csv)
- [Simple script explanations, diagrams and pseudocode](docs/script_guide.md)
- [Implementation and execution guide](docs/implementation.md)
- [Evidence verification and reproducibility limits](docs/reproducibility.md)
- [Organization record](docs/organization_log.md)
- [Expanded dataset-size and contribution comparison](docs/literature/dataset_sweep_and_novelty.md)
- [Candidate research question and proposed protocol](docs/research_question.md)

```text

Scaling-law-comm/
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── src/                   # Reusable Python implementation
├── cli/                   # Programs launched from Terminal
├── scripts/               # Shell launchers and utilities
├── configs/               # Experiment settings
├── protocols/             # Scientific plans and acceptance rules
├── tests/                 # Software checks
├── artifacts/             # Saved numerical evidence
├── docs/                  # Guides, status and literature review
└── reports/               # Research narrative, figures and tables

```

**Dated update — 28 October 2026:** reconciled the local archive with GitHub, added completed Places365 boundary HPO and final-evaluation evidence, checked 425 compact sweep records and 17,520 HPO trials, reviewed all 13 unique supplied papers for dataset-size methodology, and consolidated the research narrative. No new scientific training or final evaluation was performed for this update. Experiment execution is not being dated to this week.

Main endpoints: CIFAR-100 76.608% at D=45,000; Places365 46.668% at D=500,000. These are ten-seed means under different task-specific fixed recipes; see the report for SDs and split definitions. Fitted exponents (.5204 and .4655) describe the observed protocols and do not establish a universal law.

Next: agree the D-baseline/N-sweep scope with the supervisor; recover source/cache provenance and resolve execution-integrity gaps; specify predictive-fit and compute controls before new experiments. The original archive and paper folder remain external. Raw images, feature tensors, weights and copyrighted PDFs are not stored here.

**Follow-up — 29 October 2026:** code and compact evidence now live in root-level folders. The moved code passed 42 tests and compact-table checks; navigation and the current artifact manifest are updated. The expanded literature comparison distinguishes reported contributions from candidate gaps. The proposed channel-dependent data-efficiency study is a question for supervisor discussion, not a completed experiment or certified first-ever result.
