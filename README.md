# Thesis: Scaling Laws for Semantic Encoders Under Channel Budgets

This repository contains the working material for a master's thesis on empirical scaling laws for foundation-model semantic encoders in task-oriented communication systems.

## Topic

The thesis studies a semantic communication pipeline where an input is encoded by a foundation-model encoder, compressed or corrupted by an abstract communication channel, and then used by a receiver-side task head:

```text
input x -> semantic encoder f_N -> feature z -> channel C_(B, Landa) -> received feature z_tilde -> task head g -> prediction y_hat
```

The main research question is:

> How does downstream task performance scale with encoder size, training data size, transmitted embedding width, and channel quality?

The thesis investigates whether there is a channel-induced saturation point where using a larger encoder no longer improves task performance because the communication channel becomes the bottleneck.

## Current Contents

This repository is currently in an early thesis-planning stage. It will contain:

- thesis notes and roadmap documents
- literature notes from scaling-law, semantic communication, and JSCC papers
- experiment design documents
- reproducible code for the experimental pipeline
- result summaries, plots, and fitted scaling-law models

## Planned Experimental Direction

The planned experimental framework will compare task performance across controlled sweeps of:

- encoder size `N`
- training data size `D`
- transmitted embedding width `B`
- channel quality / noise parameter `Landa`

The first practical experiment is expected to fix a ViT-Tiny encoder and study dataset and data-size scaling under clean and channel-constrained settings.

## Candidate Scaling Laws

The project compares a channel-unaware scaling-law baseline:

```text
L = L_inf + a N^(-alpha) + b D^(-beta)
```

with channel-aware variants that include embedding budget and channel quality:

```text
L = L_inf + a N^(-alpha) + b D^(-beta) + c B^(-eta) + d landa^(-kappa)
```

The goal is to identify whether channel-aware models predict held-out configurations better than classical scaling laws.

## Repository Status

This repository has just been initialized. Code, datasets, and reproducible experiment scripts should be added as the implementation phase begins.

## Notes for Development

Avoid committing local system files, personal configuration files, caches, datasets, model checkpoints, or large experiment outputs.
