# ViT-Tiny data scaling: verified research progress

14 October 2026 · Results checked against the copied experiment files

[Read the comprehensive dataset comparison with plots, updated 15 October 2026.](COMPARISON_REPORT.md)

## What we set out to learn

Our question was straightforward: **how much does more labeled training data help when the image representation stays fixed?** We used a pretrained ViT-Tiny to turn images into 192-dimensional feature vectors, then trained only a linear classifier. This is a study of learning from a fixed representation, not training a transformer from scratch or fine-tuning its backbone. The model’s ImageNet-21k pretraining and ImageNet-1k fine-tuning are documented in its [model card](https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k).

## How the work developed

We began with exploratory CIFAR-100 sweeps using manually chosen settings. Those runs motivated a more controlled comparison: select settings using internal validation, then keep one recipe fixed across dataset sizes. Separating model selection from final evaluation addresses the selection-bias concern discussed by [Cawley and Talbot (2010)](https://www.jmlr.org/papers/v11/cawley10a.html).

The CIFAR-100 HPO completion record reports 7,920 trials. Its selected recipe was learning rate `0.0001`, weight decay `0.01`, constant schedule, and batch size `256`. The main sweep then covered 12 training sizes and ten seeds: 120 runs.

Places365 followed a similar workflow, with search extensions after boundary and training-budget warnings. The final boundary search contains 1,920 unique trials, independently checked against its configured grid. It selected learning rate `0.0002`, weight decay `1.0`, constant schedule, and batch size `256`. Its average relative validation-loss regret was 3.02% against the separately tuned winners—an explicit compromise for using one recipe everywhere, not a claim of universal optimality.

Both main sweeps used a 30,000-step maximum and validation-based checkpoint selection. Subsets were nested within each seed: balanced for CIFAR-100 and proportionally stratified for Places365. Places365’s official labeled validation split served as protected final evaluation; a separate internal validation split came from its training data.

## What the results show

The table reports means over ten seeds, recomputed from the individual records.

| Experiment | Training images | Evaluation loss, first → last | Top-1 accuracy, first → last | Fitted β |
| --- | ---: | ---: | ---: | ---: |
| CIFAR-100 | 200–45,000 | 3.1867 → 0.7894 | 31.90% → 76.61% | 0.5204 |
| Places365 | 1,300–500,000 | 3.7341 → 2.0611 | 23.47% → 46.67% | 0.4655 |

More data improved performance under both fixed recipes. Places365 showed particularly small marginal gains at the high-data end: doubling from 250,000 to 500,000 images added **0.287 percentage points** of accuracy. This is an observed diminishing return, not proof that more data cannot help or that fine-tuning would solve the limitation.

We fitted mean cross-entropy loss using `L(D) = L_inf + b D^(-beta)`. Independent recalculation reproduced both saved exponents. These are protocol- and range-dependent summaries, not dataset constants. For example, withholding the smallest Places365 size produced a prediction 13.86% **above** its observed loss; the curve should not be trusted for unrestricted extrapolation.

## What is verified—and what remains open

The reported means and standard deviations were checked against the individual run records. The original experiment results were preserved.

This does **not** establish absence of leakage. Original image-level overlap, near-duplicates, full cache provenance, and possible overlap with pretraining data remain unaudited. Leakage requires direct methodological checks, not reassurance from good-looking scores; see [Kapoor and Narayanan](https://arxiv.org/abs/2207.07048). The exact source revisions executed historically are also not identified by the result snapshots.

## Next steps

Complete the leakage/provenance audit, freeze a versioned reproducibility package, and review the findings with the supervisor. Any probe or training-policy follow-up should be specified separately, with an evaluation plan that acknowledges the existing final scores have already been seen. The completed sweeps should remain unchanged.
