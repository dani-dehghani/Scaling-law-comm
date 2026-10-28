# CIFAR-100 main dataset-size study

Current scorecard, 28 October 2026. The [comprehensive report](../thesis_progress.md) supplies methods, numerical tables, interpretation and limits.

**Question:** how does labeled head-training D affect a fixed frozen ViT-Tiny representation under one recipe?

**Design:** 12 sizes × 10 seeds; affine classifier; constant AdamW learning rate 0.0001, weight decay 0.01, batch 256, 30,000-step ceiling with validation early stopping. The final-evaluation split is official CIFAR-100 test.

![Measured loss and accuracy](../figures/cifar100_main.png)

Evidence: [runs](../../experiments/models/vit-tiny/artifacts/cifar100_main/summary.csv), [aggregates](../../experiments/models/vit-tiny/artifacts/cifar100_main/aggregated_by_D.csv), [executed snapshot](../../experiments/models/vit-tiny/artifacts/cifar100_main/config_snapshot.json), [fit](../../experiments/models/vit-tiny/artifacts/cifar100_main/scaling_fit.json), [completion](../../experiments/models/vit-tiny/artifacts/cifar100_main/completion.json).

Error bars show ±1 sample SD over seeds. Results concern downstream labels and this fixed recipe, not encoder or channel scaling. Numeric consistency was checked; original feature/image provenance remains unresolved.
