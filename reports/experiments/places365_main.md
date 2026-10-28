# Places365 main dataset-size study

Current scorecard, 28 October 2026. The [comprehensive report](../thesis_progress.md) supplies methods, numerical tables, interpretation and limits.

**Question:** how does labeled head-training D affect a fixed frozen ViT-Tiny representation under one recipe?

**Design:** 8 sizes × 10 seeds; affine classifier; constant AdamW learning rate 0.0002, weight decay 1.0, batch 256, 30,000-step ceiling with validation early stopping. The final-evaluation split is official Places365 validation, separate from the internal training-derived validation split.

![Measured loss and accuracy](../figures/places365_main.png)

Evidence: [runs](../../experiments/models/vit-tiny/artifacts/places365_main/summary.csv), [aggregates](../../experiments/models/vit-tiny/artifacts/places365_main/aggregated_by_D.csv), [executed snapshot](../../experiments/models/vit-tiny/artifacts/places365_main/config_snapshot.json), [fit](../../experiments/models/vit-tiny/artifacts/places365_main/scaling_fit.json), [completion](../../experiments/models/vit-tiny/artifacts/places365_main/completion.json).

Error bars show ±1 sample SD over seeds. Results concern downstream labels and this fixed recipe, not encoder or channel scaling. Numeric consistency was checked; original feature/image provenance remains unresolved.
