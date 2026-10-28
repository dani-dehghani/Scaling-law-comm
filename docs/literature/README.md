# Literature review: what “dataset-size sweep” actually means

Review date: 28 October 2026. Scope: all 15 supplied PDFs, representing 13 unique papers, and two local planning DOCX files. This is a focused methodological review of research questions, experiments and dataset-size controls, not an independent reproduction of every paper or proof. Page references below are **PDF page positions**, which can differ from printed page numbers. Missing details are marked unknown rather than inferred. Originals remain in the local paper folder; this repository contains paraphrases, references and hashes.

The key distinction is between **unique pretraining data**, **unique labeled downstream data**, **training-example presentations**, and **communication conditions**. Our completed experiment varies only unique labeled downstream data for a fixed frozen encoder. Closest precedents for that decision are P04 and P11. P01 supports explicit nesting but varies pretraining data. P12 offers a useful separate exposure-matched control. P09 motivates testing predictions on larger sizes rather than trusting an in-sample fit.

No reviewed paper establishes one universal D range, seed count or compute policy. Retain the present study as a documented baseline; choose any next protocol according to the thesis question and state its compute convention explicitly.

- [Machine-readable comparison](paper_comparison.csv): question, size grid, method, finding, relation and caveats.
- [Inventory and duplicate hashes](paper_inventory.csv): 15 files; duplicate copies of P05 and P08 are counted once scientifically.
- [Main synthesis and implications](../../reports/thesis_progress.md).

## P01 — Data Scaling Laws for Radiology Foundation Models

[2025 preprint](https://arxiv.org/abs/2509.12818); reviewed PDF pages: 5; methods pp.3–6.

**Question:** How does radiology pretraining data affect downstream transfer?

**Size sweep:** Unique pretraining images: 30k, 50k, 100k, 1M, 3.5M; strictly nested. Three subset/encoder repetitions at the three smallest sizes; one at the two largest; downstream fits use three seeds.

**Method:** Compare representation learners with frozen downstream features and attention pooling plus a linear readout. Match methods by wall time within each D; time increases across D (.33 to 40.83 hours).

**What we learn:** More pretraining data can improve transfer; the design separates representation learning from downstream evaluation.

**Relation to us:** Useful precedent for nesting and repeats, but our D varies downstream head-training labels, not encoder pretraining data. Attention pooling also differs from our affine head.

**Limits:** Do not interpret the abstract as constant compute across D. Exact choices are study-specific; not a universal downstream D range.

## P02 — Task-Oriented Communication with Hybrid-Precision Models

[2026 preprint](https://arxiv.org/abs/2607.16766); reviewed PDF pages: 9–12, §VI.

**Question:** Can hybrid precision reduce computation in task-oriented communication?

**Size sweep:** No standalone unique-training-D grid identified in the reviewed experiments. Fixed ImageNet training corpus; architecture, precision, SNR and communication conditions vary.

**Method:** Split ResNet models with a binarized front and full-precision back; OFDM/16-QAM experiments.

**What we learn:** Joint precision and transmission choices affect task performance and cost.

**Relation to us:** Communication-system context for future experiments; does not supply a dataset-size sampling recipe.

**Limits:** Do not read model depth, SNR or evaluation sample counts as D.

## P03 — Agent-Native Task-Oriented Communication with Joint Token Compression Coding and Modulation

[2026 preprint](https://arxiv.org/abs/2608.00368); reviewed PDF pages: 5, §IV.

**Question:** Can joint token compression, coding and modulation support agent tasks?

**Size sweep:** No standalone unique-training-D sweep identified. CLEVR/GQA tasks; 10,000 channel samples describe channel generation, not a labeled-data sweep.

**Method:** Train the communication module, then adapt the foundation model with LoRA; compare SNR/bandwidth and task performance.

**What we learn:** Task-aware token transmission can trade bandwidth against downstream utility.

**Relation to us:** Context for a later channel stage; different tasks, representation and trainable components.

**Limits:** Do not call 10,000 channel realizations a D setting for our classifier experiment.

## P04 — Scaling Laws for Data-Efficient Visual Transfer Learning

[ACM MM 2025](https://doi.org/10.1145/3746027.3754916); reviewed PDF pages: 3 §3.1; 7 §5.1.

**Question:** How do pretraining data, downstream data and model capacity jointly affect transfer?

**Size sweep:** Uniform samples at 5%, 10%, 25%, 33%, 50%, 70%, 100% for pretraining and fine-tuning data; category set held fixed.

**Method:** Vary both data axes and architecture (2/4/6/8 attention heads, head width 64); fit transfer-scaling relationships.

**What we learn:** Motivates treating upstream and downstream data as separate variables.

**Relation to us:** Direct support for a broad same-category downstream-size grid; our study isolates one downstream axis with a frozen encoder.

**Limits:** Strict nesting, repeat counts and a matched compute budget are not established by the reviewed passage. §5.1 uses inconsistent ImageNet-100/1,000-class wording; retain fractions rather than asserting a resolved absolute count.

## P05 — Budgeted Training for Vision Transformer

[ICLR 2023](https://openreview.net/pdf?id=sVzBN-DlJRi); reviewed PDF pages: 7 §5.1; experimental section.

**Question:** How can ViT training be improved under a computation budget?

**Size sweep:** Fixed ImageNet-1k corpus; not a unique-image D sweep. Training components, token fractions and stage lengths change.

**Method:** Allocate budgets across attention heads, MLPs and tokens; compare budget-specific baselines including fewer epochs.

**What we learn:** Useful efficiency study, not evidence that reducing active tokens equals reducing the dataset.

**Relation to us:** Helps distinguish computation from unique data and motivates honest exposure accounting.

**Limits:** Its B denotes a training budget and N can denote tokens. Neither should be copied into our encoder-parameter notation.

## P06 — Beyond Transmitting Bits: Context, Semantics, and Task-Oriented Communications

[IEEE JSAC 2023](https://doi.org/10.1109/JSAC.2022.3223408); reviewed PDF pages: Tutorial, introduction and system/task discussions.

**Question:** What should communication optimize when meaning and task utility matter?

**Size sweep:** Tutorial covering multiple systems; no single standardized unique-D sweep identified.

**Method:** Conceptual synthesis of context, semantics, task objectives and learning-based communication.

**What we learn:** Task performance is a meaningful system objective alongside bit-level metrics.

**Relation to us:** Supports the thesis motivation and later channel questions; not a direct D-sweep protocol.

**Limits:** Do not cite this broad tutorial as empirical validation of our sampling or fitted exponent.

## P07 — Feature-Foundation Model Evolution for Low-Latency Semantic Communication

[IEEE TWC 2026](https://doi.org/10.1109/TWC.2026.3659294); reviewed PDF pages: 10–13, especially 11.

**Question:** Can feature and foundation models evolve to handle new semantic classes with low latency?

**Size sweep:** VOC2012 segmentation with 15 old classes and online new-class images; no standalone fixed-class D grid identified.

**Method:** Transfer foundation knowledge during updates; compare SNR 0–20 dB, compression ratios .2–.6 and latency/new-class conditions.

**What we learn:** Studies adaptation to changing semantic content under communication constraints.

**Relation to us:** Useful future-task context, but class novelty differs from varying labels within our fixed task.

**Limits:** Image proportions describing novel classes are not interchangeable with distinct classifier-training D.

## P08 — How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers

[TMLR 2022](https://openreview.net/forum?id=4nPswr1KcP); reviewed PDF pages: 3–5; architecture table p.4.

**Question:** How do data, augmentation and regularization affect ViT training and transfer?

**Size sweep:** ImageNet-1k versus ImageNet-21k corpora, with 300 versus 30/300 epochs; not a fixed-category nested downstream-size grid.

**Method:** Compare model sizes and 28 augmentation/regularization settings; fine-tune downstream models; use validation for selection.

**What we learn:** Training recipe and data availability interact; larger corpora need an explicit compute comparison.

**Relation to us:** Checkpoint lineage and validation discipline are relevant; our frozen probe is a different transfer protocol.

**Limits:** The roughly compute-comparable 1k/21k comparison changes corpus/classes. Full-model parameter counts include different heads; do not substitute them for our head-removed encoder count.

## P09 — Revisiting Neural Scaling Laws in Language and Vision

[NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/8c22e5e918198702765ecff4b20d0a90-Abstract-Conference.html); reviewed PDF pages: 3–7, §4 and §5.

**Question:** Which scaling functions predict performance beyond the observed range?

**Size sweep:** Vision experiments use several pretraining corpora (12M, 100M, 200M, 300M); fitting variable includes examples seen before peak performance. Downstream few-shot tasks are fixed per curve.

**Method:** Compare multiple fit families; assess predictions on larger x after fitting lower x, including the x≤xmax/2 versus larger-x split.

**What we learn:** Interpolation fit quality does not establish extrapolation accuracy.

**Relation to us:** Supports forward-size validation and alternative-fit diagnostics. Our robust log-total-loss offset fit is an adaptation, not an exact reproduction.

**Limits:** Do not transfer their exponents or equate examples seen with unique data. Our fitted floor is not independently identified irreducible error.

## P10 — Scaling Vision Transformers

[CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Zhai_Scaling_Vision_Transformers_CVPR_2022_paper.html); reviewed PDF pages: 2–3, §2 and §2.4.

**Question:** How do model capacity, data and computation affect large ViT performance?

**Size sweep:** JFT subsets: 30M, 300M, 1B, 3B. ImageNet-21k size study: 1.3M, 2.6M, 5.1M, 10.2M, 13M.

**Method:** Vary architecture and duration; compare performance and compute envelopes, frozen few-shot transfer and full fine-tuning.

**What we learn:** Data and model scale must be interpreted together with compute.

**Relation to us:** Motivates a broad size range and a later N sweep. Its best-achievable envelope differs from our fixed encoder/recipe curve.

**Limits:** Nesting and subset repetition are not established by the reviewed passage. Some plot axes are examples seen rather than unique images.

## P11 — Tuning Pre-trained Model via Moment Probing

[ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Gao_Tuning_Pre-trained_Model_via_Moment_Probing_ICCV_2023_paper.html); reviewed PDF pages: 9 §4.7; 5 implementation.

**Question:** Can feature-moment probing improve efficient transfer and few-shot classification?

**Size sweep:** 1, 2, 4, 8, 16 labeled images per ImageNet class: 1k–16k total; three trials averaged.

**Method:** Compare linear probing, moment probing and MP+fine-tuning using an ImageNet-21k-pretrained ViT-B/16.

**What we learn:** Few-shot classification provides a direct downstream-label-size comparison.

**Relation to us:** Closest reviewed precedent for class-aware downstream D and repeated trials; our head uses first-order features and a different backbone.

**Limits:** Strict nesting and complete few-shot training budgets are not established from the main passage. MP+ changes trainable components; no data-scaling exponent is established here.

## P12 — Scaling Pre-training to One Hundred Billion Data for Vision Language Models

[CVPR Findings 2026; arXiv v2](https://arxiv.org/abs/2502.07617); reviewed PDF pages: 4 §3; experimental comparisons.

**Question:** What improves when vision-language pretraining reaches 100 billion unique pairs?

**Size sweep:** 1B and 10B random subsets of WebLI-100B, plus 100B. Nesting of 1B inside 10B is not explicit.

**Method:** SigLIP ViT-B/L/H; batch 32k, up to 100B example presentations for each corpus: about 100, 10, 1 passes respectively; evaluate multiple tasks.

**What we learn:** Benefits depend on task coverage; unique data and repeated exposure are distinct.

**Relation to us:** Strong precedent for an exposure-matched comparison as a separate future control.

**Limits:** Pretraining differs from frozen downstream probing. Source contains an unresolved Appendix reference; this review does not certify every implementation detail.

## P13 — Zero-Shot Semantic Communication With Multimodal Foundation Models

[IEEE TVT 2026; early DOI 2025](https://doi.org/10.1109/TVT.2025.3632893); reviewed PDF pages: 4 §IV.

**Question:** Can pretrained multimodal features support communication for unseen tasks/classes?

**Size sweep:** ImageNet categories partitioned into 800 training and 200 unseen categories; not a fixed-class labeled-D sweep.

**Method:** Frozen CLIP ViT-L/14@336 with communication modules; sweep SNR and bandwidth conditions.

**What we learn:** Foundation representations can support unseen-category communication objectives.

**Relation to us:** Relevant to the future channel design and task-generalization controls.

**Limits:** Class-disjoint generalization differs from our same-class data scaling. Their N=768 feature width is not our encoder parameter count.

## Planning documents and missing source material

The local `Experiment 1 — Provisional Supervisor Summary.docx` and `Experiment_1_Source_Grounded_Roadmap.docx` are planning documents, not experiments or peer-reviewed evidence. The roadmap describes an initial N sweep at fixed D, followed by a clean baseline and later communication questions. The completed frozen-encoder D study is preparatory evidence for that plan, rather than completion of the planned N sweep. The original thesis proposal was not supplied, so alignment against its exact promises remains unverified.

References named in the roadmap but absent as full papers from this folder—including Kaplan, Hestness, Bahri, Hoffmann/Chinchilla and Cherti, as well as additional communication references—are **secondary mentions, not reviewed primary evidence** in this report. Their methods should not be attributed from the planning document alone. We have not fabricated their grids or merged their claims into the comparison table.

## Protocol implications for this project

1. Define D as distinct labeled source examples used to fit the head, and keep the encoder pretraining corpus separate.
2. Keep the official final-evaluation split out of hyperparameter and checkpoint selection. Existing files show the intended separation; they cannot prove an entire historical access timeline.
3. Preserve class coverage, record subset IDs and seeds, and describe nesting explicitly. Nesting pairs sizes within a seed and therefore creates correlated measurements.
4. Report the actual optimizer steps, presentations and effective passes as well as D. Equal maximum steps plus early stopping is not equal realized compute.
5. Use a logarithmically spread D grid with enough points to reveal curvature; do not extrapolate a universal range from another task.
6. Freeze the optimization recipe for the main fixed-recipe curve. If asking about best achievable performance at each D, a separately named, independently budgeted per-D optimization study is needed.
7. Assess larger-D prediction and fit-family sensitivity. More seed repeats do not fix a misspecified curve.
8. Define N, feature width, transmitted dimension, channel noise and trainable parameter counts separately before adding a channel.
