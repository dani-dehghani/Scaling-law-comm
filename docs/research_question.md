# Candidate research question and experiment plan

Draft for supervisor discussion, 29 October 2026. **No new experiment, channel implementation, accepted hypothesis or priority claim is implied by this document.** The completed work is still the clean frozen-ViT-Tiny D baseline. The [comparison](literature/dataset_sweep_and_novelty.md) and [search log](literature/search_log_2026-10-29.csv) explain how this draft was motivated.

## 1. The question in simple language

If the receiver gets a compressed or noisy version of an image's frozen features, does collecting more labeled training images remain equally useful? Can a few smaller experiments predict how much labeled data a constrained link will need?

**Proposed primary question:** For a fixed pretrained encoder and a declared classifier-training policy, do transmitted feature dimension B and noise level gamma change only the offset/amplitude of the downstream data-learning curve, or also its rate of improvement with unique labeled D? Which description predicts reserved larger-D measurements most accurately?

A practical secondary question is: can that model predict the smallest measured D needed to reach a predeclared task-loss target under a specified link condition? “Target not reached in the measured range” is a valid outcome; do not extrapolate an unreachable target into an invented data requirement.

This is deliberately conditional on encoder, task, projection and training policy. It is not a universal law for all semantic communication. B is a count of transmitted real coordinates/channel uses in an abstract link, not bits per second or a complete wireless bandwidth model.

## 2. Why this is a candidate, rather than a certified novelty claim

The literature already contains downstream size sweeps (P04/P11), cached linear probes and transfer scaling (P14), pretrained-feature communication (P13/P16), generic-feature learning-curve theory (P17), and connections between compression and sample scaling (P18). Merely combining the words scaling, frozen encoder and channel is not enough.

The possible contribution is a controlled **empirical test of interaction and prediction**: compare a curve that shifts under channel constraints with curves whose amplitude/rate also change, validate on larger D, and quantify where label-efficiency predictions fail. The focused review did not establish this exact design in the inspected method passages, but that is not proof of absence from all literature. P18–P20 and related citation chains need deeper checking before any claim of priority. Public-code/supplement checks and supervisor review remain necessary.

The original thesis proposal is not locally available. The two supplied planning documents describe an N sweep first, at fixed D and channel conditions. This D × channel candidate may serve the later D study, but should not silently replace a promised N experiment. Confirm sequence and scope with the supervisor before implementing or scheduling formal runs.

## 3. Distinguish three possible contributions

| Candidate | What would be new evidence | Main limitation | Role |
|---|---|---|---|
| A: Protocol sensitivity | Compare current early-stopped curves with an explicitly exposure-controlled study and forward-D predictions | Related optimization/compute and learning-curve work is extensive; novelty is weak unless a specific failure is demonstrated | Useful baseline diagnostic, not automatically the thesis's novel result |
| B: Channel-dependent label efficiency | Test whether B/noise change curve shape and predict larger-D outcomes with frozen features and controlled head capacity | Channel pipeline and historical cache provenance must be recovered/validated; exact novelty remains open | Recommended candidate to discuss because it extends existing D evidence toward communication |
| C: Budget-dependent encoder ranking | Test whether a larger coherent pretrained encoder remains better at fixed D under the same link constraints | Requires a coherent checkpoint family and feature/head controls; three encoder sizes cannot justify a three-parameter scaling-law fit | Alternative if supervisor prioritizes the planned N study |

## 4. Testable competing explanations

Let c denote a link condition (B, gamma), and let L(D,c) be expected final-evaluation cross-entropy of the selected classifier under the declared training and noise policy. Consider these descriptive candidates:

- **M0 — offset only:** `L(D,c) = a_c + b × D^(-beta)`.
- **M1 — offset and amplitude:** `L(D,c) = a_c + b_c × D^(-beta)`.
- **M2 — condition-specific rate:** `L(D,c) = a_c + b_c × D^(-beta_c)`.

M0 predicts that channel conditions add an approximately D-independent loss penalty over the studied range. M1 allows label-efficiency amplitude to change without changing beta. M2 allows the rate to change. These are hypotheses, not known true forms. Include a non-power-law or simple reference predictor to detect that all three power forms may be misspecified. Do not interpret fitted a_c as irreducible error.

Fit the candidates using the same smaller-D points and objective, then compare predictions on reserved larger-D points. More parameters improving training fit is not evidence for M2. A useful conclusion could favor a simpler model, identify a boundary where predictions fail, or reject all candidates.

The primary prediction score can be RMSE of log observed versus log predicted mean loss across held-out D/conditions. Also report each held-out relative error and task accuracy. Before formal data collection, agree a practical tolerance and model-comparison rule with the supervisor. Do not select a tolerance after seeing the result. Report uncertainty on paired prediction-error differences rather than declaring an exponent difference from rounded point estimates.

## 5. Minimum experimental design

### Existing baseline

Keep the archived CIFAR-100/Places365 clean results immutable. Their original final scores have already been inspected. Any retrospective fit/holdout analysis of them must be labeled retrospective. New channel measurements can be prospectively reserved if their protocol is locked before outcomes are examined.

### Feasibility pilot — not a scaling-law result

First recover or create a separately identified feature cache with documented data, weight and preprocessing identities. Resolve the known integrity/reuse issues before new archival runs. The cache missing from the laptop is a real prerequisite; CSV results alone cannot train a new channel/classifier pipeline.

A small proposed CIFAR pilot uses four D values (500, 1,000, 5,000, 15,000), three seeds, B in {48,192}, and SNR in {5 dB, noiseless}: 48 classifier fits, excluding tuning. Its purpose is to test memory, compute, equivalence, sensible noise effects and convergence. It does not inspect the formal larger-D targets at 35,000 and 45,000. Four sizes are too few to establish a flexible three-parameter scaling law robustly; do not make a formal exponent claim from this pilot.

### Proposed formal grid — requires protocol approval

Preserve the existing 12 CIFAR D values and ten subset/training seeds. A starting channel grid is B in {48,96,192} and SNR in {0 dB,10 dB,noiseless}, giving 1,080 fits for one declared projection realization, excluding HPO and projection/noise sensitivity studies. This is an explicit proposed workload, not work launched or guaranteed affordable. Adjust it after timing the pilot.

Fit using the first ten D levels (up to 25,000), reserve D=35,000 and 45,000 for forward prediction. Fit-family/recipe selection must not use those reserved outcomes. Keep the final image split outside checkpoint/HPO selection. A separate large-D development split may be needed for pilot/convergence checks so they do not leak the formal predictive targets.

Use the same nested source IDs across conditions within a seed, fixed internal/final splits, the same pretrained encoder and documented class mapping. Keep head dimension and parameter count fixed across B using the receive-side convention below. Repeat selected conditions with independent projection seeds to test whether findings depend on one projection; if they do, expand repetitions and include that variability in uncertainty before generalizing.

### Training and exposure policy

State which scientific object is primary. A fixed-recipe study holds the update/stopping rule fixed; a per-condition tuned study estimates a different best-observed envelope. Do not mix them in one curve without labeling the difference.

Use training/internal validation for all HPO, checkpoint and convergence decisions. Preserve a common candidate budget and log recipe selections. Check that the policy does not systematically undertrain low-SNR or large-D cases. Record steps, actual examples seen, effective passes, validation effort, feature extraction and wall time. A separately named exposure-matched control is needed before claiming compute fairness; early stopping plus one step ceiling is insufficient.

## 6. A concrete, testable abstract channel convention

This subsection is a proposed controlled baseline, not a claim of an optimal semantic coding system.

- Start with frozen z in R^192. Draw a seeded orthogonal 192×192 matrix Q independently of images/labels. Let Q_B be its first B rows; the rows are orthonormal and nested across B.
- Estimate and save a scalar gain for each B using **one fixed, unlabeled calibration pool from the training split**, with no validation/final images or calibration labels. Use that same gain at every D so changing D does not also change the effective noise scale. Transmitted `x = gain × Q_B z` has average power one per real coordinate on that calibration pool. D therefore counts head-training labels; access to the fixed unlabeled calibration pool is an additional declared resource shared by all conditions. Do not imply a purely D-image-available setting. Reject zero/nonfinite calibration energy rather than dividing by it.
- For finite linear SNR gamma, transmit `r = x + noise`, where independent real AWGN has variance `1/gamma` per coordinate. Convert dB using `gamma = 10^(SNR_dB/10)`. For noiseless evaluation, use exactly zero noise.
- Feed the classifier the fixed-width backprojection `z_received = Q_B.T × r / gain`. This keeps its 192 input values and C-class parameter count constant as B changes. At B=192 with zero noise, Q.T Q restores z up to numerical precision.
- The clean reference must use the same preprocessing/feature/head conventions. Test both an explicit bypass and the full-width/noiseless composition against the archived raw-feature path on controlled inputs. Differences cannot simply be labeled channel effects.
- Average power per coordinate implies total transmitted energy grows with B. State this openly. If making energy-efficiency claims, add a separately declared fixed-total-energy control; equal SNR labels under different conventions are not automatically equivalent.

Fresh noise during training studies a channel-aware receiver, while clean-only training followed by noisy testing studies robustness to mismatch. They are different questions. Choose one as primary; a small clean-trained comparison is useful. Use fixed validation noise banks for selection and independent repeated noise draws for final evaluation. Log noise seeds separately from subset/optimizer and projection seeds. Average repeated noise outcomes within each trained run before summarizing run-level uncertainty.

No per-image normalization is silently inserted here; scalar gain is undone at the receiver in the noiseless case. A learned projection/decoder would introduce another trainable-capacity axis and needs its own control study.

## 7. What would count as an answer?

| Observed outcome | Defensible conclusion |
|---|---|
| M0 predicts reserved larger-D results as well as M1/M2 within the declared tolerance | An offset description suffices for these tested conditions; no evidence that more complex rate changes are needed |
| M1 improves held-out predictions, M2 does not | Channel conditions change finite-range label-efficiency amplitude; a different exponent is not required by this evidence |
| M2 improves prediction reproducibly across seeds/projections/tasks | Evidence for condition-dependent curve shape under the declared policy, subject to fit-family and convergence controls |
| All forms fail larger-D prediction | A useful negative result and failure map; report misspecification rather than promote a fitted beta as a law |
| A task target is not reached at any measured D | Report that fact; more labels cannot be promised to overcome the tested link restriction |
| Differences disappear under exposure or convergence controls | The apparent interaction was substantially protocol-dependent; do not attribute it solely to a channel |

The eventual deliverable would be an evidence-linked set of learning curves, a held-out prediction table, an uncertainty and compute account, and a bounded statement of where data helps under channel constraints. A new architecture is not required for that contribution, but a well-supported question, controls and a defensible distinction from prior work are required.

## 8. What we can tell the supervisor now

“We have completed clean fixed-representation data-size baselines. The reviewed papers already establish several ingredients, so our candidate contribution is narrower: testing whether bandwidth/noise change downstream label efficiency and whether small-data measurements predict larger-data outcomes. We propose competing curve models and prospective larger-D checks, with fixed head capacity, explicit power/exposure conventions and separate noise uncertainty. The channel study is not implemented or run, and the exact novelty/sequence must be checked against closest prior work and the original thesis plan.”
