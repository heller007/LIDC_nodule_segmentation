# LIDC-IDRI Nodule Segmentation — Dataset, Problems, Method

Working notes for the segmentation project: what the data actually is, what
about it constrains the model, what is currently wrong with the results, and
what the plan is. Written against the notebook experiments
(`attention-on-dplinknet50-part-2`) and the `lidcseg` refactor in this repo.

Every number below is marked **measured** (printed by a run) or **unverified**
(not yet checked). Treat the distinction as load-bearing — several
unverified quantities would change the conclusions.

---

## 1. The dataset

### Provenance

Kaggle `zhangweiled/lidcidri` — a *derived* 2D slice export of LIDC-IDRI, not
the official TCIA DICOM/XML distribution. The export has no dataset card, so
the preprocessing between the original CT volumes and these PNGs is
undocumented: window/level, intensity normalisation, in-plane resampling and
the slice-selection rule are all unknown.

This matters for a write-up. Claims about HU ranges or physical nodule size
cannot be made from these PNGs. If the thesis needs them, re-derive from TCIA
(`10.7937/K9/TCIA.2015.LO9QL9SX`) rather than inferring from here.

Upstream LIDC-IDRI: 1,010 subjects, ~244,500 DICOM images, four radiologists
annotating independently then in an unblinded second read. Nodules ≥3mm get
full boundary contours.

### Layout

```
LIDC-IDRI-slices/
└── LIDC-IDRI-0001/
    └── nodule-0/
        ├── images/0.png 1.png ...     grayscale slice crops
        ├── mask-0/0.png ...           radiologist 1
        ├── mask-1/...                 radiologist 2
        ├── mask-2/...                 radiologist 3
        └── mask-3/...                 radiologist 4
```

One directory per patient, then per nodule, then parallel `images/` and four
`mask-N/` directories keyed by identical filenames.

### Measured

Dataset downloaded and profiled in full on 2026-10-17. Full output in
[`profile.md`](profile.md) / [`profile.json`](profile.json); regenerate with
`python -m lidcseg.profile_data --config configs/improved.yaml --zip <zip>`.

| Quantity | Value |
|---|---|
| Patients | **875** |
| Nodules | **2,630** (3.01 per patient) |
| Slices, all nodules | **15,548** (17.8 per patient) |
| Slices, `nodule-0` only | **5,808** (6.64 per patient) — what the notebook used |
| Native resolution | **128x128, uniform** across all 15,548 slices |
| Reader masks per slice | **4, on every single slice** (100%) |
| Positive pixel fraction (vote >=2) | **0.584%** — background:foreground **170:1** |
| Empty merged mask (vote >=2) | **32.1%** of slices |
| Nodule equivalent diameter | median **8.7 px**, p5 **3.2 px**, p95 26.2 px, max 47 px |
| Components per non-empty slice | 1.02 |
| Reader-vs-reader Dice (pairwise) | **0.527** (sd 0.211) |
| **Reader-vs-consensus Dice** | **0.592** (sd 0.317, median 0.75) |
| Download rate from Kaggle | 170 MB at **1.3 MB/s** (~2 min) |
| Files in the export | **77,740**, mean **1.9 KB** |
| Model size (notebook) | 111.1M params (SCANeXt), 123.8M (cascaded) |

Five of these change conclusions and are called out in §3.

### Resolved, and what each answer changed

The profiler (`lidcseg.profile_data`) measures all of this. Everything
previously listed as unverified is now measured; three answers were not what
was assumed.

| # | Quantity | Result | Verdict |
|---|---|---|---|
| 1 | Positive pixel fraction | 0.584% at vote >=2, 170:1 imbalance | As feared. Accuracy is unusable: all-background scores 99.42% |
| 2 | Empty-mask fraction | **32.1%** at vote >=2 | Worse than expected — see P11 |
| 3 | Reader-mask histogram | **4 masks on 100% of slices** | **Non-issue.** No slice is unlabelable; the concern was unfounded |
| 4 | Native resolution | **128x128, uniform** | **The training resize is a no-op** — see P12 |
| 5 | Nodule size distribution | median 8.7 px diameter, p5 3.2 px | Small, but the resize is not why — see P12 |
| 6 | All nodules vs `nodule-0` | 15,548 vs 5,808 slices; 2,630 vs 875 nodules | **+167.7% data** was being discarded |
| 7 | Reader agreement | pairwise **0.527**, vs-consensus **0.592** | See P13 — the most consequential number here |

## 2. Properties that constrain modelling

### 2.1 Severe foreground/background imbalance

Pixel accuracy saturates at 0.998 while Dice sits near 0.86. A
predict-all-background model would score close to that same accuracy.
Consequences:

- **Accuracy must not be reported** as a performance figure. It is in the
  metric set only because it was in the original, and it should be read as a
  sanity check, never a result.
- The loss has to be overlap-based (Dice/Tversky) or strongly reweighted;
  plain BCE collapses to the background class.
- Thresholding at a fixed 0.5 is not justified — the operating point should be
  chosen, which is why the threshold sweep exists.

### 2.2 Small objects against a 32x-stride encoder

Nodules are small: median equivalent diameter **8.7 px**, 5th percentile
**3.2 px**, and 1,004 of 10,800 annotated components are under 4 px across.

I previously framed this as a resize problem. **That was wrong** — the export
is already 128x128, so `img_size: 128` resamples nothing, and the profiler's
survival table shows 100% of components retained at every area threshold. Two
of the fixes listed in §5 (binarise-before-resize, diffuse-before-resize) are
therefore no-ops at the default config. They remain correct for any other
`img_size`, but they buy nothing here.

The real geometric problem is the **encoder stride**. ResNet50 downsamples 32x,
so a 128x128 input reaches the bottleneck at 4x4. A median 8.7 px nodule is
~0.27 px at that depth, and a 3.2 px one is ~0.1 px. Such nodules cannot be
represented at the bottleneck at all; they survive only through the skip
connections — which is precisely why P7 (two of four skips unwired) was a
worse bug than it first appeared.

This reframes the useful experiments. Not "train at 256" (there is no extra
detail to recover — upsampling invents nothing), but:

* reconnect all skips, and ablate them per stage (`model.use_skips`)
* consider a shallower encoder, or dilated convolutions in place of the last
  stride, so the bottleneck stays at 8x8 or 16x16
* size-stratified metrics, to see whether the sub-8 px half is where the
  score is actually lost

### 2.3 Four readers who disagree

Each slice has up to four independent segmentations. Collapsing them to one
binary target is a modelling decision, not preprocessing:

- vote ≥1 — union; maximises recall, inflates nodule extent
- vote ≥2 — the notebook's choice, and the usual "majority" convention
- vote ≥3 / ≥4 — intersection-leaning; high precision, discards boundary
  disagreement

The disagreement itself is signal: pixels where 2 of 4 readers marked nodule
are genuinely ambiguous, and training against a hard label there asks the model
to be confident where experts were not. Currently discarded.

Measured: pairwise reader Dice is **0.527** (sd 0.211, median 0.50) and
reader-vs-others'-consensus Dice is **0.592** (sd 0.317, median 0.75). The
readers disagree a great deal — and the vote->=2 rule discards all content on
32.1% of slices, where at most one reader marked anything (only 0.4% of slices
are empty at vote >=1). So roughly a third of the dataset is labelled
"background" purely because the panel did not reach two votes.

### 2.4 2D slices of a 3D structure

Each slice is classified in isolation. A radiologist reads a nodule through the
stack, using whether a structure persists across slices to distinguish nodule
from vessel cross-section. The model has no access to that. Slice filenames are
ordered, so the adjacency exists in the data and is simply unused.

### 2.5 Few slices per patient — 6.64, or 17.8

6.64 under the notebook's `nodule-0` filter; 17.8 once every nodule is
included. The leakage arithmetic below uses 6.64, since that is the setting
the published numbers were produced under.

This is the unusual property of this export and it cuts two ways.

**Good:** 875 patients is a lot of independent subjects. A 15% patient-grouped
test split is ~131 patients, which is a workable test set. Patient-grouped
splitting is viable here; on a dataset with 50 patients it would not be.

**Bad:** it makes the slice-level split maximally leaky. See below.

---

## 3. Problems

Ranked by how much they affect the validity of current conclusions.

### P1 — Patient-level leakage invalidates every reported number

**The problem.** `train_test_split` was applied to the flat array of 5,808
slices. Slices are grouped by patient, and adjacent slices through one nodule
are near-duplicate images.

**How bad.** With 6.64 slices per patient under a 70/15/15 random split, for a
test slice whose patient has n slices, the chance that at least one slice of the
same patient landed in training is `1 − 0.30^(n−1)`:

| n slices | P(same-patient slice in train) |
|---|---|
| 3 | 0.910 |
| 5 | 0.992 |
| 7 | **0.999** |

At the measured mean of ~7, **essentially every test slice had a near-duplicate
of itself in the training set.** The reported test Dice is substantially a
measure of memorisation, and the test set is not held out in any meaningful
sense.

**Status.** Fixed in `lidcseg`: `split.mode: patient` uses
`GroupShuffleSplit`/`GroupKFold` on patient id, and `assert_no_leakage` fails
the run if any patient appears in two sets. The old protocol is retained as
`split.mode: slice` purely so the gap can be measured.

**Expect every Dice figure to drop.** That drop is the finding, not a
regression.

### P2 — Two different metrics both called "Dice"

The notebook flattened the whole split into one ratio (`global Dice`). Large
nodules dominate it. The per-image mean — score each slice, then average —
weights every slice equally and is what most clinical segmentation papers
report. They are not interchangeable and can differ by several points.

Reporting the global number alone makes small-nodule failure invisible. Both
are now computed everywhere; `val_dice` for model selection is per-image.

### P3 — No uncertainty, so no result is distinguishable from noise

A single test Dice with no interval cannot separate a real improvement from
split variance. The notebook's 0.8611 → 0.8826 may or may not be a difference.
Bootstrap CIs over resampled slices are now reported, and `--suite cv5` gives
5-fold patient-grouped CV.

Note that a bootstrap over slices still understates uncertainty, because slices
are not independent — they cluster by patient. The CV mean across folds is the
more honest figure.

### P4 — Changes were bundled, so nothing is attributable

The jump from 0.8432 to 0.8826 changed seven things at once: learning rate
1e-5→5e-5, Tversky γ 1.5→0.75, aux weight 0.4→0.2, noise σ 0.01→0.005, zoom
floor 0.85→0.92, early-stopping patience 15→25, plateau patience 7→10.

Given that the *previous* step (adding Tversky + augmentation + deep
supervision + threshold tuning) made things **worse** than the baseline
(0.8611 → 0.8432), the most likely reading is that the original 1e-5 learning
rate simply under-trained the new loss, and the "improvements" were mostly an
LR fix. That is a useful finding, but it is not the finding the cell comments
claim. One change per run from here; `--suite ablation` enforces it.

### P5 — Data discarded by the nodule-0 filter

`nodule-0` only. Measured cost: **15,548 slices available vs 5,808 used**, and
**2,630 nodules vs 875** — the filter discarded **+167.7%** more data than it
kept. There are 3.01 annotated nodules per patient on average and the notebook
read one. This is the cheapest available improvement in the whole list.

### P6 — Encoder input distribution mismatch

ResNet50's ImageNet weights expect caffe-style inputs: BGR, channel means
subtracted, roughly [−124, 152]. The notebook fed `[0,1]`. The pretrained
filters therefore saw inputs about two orders of magnitude off their training
distribution, and much of the benefit of pretraining was likely wasted — while
still paying its cost in a frozen-feature mismatch. Now corrected, and
ablatable via `model.imagenet_preprocess` so the size of the effect gets
measured rather than assumed.

### P7 — Decoder was missing half its skip connections

Skips were wired into `dec1` and `dec2` only. The 16×16/512ch and 8×8/1024ch
encoder stages reached the decoder solely through the 4×4 bottleneck. For small
objects that is the wrong half to drop — mid-level spatial detail is exactly
what a 3mm nodule needs. Now all four; ablatable via `model.use_skips`.

### P8 — Non-monotonic experiment history suggests instability

Across the notebook's six scored variants, test Dice went 0.8611 → 0.8432 →
0.8826 → 0.8487 → 0.8648, and the last two cells produced no metrics at all.
A spread of ~0.04 Dice across architecture variants, with the *baseline*
beating two of the "improved" ones, is more consistent with optimisation
variance than with architectural differences. Until runs are repeatable with
intervals, architecture comparisons on this dataset are not interpretable.

### P9 — No architectural control

Everything compared was a variant of DPLinkNet50 + attention. Nothing
established that this architecture beat a plain U-Net on this data at all.
Addressed: `--suite baselines` adds a from-scratch U-Net and a ResNet50-U-Net
with no attention, under the identical pipeline.

### P10 — Opaque preprocessing in the source export

See §1. The chain from CT volume to PNG is undocumented, which caps how much
can be claimed about clinical relevance regardless of how good the Dice gets.

---

### P11 — A third of the dataset is empty under the vote rule

**32.1%** of slices have an empty merged mask at vote >=2, against only
**0.4%** at vote >=1. In other words, on nearly a third of slices at most one
of four radiologists marked anything, and the majority rule turns those into
pure-background training examples.

That matters three ways:

* It inflates an already 170:1 imbalance with slices that are not obviously
  negative — they are *contested*, not clean background.
* A model is being trained to output nothing on slices where a radiologist did
  see something. That is a defensible choice, but it is a choice, and it was
  never stated.
* It makes the vote threshold a first-class hyperparameter rather than
  preprocessing. `data.vote_threshold` and the `data.drop_empty_masks` /
  `data.empty_mask_keep_frac` budget knobs should be ablated.

### P12 — The resize is a no-op; the bottleneck stride is the real constraint

The export is already 128x128, so the training `img_size: 128` resamples
nothing. I had listed mask-resize and diffusion-order as fixes; they are
correct in general but **have no effect at this configuration**, and the claim
that the resize was costing small nodules was wrong.

What is real: ResNet50's 32x stride puts a median 8.7 px nodule at ~0.27 px in
the 4x4 bottleneck. Small nodules reach the decoder only via skip connections,
which makes P7 more serious than it looked. See §2.2 for what to test instead.

### P13 — The old score sits far above the annotators' own ceiling

Reader-vs-consensus Dice is **0.592**. The notebook reported **0.8826**.

Three caveats before drawing the obvious conclusion. The 0.8826 is *global*
Dice, which pools all pixels and is systematically higher than the per-image
mean that 0.592 is computed as; the two are not directly comparable. The
ceiling is also computed from a 3-reader consensus rather than 4. And a model
genuinely can beat an individual reader at predicting a consensus, because
fusion cancels per-reader noise — that is why §1 reports the leave-one-out
figure rather than the 0.527 pairwise one.

Even allowing all three, a gap of 0.29 Dice is far too large to explain. Set
against P1 — that ~99.9% of test slices had a near-duplicate in training — the
straightforward reading is that the number measured memorisation plus an
easy metric, not segmentation skill.

The actionable part: **0.592 is roughly the target to beat**, and anything
much above it should be treated as a bug until proven otherwise. Report
per-image Dice against this ceiling, not global Dice against nothing.

## 4. Where the results currently stand

All from the notebook, **all under the leaky slice-level split**, all global
Dice, all single runs without intervals. Recorded for reference, not as
baselines to beat.

| cell | variant | Dice | IoU | Prec | Recall |
|---|---|---|---|---|---|
| 2 | SCANeXt + Dice-Focal (baseline) | 0.8611 | 0.7560 | 0.9024 | 0.8233 |
| 5 | + Tversky, aug, deep sup, thresh (lr 1e-5) | 0.8432 | 0.7289 | 0.8592 | 0.8277 |
| 6 | same, lr 5e-5, γ 0.75, aux 0.2 | **0.8826** | **0.7899** | 0.8705 | 0.8951 |
| 26 | variant | 0.8487 | 0.7371 | 0.8920 | 0.8094 |
| 28 | cascaded attention decoder | 0.8648 | 0.7618 | 0.8895 | 0.8414 |
| 30, 31 | CAB variants | — | — | — | — (did not complete) |

Reading it: the best cell traded precision (0.90 → 0.87) for recall
(0.82 → 0.90), which for screening is the right direction — but that is a
threshold-placement effect as much as a model effect, and the best cell's
chosen threshold (0.575) sat at the edge of its own search window [0.25, 0.60],
so the true optimum may have been outside it.

**The first job is re-measuring all of this under the patient split.** The
profiler has now run (§1), and its answer reframes the table above: the
annotators' own reader-vs-consensus Dice is **0.592**, so every number in this
table sits 0.26–0.30 above what a radiologist achieves against their
colleagues' consensus. Combined with P1, treat the whole table as
uninterpretable rather than as a baseline to beat.

---

## 5. Methods now in place

Implemented in `lidcseg` (see the repo README for the full 25-item list).

**Protocol**
- patient-grouped splits with a runtime leakage assertion
- 5-fold patient-grouped CV (`--suite cv5`)
- bootstrap CIs on test metrics
- per-image *and* global Dice/IoU reported side by side
- threshold tuned on validation, applied unchanged to test, over a widened
  [0.10, 0.90] window

**Data**
- all nodule directories indexed, not just `nodule-0`
- configurable reader-vote threshold
- masks binarised before nearest-neighbour resize
- diffusion before resize, edge-replicated, γ stability-checked
- `.npy` memmap cache keyed by a hash of every preprocessing parameter
- fresh augmentation each epoch in `tf.data` instead of one static 2× copy

**Model**
- three architectures behind one factory: `dplinknet`, `resnet50_unet`, `unet`
- attention blocks as serializable Keras layers
- ImageNet-correct encoder input scaling
- all four decoder skips
- float32 output head under mixed precision

**Training**
- AdamW, cosine schedule with warmup, optional frozen-encoder phase
- per-image losses (Tversky-focal / Dice-focal / Dice-BCE)
- stateful streaming metrics
- gradient clipping, `TerminateOnNaN`

**Inference**
- 8-way dihedral TTA averaged in probability space
- small-component removal

**Measurement**
- `lidcseg.profile_data` — native-resolution dataset profiler covering all
  seven quantities above, with a paste-ready markdown report

**Reproducibility**
- every run writes its fully resolved config, split summary, index, history,
  results and figures
- 89 tests with no TF dependency; a TF test that builds every architecture

---

## 6. Research directions

Ranked by expected value per unit of effort. The first three are measurement,
not modelling — on current evidence (P8) the measurement is the bottleneck.

### Near term

0. ~~**Profile the dataset.**~~ **Done** (2026-10-17) — see §1 and
   [`profile.md`](profile.md). Re-run it if the vote threshold or `img_size`
   changes, since both alter the label set.
1. **Re-baseline everything under the patient split.** Run
   `notebook_best` → `notebook_best_patient` → `improved`. This produces the
   one number the thesis most needs: how much of the published 0.8826 was
   leakage.
2. **`--suite baselines`.** Establish that the architecture beats a plain
   U-Net at all. If it does not, that redirects the whole project.
3. **`--suite cv5` with CIs.** Replace single-split point estimates with a
   mean and interval.
4. **Stratify metrics by nodule size.** Report Dice in bins of ground-truth
   area. This is where §2.2 either is or is not a real problem, and it is the
   most likely source of a genuine, defensible contribution: a method that
   specifically improves small-nodule segmentation.
5. **Separate detection from delineation.** Report per-nodule detection rate
   (did any overlap occur) alongside Dice-given-detection. A Dice of 0 from a
   total miss and a Dice of 0.4 from a sloppy boundary are different failures
   and currently average together.

### Medium term

6. **2.5D input.** Stack adjacent slices as the three input channels instead of
   replicating one slice three times. The channels already exist, the encoder
   already expects three, and the slice ordering is in the filenames — so this
   is close to free and directly addresses §2.4. Strongest
   cost/benefit of the modelling ideas.
7. **Use reader disagreement instead of discarding it.** Options, roughly in
   order of ambition: soft targets from the vote count (0, 0.25, 0.5, 0.75, 1)
   with a soft Dice; per-reader multi-head prediction; or explicit aleatoric
   uncertainty estimation. A model that reports "experts disagreed here" is
   more clinically useful than one that guesses confidently, and it is a real
   research angle rather than a tuning exercise.
8. **Resolution study.** Train at 256×256 and compare, with the size-stratified
   metrics from (4). Tests directly whether the 128×128 resize was costing
   small nodules.
9. **Boundary metrics.** Add Hausdorff distance (95th percentile) and average
   surface distance. Overlap metrics are insensitive to boundary quality, which
   is what a radiologist actually looks at.
10. **Calibration.** Report reliability curves / ECE. A segmentation model whose
    0.7 means 70% is far more useful downstream than one that is merely
    well-thresholded.

### Longer term

11. **A proper 3D or hybrid baseline.** If 2.5D helps, full 3D on volumetric
    LIDC is the natural next step — but that means leaving this export and
    going back to TCIA, which is a separate data-engineering project.
12. **External validation.** LUNA16 derives from LIDC-IDRI, so it shares
    patients and is *not* independent — check the subject overlap before using
    it as external. Genuine external validation needs a different cohort.
13. **nnU-Net as a strong reference.** A self-configuring baseline is the
    standard reviewers ask for; beating it is a much stronger claim than
    beating a hand-tuned U-Net.

---

## 7. Protocol rules for this project

Fixed so that results stay comparable as the work continues.

1. **Patient-grouped splits always.** `split.mode: slice` exists only to
   quantify leakage and its numbers are never reported as performance.
2. **One change per run.** If several things must change together, say so
   explicitly and treat the result as unattributed.
3. **Report per-image Dice as the headline**, global Dice alongside it, never
   accuracy as a result.
4. **No comparison without an interval.** CV mean + CI, or at minimum a
   bootstrap CI.
5. **Thresholds are tuned on validation only.** Never on test.
6. **Baselines run through the identical pipeline.** A baseline that got worse
   augmentation or a worse LR is not a baseline.
7. **Seeds and configs are committed.** Every run's resolved config is written
   to its run directory; quote it when quoting a number.
8. **Note what is not parameter-matched.** Cross-architecture comparisons print
   a params column for this reason.

---

## 8. Open questions

Answered by the profiler:

- ~~What is the inter-reader Dice?~~ **0.527 pairwise, 0.592 vs consensus.**
- ~~What fraction of slices have fewer than 2 reader masks?~~ **None — all
  15,548 have 4.**
- ~~How many slices does including all nodules add?~~ **+9,740 (+167.7%).**
- ~~Native resolution, and does the resize cost small nodules?~~ **128x128
  already; the resize is a no-op.**

Still open:

- How much of 0.8826 was leakage? *(the first two training runs answer this)*
- Does the attention decoder beat a plain ResNet50-U-Net under an honest
  split, by more than the CIs overlap?
- Was the cell-5 → cell-6 gain mostly the learning-rate fix? *(`--suite loss`
  with LR held fixed)*
- What is the per-size Dice profile? With a median of 8.7 px and a 4x4
  bottleneck, is the sub-8 px half where the score is lost?
- Is vote >=2 the right fusion rule, given it empties 32.1% of slices? Does
  soft labelling from the vote count beat it?
- Does keeping the bottleneck at 8x8 or 16x16 (dilation in place of the last
  stride) help the small-nodule half?
- Can a model trained on consensus exceed 0.592 per-image Dice on a
  patient-grouped split at all?
