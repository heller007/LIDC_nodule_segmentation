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

| Quantity | Value | Source |
|---|---|---|
| Patients | **875** | every notebook cell printed `Patients Selected: 875` |
| Slices (`nodule-0` only) | **5,808** | `Total slices: 5808` |
| Slices per patient | **6.64** | 5808 / 875 |
| Radiologist masks per slice | 4 | directory structure |
| Split used | 4,065 / 871 / 872 | 70/15/15 over slices |
| Model size | 111.1M params (SCANeXt), 123.8M (cascaded) | Keras summary |
| Pixel accuracy at convergence | ~0.998 | every cell |

### Unverified — run the profiler

```bash
python -m lidcseg.profile_data --config configs/improved.yaml
```

Measures all of the following and writes `profile.json`, `profile.md` (tables
ready to paste into this section) and figures. Masks are read at **native**
resolution, which the training cache cannot do because it has already resized.
Pure numpy/OpenCV, no TensorFlow, a few minutes over the full set, with
`--max-slices` for a strided sample.

| # | Quantity | Why it matters |
|---|---|---|
| 1 | Positive pixel fraction, per vote threshold | How severe the imbalance is; implied tiny by the 0.998 accuracy but never printed |
| 2 | Empty-merged-mask fraction, per vote threshold | Sets the negative-sample budget |
| 3 | Reader-mask histogram | Slices with <2 masks on disk *cannot* be positive under a ≥2 vote — they are silently all-background and poison the negative class |
| 4 | Native resolution | Needed before any claim about what the 128×128 resize costs |
| 5 | Nodule size distribution, native **and** post-resize | The size-stratified metrics in §6.4 need these bins; the post-resize half directly tests §2.2 |
| 6 | Slice/patient/nodule counts, all nodules vs `nodule-0` | How much data P5 discarded |
| 7 | **Inter-reader pairwise Dice** | An approximate *ceiling* on achievable Dice — see below |

Item 7 was not on the original list and is arguably the most consequential.
The four radiologists disagree with each other by some amount; their mean
pairwise Dice is roughly the best a model can be expected to score against a
merged target. Published LIDC inter-reader agreement sits around 0.7–0.85
depending on the fusion rule.

If that number comes back near 0.85 while the notebook reported 0.8826 on a
leaky split, the model was scoring **at or above the agreement between the
experts who made the labels** — which would mean it was fitting the vote-≥2
fusion rule rather than the anatomy. That single comparison would recast the
whole result, and it costs one profiler run.

---

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

### 2.2 Small objects at low resolution

Nodules are 3–30mm. Slices are resized to 128×128 before training, and the
native resolution is unverified. If the export is 512×512 crops, a 4× downscale
puts a 3mm nodule in the range of a handful of pixels, where nearest-neighbour
mask resizing and a 32× encoder stride both start to hurt. The deepest ResNet50
tap is 4×4 for a 128×128 input — a small nodule may not survive to the
bottleneck at all.

This is the most likely place where headline Dice hides a real weakness:
large nodules carry the score while small ones are missed.

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

The profiler quantifies this (`inter_reader_agreement`): mean pairwise Dice
between readers, which bounds what any model can score against a fused target.
Compare it against the model's Dice before claiming the model is good.

### 2.4 2D slices of a 3D structure

Each slice is classified in isolation. A radiologist reads a nodule through the
stack, using whether a structure persists across slices to distinguish nodule
from vessel cross-section. The model has no access to that. Slice filenames are
ordered, so the adjacency exists in the data and is simply unused.

### 2.5 Few slices per patient — 6.64

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

`nodule-0` only. Every second and later nodule of a multi-nodule patient was
dropped. Amount unknown until the index runs, but it is free data in a dataset
where more data is the cheapest available improvement.

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

**The first job is re-measuring all of this under the patient split** — and
before that, one profiler run, because the inter-reader Dice determines whether
these numbers were ever physically meaningful.

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

0. **Profile the dataset.** `python -m lidcseg.profile_data`. Minutes, no GPU,
   and it fills §1's table — including the inter-reader Dice ceiling, which
   determines whether any of the existing numbers were meaningful.
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

- How much of 0.8826 was leakage? *(answered by the first two runs)*
- Does the attention decoder beat a plain ResNet50-U-Net under an honest
  split, by more than the CIs overlap?
- Was the cell-5 → cell-6 gain mostly the learning-rate fix? *(`--suite loss`
  with LR held fixed answers this)*
- What is the per-size Dice profile, and is small-nodule performance the real
  weakness?
- How many slices does including all nodules actually add?
- What fraction of slices have fewer than 2 reader masks, and are they
  silently poisoning the negative class?
- What is the inter-reader Dice, and did the reported 0.8826 exceed it?
