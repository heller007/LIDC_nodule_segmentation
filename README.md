# LIDC-IDRI Nodule Segmentation — DPLinkNet50 with Attention Decoders

Lung-nodule segmentation on the LIDC-IDRI slice dataset (Kaggle
`zhangweiled/lidcidri`: 875 patients, 5,808 slices with `nodule-0` only, four
radiologist masks per slice). A ResNet50 encoder feeds a LinkNet-style decoder
whose every stage carries an attention block (SCANeXt dual-attention or a
3-gate cascaded block), trained with focal Tversky loss and deep supervision.

**[docs/RESEARCH_NOTES.md](docs/RESEARCH_NOTES.md)** covers the dataset's
properties, the open problems, and the research plan. Read it before
interpreting any number here.

Refactored from an exploratory notebook into a configurable, reproducible
pipeline. One command trains; one command runs an ablation suite.

```bash
python -m lidcseg.train --config configs/improved.yaml
python -m lidcseg.experiments --config configs/improved.yaml --suite ablation
```

On Kaggle, open [`build/make_standalone.py     generator for the offline single-file notebook
notebooks/
  kaggle_train.ipynb             full pipeline (needs internet)
  kaggle_profile_standalone.ipynb   profiler only, fully offline`](notebooks/kaggle_train.ipynb)
and run it top to bottom.

---

## Read this before comparing against the old numbers

The notebook reported **test Dice 0.8826 / IoU 0.7899** at its best. That run
split the data with `train_test_split` over the flat array of slices.

Neighbouring CT slices through one nodule are near-identical images. A
slice-level split therefore puts copies of the same nodule in train and in
test, and the test score partly measures memorisation.

The dataset has 5,808 slices across 875 patients — **6.64 slices per patient**.
Under a 70/15/15 random slice split, a test slice whose patient has n slices
has a `1 − 0.30^(n−1)` chance that at least one slice of the same patient
landed in training: 0.91 at n=3, 0.99 at n=5, **0.999 at n=7**. At the measured
mean, essentially every test slice had a near-duplicate of itself in training.

This repo splits by **patient** (`split.mode: patient`), verified by an
assertion that fails the run if any patient appears in two sets.

**Expect the headline Dice to drop.** That is the leakage being removed, not a
regression. Two configs are provided to measure it directly:

| config | recipe | split | what it tells you |
|---|---|---|---|
| `notebook_best.yaml` | original | slice | reproduces the old number; checks the refactor is faithful |
| `notebook_best_patient.yaml` | original | patient | same model, honest protocol |
| `improved.yaml` | improved | patient | the proposal |

Run all three. `notebook_best` minus `notebook_best_patient` is the leakage.
`improved` minus `notebook_best_patient` is the actual modelling gain.

## Baselines

Ablations compare the model against versions of itself, which cannot say
whether the architecture was a good choice. Two external baselines run through
the identical pipeline — same split, loss, augmentation, schedule and
evaluation — so a difference is architectural:

| config | arch | encoder | decoder | attention |
|---|---|---|---|---|
| `improved.yaml` | `dplinknet` | ResNet50 (ImageNet) | LinkNet: additive skips, 1x1 squeeze/expand | SCANeXt per stage |
| `baseline_resnet50_unet.yaml` | `resnet50_unet` | ResNet50 (ImageNet) | U-Net: concat skips, two 3x3 convs | none |
| `baseline_unet.yaml` | `unet` | none (random init) | U-Net | none |

How to read the result:

- **`improved` − `resnet50_unet`** — what the LinkNet decoder and the attention
  blocks are worth, holding the encoder fixed. `--suite attention` then splits
  that between the decoder and the attention itself.
- **`resnet50_unet` − `unet`** — what the ImageNet encoder is worth.

```bash
python -m lidcseg.experiments --config configs/improved.yaml --suite baselines
```

The five-variant `baselines` suite runs all three architectures plus the two
other attention settings and writes one leaderboard.

**These models are not parameter-matched.** DPLinkNet50 + SCANeXt measured
111.1M parameters in the original notebook; the two baselines are smaller, and
the leaderboard prints the exact count per run. If DPLinkNet50 wins by a
little, test whether capacity explains it before crediting the attention
block: raise `model.base_filters` on the U-Net, or compare against
`--suite attention`'s `none` row, which is parameter-matched by construction.

Two config notes, both deliberate:

- `baseline_unet.yaml` sets `imagenet_preprocess: false` — there are no
  pretrained weights expecting caffe-style inputs, so it trains on `[0,1]`.
- It also sets `lr: 1.0e-3` against `3.0e-4` for the fine-tunes. A
  from-scratch net needs a larger step; handing it a fine-tuning LR would
  understate it and make the baseline a straw man.

Inputs stay 3-channel for every architecture so one pipeline serves all three.

---

## What changed, and why

### Protocol

| # | Change | Rationale |
|---|---|---|
| 1 | **Patient-grouped splits** | The leakage above. `GroupShuffleSplit` / `GroupKFold` on patient id, asserted at runtime. |
| 2 | **5-fold grouped CV** | A single hold-out split of a few hundred patients is noisy; two configs differing by 0.01 Dice are not distinguishable from it. `--suite cv5`. |
| 3 | **Bootstrap CIs** | Test Dice now ships a 95% interval over resampled slices, so "better" can be checked rather than asserted. |
| 4 | **Per-image Dice reported alongside global** | The old metric pooled every pixel in the split into one ratio, which large nodules dominate. Per-image mean weights each slice equally and is the stricter, more clinically meaningful number. Both appear in every result. |

### Data

| # | Change | Rationale |
|---|---|---|
| 5 | **All nodules per patient** | The notebook read only `nodule-0` and silently dropped nodules 1..N of every multi-nodule patient. The notebook prints how much data this recovers. |
| 6 | **Dataset root auto-discovery** | Mount paths differ: `kagglehub.dataset_download` gives `/kaggle/input/datasets/<owner>/<slug>/` (what the notebook hard-coded, and correct for that setup), while the Add Data panel gives `/kaggle/input/<slug>/`. Now resolved by searching for `LIDC-IDRI-<number>` directories, so either works. |
| 7 | **Binarise masks before resize** | Resizing a 0-255 mask bilinearly and thresholding after invents labels along nodule boundaries. Now: threshold, then nearest-neighbour resize. |
| 8 | **Diffusion before resize, no wrap-around** | Denoising after downsampling works on already-mixed pixels. The old `np.roll` gradients also wrapped the top row into the bottom; now edges replicate. `gamma` is checked against the 0.25 stability bound. |
| 9 | **`.npy` memmap cache** | Preprocessing ~6k slices took minutes on every run and held the whole float32 stack in RAM twice. Now paid once, keyed by a hash of every preprocessing parameter, then memmapped. |
| 10 | **Per-epoch augmentation in `tf.data`** | The old code built one static augmented copy and concatenated it, so every epoch saw the same 2x set — double the memory for a fraction of the regularisation. Transforms are now drawn fresh each epoch via stateless RNG (reproducible, parallel-safe). |

### Model

| # | Change | Rationale |
|---|---|---|
| 11 | **ImageNet-correct input scaling** | ResNet50's pretrained weights expect caffe-style inputs (mean-subtracted, ~[-124, 152]). The notebook fed raw [0,1], so the pretrained filters saw inputs an order of magnitude off their training distribution. Ablatable via `model.imagenet_preprocess`. |
| 12 | **All four decoder skips reconnected** | The notebook wired skips only into `dec1`/`dec2`; `conv3_block4_out` (16x16, 512ch) and `conv4_block6_out` (8x8, 1024ch) reached the decoder only through the 4x4 bottleneck. LinkNet specifies all four. Ablatable via `model.use_skips`. |
| 13 | **Attention as registered Keras layers** | The `Lambda(lambda t: tf.reduce_mean(...))` wrappers blocked `model.save()` without `safe_mode=False`, which is why only weights could be checkpointed. Now serializable `Layer` subclasses. |
| 14 | **float32 output head** | Under `mixed_float16` a float16 sigmoid saturates and the Dice gradient vanishes. The final activation and loss stay full precision. |
| 15 | **Residual in the cascaded block** | Three multiplicative gates in series can collapse the signal; `x + gated` keeps the ungated path available. |

### Training

| # | Change | Rationale |
|---|---|---|
| 16 | **Per-image loss** | `tf.reshape(y, [-1])` pooled the whole batch into one Tversky ratio, so a batch with one large nodule and three small ones was dominated by the large one. Now scored per image, then averaged. Ablatable via `loss.per_image`. |
| 17 | **Streaming metrics** | `dice_coef` was a stateless function, which Keras averages across batches — the mean of per-batch ratios is not the epoch ratio. Now stateful `Metric` classes. |
| 18 | **Empty-and-correct scores 1.0** | A slice with no nodule, predicted empty, was `0/0`. Now it is a perfect score, not a zero. |
| 19 | **AdamW + cosine schedule with warmup** | Replaces constant LR + `ReduceLROnPlateau`. Decoupled weight decay regularises an 80M-parameter model better than none. `ReduceLROnPlateau` is attached only when `schedule: none`, since the two fight. |
| 20 | **Batch 16, mixed precision** | Batch 4 at 128x128 wastes a T4 and makes BatchNorm statistics noisy. |
| 21 | **Optional frozen-encoder warmup** | Two epochs training the random decoder against frozen ImageNet features before fine-tuning, so early large gradients do not wreck the pretrained stack. |
| 22 | **Gradient clipping, `TerminateOnNaN`** | Focal Tversky with small `gamma` can spike. |

### Inference

| # | Change | Rationale |
|---|---|---|
| 23 | **8-way dihedral TTA** | Averaged in probability space so the threshold sweep still works on soft scores. |
| 24 | **Small-component removal** | LIDC nodules are >=3mm; single-pixel responses cost precision and can never be real. `eval.min_component_px`. |
| 25 | **Wider threshold sweep** | Kept from the notebook (tune on val, apply to test — the right order), widened from [0.25, 0.60] to [0.10, 0.90]. The old window's optimum sat at its upper edge (0.575), so the true optimum may have been outside it. |

---

## Profiling the dataset

Before training anything, measure what you are training on:

```bash
python -m lidcseg.profile_data --config configs/improved.yaml
python -m lidcseg.profile_data --config configs/improved.yaml --max-slices 1500
```

**No internet or GPU available?** `notebooks/kaggle_profile_standalone.ipynb`
is a single self-contained notebook with the profiler inlined -- no clone, no
code dataset, nothing to install. Regenerate it with
`python build/make_standalone.py` after changing `profile_data.py` or
`data/index.py`; the generated file is verified to produce output identical to
the package version.

Minutes, no GPU, no TensorFlow. Reads masks at **native** resolution — the
training cache has already resized, so it cannot answer these — and writes
`profile.json`, `profile.md` (paste-ready tables) and `profile.png` to
`<out_dir>/profile/`:

- positive pixel fraction and background:foreground ratio, **per vote threshold**
- empty-merged-mask fraction, per vote threshold
- reader-mask histogram, flagging slices with fewer masks than the vote needs
  (those cannot be positive by construction)
- native resolution, and whether it is uniform
- nodule size distribution native **and** after the training resize, plus a
  survival table: how many components fall below 1/4/9/16/25 px
- slice, patient and nodule counts with all nodules vs `nodule-0` only
- **inter-reader pairwise Dice** — roughly the ceiling on achievable Dice,
  since a model cannot meaningfully beat the agreement between the
  radiologists who drew the labels

That last one is worth running before trusting any score.
[docs/RESEARCH_NOTES.md](docs/RESEARCH_NOTES.md) explains why.

## Layout

```
configs/                 experiment definitions (YAML, with _base_ inheritance)
  base.yaml                shared defaults
  smoke.yaml               2-epoch wiring check — run this first
  improved.yaml            the proposal
  notebook_best.yaml       faithful reproduction of the old best cell
  notebook_best_patient.yaml   same recipe, honest split
  improved_cab.yaml        cascaded attention instead of SCANeXt
  improved_noattn.yaml     attention removed
  improved_cv5.yaml        5-fold grouped CV
  baseline_unet.yaml           plain U-Net, random init
  baseline_resnet50_unet.yaml  ResNet50 encoder + U-Net decoder, no attention
src/lidcseg/
  config.py              dataclasses, YAML loading, dotted CLI overrides
  data/
    index.py               dataset discovery; carries patient id
    preprocess.py          diffusion, reader-vote fusion
    cache.py               .npy memmap cache
    splits.py              patient-grouped splits + leakage assertion
    pipeline.py            tf.data, per-epoch augmentation
  models/
    attention.py           SCANeXt and cascaded blocks (serializable)
    dplinknet.py           DPLinkNet50: encoder taps, LinkNet decoder
    unet.py                U-Net and ResNet50-U-Net baselines
    factory.py             arch dispatch
  losses.py              focal Tversky, Dice+Focal, Dice+BCE
  metrics.py             numpy metrics (importable without TensorFlow)
  metrics_tf.py          streaming Keras metrics
  evaluate.py            TTA, threshold tuning, bootstrap CIs
  profile_data.py        dataset profiler (native-resolution statistics)
  report.py              curves, metric bars, qualitative panels
  train.py               entry point
  experiments.py         ablation suites + leaderboard
docs/RESEARCH_NOTES.md   dataset properties, problems, research plan
notebooks/kaggle_train.ipynb
tests/
  test_no_tf.py          89 checks: splits, metrics, configs -- no TF needed
  test_profile.py        51 checks: profiler against a synthetic tree, no TF
  test_models_tf.py      builds every arch, forward/backward, needs TF
```

## Configuration

Configs inherit via `_base_` and accept dotted overrides:

```bash
python -m lidcseg.train --config configs/improved.yaml \
    --set train.lr=3.0e-4 model.attention=cab aug.enabled=false
```

Write override floats as `3.0e-4`, not `3e-4` — YAML 1.1 reads the latter as a
string. (The loader coerces to the declared field type, so both work, but the
explicit form is clearer.)

Unknown keys are rejected rather than silently ignored.

## Running an ablation

```bash
python -m lidcseg.experiments --config configs/improved.yaml --suite ablation
python -m lidcseg.experiments --config configs/improved.yaml --suite attention
python -m lidcseg.experiments --config configs/improved.yaml --suite cv5
python -m lidcseg.experiments --list          # show every suite
```

Suites: `ablation` (11 variants, one change each), `baselines` (5 -- the
architecture comparison), `attention`, `loss`, `leakage`, `cv5`. The
leaderboard is rewritten after each variant, so an interrupted sweep keeps its
results, and a variant that crashes does not lose the others.

## Each run writes

```
runs/<name>/
  config.yaml          fully resolved — the run is reproducible from this alone
  index.json           every slice used, with patient ids
  split.json           set sizes and the patient-overlap check
  best.weights.h5
  history.csv
  results.json         val + test metrics, threshold, sweep trace, CIs
  summary.json         hardware, params, timing
  training_curves.png  test_metrics.png  threshold_sweep.png  predictions.png
```

## Tests

```bash
python tests/test_no_tf.py
```

54 checks over the split protocol, vote fusion, diffusion, metric conventions
and config loading — no TensorFlow or GPU required. These cover the parts most
likely to be silently wrong, including that a slice-level split *does* leak and
that the leakage assertion catches it.

## Requirements

Kaggle's GPU image already has everything. Locally:

```bash
pip install -r requirements.txt
pip install -e .          # or put src/ on PYTHONPATH
```

The package lives under `src/`, so `python -m lidcseg.train` needs either an
editable install or `PYTHONPATH=src`. The Kaggle notebook sets `PYTHONPATH`
itself — and asserts a subprocess can import the package before running
anything long, since `sys.path` set in a kernel does not reach `!python`.

TensorFlow 2.15+ (Keras 3 supported), numpy, opencv, scikit-learn, pyyaml,
matplotlib.

---

## Suggested order of work

0. `python -m lidcseg.profile_data` — measure the dataset before modelling it.
   Minutes, no GPU, and the inter-reader Dice it reports tells you what score
   is even achievable.
1. `configs/smoke.yaml` — confirm the path end to end (~3 min).
2. `notebook_best.yaml` — confirm the refactor reproduces ~0.88 global Dice
   under the old protocol. If it does not, the refactor changed the model and
   that needs resolving before anything else.
3. `notebook_best_patient.yaml` — measure the leakage.
4. `improved.yaml` — the proposal.
5. `--suite attention`, then `--suite ablation` — attribute the gains.
6. `--suite cv5` — the number to quote, with a CI.

Baselines worth adding for a write-up: plain U-Net and a U-Net with the same
ResNet50 encoder, both under the patient split. Without them it is hard to
claim the attention decoder specifically is doing the work — `--suite ablation`
answers that only within this architecture.
