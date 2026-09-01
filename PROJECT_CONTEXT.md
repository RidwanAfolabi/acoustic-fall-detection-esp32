# Acoustic Emergency Alerting System — Project Context

> Handoff / memory document. Written 2026-08-26. Paste this whole file into a new
> LLM session to bring it up to speed. Everything stated as a number here was
> measured, not estimated, unless explicitly flagged.

---

## 1. What this project is

Detect the sound of a person falling in a **health-centre restroom** and raise an
alert. Phase 1 is a proof of concept.

- **Target device:** ESP32-S3-BOX-3, TensorFlow Lite Micro, 8 MB external PSRAM
- **Approach:** transfer learning on **YAMNet** (Google's MobileNetV1 audio model,
  pretrained on AudioSet ~2M clips)
- **Working directory:** `c:\Users\USER\Documents\ESP32-S3-BOX-3-Voice-project`
- **Project root:** `acoustic-alert-system/` (a sibling `stt-tts-agentic-project/`
  is a *different, unrelated* project — do not touch it)

### Status at a glance

| Phase | State |
| ----- | ----- |
| Steps 1–5 v1 (audit → preprocess → split → train → INT8 export) | ✅ **complete** — synthetic RIRs, baseline model |
| Steps 2–5 v2 (real MIT bathroom RIRs) | 🟡 **in progress** — Step 2 ✅, Step 3 ✅, Step 4 🔄 running, Step 5 ⏳ |
| PC demo bundle for colleagues (.keras) | ✅ `share/fall-detector-demo.zip` (14 MB) |
| PC demo bundle (.tflite, lightweight) | ✅ `share/fall-detector-tflite-demo.zip` (3.6 MB) |
| Confuser gap closed (hard negatives) | ❌ blocked — MUSAN/OpenSLR-28 archives corrupt/truncated (see §13) |
| ESP32 firmware (C++ front-end + TFLM integration) | ❌ not started |
| Phase 2 (external data, emergency/normal taxonomy) | 🟡 scaffolding written, data partly downloaded, **not run** |

---

## 2. Environment — read this first, it has bitten us twice

Two hard Windows constraints:

1. **Python must be 3.11.** The machine default is **3.14**, and TensorFlow
   publishes no cp314 wheels. `pip install tensorflow` fails outright on 3.13/3.14.
2. **The venv must live at a short path.** TensorFlow's bundled headers exceed the
   260-char Windows `MAX_PATH`. `LongPathsEnabled` is `0` on this machine and
   enabling it needs admin.

### Working interpreter

```
C:\aesv\Scripts\python.exe        <-- use this for everything
```

Verified: TensorFlow 2.21.0, tensorflow-hub 0.16.1, librosa 0.11.0, numpy 2.4.6,
pandas 3.0.5, Keras 3.

### Other environments on this machine

| Path | Python | Notes |
| ---- | ------ | ----- |
| `C:\aesv` | 3.11.0 | ✅ **the one to use** |
| `acoustic-alert-system\.venv` | 3.11.0 | Also works now (TF installed via junction trick) |
| `<root>\.venv` | 3.14.0 | **Belongs to `stt-tts-agentic-project`. Do not delete or modify.** |
| Global `C:\Python314` | 3.14.0 | Cannot run TF |

### The MAX_PATH failure and its fixes

Symptom — looks like a corrupt download, is actually path length:

```
OSError: [Errno 2] No such file or directory:
'...\client_side_weighted_round_robin.upb_minitable.h'
```

Fixes, all verified:
- **Short venv path** (`C:\aesv`) — what we use
- **Directory junction, no admin needed:**
  ```powershell
  New-Item -ItemType Junction -Path "C:\aeslink" -Target "<deep project path>"
  C:\aeslink\.venv\Scripts\python.exe -m pip install tensorflow
  ```
  Python resolves `sys.prefix` through the short path (98 chars → 34). Only needed
  during install; afterwards the venv works via its real path.
- **`LongPathsEnabled = 1`** (needs admin) — the proper permanent fix

### PowerShell activation gotcha (verified by experiment)

`PATHEXT` does **not** include `.PS1`. So `...\Scripts\activate` runs
`activate.bat`, which sets variables in a cmd subprocess and **silently does
nothing** in PowerShell — no error, `python` stays wrong. Use
`...\Scripts\Activate.ps1`. Activation is session-wide and **not** affected by
current directory.

---

## 3. The DSP contract — every constant is load-bearing

Defined in `src/config.py`. These bind the Python pipeline *and* the future
ESP32 C++ front-end. Changing one without the other silently breaks accuracy.

| Parameter | Value |
| --------- | ----- |
| Sample rate | 16 000 Hz, mono |
| Window | 0.975 s = **15 600 samples** |
| Hop (50 % overlap) | 0.4875 s = **7 800 samples** |
| STFT frame / hop | 400 / 160 samples (25 ms / 10 ms) |
| FFT length | 512, periodic Hann, **magnitude** (not power) |
| Mel bins | 64, **125 Hz – 7500 Hz** |
| Log offset | `log(mel + 0.001)` |
| Patch | 96 frames × 64 bins → 1 patch per window |
| Embedding | 1024-d |

**Why 0.975 s:** a 96-frame patch at 10 ms hop needs the last frame's full 25 ms
window → 95 × 0.010 + 0.025 = 0.975 s.

**Mel floor discrepancy — still an open decision.** The original brief said 50 Hz;
upstream YAMNet uses **125 Hz**. We kept 125 Hz, because a mismatched floor would
shift every mel bin relative to what the frozen backbone was trained on.

The reference implementation is `yamnet_model.build_log_mel()`. **This is the
specification the C++ code must reproduce.**

---

## 4. Dataset — SAFE

**SAFE: Sound Analysis for Fall Event detection**, Kaggle
`antonygarciag/fall-audio-detection-dataset`. 950 clips, 274 MB, in `data/raw/`.
All **48 kHz mono PCM_16, exactly 3.000 s**. Perfectly balanced 475/475.

### Filename schema — the brief was WRONG, this was derived empirically

```
01-430-01-001-01.wav     AA-BBB-CC-DDD-FF
                         AA  fold id     01–10, exactly 95 files each
                         BBB clip uid    001–950, unique per file
                         CC  event code  00 = background, 01–07 = fall subtypes
                         DDD take id     001–475, a plain index
                         FF  class code  01 = Fall, 02 = Non-Fall
```

The brief said class was **first** and fold **last**. It is the reverse. Evidence:
`FF` has exactly 2 values split 475/475 and is perfectly determined by `CC`
(`CC==00` ⟺ non-fall, for all 950 files, zero exceptions); `AA` has 10 values with
exactly 95 files each — a textbook 10-fold CV design.

**Had this gone unnoticed**, Step 3 would have split on a 2-valued "fold" that was
actually the label, putting all falls in train and all non-falls in test. Silent
catastrophic failure, no crash.

`DDD` groups straddle folds, which would imply leakage *if* it identified a source
recording. Tested: comparing corpus-whitened background mel profiles, within-DDD
pairs are no more similar than random pairs (**Cohen's d = −0.04**). It is a plain
index; safe to ignore when splitting.

### How SAFE was recorded (from the source paper)

Falls were a **grappling dummy** dropped onto **carpet, wood and concrete**, in
**basements, laboratories, stairwells**. Non-fall is household background noise.
Fall clips were built by **mixing fall events into background taken from the
non-fall recordings** — so the two classes share identical backgrounds and differ
only by the added impulse.

---

## 5. Pipeline — Steps 1–5, all complete

### Step 1 — `src/1_dataset_audit.py`

Integrity, class/fold distribution, window-slicing preview. Exit codes: `0` clean,
`1` issues, `2` no dataset. Result: 950/950 decode, zero clipped/silent/DC-biased,
event-code invariant holds for all files. Projected **4,749 windows**.

### Step 2 — `src/2_preprocess.py` (+ `src/rir.py`)

Resample to 16 kHz mono, apply bathroom RIR, slice to 15,600-sample windows, save
`int16`. **14,247 windows** (4,749 × 3 variants), 444 MB.

Variants per clip:
- **0 = canonical RIR** — one fixed bathroom, identical every run (used by val/test)
- **1 = dry** — resampled only
- **2+ = random RIR** — randomised geometry, RT60 0.45–0.80 s, DRR −9…+2 dB

**RIR synthesis is a hybrid** (in `src/rir.py`): image-source method for
geometrically correct direct sound and early reflections, plus a *statistically
constructed* exponentially-decaying late tail. Pure ISM always under-builds the
diffuse tail and undershoots RT60. Every RIR is measured back by Schroeder T30 —
**measured RT60 tracks target to 1.2 % mean error**.

Three bugs found by that self-verification, all fixed — worth knowing because they
are easy to reintroduce:

1. **RT60 undershot ~10 %** — HF shaping steepens the broadband decay. Fixed with
   `RT60_CALIBRATION = 1.113`. Error 10.2 % → 1.1 %.
2. **`argmax` found the wrong direct path** (75 ms vs the geometric 7.26 ms). In a
   small hard room early reflections sum *above* the direct sound. Now uses **first
   significant arrival**. Trimming to the peak would have shifted every event in the
   corpus backwards in time, silently.
3. **Peak-normalising the RIR multiplied levels ~10×**, forcing a headroom cut on
   61 % of windows (worst gain 0.096). Switched to **unit-energy** normalisation:
   8.3 % of windows, worst gain 0.705. Loudness is real evidence for a fall, so a
   silent 10× attenuation would have corrupted it.

Pluggable sources (both implemented):
```
--rir-source measured --rir-dir DIR     # real RIRs
--negatives-dir DIR                     # hard negatives, OFF by default
```

### Step 3 — `src/3_prepare_splits.py`

Whole-fold splits. Default `--strategy fold` → **7 train / 2 val / 1 test folds**
(70/20/10 by clip). `--strategy ratio` gives exactly 70/15/15 with a 143-clip test
set (val and test then share a fold pool, never a clip).

Fold assignment is **computed, not hardcoded**: folds are ranked by class skew and
the most skewed go to train. Fold 10 (45.3 % fall vs ~50.5 % elsewhere) lands in
train automatically, leaving val/test balanced.

**Variant policy:** train uses all variants; **val/test use variant 0 only**. So
evaluation is reverberant like deployment, reproducible, and not inflated by
counting three augmentations of one clip as three test cases.

Verified: 0 clips span >1 split, 0 variants separated, 0 folds shared between train
and eval, index arrays disjoint and in range.

Splits: train 665 clips / 9,972 windows · val 190 / 950 · test 95 / 475.

### Step 4 — `src/4_train.py` (+ `src/yamnet_model.py`)

**This is transfer learning, specifically frozen feature extraction — not
pretraining, and not fine-tuning.** The backbone's 3,217,344 parameters never move;
only a 131,458-parameter head trains:

```
1024-d embedding -> Dense(128, relu) -> Dropout(0.4) -> Dense(2, softmax)
```

Adam, EarlyStopping (patience 15, restore best), ReduceLROnPlateau. Backbone runs
**once**; all 14,247 embeddings cached to `data/processed/embeddings.npy` (58 MB),
so epochs take a fraction of a second.

#### The patch-input rebuild — critical to understand

TFLite Micro supports **neither `tf.signal.stft` nor the mel filterbank matmul**.
So the deployed model **cannot take a waveform**. TF-Hub's YAMNet is a sealed
waveform-in graph with no patch entry point and does not expose its variables.

`src/yamnet_model.py` therefore **rebuilds** YAMNet with a fixed `(96, 64, 1)` input
and loads the official `yamnet.h5` weights (downloaded to `models/yamnet.h5`).
Keras 3 forbids `/` in layer names (the checkpoint uses it), so layers use
underscores with an explicit `yamnet_h5_map` that **fails loudly** on any unmapped
layer rather than leaving it randomly initialised.

**Training refuses to start unless the rebuild matches TF-Hub:**

| Check | Result |
| ----- | ------ |
| Backbone parameters | 3,217,344 — exact match |
| Max log-mel difference | 8.6e-06 |
| Max embedding difference | 1.4e-05 (scale 4.93) |
| Relative embedding error | **2.9e-06** — floating-point noise |

Embeddings are extracted through the **rebuilt** model, not TF-Hub, so the head
trains on exactly what gets exported.

#### Window-level label noise — the key modelling insight

Window accuracy looked poor (0.80) until diagnosed. Mean P(fall) by window position
within fall clips:

```
w0=0.49   w1=0.85   w2=0.92   w3=0.92   w4=0.60
```

The impact sits mid-clip. A 3 s clip is sliced into 5 windows and all five inherit
the clip's label, but windows 0 and 4 are largely background wearing a "fall"
label. **No model can fit that.** Aggregating per clip recovers the signal:
0.80 → 0.92.

**Use mean, not max.** Max measured far worse (0.68 vs 0.92) because one spurious
window becomes a false alarm for the whole clip. **The firmware should likewise
integrate over ~1–2 s before alarming, not fire on a single inference.**

### Step 5 — `src/5_export_tflite.py` — INT8 quantization and C export

> Note: Step 5 and all Phase 2 scripts were authored **outside** the sessions that
> produced Steps 1–4. Their outputs have been read and are reported accurately
> below, but that code has not been line-by-line audited here.

Outputs in `models/`:
```
yamnet_fall_detector_int8.tflite      3.67 MB
yamnet_fall_detector_float32.tflite  13.35 MB
yamnet_model_data.h                   header with dims + quant params
yamnet_model_data.cc                 23.6 MB C byte array
```

**Quantization parameters the C++ front-end MUST use:**

```c
#define YAMNET_INPUT_SCALE        4.20091972e-02f
#define YAMNET_INPUT_ZERO_POINT   31
#define YAMNET_OUTPUT_SCALE       3.90625000e-03f
#define YAMNET_OUTPUT_ZERO_POINT  -128
// quantized = round(real / SCALE) + ZERO_POINT
```

Getting these wrong produces silently wrong predictions with no error.

---

## 6. Final Phase 1 results (measured)

Held-out test set, 95 clips / 475 windows:

| Model | Level | Acc | AUC | Fall P | Fall R | Fall F1 |
| ----- | ----- | --- | --- | ------ | ------ | ------- |
| float32 | window | 0.804 | 0.865 | 0.808 | 0.804 | 0.806 |
| INT8 | window | 0.783 | 0.855 | 0.764 | 0.825 | 0.794 |
| float32 | **clip** | **0.916** | 0.955 | 0.857 | **1.000** | 0.923 |
| INT8 | **clip** | **0.916** | 0.942 | 0.870 | 0.979 | 0.922 |

**Quantization cost:** clip accuracy unchanged (0.916), but **recall fell 1.000 →
0.979 — one fall missed**. Per-window parity drift is non-trivial: max |Δ| 0.43,
mean 0.071.

### Threshold sweep (INT8, clip level, test set) — actionable

| Threshold | Acc | Precision | Recall | F1 |
| --------- | --- | --------- | ------ | -- |
| 0.40 | 0.895 | 0.828 | **1.000** | 0.906 |
| **0.45** | **0.916** | **0.857** | **1.000** | **0.923** |
| 0.50 | 0.916 | 0.870 | 0.979 | 0.922 |
| 0.60 | 0.905 | 0.898 | 0.917 | 0.907 |

**Recommendation: use threshold 0.45, not 0.50.** It restores **recall 1.000** on
the INT8 model at identical accuracy and the best F1 in the sweep — i.e. it
recovers the fall that quantization lost, for free. For a safety system, missing a
fall is far worse than a false alarm.

---

## 7. Known limitations — read before trusting any number above

**1. The confuser gap — the most important open risk.** YAMNet's own AudioSet head
was run over all 950 clips to inventory them semantically. Result, clips scoring
>0.2 per family (out of 475 per class):

| Confuser | In non-fall | In fall |
| -------- | ----------- | ------- |
| Toilet flush | **0** | 0 |
| Hand/hair dryer | **0** | 0 |
| Footsteps | **0** | 0 |
| **Impact / thud / bang** | **0** | 7 |
| Door / slam | 2 | 16 |

**Zero** non-fall clips contain an impulsive impact; every impulsive event in the
corpus is labelled fall. The model's easiest available rule is
*"impulsive transient ⇒ fall"* — which scores ~99 % here and fires on every door
slam in a real restroom. **The 0.857 precision figure is measured against a test
set containing no confusers and is therefore not a real-world number.**

**2. Domain gap has two axes, only one is addressed.** RIR convolution fixes the
*room*. It does not fix the *floor*: SAFE falls land on carpet/wood/concrete, never
tile.

**3. Speech dominance.** ~52 % of clips in *both* classes are Speech-dominant per
YAMNet (people talking during recording). Roughly symmetric, so not a class
shortcut, but over half the acoustic content is irrelevant to the task.

**4. Synthetic-RIR self-reference.** Train and test both use RIRs from the same
generator, so evaluation shares its quirks. Real measured RIRs are now available
(see Phase 2) but the Phase 1 model was **not** trained with them.

**5. Frozen backbone.** No fine-tuning was attempted. Unfreezing the last 2–3 blocks
is an untried lever, ranked below the data problem.

---

## 8. Phase 2 — scaffolding exists, not yet run

Goal: replace the 2-class fall/non-fall taxonomy with **emergency / normal**, and
fix the confuser gap using external corpora. Config in `src/config_phase2.py`.

### Scripts written (not yet executed end to end)

| Script | Purpose |
| ------ | ------- |
| `6b_download_mit_ir.py` | MIT IR Survey bathroom RIRs |
| `6c_download_musan_desed.py` | MUSAN noise subset for SNR augmentation |
| `6d_download_freesound.py` | Freesound API, **CC0/CC-BY only** (commercial-safe) |
| `6e_audit_fsd50k.py` | Filter FSD50K to CC0/CC-BY clips → `commercial_clips.csv` |
| `7_preprocess_external.py` | External audio → 16 kHz → RIR + MUSAN noise variants → windows |
| `8_merge_datasets.py` | *(referenced by Step 7, not yet present)* |

Planned Phase 2 training (from `config_phase2.py`): Dense(256) head, dropout
0.4/0.3, **two-stage** — `P2_HEAD_EPOCHS = 40` frozen, then
`P2_FINETUNE_EPOCHS = 25` unfreezing **YAMNet layers 13–14** at `lr = 1e-4`.
So Phase 2 *does* intend fine-tuning, unlike Phase 1.

### Data actually on disk

| Dataset | Files | Size | State |
| ------- | ----- | ---- | ----- |
| MIT IR Survey | 282 | 29 MB | ✅ downloaded, incl. **8 bathroom RIRs at 16 kHz** in `data/external/mit_ir/bathroom_irs/` |
| MUSAN | 1 | 271 MB | 🟡 archive only, not extracted |
| OpenSLR-28 | 1 | 513 MB | 🟡 archive only, not extracted |
| FSD50K | 0 | — | ❌ dirs exist, empty (~26 GB download) |
| DESED, Freesound | 0 | — | ❌ empty |
| `processed_p2/`, `combined/` | 0 | — | ❌ empty — Step 7 never run |

The 8 real bathroom RIRs are immediately usable with the **existing** Phase 1
pipeline — no Phase 2 code needed:

```powershell
C:\aesv\Scripts\python.exe src\2_preprocess.py `
  --rir-source measured --rir-dir data\external\mit_ir\bathroom_irs
```

---

## 9. Shareable PC demo — built and verified

`share/fall-detector-demo.zip` (14 MB): `try_fall_detector.py` (self-contained,
zero project imports), `yamnet_fall_detector.keras`, six held-out sample clips
with the canonical RIR applied, and a README.

Recipient needs `pip install tensorflow soundfile scipy`. Verified from a clean
extraction with no project modules importable: **6/6 correct**.

Formats: wav/flac/ogg/aiff/mp3 direct; **m4a/mp4/aac via ffmpeg fallback**. Flag is
`--audio` (`--wav` kept as alias). Lossy costs a little: same clip scored 0.653 as
WAV, 0.643 as AAC.

Local testing tool: `src/predict.py` with `--test-set N`, `--audio FILE`,
`--mic`, `--simulate-bathroom`.

---

## 10. Open decisions

1. **Threshold 0.45 vs 0.50** — sweep says 0.45 restores INT8 recall to 1.000 at
   equal accuracy. Recommended but not yet baked into deployment notes.
2. **Mel floor 125 Hz vs the brief's 50 Hz** — kept at 125 Hz; needs sign-off since
   it binds the C++ front-end.
3. **Model size** — INT8 is **3.67 MB** against the brief's ~1.2 MB target. Fits
   8 MB PSRAM comfortably. If 1.2 MB is hard, that needs a width-multiplier variant
   or pruning, i.e. a retrain.
4. **"Model in PSRAM"** — the brief says this, but a `const` byte array lands in
   **flash** (memory-mapped, costs zero PSRAM). What belongs in PSRAM is the
   **tensor arena**. Arena size has not yet been measured.
5. **Hard negatives** — hook exists, still off. Decides whether the device survives
   a real restroom.
6. **Split strategy** — currently `fold` (95-clip test). `ratio` gives 143 clips
   for more trustworthy final numbers.

---

## 11. Immediate next steps, in priority order

1. **Close the confuser gap.** Either run the Phase 2 external-data path, or take
   the cheap route: extract MUSAN and use the existing
   `2_preprocess.py --negatives-dir`. Everything downstream is a re-run.
2. **Retrain with the 8 real bathroom RIRs** (hybrid), removing the synthetic
   self-reference in evaluation.
3. **Gather colleague false-alarm data** with the PC bundle — ask them to try to
   *break* it (claps, door slams, dropped keys). That is the missing empirical
   input on real-world precision.
4. **ESP32 firmware:** C++ log-mel front-end matching `build_log_mel()`, TFLM
   integration, `MicroMutableOpResolver` registrations, tensor arena sizing. Not
   started. Deliberately deferred until model quality is settled, since firmware is
   where time gets sunk.

---

## 12. Working conventions that proved valuable

- **Verify, don't assume.** The schema correction, the three RIR bugs, the
  rebuild-vs-TF-Hub check and the label-noise diagnosis were all caught by scripts
  that measure their own output. Keep that habit.
- **Every script self-verifies and exits non-zero on problems.**
- **Report clip-level metrics**, and say so explicitly — window-level numbers look
  worse and mean something different.
- Be explicit that the corpus is clean-recorded, dummy-dropped, and confuser-free
  whenever quoting accuracy. Overstating a 91.6 % prototype in a health-centre
  context is the failure mode to avoid.

---

## 13. Data Archive Status — Action Required

Discovered 2026-08-31 when attempting to run the confuser-gap fix.

### MUSAN (`data/external/musan/musan.tar.gz`, 270 MB)

**Status: TRUNCATED.** The archive yields only 41 of ~1000+ expected members
(stops partway through `musan/music/fma/`, never reaches `musan/noise/`).

```
EOFError: Compressed file ended before the end-of-stream marker was reached
```

**Action:** Re-download fresh from [https://www.openslr.org/17/](https://www.openslr.org/17/).
Place the new file at `data/external/musan/musan.tar.gz` (overwrite).
Only the `noise/` subdirectory is needed; the download script filters to it.

### OpenSLR-28 (`data/external/openslr28/rirs_noises.zip`, 512 MB)

**Status: CORRUPTED.** The file has valid ZIP magic bytes but `zipfile`,
`Expand-Archive` (PowerShell), and `tarfile` all refuse it. The end-of-central-directory
signature appears at offset 105 MB rather than near end-of-file, suggesting
a failed/partial download that padded to apparent size.

**Action:** Re-download fresh from [https://www.openslr.org/28/](https://www.openslr.org/28/).
Place the new file at `data/external/openslr28/rirs_noises.zip` (overwrite).

### What unblocks after re-download

Once either MUSAN noise or OpenSLR-28 is re-downloaded and extracted,
the confuser-gap fix is a single command:

```powershell
# Using MUSAN noise (recommended - smaller, noise-only relevant subset)
C:\aesv\Scripts\python.exe src\2_preprocess.py `
  --rir-source measured `
  --rir-dir data\external\mit_ir\bathroom_irs `
  --negatives-dir data\external\musan\musan\noise `
  --negative-snr 0 15

# Then re-run Steps 3 → 4 → 5 as normal
```

This will inject impulsive non-fall sounds (noise recordings) into the non-fall
class during preprocessing, closing the rule `"transient => fall"` that currently
gives the model its spuriously high corpus precision.

### v2 Model (Real RIRs, No Confuser Fix)

Completed 2026-08-31. Steps 2–5 re-run with the 8 MIT bathroom RIRs.
This removes synthetic self-reference from evaluation. Outputs:

```
models/yamnet_fall_detector.keras          float32 v2 model (real-RIR trained)
models/yamnet_fall_detector_int8.tflite   INT8 v2 model
share/fall-detector-tflite-demo.zip       lightweight demo (no TF needed)
```

Results compared to v1 (synthetic RIRs) are in `outputs/tflite_metrics.json`
(written after Step 5 completes). The confuser gap remains open until
MUSAN/OpenSLR-28 archives are re-downloaded.

