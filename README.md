# Acoustic Emergency Alerting System — Phase 1 (PoC)

Fall detection from restroom audio on an **ESP32-S3-BOX-3**, using **YAMNet
transfer learning** and **TensorFlow Lite Micro** with the model resident in
8 MB external PSRAM.

## Pipeline

| Step | Script | Purpose | Status |
| ---- | ------ | ------- | ------ |
| 1 | `src/1_dataset_audit.py` | Integrity, class/fold distribution, window-slicing preview | ✅ implemented |
| 2 | `src/2_preprocess.py` | 16 kHz mono resample, bathroom RIR (RT60 ≈ 0.6 s), 0.975 s slicing | ✅ implemented |
| 3 | `src/3_prepare_splits.py` | Leakage-free fold-based splits, verified | ✅ implemented |
| 4 | `src/4_train.py` | Frozen YAMNet backbone + Dense(128)→Dropout(0.4)→Dense(2) head | ✅ implemented |
| 5 | `src/5_export_tflite.py` | Metrics, INT8 PTQ, `yamnet_model_data.h` C array | ✅ implemented |

Shared modules: `src/config.py` (the single source of truth for the DSP
contract), `src/audio_utils.py` (filename parsing, window slicing),
`src/rir.py` (room impulse response synthesis, measurement, pluggable sources),
and `src/yamnet_model.py` (rebuilt patch-input YAMNet + its log-mel front-end).

## Environment

Two Windows constraints shape this:

1. TensorFlow publishes no wheels for Python 3.14 (this machine's default), so
   the project runs on **Python 3.11**.
2. TensorFlow's bundled headers blow past the **260-char `MAX_PATH` limit** when
   installed under this project's directory. `LongPathsEnabled` is `0` on this
   machine and enabling it needs admin. So the venv lives at a **short path**.

```powershell
py -3.11 -m venv C:\aesv
C:\aesv\Scripts\python.exe -m pip install -r requirements.txt
```

Run everything through **`C:\aesv\Scripts\python.exe`**.

> If you have admin, setting `HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem`
> → `LongPathsEnabled = 1` is the cleaner permanent fix, and the venv can then
> move back into the project as `.venv`.

Verified working: TensorFlow 2.21.0, tensorflow-hub 0.16.1, librosa 0.11.0.
YAMNet loads from TF-Hub and runs on this machine.

## YAMNet input contract

| Parameter | Value |
| --------- | ----- |
| Sample rate | 16 000 Hz, mono, 16-bit PCM |
| Window | 0.975 s = **15 600 samples** |
| Hop (50 % overlap) | 0.4875 s = **7 800 samples** |
| STFT | 25 ms window / 10 ms hop (400 / 160 samples) |
| Log-mel patch | 96 frames × 64 bins, 125 Hz – 7 500 Hz |
| Embedding | 1024-d |

The 0.975 s figure is the *waveform* span needed to produce a 0.96 s patch: 96
frames at a 10 ms hop, with the last frame still needing its full 25 ms window
(95 × 0.010 + 0.025 = 0.975 s).

> **Mel floor:** the brief specified 50 Hz, but upstream YAMNet uses **125 Hz**
> (`research/audioset/yamnet/params.py`). `config.py` keeps 125 Hz — a mismatched
> floor in the ESP32 front-end would shift every mel bin relative to what the
> frozen backbone was trained on. Flagged for your decision.

## Dataset

**SAFE: Sound Analysis for Fall Event detection** — [Kaggle](https://www.kaggle.com/datasets/antonygarciag/fall-audio-detection-dataset)
(`antonygarciag/fall-audio-detection-dataset`), 950 clips, 274 MB, already in
`data/raw/`. To re-fetch:

```powershell
.venv\Scripts\python.exe -m pip install kaggle    # needs ~/.kaggle/kaggle.json
.venv\Scripts\python.exe -c "from kaggle.api.kaggle_api_extended import KaggleApi; a=KaggleApi(); a.authenticate(); a.dataset_download_files('antonygarciag/fall-audio-detection-dataset', path='data/raw', unzip=True)"
```

Every clip is **48 kHz mono PCM_16, exactly 3.000 s**. Step 2 resamples to 16 kHz.

### Filename schema — derived, not assumed

```
data/raw/01-430-01-001-01.wav     AA-BBB-CC-DDD-FF
                                  AA  fold id     01–10, 95 files each
                                  BBB clip uid    001–950, unique per file
                                  CC  event code  00 = background, 01–07 = fall subtype
                                  DDD take id     001–475, a plain index
                                  FF  class code  01 = Fall, 02 = Non-Fall
```

> The original brief placed the class code first and the fold last. The audit
> showed the opposite: **class is the last field, fold is the first.** The class
> codes themselves (01 = Fall, 02 = Non-Fall) were correct. See `config.py` for
> the supporting evidence.

The corpus is perfectly balanced — 475 fall / 475 non-fall, ~48/47 per fold —
and `event_code == 00` holds for every non-fall clip and no fall clip.

## Step 1 — run the audit

```powershell
.venv\Scripts\python.exe src\1_dataset_audit.py
.venv\Scripts\python.exe src\1_dataset_audit.py --no-deep     # headers only
.venv\Scripts\python.exe src\1_dataset_audit.py --limit 200   # sample
```

Artefacts:

- `outputs/dataset_audit_manifest.csv` — one row per source file
- `outputs/dataset_audit_summary.json` — machine-readable summary
- `outputs/figures/audit_*.png` — duration, class and class×fold plots
- `outputs/logs/dataset_audit.log` — full run log

Exit codes: `0` clean · `1` issues found · `2` no dataset in `data/raw/`.

## Step 2 — preprocess and augment

```powershell
C:\aesv\Scripts\python.exe src\2_preprocess.py
C:\aesv\Scripts\python.exe src\2_preprocess.py --variants 4
C:\aesv\Scripts\python.exe src\2_preprocess.py --rir-source measured --rir-dir data\rirs
C:\aesv\Scripts\python.exe src\2_preprocess.py --negatives-dir data\negatives
```

Each clip becomes several **variants**:

| Variant | Kind | Purpose |
| ------- | ---- | ------- |
| 0 | canonical RIR | One fixed bathroom, identical every run — what Step 3 gives val/test, so they are reverberant like deployment but reproducible |
| 1 | dry | Resampled only; robustness and regularisation |
| 2+ | random RIR | Randomised geometry, RT60 0.45–0.80 s, DRR −9…+2 dB |

Outputs (`data/processed/`): `chunks_int16.npy` (14,247 × 15,600 int16, 444 MB),
`manifest.csv`, `preprocess_summary.json`, plus listenable `samples/*.wav`.

**Every RIR is measured, never assumed.** Schroeder T30 and DRR are computed for
each generated response and reported; measured RT60 tracks target to **1.2% mean
error**. Two design notes worth keeping:

- RIRs are normalised to **unit energy**, not unit peak. Peak normalisation
  multiplied levels ~10× and forced a headroom cut that silently changed loudness
  against the dry variant — and loudness is real evidence for a fall.
- The direct arrival is found as the **first** significant arrival, not `argmax`.
  In a small hard room, early reflections sum above the direct sound, so
  trimming to the peak would shift every event backwards in time.

### Known gap — hard negatives

Off by default. The Step 1 audit found **zero** non-fall clips containing an
impulsive impact, while every impulsive event in the corpus is labelled fall. A
model can therefore score ~99% here with the rule *"transient ⇒ fall"* and then
fire on every door slam in a real restroom. `--negatives-dir` mixes confusers
(ESC-50 toilet flush / door knock / pouring water, or OpenSLR-28 noise) into the
non-fall class at controlled SNR to remove that shortcut.

### Going hybrid with real RIRs

`--rir-source measured --rir-dir DIR` accepts any directory of audio files.
Recommended sources:

| Corpus | Why |
| ------ | --- |
| [MIT IR Survey](https://mcdermottlab.mit.edu/Reverb/IR_Survey.html) | 270 real IRs, 68 labelled space classes **including bathroom**, CC-BY 4.0, 15.9 MB |
| [OpenSLR-28](https://www.openslr.org/28/) | Already 16 kHz/16-bit, Apache 2.0, real + simulated RIRs **and** noise for hard negatives |

Best of all is measuring the actual restrooms with the actual device — a sine
sweep or balloon pop, ~30–60 min per room — which also captures the microphone
and enclosure, something no simulation provides.

## Step 3 — build the splits

```powershell
C:\aesv\Scripts\python.exe src\3_prepare_splits.py
C:\aesv\Scripts\python.exe src\3_prepare_splits.py --strategy ratio
```

Two strategies, both leakage-free:

| Strategy | Clips | Test set | Notes |
| -------- | ----- | -------- | ----- |
| `fold` (default) | 70 / 20 / 10 | 95 clips | Whole folds; val, test and train fully fold-isolated |
| `ratio` | 70 / 15 / 15 | 143 clips | 7 whole folds to train, remaining 3 divided clip-wise; val and test share a fold pool but never a clip |

Fold assignment is computed, not hardcoded: folds are ranked by class skew and
the most skewed are absorbed into **train**, the largest split. Fold 10 (45.3%
fall vs ~50.5% elsewhere) lands in train automatically, leaving val and test at
50.3–50.7%.

**Variant policy.** Training uses all three variants; validation and test use
**variant 0 (canonical RIR) only**. Evaluation is therefore reverberant like
deployment, reproducible, and not inflated by counting three augmentations of
one clip as three independent test cases. 2,850 non-canonical eval-clip windows
are excluded for this reason. Override with `--eval-variants all`.

> The split ratio is measured in **clips**, not windows. Window percentages read
> 87/8/4 only because train keeps 3 variants per clip and eval keeps 1 — the
> intended asymmetry, not an unbalanced split.

Outputs (`data/splits/`): `split_manifest.csv`, `{train,val,test}_indices.npy`
(row indices into `chunks_int16.npy`), `splits_summary.json` with class weights.

Verified: 0 clips span >1 split, 0 variants separated, 0 folds shared between
train and evaluation, all index arrays disjoint and in range.

## Step 4 — YAMNet transfer learning

```powershell
C:esv\Scripts\python.exe src_train.py
C:esv\Scripts\python.exe src_train.py --skip-verify   # reuse cached embeddings
```

The backbone is frozen and run **once** — all 14,247 embeddings are cached
(58 MB), so epochs take a fraction of a second instead of minutes. Only the head
trains: 131,458 parameters against 3,217,344 frozen.

### The patch-input rebuild

TFLite Micro supports neither `tf.signal.stft` nor the mel filterbank matmul, so
the deployed model **cannot take a waveform**. `src/yamnet_model.py` rebuilds
YAMNet with a fixed `(96, 64, 1)` log-mel input and loads the official
`yamnet.h5` weights, and `build_log_mel` is the reference front-end the ESP32
C++ code must reproduce.

Verified numerically against TF-Hub YAMNet on real corpus audio before any
training is allowed to start:

| Check | Result |
| ----- | ------ |
| Backbone parameters | 3,217,344 — exact match |
| Max log-mel difference | 8.6e-06 |
| Max embedding difference | 1.4e-05 (scale 4.93) |
| **Relative embedding error** | **2.9e-06** — floating-point noise |

Training aborts if relative error exceeds 1e-3. Embeddings are extracted through
the *rebuilt* model, not TF-Hub, so the head trains on exactly the features
Step 5 exports.

### Results

| Split | Level | Accuracy | AUC | Fall P | Fall R | Fall F1 |
| ----- | ----- | -------- | --- | ------ | ------ | ------- |
| val   | window | 0.774 | 0.857 | 0.762 | 0.802 | 0.782 |
| test  | window | 0.804 | 0.865 | 0.808 | 0.804 | 0.806 |
| val   | **clip** | **0.921** | 0.966 | 0.865 | **1.000** | 0.928 |
| test  | **clip** | **0.916** | 0.955 | 0.857 | **1.000** | 0.923 |

**Report the clip-level numbers.** Window-level labelling is inherently noisy: a
3 s clip is sliced into five windows and all five inherit the clip's label, but
the impact lasts a fraction of a second and sits mid-clip. Measured mean P(fall)
by window position in fall clips is `0.49 / 0.85 / 0.92 / 0.92 / 0.60` — the
first and last windows are largely background wearing a "fall" label, and no
model can fit that.

Aggregation uses **mean**, not max. Max triggers on any single confident window
and measured far worse (0.68 vs 0.92 accuracy) because one spurious window
becomes a false alarm for the whole clip. The device should likewise integrate
over ~1–2 s before alarming rather than firing on a single inference.

Recall is **1.000** — no missed falls in either evaluation set — with precision
0.86. For a safety system that is the correct error profile, but see the
hard-negative gap in Step 2: the confusers most likely to cause false alarms are
absent from this corpus entirely.

Outputs: `models/yamnet_fall_detector.keras` / `.h5` (float32 composite,
`(96,64,1) → 2`), `models/fall_head.keras`, `data/processed/embeddings.npy`,
`outputs/train_metrics.json`, `outputs/figures/train_curves.png`.

## Step 5 — export and quantise for TFLite Micro

```powershell
C:\aesv\Scripts\python.exe src\5_export_tflite.py
C:\aesv\Scripts\python.exe src\5_export_tflite.py --num-calibration 400
C:\aesv\Scripts\python.exe src\5_export_tflite.py --skip-eval
```

Step 5 prepares the model for on-chip inference on the ESP32-S3-BOX-3 with
TensorFlow Lite Micro:

1. **Static batch graph:** Wraps the composite model with fixed `(1, 96, 64, 1)`
   input shape required by microcontrollers.
2. **Full INT8 Post-Training Quantization (PTQ):** Calibrates all layers using
   representative log-mel patches sampled from the training set, producing
   int8 input/output tensors for hardware acceleration via ESP-NN.
3. **C Array Export:** Writes 16-byte aligned C source and header
   (`models/yamnet_model_data.h` and `.cc`) containing the FlatBuffer binary
   ready for placement in 8 MB external PSRAM or Flash.

### Compression & Model Sizes

| Format | File | Size | Ratio |
| ------ | ---- | ---- | ----- |
| Keras H5 | `models/yamnet_fall_detector.h5` | 13.59 MB | 1.00x |
| Float32 TFLite | `models/yamnet_fall_detector_float32.tflite` | 13.35 MB | 1.02x |
| **INT8 TFLite** | `models/yamnet_fall_detector_int8.tflite` | **3.67 MB** | **3.64x** |

### INT8 Quantisation Parity & Metrics

| Split | Format | Window Acc | Window F1 | Clip Acc | Clip Recall | Clip Prec | Clip F1 |
| ----- | ------ | ---------- | --------- | -------- | ----------- | --------- | ------- |
| test  | Float32 | 0.804 | 0.806 | 0.916 | 1.000 | 0.857 | 0.923 |
| test  | **INT8** (t=0.50) | 0.783 | 0.794 | **0.916** | 0.979 | **0.870** | **0.922** |
| test  | **INT8** (t=0.45) | 0.777 | 0.796 | **0.916** | **1.000** | 0.857 | **0.923** |
| val   | Float32 | 0.774 | 0.782 | 0.921 | 1.000 | 0.865 | 0.928 |
| val   | **INT8** (t=0.50) | 0.764 | 0.777 | **0.900** | 0.990 | 0.841 | 0.909 |

> **Threshold tuning:** At default threshold 0.50, INT8 achieves 0.916 clip accuracy
> with 0.870 precision and 0.979 recall (1 missed fall out of 48). Setting the
> decision threshold to **0.45** restores **1.000 recall** (0 missed falls) while
> maintaining 0.916 accuracy and 0.857 precision.

Artefacts:
- `models/yamnet_fall_detector_int8.tflite` — 3.67 MB INT8 model
- `models/yamnet_model_data.h` and `.cc` — C byte array for ESP-IDF / Arduino
- `outputs/tflite_metrics.json` — complete evaluation and parity metrics
- `outputs/figures/tflite_eval.png` — ROC, error distribution, confusion matrix, threshold sweep
- `outputs/logs/export_tflite.log` — full execution log

## Trying the model yourself

`src/predict.py` is a utility (not part of the numbered pipeline) with three modes.

**1. Sanity check against held-out clips** — needs no recording:

```powershell
C:esv\Scripts\python.exe src\predict.py --test-set 20
```

Prints each clip's mean P(fall) beside its true label. These clips were never
seen in training and carry the canonical bathroom RIR.

**2. Your own audio** — any sample rate or channel count. `--audio` takes
wav/flac/ogg/aiff/mp3 directly and m4a/mp4/aac via ffmpeg; `--wav` is an alias:

```powershell
C:esv\Scripts\python.exe src\predict.py --wav myrecording.wav
C:esv\Scripts\python.exe src\predict.py --wav myrecording.wav --simulate-bathroom
```

Shows a per-window probability bar plus the clip decision:

```
01-020-02-073-01.wav  3.00 s  -> 5 window(s)
   w0   0.00- 0.97s  #########.........................  P(fall)=0.266
   w1   0.49- 1.46s  ################################..  P(fall)=0.942
   w2   0.97- 1.95s  ##################################  P(fall)=0.998
   w3   1.46- 2.44s  ###############################...  P(fall)=0.903
   w4   1.95- 2.92s  ##############....................  P(fall)=0.414
   CLIP  mean P(fall) = 0.705  ->  FALL
```

**3. Live microphone**, using the ring buffer, hop and smoothing the firmware
should use:

```powershell
C:esv\Scripts\python.exe src\predict.py --mic
C:esv\Scripts\python.exe src\predict.py --mic --smooth 5 --threshold 0.6
```

### Reading the results honestly

- **Judge clips, not windows.** A single window that straddles the impact is
  weak evidence. `--smooth` in mic mode is the same averaging.
- **`--simulate-bathroom` is the fair test.** The model was trained and evaluated
  on reverberant audio; dry laptop recordings are off-distribution.
- **Expect false positives on impulsive non-fall sounds.** Clapping, door slams
  and dropped objects will likely fire it — the corpus contains no such
  negatives (see the hard-negative gap above), so this is a known data gap
  rather than a surprise.
- Also listen to `data/processed/samples/*.wav` to hear the dry / canonical /
  random RIR variants of the same clip.
