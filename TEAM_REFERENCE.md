# Acoustic Fall Detection — Team Reference Guide

> **Audience:** Engineers, ML practitioners, and clinical stakeholders joining this project.  
> **Purpose:** Give every new contributor the full honest picture — what works, what doesn't,
> what the hard problems are, and where the effort should go next.  
> **Last updated:** 2026-09-01

---

## 1. What This Project Is Trying to Do

Build a **small, self-contained device** that detects when a person falls in a health-centre
restroom and raises an alert — without any camera (privacy-preserving), without any
wearable on the patient, and with enough reliability to actually be trusted in a clinical setting.

**Target hardware:** ESP32-S3-BOX-3 (Espressif)
- Dual-core 240 MHz, 512 KB SRAM, 16 MB flash
- On-board microphone
- Runs TensorFlow Lite Micro (TFLM)
- Consumes ~1W — suitable for permanent installation

**Target environment:** Single-occupancy patient/public restrooms in health centres.  
**Target users:** Elderly or clinically vulnerable patients who may fall while unattended.

The device must:
- Detect a fall impact sound within a few seconds of it happening
- Raise an alert (buzzer, network message, or nurse-call signal)
- Not produce so many false alarms that staff stop trusting it
- Process audio entirely on-device (no audio leaves the device — privacy requirement)

---

## 2. The ML Approach Being Used

### Transfer learning on Google's YAMNet model

YAMNet is a pre-trained audio neural network from Google, trained on **AudioSet** —
2 million labelled YouTube clips covering 521 sound categories, including "Fall",
"Impact", "Crash", "Slam", and many related classes.

**Architecture:**
- **Input:** 96 × 64 log-mel patch (0.975s of audio at 16 kHz)
- **Backbone (frozen):** MobileNetV1, 3,217,344 parameters — never trained on our data
- **Head (trained by us):** Dense(128) → Dropout(0.4) → Dense(2, softmax), 131,458 parameters
- **Output:** P(non_fall), P(fall)

### Why this approach

The labelled fall dataset available (SAFE corpus) contains only 665 clips.
You **cannot** train a 3.2M-parameter audio CNN from scratch on 665 examples — it would
memorise the training set and generalise to nothing. Transfer learning solves this:
YAMNet already knows how to extract acoustically meaningful features from any sound;
we train only a tiny head to re-map those features to our 2-class problem.

This is a **legitimate, professional approach — not a shortcut.** It has hard limits
(described in §5), but it is the correct choice given the dataset size.

### DSP contract (binding — do not change without full retraining)

Every component — Python training, PC demo, ESP32 firmware — must use exactly these
parameters or inference will produce wrong results:

| Parameter | Value |
|-----------|-------|
| Sample rate | 16,000 Hz mono |
| Window | 15,600 samples (0.975 s) |
| Hop | 7,800 samples (50% overlap) |
| STFT frame | 400 samples, 160-sample hop, 512-point FFT, periodic Hann |
| Mel bins | 64, range 125–7500 Hz |
| Log-mel | log(mel + 0.001) |

---

## 3. Current Model Performance — Honest Numbers

Two model versions have been trained and evaluated.

### How to read the metrics

- **Clip-level** = the correct metric. A 3s fall clip is sliced into ~5 windows; we
  average window probabilities across a clip and threshold at 0.50. This is how the
  device will actually behave.
- **Window-level** numbers look worse — don't quote them in isolation.
- **"Corpus accuracy"** means accuracy on the SAFE test corpus, which has a known
  structural flaw described in §4. These are **NOT** real-world deployment numbers.

### v1 Model — Synthetic room acoustics (trained 2026-08)

| Split | Accuracy | Recall | Precision | F1 |
|-------|----------|--------|-----------|-----|
| Test clip (INT8, threshold 0.50) | 91.6% | 97.9% | 87.0% | 92.3% |
| Val clip (INT8, threshold 0.50) | 92.1% | 100% | 86.5% | 92.8% |

### v2 Model — Real MIT bathroom RIRs (trained 2026-08-31) ← current

| Split | Accuracy | Recall | Precision | F1 |
|-------|----------|--------|-----------|-----|
| Test clip (INT8, threshold 0.50) | **91.6%** | 91.7% | **91.7%** | 91.7% |
| Val clip (INT8, threshold 0.50) | 91.1% | 94.8% | **88.4%** | 91.8% |

v2 is the current production model. Accuracy is identical; precision improved ~5 pp
because evaluation is on genuinely unseen acoustic conditions (real measured bathroom
impulse responses, not synthetic ones from the same generator used in training).

### v2 INT8 quantisation parameters (must match in ESP32 firmware)

```c
#define YAMNET_INPUT_SCALE       4.22173813e-02f
#define YAMNET_INPUT_ZERO_POINT  28
#define YAMNET_OUTPUT_SCALE      3.90625000e-03f
#define YAMNET_OUTPUT_ZERO_POINT -128
```

### Recommended inference threshold: 0.50

At 0.50: F1 = 0.917. If you prefer recall over precision (fewer missed falls):
use 0.45 — recall 0.958, precision 0.868, F1 0.911.

---

## 4. The Critical Problem: The Confuser Gap

> **This is the most important section. Read it before drawing any conclusions
> from the accuracy numbers.**

### What the problem is

The SAFE corpus has a fundamental structural flaw:

- Every **non-fall** clip contains only quiet, ambient, non-impulsive sounds
  (room noise, walking, calm voice, running water).
- Every **impulsive sound** in the dataset (thuds, impacts, bangs) is labelled **fall**.

This means the model never sees a training example of:
> *"here is a loud impact transient → this is NOT a fall"*

The model therefore learned the rule: **`any sharp loud transient → FALL`**

This rule scores 91.6% on a test set that has exactly the same flaw. But in a real
restroom, this rule fires on every:
- Toilet seat dropped
- Cistern lid placed down
- Bathroom door slammed
- Cabinet door closed hard
- Object dropped on tile (bottle, bin, keys)
- Flush valve impact

**In a real deployment, the false alarm rate could be multiple per hour.** The 91.6%
corpus precision is not a real-world number. Without confuser training data, real-world
precision is likely in the **50–70% range** — meaning 30–50% of alerts would be false.

### Why this is a data problem, not a model problem

No architecture — however sophisticated — can learn to distinguish a fall impact from
a toilet lid slam if its training data labels all impacts as falls and all non-falls
as quiet sounds. The model is doing exactly what the data instructs.

**The fix is not a better architecture. It is labelled examples of impulsive
non-fall sounds in the training set.**

---

## 5. Architectural Limitations and Improvements

The frozen YAMNet head approach is correct for the dataset size, but has a ceiling:

### Current limitations

**1. The embedding space wasn't designed for this boundary.**
YAMNet's 1024-d embedding separates 521 AudioSet categories. The boundary between
"fall" and "non-fall" in that space is incidental, not engineered for our problem.

**2. Temporal dynamics are collapsed to one vector.**
A fall has a specific time structure: `silence → impact (10–30ms) → settling (100–500ms)`.
YAMNet converts a 975ms window into one 1024-d vector — the temporal arc is lost.
A door slam has a similar vector. A temporal model (LSTM over consecutive windows)
would see the difference.

**3. Backbone cannot adapt to bathroom acoustics.**
Hard tiles create specific reverb. With a frozen backbone, the feature extractor
cannot shift toward bathroom-specific spectral signatures.

### Improvements in priority order

| Improvement | Expected gain | Difficulty |
|-------------|--------------|------------|
| Add confuser data (fix the gap) | **High — most impactful** | Medium |
| Temporal LSTM head (sequence of embeddings) | Medium | Medium |
| Partial fine-tuning of YAMNet layers 13–14 | Medium | Medium (code exists in `config_phase2.py`) |
| Full Phase 2 pipeline (emergency/normal taxonomy + external data) | High | High |

---

## 6. Dataset Status

### What is ready to use

| Dataset | Location | Status |
|---------|----------|--------|
| SAFE corpus (950 clips) | `data/raw/` | ✅ Ready |
| MIT IR Survey (8 bathroom RIRs) | `data/external/mit_ir/bathroom_irs/` | ✅ Ready |

### What is broken and needs re-download

| Dataset | Problem | Fix |
|---------|---------|-----|
| MUSAN (`musan.tar.gz`, 270 MB) | **TRUNCATED** — stops at 41 members, noise/ never reached | Re-download: https://www.openslr.org/17/ |
| OpenSLR-28 (`rirs_noises.zip`, 512 MB) | **CORRUPTED** — ZIP header valid but unreadable by any tool | Re-download: https://www.openslr.org/28/ |

### What was never downloaded

| Dataset | Relevance to this problem | Source | Size |
|---------|--------------------------|--------|------|
| **ESC-50** | 2000 clips, 50 classes. Includes door knock, footsteps, glass breaking, toilet flush — directly the confusers we need. Small and immediately useful. | https://github.com/karoldvl/ESC-50 | ~600 MB |
| **FSD50K** | 51K clips, 200 AudioSet classes including door events, impacts, bathroom sounds. CC-licensed. Large but comprehensive. | https://zenodo.org/record/4060432 | ~30 GB |

### Honest assessment of MUSAN

MUSAN's noise subset is often cited as a useful hard-negative source, but for our
specific problem it is **less targeted than commonly assumed.** It contains HVAC,
traffic, office ambient noise, and field recordings — not toilet-lid drops or door
slams. It improves robustness against sustained background noise but will not solve
the impulsive-confuser problem.

**ESC-50 is more directly useful** and far smaller. Self-recorded restroom confusers
(see §7a) are the most targeted option of all.

---

## 7. Data Collection Strategy

### The strategic question: collect first, or deploy first?

**Option A — Collect confuser audio now, then deploy:**
Improves the model before it goes live. Reduces initial false alarm rate. Safer start.

**Option B — Deploy now, log real audio, label it, retrain:**
The "data flywheel" approach used by all major voice assistant products. Real
deployment surfaces confusers you never thought to record in a lab. But the initial
model may have such a high false alarm rate that it becomes unacceptable and gets
switched off before it can collect useful data.

### Recommended: Hybrid approach

```
Stage 1:  Record restroom confuser audio yourself (see §7a) — a morning of effort
          Retrain v3 model. Deploy v3.

Stage 2:  Firmware logs audio from every alert + random negative samples.
          Human labels the saved clips.
          Retrain v4 with the new real-world data.

Stage 3:  Controlled fall simulation in actual deployment rooms (see §7b).
          These are the highest-value positive examples possible.

Ongoing:  Label → retrain cycle every few weeks.
```

### 7a. Recording confuser audio yourself

Use any smartphone recorder. Any bathroom with hard surfaces works.
Label everything `non_fall`. This is the highest-impact thing you can do right now.

| Sound to record | Priority | Notes |
|----------------|----------|-------|
| Toilet seat dropped from full height | 🔴 Critical | Most common restroom transient |
| Toilet cistern lid placed down hard | 🔴 Critical | Heavy ceramic — very loud impact |
| Bathroom door slammed (3 force levels) | 🔴 Critical | Acoustically very similar to body impact |
| Cabinet/cupboard door slammed | 🔴 Critical | Sharp transient on rigid surface |
| Plastic bottle / container dropped on tile | 🔴 Critical | Common accident sound |
| Metal bin lid dropped on tile | 🔴 Critical | High-energy impulsive |
| Toilet flushing (multiple types) | 🟡 Important | Valve impact + flow |
| Footsteps on tile at various speeds | 🟡 Important | Rhythmic transients |
| Object slid across tile floor | 🟡 Important | Scraping confuser |
| Tap/shower water running | 🟡 Important | Background masking |

**Recording protocol:**
- Each sound at **3 distances**: ~0.5m, ~1m, ~2m from the microphone
- **10–15 repetitions** per sound per distance
- Try different rooms (different acoustics = more diversity)
- Any audio format works; pipeline resamples to 16 kHz mono

One morning of recording yields ~500–800 clips. That is enough to meaningfully
improve the model. You do **not** need to record falls — the SAFE corpus has 480 fall clips.
The gap is entirely on the non-fall side.

### 7b. Getting better positive (fall) examples

Real falls in deployment are rare and unpredictable. The solution is controlled
fall simulation **in your actual deployment rooms** — not in a lab:

- A healthy volunteer (or weighted bag/mannequin) falling onto a crash mat
- 20–30 falls from different angles and positions
- Recorded in the actual restroom

These are worth far more per clip than SAFE corpus clips, because they capture your
specific room acoustics.

### 7c. Privacy and consent requirements

Audio recording in healthcare restrooms requires at minimum:
- Explicit notice signage for users
- Data handling and retention policy
- Potentially ethics committee review (jurisdiction-dependent)
- GDPR / HIPAA equivalent compliance

The on-device inference architecture addresses the core concern: **raw audio never
leaves the ESP32.** Only post-trigger clips (with explicit consent framework in place)
or anonymised prediction logs are transmitted.

---

## 8. The Active Learning Deployment Loop

Once a model is deployed, design the firmware to support:

1. **Circular audio buffer** — always hold last 10 seconds in RAM
2. **On-trigger clip save** — when model fires an alert, write the preceding 10s to
   SD card or transmit to server
3. **Random negative sampling** — 1-in-200 non-triggering windows: also save a clip
   (gives you negative examples of real in-situ sounds you never anticipated)
4. **Raw score logging** — log the float32 probability per window, not just binary;
   enables offline threshold analysis

A human then labels saved clips: `fall`, `non_fall_confuser`, `non_fall_background`.
Feed into the next retrain.

This loop is how you move from 91% corpus accuracy to a system trusted in clinical use.
There is no shortcut around real-world data.

---

## 9. Pipeline and Files

```
acoustic-alert-system/
├── src/
│   ├── 1_dataset_audit.py        Validates SAFE corpus structure
│   ├── 2_preprocess.py           Audio → 16kHz windows, RIR augmentation, hard negatives
│   ├── 3_prepare_splits.py       Leakage-free train/val/test split (fold-based)
│   ├── 4_train.py                YAMNet transfer learning (embeddings auto-cached + hash-guarded)
│   ├── 5_export_tflite.py        INT8 PTQ, C header export, threshold sweep
│   ├── config.py                 All DSP constants (single source of truth)
│   ├── config_phase2.py          Phase 2 taxonomy, external data, fine-tuning config
│   └── predict.py                Single-file inference for testing
│
├── data/
│   ├── raw/                      950 SAFE corpus WAVs at 48kHz
│   ├── processed/                Step 2 output: chunks, embeddings, hash sidecar
│   ├── splits/                   Step 3 output: train/val/test index files
│   └── external/
│       ├── mit_ir/bathroom_irs/  8 real bathroom RIRs at 16kHz ✅
│       ├── musan/musan.tar.gz    ❌ Truncated — re-download
│       └── openslr28/rirs_noises.zip  ❌ Corrupted — re-download
│
├── models/
│   ├── yamnet_fall_detector.keras        Current float32 model
│   ├── yamnet_fall_detector_int8.tflite  INT8 model for deployment (3.67 MB)
│   ├── yamnet_model_data.h               C array header for ESP32
│   └── yamnet_model_data.cc              C source (23 MB)
│
├── outputs/
│   ├── tflite_metrics.json       Full threshold sweep results
│   └── figures/                  Training curves, RIR pool plots, eval figures
│
├── PROJECT_CONTEXT.md            Full technical history and decisions log
└── TEAM_REFERENCE.md             This file (team onboarding guide)
```

### Run the pipeline

```powershell
# Working directory: acoustic-alert-system/
# IMPORTANT: Use C:\aesv\Scripts\python.exe — NOT system Python 3.14

python src\1_dataset_audit.py
python src\2_preprocess.py --rir-source measured --rir-dir data\external\mit_ir\bathroom_irs
python src\3_prepare_splits.py
python src\4_train.py
python src\5_export_tflite.py
```

### Add confuser data (once recorded)

```powershell
python src\2_preprocess.py `
  --rir-source measured `
  --rir-dir data\external\mit_ir\bathroom_irs `
  --negatives-dir path\to\your\confuser\recordings `
  --negative-snr 0 15

# Then re-run Steps 3 → 4 → 5 as normal
# The embeddings cache is auto-invalidated when chunks change (no manual deletion needed)
```

---

## 10. Immediate Action Checklist

- [ ] **Record restroom confuser audio** (§7a) — any bathroom, any smartphone, ~2–3 hours
- [ ] **Re-download MUSAN** — https://www.openslr.org/17/ → replace `data/external/musan/musan.tar.gz`
- [ ] **Download ESC-50** — https://github.com/karoldvl/ESC-50 → extract to `data/external/esc50/`
- [ ] **Retrain v3** with confuser data via `2_preprocess.py --negatives-dir` → Steps 3–5
- [ ] **Controlled fall simulation** in actual deployment room (weighted bag + crash mat, 20–30 falls)
- [ ] **ESP32 firmware**: C++ audio capture → 16kHz DSP → TFLM inference → alert output
- [ ] **Deploy firmware with audio logging** (consent framework in place first)
- [ ] **Active learning cycle**: label saved clips → retrain every 2–4 weeks

---

## 11. Key Lessons Learned

1. **Corpus accuracy ≠ real-world accuracy.** 91.6% on SAFE corpus does not mean
   91.6% in a real restroom. The corpus has no confusers. Always state this caveat.

2. **The confuser gap is a data problem.** No architecture fixes a training set where
   every impact is labelled "fall". Fix the data first.

3. **When Step 2 is re-run, embeddings auto-invalidate.** `4_train.py` uses a SHA-256
   hash sidecar to detect when `chunks_int16.npy` has changed. A silent version of this
   bug caused the first v2 training run to train on synthetic-RIR embeddings while
   evaluating on real-RIR audio — producing a false apparent regression.

4. **Real RIRs give more honest evaluation.** v1 and v2 have nearly identical accuracy,
   but v2 numbers are more trustworthy: the test set used genuinely unseen acoustic
   measurements rather than a different seed of the same synthetic generator.

5. **MUSAN is less targeted than commonly cited.** Its noise subset contains HVAC,
   traffic, and office sounds — not toilet-lid drops or door slams. ESC-50 and
   self-recorded restroom audio are more directly useful for this problem.

6. **The data flywheel beats lab collection for the negative class.** Real deployment
   surfaces confusers you never anticipated. Build audio logging into the firmware
   from day one.

7. **On-device inference is the right privacy architecture.** Raw audio never leaves
   the ESP32. This is essential for healthcare restroom deployment.
