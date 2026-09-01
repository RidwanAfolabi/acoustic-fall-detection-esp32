"""
Central configuration for the Acoustic Emergency Alerting System (Phase 1 PoC).

Every stage of the pipeline (audit -> preprocess -> split -> train -> export)
imports its constants from here so that the DSP contract used to build the
training set is bit-for-bit the contract the ESP32-S3 firmware must implement.

Target device : ESP32-S3-BOX-3, TensorFlow Lite Micro, 8 MB external PSRAM
Feature model : YAMNet (MobileNetV1 audio backbone, trained on AudioSet)
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
LOGS_DIR = OUTPUTS_DIR / "logs"

ALL_DIRS = (
    RAW_DIR,
    PROCESSED_DIR,
    SPLITS_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    FIGURES_DIR,
    LOGS_DIR,
)


def ensure_dirs() -> None:
    """Create every project directory that the pipeline writes into."""
    for directory in ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# YAMNet input contract
# ---------------------------------------------------------------------------
# YAMNet consumes a mono float32 waveform in [-1.0, 1.0] at 16 kHz and frames it
# internally into 96 x 64 log-mel patches.
#
# Why 15,600 samples for a "0.96 s" patch: the patch is 96 STFT frames at a
# 10 ms hop, and the final frame still needs its full 25 ms window. So the
# waveform span is 95 * 0.010 + 0.025 = 0.975 s -> 15,600 samples at 16 kHz.
# Feeding exactly this many samples yields exactly one embedding vector.

SAMPLE_RATE = 16_000
SAMPLE_WIDTH_BITS = 16  # int16 PCM on the wire / on the microphone
CHANNELS = 1

WINDOW_SECONDS = 0.975
WINDOW_SAMPLES = 15_600  # int(round(WINDOW_SECONDS * SAMPLE_RATE))

# 50 % overlap between consecutive training windows.
WINDOW_OVERLAP = 0.5
HOP_SAMPLES = 7_800  # int(round(WINDOW_SAMPLES * (1 - WINDOW_OVERLAP)))
HOP_SECONDS = HOP_SAMPLES / SAMPLE_RATE

# Log-mel front-end parameters. These are *not* used by the Python pipeline
# (TF-Hub YAMNet computes them internally from the raw waveform) but they are
# the exact numbers the ESP32-S3 C++ front-end must reproduce.
STFT_WINDOW_SECONDS = 0.025
STFT_HOP_SECONDS = 0.010
STFT_WINDOW_SAMPLES = 400  # 0.025 * 16000
STFT_HOP_SAMPLES = 160  # 0.010 * 16000

MEL_BANDS = 64
PATCH_FRAMES = 96
# NOTE: upstream YAMNet (research/audioset/yamnet/params.py) uses a 125 Hz mel
# floor, not 50 Hz. Keep the upstream value while YAMNet's own front-end is in
# use -- a mismatched floor on the device would shift every mel bin and silently
# degrade accuracy. Revisit only if the backbone's front-end is retrained.
MEL_MIN_HZ = 125.0
MEL_MAX_HZ = 7_500.0
LOG_OFFSET = 0.001

EMBEDDING_DIM = 1024  # YAMNet penultimate global-pooled feature vector

# ---------------------------------------------------------------------------
# SAFE dataset conventions
# ---------------------------------------------------------------------------
# Filename stem: AA-BBB-CC-DDD-FF
#
#   AA  -> fold id     10 folds (01-10), exactly 95 files each
#   BBB -> clip uid    contiguous 001-950, unique per file
#   CC  -> event code  00 = background/no event; 01-07 = fall subtypes
#   DDD -> take id     contiguous 001-475, a plain index
#   FF  -> class code  01 = Fall, 02 = Non-Fall
#
# This mapping was derived empirically from the 950-file corpus, NOT assumed:
# the class code is the LAST field, and the fold id is the FIRST. Evidence:
#   * FF has exactly 2 values split 475/475, and is perfectly determined by CC
#     (CC=00 <=> FF=02 for all 475 non-fall clips; CC in 01..07 <=> FF=01).
#   * AA has 10 values holding exactly 95 files each, ~48/47 per class -- the
#     balanced design of a 10-fold CV assignment.
#   * BBB is a bijection onto 1..950, so it identifies a file, not a session.
#   * DDD groups span multiple folds, which would imply cross-fold leakage if it
#     identified a source recording. It does not: comparing corpus-whitened
#     background mel profiles, within-DDD pairs are no more similar than random
#     pairs (Cohen's d = -0.04). DDD is therefore safe to ignore when splitting.

FILENAME_FIELDS = ("fold_id", "clip_uid", "event_code", "take_id", "class_code")
FILENAME_REGEX = r"^(\d+)-(\d+)-(\d+)-(\d+)-(\d+)$"

CLASS_CODE_TO_LABEL = {"01": "fall", "02": "non_fall"}

# Every non-fall clip carries event code 00; fall subtypes are 01-07. The audit
# cross-checks this invariant, since a violation would mean the label is wrong.
NON_FALL_EVENT_CODE = "00"
LABEL_TO_INDEX = {"fall": 1, "non_fall": 0}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}
NUM_CLASSES = len(LABEL_TO_INDEX)

AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg", ".mp3", ".m4a")

# Nominal SAFE clip length, used only to sanity-check the corpus.
EXPECTED_CLIP_SECONDS = 3.0
CLIP_DURATION_TOLERANCE = 0.25

# ---------------------------------------------------------------------------
# Quality thresholds used by the audit
# ---------------------------------------------------------------------------

CLIPPING_SAMPLE_THRESHOLD = 0.999  # |x| above this counts as a clipped sample
CLIPPING_RATIO_WARN = 0.001  # >0.1 % clipped samples -> flag the file
SILENCE_DBFS_THRESHOLD = -60.0  # whole-file RMS below this -> flag as silent
DC_OFFSET_WARN = 0.01  # |mean(x)| above this -> flag DC bias

# ---------------------------------------------------------------------------
# Split configuration (Step 3)
# ---------------------------------------------------------------------------

SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 1337
