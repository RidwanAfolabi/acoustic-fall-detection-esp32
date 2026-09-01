"""
Phase 2 configuration extensions for the Acoustic Emergency Alerting System.

Adds:
  - Binary emergency / normal taxonomy (replaces 2-class fall / non_fall)
  - External dataset paths (FSD50K, MUSAN, DESED, Freesound, MIT IR, OpenSLR-28)
  - FSD50K class-to-label mapping (which sound classes map to emergency vs normal)
  - Preprocessing parameters for external audio (augmentation, SNR ranges)
  - Combined dataset paths (merged SAFE + external)

Import this alongside config.py. Constants here override or extend config.py.
"""

from __future__ import annotations

from pathlib import Path

import config  # noqa: F401 — re-exported for one-stop import

# ---------------------------------------------------------------------------
# Phase 2 label taxonomy (binary: emergency / normal)
# ---------------------------------------------------------------------------
# Falls and vocal distress are merged into a single "emergency" class.
# This simplifies the classification boundary and the on-device threshold logic.

P2_LABEL_TO_INDEX: dict[str, int] = {
    "normal":    0,   # ambient background, bathroom sounds, calm speech
    "emergency": 1,   # physical fall impact + vocal distress (screams/groans/yells)
}
P2_INDEX_TO_LABEL: dict[int, str] = {v: k for k, v in P2_LABEL_TO_INDEX.items()}
P2_NUM_CLASSES = 2

# SAFE corpus class code → Phase 2 label
# fall (01) → emergency;  non_fall (02) → normal
SAFE_CLASS_TO_P2_LABEL: dict[str, str] = {
    "01": "emergency",  # all SAFE fall subtypes
    "02": "normal",
}

# ---------------------------------------------------------------------------
# FSD50K class-name → Phase 2 label mapping
# ---------------------------------------------------------------------------
# Source: FSD50K dev/eval vocabulary.csv (200 AudioSet ontology class names)
# All names are exactly as they appear in the FSD50K vocabulary CSV.
# Multi-labeled clips that mix emergency + normal classes are excluded.

FSD50K_EMERGENCY_CLASSES: frozenset[str] = frozenset({
    # Vocal distress
    "Screaming",
    "Scream",
    "Shout",
    "Shouting",
    "Yell",
    "Crying and sobbing",
    "Crying",
    "Sobbing",
    "Whimper",
    "Groan",
    "Grunt",
    "Gasp",
    "Pant",
    "Wheezing",
    "Breathing",            # heavy/distressed breathing
    # Physical impact (emergency)
    "Thud",
    "Slam",
    "Bang",
    "Crash",
    "Breaking",
    "Glass",
    "Fall",
    "Impact",
})

FSD50K_NORMAL_CLASSES: frozenset[str] = frozenset({
    # Bathroom sounds (primary confusers)
    "Toilet flush",
    "Toilet",
    "Water tap and faucet",
    "Faucet",
    "Sink (filling or washing)",
    "Bathtub (filling or washing)",
    "Bathtub",
    "Shower",
    "Drip",
    "Pour",
    "Running water",
    "Water",
    "Liquid",
    "Gargling",
    "Toilet paper",
    "Electric shaver and toothbrush",
    "Electric toothbrush",
    # Doors and impacts (non-fall)
    "Door",
    "Drawer open or close",
    "Knock",
    "Tap",
    "Click",
    # Footsteps and movement
    "Walk and footsteps",
    "Footsteps",
    # Background / ambient
    "Domestic sounds and home sounds",
    "Silence",
    "White noise",
    "Hum",
    "Fan",
    "Air conditioning",
    "Speech",
    "Whispering",
    "Inside, small room",
    "Inside, large room or hall",
    "Reverberation",
    "Echo",
})

# FSD50K clips whose labels contain BOTH emergency and normal classes
# will be excluded from training (ambiguous ground truth).

# ---------------------------------------------------------------------------
# Freesound API targeted queries
# ---------------------------------------------------------------------------
# Each entry: query string, licence filter (CC0 or CC-BY only), target label.
# The download script uses these to cap per-query clip counts.

FREESOUND_QUERIES: list[dict] = [
    # --- Emergency: vocal distress ---
    {"query": "scream pain fear",       "label": "emergency", "max_clips": 200},
    {"query": "groan pain moan",        "label": "emergency", "max_clips": 150},
    {"query": "yell shout alarm help",  "label": "emergency", "max_clips": 150},
    {"query": "cry sob distress human", "label": "emergency", "max_clips": 100},
    {"query": "gasp breathe heavy",     "label": "emergency", "max_clips":  80},
    # --- Emergency: physical impact ---
    {"query": "thud fall body hit",     "label": "emergency", "max_clips": 100},
    {"query": "crash bang impact hard", "label": "emergency", "max_clips": 100},
    # --- Normal: bathroom ---
    {"query": "toilet flush",           "label": "normal",    "max_clips": 200},
    {"query": "shower running water",   "label": "normal",    "max_clips": 150},
    {"query": "sink faucet water tap",  "label": "normal",    "max_clips": 150},
    {"query": "bathtub filling water",  "label": "normal",    "max_clips": 100},
    # --- Normal: impact confusers ---
    {"query": "door slam knock bang",   "label": "normal",    "max_clips": 150},
    {"query": "drop object hit floor",  "label": "normal",    "max_clips": 150},
    {"query": "drawer open close",      "label": "normal",    "max_clips":  80},
]

FREESOUND_LICENSE_FILTER = 'license:"Creative Commons 0" OR license:"Attribution"'
FREESOUND_FIELDS = "id,name,license,previews,duration,samplerate,channels,filesize"
FREESOUND_MAX_DURATION_SEC = 30.0   # skip clips longer than this
FREESOUND_MIN_DURATION_SEC = 0.5    # skip clips shorter than this

# ---------------------------------------------------------------------------
# Phase 2 project paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent

EXTERNAL_DIR = _ROOT / "data" / "external"

# External dataset roots
FSD50K_DIR      = EXTERNAL_DIR / "fsd50k"
FSD50K_DEV_DIR  = FSD50K_DIR / "dev"
FSD50K_EVAL_DIR = FSD50K_DIR / "eval"
FSD50K_MANIFEST = FSD50K_DIR / "commercial_clips.csv"   # filtered CC0+CC-BY

MUSAN_DIR       = EXTERNAL_DIR / "musan"
MUSAN_NOISE_DIR = MUSAN_DIR / "noise"

DESED_DIR       = EXTERNAL_DIR / "desed"
DESED_AUDIO_DIR = DESED_DIR / "audio"

FREESOUND_DIR   = EXTERNAL_DIR / "freesound"

OPENSLR28_DIR   = EXTERNAL_DIR / "openslr28"
MIT_IR_DIR      = EXTERNAL_DIR / "mit_ir"

# Processed output for Phase 2
P2_PROCESSED_DIR   = _ROOT / "data" / "processed_p2"
P2_CHUNKS_FILE     = P2_PROCESSED_DIR / "chunks_external.npy"   # (N, 15600) int16
P2_MANIFEST_FILE   = P2_PROCESSED_DIR / "manifest_external.csv"

# Combined (SAFE + external)
COMBINED_DIR         = _ROOT / "data" / "combined"
COMBINED_CHUNKS_FILE = COMBINED_DIR / "chunks_combined.npy"
COMBINED_MANIFEST    = COMBINED_DIR / "manifest_combined.csv"

# Phase 2 model outputs
P2_MODEL_DIR        = _ROOT / "models"
P2_KERAS_MODEL      = P2_MODEL_DIR / "yamnet_emergency_v2.keras"
P2_TFLITE_INT8      = P2_MODEL_DIR / "yamnet_emergency_v2_int8.tflite"

P2_ALL_DIRS = (
    EXTERNAL_DIR,
    FSD50K_DIR, FSD50K_DEV_DIR, FSD50K_EVAL_DIR,
    MUSAN_DIR, MUSAN_NOISE_DIR,
    DESED_DIR, DESED_AUDIO_DIR,
    FREESOUND_DIR,
    OPENSLR28_DIR,
    MIT_IR_DIR,
    P2_PROCESSED_DIR,
    COMBINED_DIR,
)


def ensure_p2_dirs() -> None:
    """Create every Phase 2 directory that the pipeline writes into."""
    for d in P2_ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Augmentation parameters for external audio preprocessing
# ---------------------------------------------------------------------------

# Signal-to-Noise Ratio range when mixing noise onto clean clips
NOISE_SNR_DB_RANGE = (-3.0, 15.0)

# Gain jitter applied to source clips before windowing
GAIN_JITTER_DB = 2.0

# Number of augmented variants to produce per external clip
# (vs 3 in Phase 1 which used: original, rir_synth, noise_mixed)
EXTERNAL_AUG_VARIANTS = 3   # original + rir_convolved + noise_mixed

# Fraction of external clips to convolve with bathroom RIRs
RIR_AUG_FRACTION = 0.70

# ---------------------------------------------------------------------------
# Training hyperparameters (Phase 2)
# ---------------------------------------------------------------------------

P2_HEAD_EPOCHS      = 40    # Phase C1: frozen backbone, head only
P2_FINETUNE_EPOCHS  = 25    # Phase C2: unfreeze YAMNet layers 13–14
P2_HEAD_LR          = 1e-3
P2_FINETUNE_LR      = 1e-4  # 10× lower to protect pretrained weights
P2_BATCH_SIZE       = 64
P2_DROPOUT_1        = 0.4
P2_DROPOUT_2        = 0.3
P2_DENSE_1_UNITS    = 256
P2_DENSE_2_UNITS    = 128
P2_L2_REG           = 1e-4

# Layers 13 and 14 in the YAMNet MobileNetV1 backbone are the last two
# depthwise-separable blocks. Unfreezing them lets the model adapt its
# high-level spectral features to scream / groan acoustics without
# disturbing the low-level filter banks learned on AudioSet.
YAMNET_FINETUNE_FROM_LAYER = 13   # inclusive (0-indexed within backbone)

# ---------------------------------------------------------------------------
# On-device temporal state machine parameters (simulate in Python first)
# ---------------------------------------------------------------------------

# Exponential moving average weight on per-window P_emergency
TEMPORAL_EMA_ALPHA = 0.6

# Number of consecutive windows above threshold before alarm fires
TEMPORAL_CONSEC_TRIGGER = 2

# Number of consecutive windows below low threshold before returning to IDLE
TEMPORAL_CONSEC_RESET = 5

# Cooldown period after an alarm before the system can re-arm (seconds)
ALARM_COOLDOWN_SEC = 30.0

# Initial threshold sweep range for calibration
THRESHOLD_SWEEP = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
