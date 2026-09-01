"""
Step 7 — Preprocess external audio sources into the Phase 2 training format.

Reads external audio clips from:
  - FSD50K (CC0/CC-BY filtered, from 6e_audit_fsd50k.py manifest)
  - Freesound targeted downloads (from 6d_download_freesound.py)
  - DESED domestic environment sounds (from 6c_download_musan_desed.py)

For each clip:
  1. Load & resample to 16 kHz mono
  2. Apply quality gate (too quiet or too short → skip)
  3. Produce augmented variants:
       variant 0: original (gain-normalized)
       variant 1: convolved with a random bathroom RIR
       variant 2: RIR + additive MUSAN noise at random SNR
  4. Slice into 0.975 s windows (same contract as Step 2)
  5. Save windows as int16 to chunks_external.npy and manifest_external.csv

Output is designed to be merged with the original SAFE-derived chunks
by Step 8 (8_merge_datasets.py) before training.

Usage:
    C:\\aesv\\Scripts\\python.exe src\\7_preprocess_external.py [options]

Options:
    --sources          Which sources to process: fsd50k freesound desed all (default: all)
    --max-per-label    Maximum clips per label class to process (0 = unlimited)
    --dry-run          Count clips and print stats without writing output
    --workers          Number of parallel audio-loading workers (default: 4)
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import config_phase2 as cfg

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("preprocess_external")

# ---------------------------------------------------------------------------
# Quality gates (same philosophy as 2_preprocess.py)
# ---------------------------------------------------------------------------

MIN_CLIP_SECONDS   = 0.5   # skip clips shorter than this
RMS_FLOOR_DBFS     = -55.0 # skip near-silent clips
CLIPPING_WARN      = 0.01  # warn if >1% of samples clipped


# ---------------------------------------------------------------------------
# Audio loading / resampling
# ---------------------------------------------------------------------------


def load_wav_16k(path: Path) -> np.ndarray | None:
    """Load an audio file, resample to 16 kHz mono float32 in [-1, 1].

    Returns None if the file is unreadable or too short.
    """
    try:
        import librosa
        y, _ = librosa.load(str(path), sr=config.SAMPLE_RATE, mono=True, res_type="kaiser_fast")
    except Exception as exc:
        log.debug("  Skipping %s: %s", path.name, exc)
        return None

    if y.shape[0] < config.WINDOW_SAMPLES:
        return None  # too short for even one window

    rms = float(np.sqrt(np.mean(y ** 2)))
    if rms < 1e-6:
        return None  # near-silent

    rms_db = 20.0 * np.log10(rms + 1e-12)
    if rms_db < RMS_FLOOR_DBFS:
        return None  # too quiet

    # Normalize to -3 dBFS peak
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y * (0.708 / peak)  # 0.708 ≈ -3 dBFS

    return y.astype(np.float32)


# ---------------------------------------------------------------------------
# Room impulse responses
# ---------------------------------------------------------------------------


_RIR_CACHE: list[np.ndarray] | None = None


def _load_rir_pool() -> list[np.ndarray]:
    """Load all available bathroom RIRs at 16 kHz into memory."""
    global _RIR_CACHE
    if _RIR_CACHE is not None:
        return _RIR_CACHE

    rir_paths: list[Path] = []

    # MIT IR Survey bathroom IRs (primary)
    mit_ir_dir = cfg.MIT_IR_DIR / "bathroom_irs"
    if mit_ir_dir.exists():
        rir_paths.extend(sorted(mit_ir_dir.glob("*.wav")))

    # OpenSLR-28 simulated RIRs (supplementary)
    if cfg.OPENSLR28_DIR.exists():
        rir_paths.extend(sorted(cfg.OPENSLR28_DIR.rglob("sim_rir*.wav"))[:200])
        # Also add real RIRs from RWCP/REVERB/AIR
        rir_paths.extend(sorted(cfg.OPENSLR28_DIR.rglob("real_rir*.wav"))[:100])

    rirs: list[np.ndarray] = []
    for rp in rir_paths:
        try:
            ir, sr = sf.read(str(rp))
            if sr != config.SAMPLE_RATE:
                import librosa
                ir = librosa.resample(ir.astype(np.float32), orig_sr=sr, target_sr=config.SAMPLE_RATE)
            if ir.ndim > 1:
                ir = ir[:, 0]  # take first channel
            ir = ir.astype(np.float32)
            if ir.shape[0] > 0:
                rirs.append(ir)
        except Exception:
            pass

    if not rirs:
        log.warning("No RIR files found — RIR augmentation will be skipped.")

    _RIR_CACHE = rirs
    log.info("Loaded %d room impulse responses for augmentation", len(rirs))
    return rirs


def apply_rir(waveform: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve waveform with a room impulse response (fast FFT convolution)."""
    from scipy.signal import fftconvolve
    convolved = fftconvolve(waveform, rir, mode="full")[: len(waveform)]
    # Normalize so peak is preserved
    peak = np.max(np.abs(convolved))
    if peak > 0:
        original_peak = np.max(np.abs(waveform))
        convolved *= original_peak / peak
    return convolved.astype(np.float32)


# ---------------------------------------------------------------------------
# Noise mixing
# ---------------------------------------------------------------------------


_NOISE_CACHE: list[np.ndarray] | None = None
_NOISE_CACHE_MAX = 500   # keep up to N noise clips in memory


def _load_noise_pool(max_clips: int = _NOISE_CACHE_MAX) -> list[np.ndarray]:
    """Load a pool of MUSAN noise clips at 16 kHz for SNR mixing."""
    global _NOISE_CACHE
    if _NOISE_CACHE is not None:
        return _NOISE_CACHE

    noise_paths: list[Path] = []
    if cfg.MUSAN_NOISE_DIR.exists():
        noise_paths.extend(sorted(cfg.MUSAN_NOISE_DIR.rglob("*.wav")))
    if not noise_paths:
        log.warning("MUSAN noise not found — noise augmentation will be skipped.")
        _NOISE_CACHE = []
        return []

    random.shuffle(noise_paths)
    noise_clips: list[np.ndarray] = []
    for np_path in noise_paths[:max_clips]:
        try:
            n, sr = sf.read(str(np_path))
            if sr != config.SAMPLE_RATE:
                import librosa
                n = librosa.resample(n.astype(np.float32), orig_sr=sr, target_sr=config.SAMPLE_RATE)
            if n.ndim > 1:
                n = n[:, 0]
            n = n.astype(np.float32)
            if n.shape[0] >= config.WINDOW_SAMPLES:
                noise_clips.append(n)
        except Exception:
            pass

    _NOISE_CACHE = noise_clips
    log.info("Loaded %d MUSAN noise clips for SNR mixing", len(noise_clips))
    return noise_clips


def add_noise_at_snr(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Mix a random noise clip into signal at the specified SNR."""
    noise_pool = _load_noise_pool()
    if not noise_pool:
        return signal  # no noise available, return unchanged

    noise = random.choice(noise_pool)
    # Random crop / tile noise to match signal length
    if noise.shape[0] < signal.shape[0]:
        reps = (signal.shape[0] // noise.shape[0]) + 1
        noise = np.tile(noise, reps)
    start = random.randint(0, max(0, noise.shape[0] - signal.shape[0]))
    noise = noise[start : start + signal.shape[0]]

    # Scale noise to target SNR
    sig_rms = np.sqrt(np.mean(signal ** 2) + 1e-12)
    noi_rms = np.sqrt(np.mean(noise ** 2) + 1e-12)
    scale = sig_rms / (noi_rms * (10 ** (snr_db / 20)))
    mixed = signal + scale * noise

    # Re-normalize peak
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed /= peak
    return mixed.astype(np.float32)


# ---------------------------------------------------------------------------
# Window slicing (reuses audio_utils contract)
# ---------------------------------------------------------------------------


def slice_to_windows(waveform: np.ndarray) -> list[np.ndarray]:
    """Slice waveform into 0.975 s windows with 50% overlap. Returns list of windows."""
    windows = []
    for start in config.window_offsets(waveform.shape[0]):
        w = waveform[start : start + config.WINDOW_SAMPLES]
        windows.append(w)
    return windows


# ---------------------------------------------------------------------------
# Source inventory builders
# ---------------------------------------------------------------------------


def _clip_records_from_fsd50k() -> list[dict]:
    """Load FSD50K records from the commercial manifest CSV."""
    manifest = cfg.FSD50K_MANIFEST
    if not manifest.exists():
        log.warning("FSD50K manifest not found: %s", manifest)
        log.warning("  Run 6e_audit_fsd50k.py first after downloading FSD50K.")
        return []

    records = []
    with open(manifest, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            p = Path(row["wav_path"])
            if p.exists():
                records.append({"path": p, "label": row["label"],
                                 "source": "fsd50k", "classes": row["fsd50k_classes"]})
    log.info("FSD50K: %d usable clips from commercial manifest", len(records))
    return records


def _clip_records_from_freesound() -> list[dict]:
    """Collect all WAV files downloaded by 6d_download_freesound.py."""
    records = []
    for label in ("emergency", "normal"):
        label_dir = cfg.FREESOUND_DIR / label
        if not label_dir.exists():
            continue
        for wav in sorted(label_dir.glob("*.wav")):
            records.append({"path": wav, "label": label, "source": "freesound", "classes": ""})
    log.info("Freesound: %d clips across emergency/normal", len(records))
    return records


def _clip_records_from_desed() -> list[dict]:
    """Collect DESED clips and map to normal label (all DESED classes are confusers)."""
    records = []
    if not cfg.DESED_DIR.exists():
        return records
    for wav in sorted(cfg.DESED_DIR.rglob("*.wav")):
        records.append({"path": wav, "label": "normal", "source": "desed", "classes": ""})
    log.info("DESED: %d clips (all → normal label)", len(records))
    return records


# ---------------------------------------------------------------------------
# Core processing loop
# ---------------------------------------------------------------------------


def process_clips(
    clip_records: list[dict],
    dry_run: bool = False,
) -> tuple[list[np.ndarray], list[dict]]:
    """Process a list of clip records into windows with augmentation.

    Returns (list_of_int16_windows, manifest_rows).
    """
    rirs = _load_rir_pool()
    _load_noise_pool()   # warm the noise cache

    windows_out: list[np.ndarray] = []
    manifest_rows: list[dict] = []
    window_idx = 0

    skipped_load = 0
    skipped_short = 0
    total_clips = 0

    for rec in tqdm(clip_records, desc="Processing clips", unit="clip"):
        wav_path: Path = rec["path"]
        label: str = rec["label"]
        source: str = rec["source"]

        y = load_wav_16k(wav_path)
        if y is None:
            skipped_load += 1
            continue
        total_clips += 1

        # Build variants
        variants: list[tuple[np.ndarray, str]] = []

        # Variant 0: original (gain-jittered)
        gain = 10 ** (random.uniform(-cfg.GAIN_JITTER_DB, cfg.GAIN_JITTER_DB) / 20.0)
        v0 = np.clip(y * gain, -1.0, 1.0)
        variants.append((v0, "original"))

        # Variant 1: RIR convolution (if RIRs available)
        if rirs and random.random() < cfg.RIR_AUG_FRACTION:
            rir = random.choice(rirs)
            v1 = apply_rir(y, rir)
            variants.append((v1, "rir"))
        else:
            variants.append((y.copy(), "original_b"))  # duplicate clean if no RIR

        # Variant 2: RIR + noise
        if rirs:
            rir = random.choice(rirs)
            v2_base = apply_rir(y, rir)
        else:
            v2_base = y.copy()
        snr = random.uniform(*cfg.NOISE_SNR_DB_RANGE)
        v2 = add_noise_at_snr(v2_base, snr_db=snr)
        variants.append((v2, f"noise_snr{snr:.0f}"))

        for waveform, variant_kind in variants[:cfg.EXTERNAL_AUG_VARIANTS]:
            for start_sample, window in [(s, y[s:s+config.WINDOW_SAMPLES])
                                          for s in config.window_offsets(waveform.shape[0])]:
                if dry_run:
                    window_idx += 1
                    continue

                win_int16 = (window * 32767.0).clip(-32768, 32767).astype(np.int16)
                windows_out.append(win_int16)

                manifest_rows.append({
                    "window_idx": window_idx,
                    "source": source,
                    "clip_path": str(wav_path),
                    "label": label,
                    "label_index": cfg.P2_LABEL_TO_INDEX[label],
                    "variant_kind": variant_kind,
                    "start_sample": start_sample,
                    "fsd50k_classes": rec.get("classes", ""),
                })
                window_idx += 1

    log.info("  Total source clips processed: %d", total_clips)
    log.info("  Skipped (load error / too short): %d", skipped_load + skipped_short)
    log.info("  Total windows generated: %d", window_idx)

    return windows_out, manifest_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", nargs="+",
                   choices=["fsd50k", "freesound", "desed", "all"],
                   default=["all"])
    p.add_argument("--max-per-label", type=int, default=0,
                   help="Max clips per label (0=unlimited)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg.ensure_p2_dirs()

    log.info("=" * 60)
    log.info("Step 7 — Preprocess External Audio Sources")
    log.info("=" * 60)

    # ── Collect clip records ──────────────────────────────────────────────
    sources = set(args.sources)
    use_all = "all" in sources
    all_records: list[dict] = []

    if use_all or "fsd50k" in sources:
        all_records.extend(_clip_records_from_fsd50k())
    if use_all or "freesound" in sources:
        all_records.extend(_clip_records_from_freesound())
    if use_all or "desed" in sources:
        all_records.extend(_clip_records_from_desed())

    if not all_records:
        log.warning("No external clips found. Check that downloads completed.")
        log.warning("Expected sources:")
        log.warning("  FSD50K:    %s", cfg.FSD50K_DIR)
        log.warning("  Freesound: %s", cfg.FREESOUND_DIR)
        log.warning("  DESED:     %s", cfg.DESED_DIR)
        sys.exit(0)

    # Apply per-label cap
    if args.max_per_label > 0:
        from collections import defaultdict
        by_label: dict[str, list[dict]] = defaultdict(list)
        for r in all_records:
            by_label[r["label"]].append(r)
        capped = []
        for lbl, recs in by_label.items():
            random.shuffle(recs)
            capped.extend(recs[: args.max_per_label])
            log.info("  Label %-10s: %d clips (capped from %d)", lbl, len(recs[:args.max_per_label]), len(recs))
        all_records = capped

    random.shuffle(all_records)
    log.info("Total clips to process: %d", len(all_records))

    # Breakdown by label
    for lbl in ("emergency", "normal"):
        count = sum(1 for r in all_records if r["label"] == lbl)
        log.info("  %-10s: %d clips", lbl, count)

    if args.dry_run:
        log.info("\nDRY RUN — estimating window count only …")
        _, rows = process_clips(all_records, dry_run=True)
        log.info("Estimated windows (×%d variants): ~%d",
                 cfg.EXTERNAL_AUG_VARIANTS, len(rows) * cfg.EXTERNAL_AUG_VARIANTS)
        return

    # ── Process ──────────────────────────────────────────────────────────
    log.info("\nProcessing …")
    windows, manifest_rows = process_clips(all_records)

    if not windows:
        log.error("No windows produced. Check audio files.")
        sys.exit(1)

    # ── Save chunks numpy array ───────────────────────────────────────────
    chunks = np.stack(windows, axis=0)  # (N, 15600) int16
    np.save(str(cfg.P2_CHUNKS_FILE), chunks)
    log.info("Saved: %s  shape=%s  dtype=%s  (%.1f MB)",
             cfg.P2_CHUNKS_FILE, chunks.shape, chunks.dtype,
             chunks.nbytes / 1e6)

    # ── Save manifest CSV ─────────────────────────────────────────────────
    fieldnames = ["window_idx", "source", "clip_path", "label", "label_index",
                  "variant_kind", "start_sample", "fsd50k_classes"]
    with open(cfg.P2_MANIFEST_FILE, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    log.info("Saved: %s  (%d rows)", cfg.P2_MANIFEST_FILE, len(manifest_rows))

    # Summary
    emergency_windows = sum(1 for r in manifest_rows if r["label"] == "emergency")
    normal_windows    = sum(1 for r in manifest_rows if r["label"] == "normal")

    log.info("\n" + "=" * 60)
    log.info("Preprocessing complete.")
    log.info("  Emergency windows : %d", emergency_windows)
    log.info("  Normal windows    : %d", normal_windows)
    log.info("  Total windows     : %d", len(manifest_rows))
    log.info("  chunks file       : %s", cfg.P2_CHUNKS_FILE)
    log.info("  manifest file     : %s", cfg.P2_MANIFEST_FILE)
    log.info("  Next: run 8_merge_datasets.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
