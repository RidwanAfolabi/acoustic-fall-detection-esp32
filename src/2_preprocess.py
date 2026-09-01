"""
STEP 2 - Preprocessing and bathroom acoustic augmentation.

Turns the raw 48 kHz SAFE corpus into the exact tensor YAMNet consumes:
16 kHz mono, 15,600-sample windows at 50 % overlap, stored as int16.

Per source clip, several *variants* are produced:

    variant 0  canonical RIR  a single fixed bathroom, identical every run
    variant 1  dry            resampled only, no room
    variant 2+ random RIR     randomised geometry, RT60 and DRR

Variant 0 exists so Step 3 can build validation and test sets that are
reverberant (matching deployment) yet perfectly reproducible, while training
draws on every variant for diversity. Every chunk carries its source clip_uid,
fold_id and variant_id, so Step 3 can split without leaking a clip's own
variants across the train/test boundary.

Two augmentation sources are pluggable:

    --rir-source synthetic          simulated bathrooms (default)
    --rir-source measured --rir-dir DIR
                                    real measured RIRs, e.g. MIT IR Survey
                                    bathrooms or OpenSLR-28

    --negatives-dir DIR             OPTIONAL hard negatives mixed into the
                                    non-fall class

The hard-negative hook is off by default but matters more than it looks. The
Step 1 audit found ZERO non-fall clips containing an impulsive impact, while
every impulsive event in the corpus is labelled fall. A model trained on that
can score ~99% with the rule "transient => fall", then fire on every door slam
in a real restroom. Pointing --negatives-dir at ESC-50 (toilet flush, door
knock, pouring water, ...) or OpenSLR-28 noise removes that shortcut.

Usage
    python src/2_preprocess.py
    python src/2_preprocess.py --variants 4
    python src/2_preprocess.py --rir-source measured --rir-dir data/rirs
    python src/2_preprocess.py --negatives-dir data/negatives
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import fftconvolve
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audio_utils
import config
import rir as rir_lib

LOGGER = logging.getLogger("preprocess")

DRY_VARIANT = 1
CANONICAL_VARIANT = 0
INT16_FULL_SCALE = 32767.0
PEAK_CEILING = 0.999


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(console)
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    LOGGER.addHandler(handler)


def section(title: str) -> None:
    LOGGER.info("\n" + "=" * 78)
    LOGGER.info(title)
    LOGGER.info("=" * 78)


def indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def stable_seed(name: str, salt: int) -> int:
    """Deterministic per-file seed.

    Python's built-in hash() is randomised per process, so re-running would
    silently produce a different dataset. CRC32 is stable across runs.
    """
    return (zlib.crc32(name.encode("utf-8")) ^ (salt & 0xFFFFFFFF)) & 0x7FFFFFFF


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def load_resampled(path: Path, target_length: int) -> np.ndarray:
    """Load as 16 kHz mono float32, forced to exactly ``target_length`` samples.

    The length is pinned because the output array is preallocated from header
    metadata before any audio is decoded. Resamplers may land a sample either
    side of the predicted length; a one-sample pad or trim is inaudible, while a
    mismatch against the preallocated plan would corrupt the whole array.
    """
    import librosa

    wave, _ = librosa.load(
        str(path), sr=config.SAMPLE_RATE, mono=True, res_type="soxr_hq"
    )
    if wave.shape[0] < target_length:
        wave = np.pad(wave, (0, target_length - wave.shape[0]))
    return wave[:target_length].astype(np.float32)


def mix_negative(
    clip: np.ndarray, negative: np.ndarray, snr_db: float, rng: np.random.Generator
) -> np.ndarray:
    """Add a hard negative to ``clip`` at the requested clip-to-negative SNR."""
    need = clip.shape[0]
    if negative.shape[0] < need:
        repeats = int(np.ceil(need / max(negative.shape[0], 1)))
        negative = np.tile(negative, repeats)
    start = int(rng.integers(0, negative.shape[0] - need + 1))
    segment = negative[start : start + need]

    clip_rms = float(np.sqrt(np.mean(clip**2)))
    seg_rms = float(np.sqrt(np.mean(segment**2)))
    if seg_rms <= 0.0 or clip_rms <= 0.0:
        return clip
    gain = clip_rms / (seg_rms * (10.0 ** (snr_db / 20.0)))
    return (clip + gain * segment).astype(np.float32)


def apply_room(clip: np.ndarray, impulse: np.ndarray) -> np.ndarray:
    """Convolve with a RIR, preserving both onset alignment and clip length.

    The convolution is trimmed from the direct arrival rather than from sample
    zero, so the event stays where it was in time. Everything past the original
    duration is discarded, keeping a constant 5 windows per clip; the lost tail
    is decaying reverb of an onset already captured.
    """
    wet = fftconvolve(clip, impulse, mode="full")
    onset = rir_lib.direct_path_index(impulse)
    return wet[onset : onset + clip.shape[0]].astype(np.float32)


def to_int16(wave: np.ndarray) -> tuple[np.ndarray, float]:
    """Convert float audio to int16, guarding against overflow.

    Reverberation adds energy, so a convolved clip can exceed full scale. Rather
    than let it wrap, it is scaled down and the applied gain is recorded in the
    manifest so the level change stays auditable.
    """
    peak = float(np.max(np.abs(wave))) if wave.size else 0.0
    gain = 1.0
    if peak > PEAK_CEILING:
        gain = PEAK_CEILING / peak
        wave = wave * gain
    quantised = np.clip(np.rint(wave * INT16_FULL_SCALE), -32768, 32767)
    return quantised.astype(np.int16), gain


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def plan_corpus(raw_dir: Path, variants: int) -> tuple[list[dict[str, Any]], int]:
    """Inventory the corpus and compute the exact output size up front."""
    paths = audio_utils.find_audio_files(raw_dir)
    if not paths:
        section("NO INPUT AUDIO")
        LOGGER.info("Nothing found under %s", raw_dir)
        LOGGER.info("Run Step 1 first, or check the path.")
        raise SystemExit(2)

    plan: list[dict[str, Any]] = []
    skipped = 0
    for path in paths:
        parsed = audio_utils.parse_filename(path)
        if parsed is None or parsed.label is None:
            skipped += 1
            continue
        try:
            info = sf.info(str(path))
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        length = int(round(info.frames * config.SAMPLE_RATE / info.samplerate))
        windows = audio_utils.num_windows(length)
        if windows == 0:
            skipped += 1
            continue
        plan.append(
            {
                "path": path,
                "parsed": parsed,
                "length": length,
                "windows": windows,
            }
        )
    total = sum(item["windows"] for item in plan) * variants
    if skipped:
        LOGGER.info("Skipped %d unusable file(s).", skipped)
    return plan, total


def build_rir_pool(source, pool_size: int, rng: np.random.Generator) -> list:
    """Pre-generate a pool of RIRs and reuse it across clips.

    Synthesising a fresh RIR for every clip would dominate runtime for no real
    gain in diversity: a pool of a couple of hundred rooms already gives each
    clip an essentially unique acoustic context.
    """
    pool = []
    for _ in tqdm(
        range(pool_size), desc="Building RIR pool", unit="rir", ncols=78,
        disable=not sys.stderr.isatty(),
    ):
        pool.append(source.sample(rng))
    return pool


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------


def process(args: argparse.Namespace) -> int:
    config.ensure_dirs()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    section("PREPROCESSING & BATHROOM AUGMENTATION  -  Step 2")
    LOGGER.info("Raw data ........... %s", args.raw_dir)
    LOGGER.info("Output ............. %s", out_dir)
    LOGGER.info("Variants per clip .. %d", args.variants)
    LOGGER.info("RIR source ......... %s", args.rir_source)
    LOGGER.info(
        "Hard negatives ..... %s",
        args.negatives_dir if args.negatives_dir else "disabled",
    )

    plan, total_windows = plan_corpus(args.raw_dir, args.variants)
    if args.limit > 0:
        plan = plan[: args.limit]
        total_windows = sum(i["windows"] for i in plan) * args.variants
    LOGGER.info("Source clips ....... %d", len(plan))
    LOGGER.info("Projected windows .. %d", total_windows)
    LOGGER.info(
        "Array size ......... %.1f MB (int16)",
        total_windows * config.WINDOW_SAMPLES * 2 / 1e6,
    )

    rng = np.random.default_rng(args.seed)
    source = rir_lib.make_rir_source(args.rir_source, config.SAMPLE_RATE, args.rir_dir)
    canonical_rir, canonical_meta = source.canonical()
    pool = build_rir_pool(source, args.rir_pool_size, rng) if args.variants > 2 else []

    negatives: list[tuple[np.ndarray, str]] = []
    if args.negatives_dir:
        import librosa

        neg_paths = sorted(
            p for p in Path(args.negatives_dir).rglob("*")
            if p.is_file() and p.suffix.lower() in config.AUDIO_EXTENSIONS
        )
        for p in tqdm(neg_paths, desc="Loading negatives", unit="file", ncols=78,
                      disable=not sys.stderr.isatty()):
            data, _ = librosa.load(str(p), sr=config.SAMPLE_RATE, mono=True)
            if data.size and float(np.max(np.abs(data))) > 0.0:
                negatives.append((data.astype(np.float32), p.stem))
        LOGGER.info("Loaded %d hard-negative file(s).", len(negatives))
        if not negatives:
            raise SystemExit(f"no usable audio under {args.negatives_dir}")

    chunk_path = out_dir / "chunks_int16.npy"
    chunks = np.lib.format.open_memmap(
        chunk_path, mode="w+", dtype=np.int16,
        shape=(total_windows, config.WINDOW_SAMPLES),
    )

    rows: list[dict[str, Any]] = []
    cursor = 0
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    sample_saved = 0
    example: dict[str, np.ndarray] = {}

    progress = tqdm(plan, desc="Preprocessing", unit="clip", ncols=78,
                    disable=not sys.stderr.isatty())
    for item in progress:
        path: Path = item["path"]
        parsed = item["parsed"]
        base = load_resampled(path, item["length"])
        file_rng = np.random.default_rng(stable_seed(path.name, args.seed))
        is_non_fall = parsed.label == "non_fall"

        for variant in range(args.variants):
            clip = base.copy()

            neg_name, neg_snr = "", float("nan")
            if (
                negatives
                and is_non_fall
                and file_rng.random() < args.negative_prob
            ):
                neg_wave, neg_name = negatives[int(file_rng.integers(len(negatives)))]
                neg_snr = float(file_rng.uniform(*args.negative_snr))
                clip = mix_negative(clip, neg_wave, neg_snr, file_rng)

            if variant == DRY_VARIANT:
                kind, impulse, meta = "dry", None, {}
                wet = clip
            elif variant == CANONICAL_VARIANT:
                kind, impulse, meta = "canonical", canonical_rir, canonical_meta
                wet = apply_room(clip, impulse)
            else:
                impulse, meta = pool[int(file_rng.integers(len(pool)))]
                kind = "random"
                wet = apply_room(clip, impulse)

            quantised, gain = to_int16(wet)
            for window_index, start in enumerate(
                audio_utils.window_offsets(quantised.shape[0])
            ):
                chunks[cursor] = quantised[start : start + config.WINDOW_SAMPLES]
                rows.append(
                    {
                        "chunk_index": cursor,
                        "source_filename": path.name,
                        "clip_uid": parsed.clip_uid,
                        "fold_id": parsed.fold_id,
                        "class_code": parsed.class_code,
                        "label": parsed.label,
                        "label_index": config.LABEL_TO_INDEX[parsed.label],
                        "event_code": parsed.event_code,
                        "variant_id": variant,
                        "variant_kind": kind,
                        "rir_id": meta.get("rir_id", ""),
                        "rt60_target": meta.get("rt60_target", float("nan")),
                        "drr_target_db": meta.get("drr_target_db", float("nan")),
                        "window_index": window_index,
                        "start_sample": start,
                        "negative_mixed": bool(neg_name),
                        "negative_source": neg_name,
                        "negative_snr_db": neg_snr,
                        "gain_applied": gain,
                    }
                )
                cursor += 1

            if sample_saved < args.n_samples and parsed.label == "fall":
                sf.write(
                    samples_dir / f"{path.stem}__v{variant}_{kind}.wav",
                    quantised, config.SAMPLE_RATE, subtype="PCM_16",
                )
                if kind in ("dry", "canonical"):
                    example[kind] = wet.copy()
                if variant == args.variants - 1:
                    sample_saved += 1

    chunks.flush()
    if cursor != total_windows:
        LOGGER.info(
            "WARNING: wrote %d windows but planned %d; trimming.", cursor, total_windows
        )

    manifest = pd.DataFrame.from_records(rows)
    manifest_path = out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    report(manifest, pool, canonical_rir, canonical_meta, args)
    figure_paths = write_figures(pool, canonical_rir, example, args)

    summary = {
        "raw_dir": str(args.raw_dir),
        "variants": args.variants,
        "rir_source": args.rir_source,
        "rir_dir": str(args.rir_dir) if args.rir_dir else None,
        "negatives_dir": str(args.negatives_dir) if args.negatives_dir else None,
        "source_clips": len(plan),
        "total_windows": int(cursor),
        "sample_rate": config.SAMPLE_RATE,
        "window_samples": config.WINDOW_SAMPLES,
        "hop_samples": config.HOP_SAMPLES,
        "windows_per_label": manifest["label"].value_counts().to_dict(),
        "windows_per_variant": manifest["variant_kind"].value_counts().to_dict(),
        "negatives_mixed": int(manifest["negative_mixed"].sum()),
        "seed": args.seed,
    }
    (out_dir / "preprocess_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    section("ARTEFACTS")
    LOGGER.info("chunks ..... %s  (%.1f MB)", chunk_path,
                chunk_path.stat().st_size / 1e6)
    LOGGER.info("manifest ... %s", manifest_path)
    LOGGER.info("summary .... %s", out_dir / "preprocess_summary.json")
    LOGGER.info("samples .... %s", samples_dir)
    for p in figure_paths:
        LOGGER.info("figure ..... %s", p)

    section("RESULT")
    LOGGER.info("Wrote %d windows from %d clips. Ready for Step 3.", cursor, len(plan))
    return 0


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def report(manifest, pool, canonical_rir, canonical_meta, args) -> None:
    section("RIR VERIFICATION")
    LOGGER.info("Every RIR is measured back, never assumed.\n")

    measured_rt60 = rir_lib.measure_rt60(canonical_rir, config.SAMPLE_RATE)
    measured_drr = rir_lib.measure_drr(canonical_rir, config.SAMPLE_RATE)
    LOGGER.info("Canonical RIR (variant 0, used by val/test in Step 3):")
    LOGGER.info("  id ................ %s", canonical_meta.get("rir_id", "?"))
    LOGGER.info("  RT60 target ....... %.3f s",
                canonical_meta.get("rt60_target", float("nan")))
    LOGGER.info("  RT60 measured ..... %.3f s  (Schroeder T30)", measured_rt60)
    LOGGER.info("  DRR  measured ..... %+.2f dB", measured_drr)

    if pool:
        rt = np.array([rir_lib.measure_rt60(r, config.SAMPLE_RATE) for r, _ in pool])
        dr = np.array([rir_lib.measure_drr(r, config.SAMPLE_RATE) for r, _ in pool])
        tgt = np.array([m.get("rt60_target", np.nan) for _, m in pool])
        ok = np.isfinite(rt) & np.isfinite(tgt)
        LOGGER.info("\nRandom RIR pool (%d rooms):", len(pool))
        LOGGER.info("  RT60 measured ..... %.3f - %.3f s (mean %.3f)",
                    np.nanmin(rt), np.nanmax(rt), np.nanmean(rt))
        LOGGER.info("  DRR  measured ..... %+.2f - %+.2f dB (mean %+.2f)",
                    np.nanmin(dr), np.nanmax(dr), np.nanmean(dr))
        if ok.any():
            err = np.abs(rt[ok] - tgt[ok]) / tgt[ok] * 100.0
            LOGGER.info("  RT60 error vs target  mean %.1f%%  max %.1f%%",
                        err.mean(), err.max())
            if err.mean() > 5.0:
                LOGGER.info("  WARNING: generator is drifting from its RT60 target.")
        bad = int((~np.isfinite(rt)).sum())
        if bad:
            LOGGER.info("  WARNING: %d RIR(s) too short to measure T30.", bad)

    section("OUTPUT COMPOSITION")
    LOGGER.info("Windows per class:")
    LOGGER.info(indent(manifest["label"].value_counts().to_frame().to_string()))
    LOGGER.info("\nWindows per variant:")
    LOGGER.info(indent(
        manifest.groupby(["variant_id", "variant_kind"]).size()
        .rename("windows").to_frame().to_string()
    ))
    LOGGER.info("\nWindows per fold:")
    LOGGER.info(indent(
        pd.crosstab(manifest["fold_id"], manifest["label"], margins=True).to_string()
    ))

    clipped = manifest.loc[manifest["gain_applied"] < 1.0]
    LOGGER.info(
        "\nWindows needing headroom reduction after reverb: %d (%.1f%%)",
        len(clipped), 100.0 * len(clipped) / max(len(manifest), 1),
    )
    if len(clipped):
        LOGGER.info("  smallest gain applied: %.3f", clipped["gain_applied"].min())

    if manifest["negative_mixed"].any():
        n = int(manifest["negative_mixed"].sum())
        LOGGER.info("\nHard negatives mixed into %d window(s) (%.1f%% of non-fall).",
                    n, 100.0 * n / max((manifest["label"] == "non_fall").sum(), 1))
    else:
        LOGGER.info(
            "\nNo hard negatives mixed. The non-fall class therefore still "
            "contains\nno impulsive impacts (see Step 1 audit), so the model can "
            "learn\n'transient => fall'. Pass --negatives-dir to close this."
        )


def write_figures(pool, canonical_rir, example, args) -> list[Path]:
    if not args.figures:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = config.FIGURES_DIR
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    sr = config.SAMPLE_RATE

    # --- canonical RIR: waveform + energy decay curve ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    t = np.arange(canonical_rir.shape[0]) / sr
    axes[0].plot(t, canonical_rir, lw=0.4, color="#3b7dd8")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("amplitude")
    axes[0].set_title("Canonical bathroom RIR")

    edc = rir_lib.energy_decay_curve(canonical_rir)
    rt = rir_lib.measure_rt60(canonical_rir, sr)
    axes[1].plot(t, edc, color="#3b7dd8")
    axes[1].axhline(-5, ls="--", lw=0.8, color="#888")
    axes[1].axhline(-35, ls="--", lw=0.8, color="#888")
    axes[1].set_ylim(-70, 2)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("energy (dB)")
    axes[1].set_title(f"Schroeder decay - measured T30 = {rt:.3f} s")
    fig.tight_layout()
    p = out / "preprocess_canonical_rir.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(p)

    # --- dry vs reverberant mel spectrogram of the same clip ---
    if "dry" in example and "canonical" in example:
        import librosa

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ax, key, title in (
            (axes[0], "dry", "Dry (as recorded)"),
            (axes[1], "canonical", "After bathroom RIR"),
        ):
            mel = librosa.feature.melspectrogram(
                y=example[key], sr=sr, n_fft=config.STFT_WINDOW_SAMPLES,
                hop_length=config.STFT_HOP_SAMPLES, n_mels=config.MEL_BANDS,
                fmin=config.MEL_MIN_HZ, fmax=config.MEL_MAX_HZ,
            )
            ax.imshow(np.log(mel + config.LOG_OFFSET), aspect="auto", origin="lower",
                      cmap="magma")
            ax.set_title(title)
            ax.set_xlabel("frame (10 ms)")
        axes[0].set_ylabel("mel bin")
        fig.suptitle("Reverberation smears the transient the model relies on")
        fig.tight_layout()
        p = out / "preprocess_dry_vs_wet.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        written.append(p)

    # --- distribution of the pool's measured acoustics ---
    if pool:
        rt = np.array([rir_lib.measure_rt60(r, sr) for r, _ in pool])
        dr = np.array([rir_lib.measure_drr(r, sr) for r, _ in pool])
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
        axes[0].hist(rt[np.isfinite(rt)], bins=25, color="#3b7dd8", edgecolor="white")
        axes[0].set_xlabel("measured RT60 (s)")
        axes[0].set_ylabel("rooms")
        axes[0].set_title("RIR pool - reverberation time")
        axes[1].hist(dr[np.isfinite(dr)], bins=25, color="#3b7dd8", edgecolor="white")
        axes[1].set_xlabel("measured DRR (dB)")
        axes[1].set_title("RIR pool - direct-to-reverberant ratio")
        fig.tight_layout()
        p = out / "preprocess_rir_pool.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        written.append(p)

    return written


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    p.add_argument("--out-dir", type=Path, default=config.PROCESSED_DIR)
    p.add_argument("--variants", type=int, default=3,
                   help="variants per clip: 0=canonical RIR, 1=dry, 2+=random")
    p.add_argument("--rir-source", choices=("synthetic", "measured"),
                   default="synthetic")
    p.add_argument("--rir-dir", type=Path, default=None,
                   help="directory of measured RIRs (with --rir-source measured)")
    p.add_argument("--rir-pool-size", type=int, default=200)
    p.add_argument("--negatives-dir", type=Path, default=None,
                   help="OPTIONAL directory of hard-negative audio to mix into "
                        "the non-fall class")
    p.add_argument("--negative-prob", type=float, default=0.5)
    p.add_argument("--negative-snr", type=float, nargs=2, default=(0.0, 15.0),
                   metavar=("MIN_DB", "MAX_DB"))
    p.add_argument("--n-samples", type=int, default=3,
                   help="how many example clips to write out as listenable wavs")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--no-figures", dest="figures", action="store_false")
    args = p.parse_args(argv)
    if args.variants < 1:
        p.error("--variants must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config.ensure_dirs()
    setup_logging(config.LOGS_DIR / "preprocess.log")
    return process(args)


if __name__ == "__main__":
    raise SystemExit(main())
