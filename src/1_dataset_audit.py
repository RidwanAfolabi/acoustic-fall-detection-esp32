"""
STEP 1 - SAFE dataset audit.

Verifies that the raw corpus in ``data/raw/`` is fit to feed a YAMNet transfer
learning pipeline, before a single sample is preprocessed. Specifically:

  1. File integrity  - every file decodes; format, sample rate, channel count,
                       bit depth and duration are recorded and checked.
  2. Class balance   - distribution over Fall (01) / Non-Fall (02), plus the
                       fold distribution that Step 3 will split on.
  3. Window preview  - simulates the 0.975 s / 50 %-overlap slicing at 16 kHz
                       and reports exactly how many YAMNet patches the corpus
                       will yield, per class and per fold.

Outputs
    outputs/dataset_audit_manifest.csv  one row per source file
    outputs/dataset_audit_summary.json  machine-readable summary
    outputs/figures/*.png               distribution plots
    outputs/logs/dataset_audit.log      full run log

Usage
    python src/1_dataset_audit.py
    python src/1_dataset_audit.py --no-deep          # header scan only (fast)
    python src/1_dataset_audit.py --limit 200        # sample the corpus

Exit codes
    0  audit passed with no blocking issues
    1  audit completed but found issues needing attention
    2  no dataset found in data/raw/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audio_utils
import config

LOGGER = logging.getLogger("dataset_audit")


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

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")
    )
    LOGGER.addHandler(file_handler)


def section(title: str) -> None:
    LOGGER.info("\n" + "=" * 78)
    LOGGER.info(title)
    LOGGER.info("=" * 78)


# ---------------------------------------------------------------------------
# Per-file inspection
# ---------------------------------------------------------------------------


def inspect_header(path: Path) -> dict[str, Any]:
    """Read container metadata without decoding audio."""
    record: dict[str, Any] = {
        "path": str(path),
        "filename": path.name,
        "relative_path": str(path.relative_to(config.RAW_DIR))
        if config.RAW_DIR in path.parents
        else path.name,
        "size_bytes": path.stat().st_size,
        "readable": False,
        "error": "",
    }
    try:
        info = sf.info(str(path))
    except Exception as exc:  # noqa: BLE001 - any decoder failure is a finding
        record["error"] = f"header: {type(exc).__name__}: {exc}"
        return record

    record.update(
        readable=True,
        native_sample_rate=int(info.samplerate),
        native_channels=int(info.channels),
        native_frames=int(info.frames),
        duration_sec=float(info.frames) / float(info.samplerate)
        if info.samplerate
        else 0.0,
        container=info.format,
        subtype=info.subtype,
    )
    return record


def inspect_signal(path: Path, record: dict[str, Any]) -> None:
    """Fully decode the file and measure signal quality. Mutates ``record``."""
    import librosa  # imported lazily: pulls in numba, ~2 s of import time

    try:
        waveform, sample_rate = librosa.load(str(path), sr=None, mono=False)
    except Exception as exc:  # noqa: BLE001
        record["readable"] = False
        record["error"] = f"decode: {type(exc).__name__}: {exc}"
        return

    if waveform.ndim > 1:
        waveform = librosa.to_mono(waveform)

    if waveform.size == 0:
        record["error"] = "decode: empty waveform"
        record.update(peak_amplitude=0.0, rms_dbfs=float("-inf"))
        return

    peak = float(np.max(np.abs(waveform)))
    record.update(
        decoded_sample_rate=int(sample_rate),
        decoded_samples=int(waveform.size),
        peak_amplitude=peak,
        rms_dbfs=audio_utils.rms_dbfs(waveform),
        clipping_ratio=audio_utils.clipping_ratio(waveform),
        dc_offset=float(np.mean(waveform)),
        is_silent=peak == 0.0
        or audio_utils.rms_dbfs(waveform) < config.SILENCE_DBFS_THRESHOLD,
    )


def build_manifest(paths: list[Path], deep: bool) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    progress = tqdm(
        paths,
        desc="Inspecting audio",
        unit="file",
        ncols=78,
        # A redirected stderr turns the bar into thousands of redraw lines in
        # the captured log, so only draw it on a real terminal.
        disable=not sys.stderr.isatty(),
    )
    for path in progress:
        record = inspect_header(path)
        if deep and record["readable"]:
            inspect_signal(path, record)

        parsed = audio_utils.parse_filename(path)
        record["filename_conforms"] = parsed is not None
        if parsed is not None:
            record.update(parsed.as_dict())
            record["label"] = parsed.label or "UNKNOWN"
        else:
            record.update({field: "" for field in config.FILENAME_FIELDS})
            record["label"] = "UNPARSED"

        records.append(record)

    frame = pd.DataFrame.from_records(records)

    # Window yield is computed from the duration *after* the 16 kHz resample in
    # Step 2, which is the number of samples the slicer will actually see.
    duration = frame.get("duration_sec", pd.Series(0.0, index=frame.index)).fillna(0.0)
    resampled_samples = np.floor(duration * config.SAMPLE_RATE).astype("int64")
    frame["resampled_samples"] = resampled_samples
    frame["n_windows"] = [audio_utils.num_windows(int(n)) for n in resampled_samples]
    frame["covered_samples"] = [
        audio_utils.covered_samples(int(n)) for n in resampled_samples
    ]
    frame["uncovered_sec"] = (
        frame["resampled_samples"] - frame["covered_samples"]
    ) / config.SAMPLE_RATE
    return frame


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def report_integrity(frame: pd.DataFrame, deep: bool) -> list[str]:
    section("1. FILE INTEGRITY & FORMAT")
    issues: list[str] = []

    total = len(frame)
    unreadable = frame.loc[~frame["readable"]]
    LOGGER.info("Files discovered ............ %d", total)
    LOGGER.info("Decoded successfully ........ %d", total - len(unreadable))
    LOGGER.info("Failed ...................... %d", len(unreadable))
    LOGGER.info(
        "Total size .................. %.1f MB", frame["size_bytes"].sum() / 1e6
    )

    if len(unreadable):
        issues.append(f"{len(unreadable)} file(s) failed to decode")
        LOGGER.info("\nUnreadable files:")
        for _, row in unreadable.head(20).iterrows():
            LOGGER.info("  %-45s %s", row["filename"], row["error"])

    good = frame.loc[frame["readable"]]
    if good.empty:
        return issues

    LOGGER.info("\nContainer / subtype:")
    LOGGER.info(
        indent(
            good.groupby(["container", "subtype"], dropna=False)
            .size()
            .rename("files")
            .to_frame()
            .to_string()
        )
    )

    LOGGER.info("\nNative sample rates (Hz):")
    rates = good["native_sample_rate"].value_counts().sort_index()
    for rate, count in rates.items():
        marker = "  <- target" if rate == config.SAMPLE_RATE else ""
        LOGGER.info("  %8d Hz  %5d file(s)%s", rate, count, marker)
    off_rate = int((good["native_sample_rate"] != config.SAMPLE_RATE).sum())
    if off_rate:
        LOGGER.info(
            "  NOTE: %d file(s) are not 16 kHz; Step 2 will resample them.", off_rate
        )

    LOGGER.info("\nChannel counts:")
    for channels, count in good["native_channels"].value_counts().sort_index().items():
        LOGGER.info("  %d channel(s)  %5d file(s)", channels, count)
    non_mono = int((good["native_channels"] != config.CHANNELS).sum())
    if non_mono:
        LOGGER.info(
            "  NOTE: %d file(s) are not mono; Step 2 will downmix them.", non_mono
        )

    # Durations
    section("1b. DURATIONS")
    durations = good["duration_sec"]
    LOGGER.info("count %6d", durations.count())
    LOGGER.info("mean  %8.3f s", durations.mean())
    LOGGER.info("std   %8.3f s", durations.std(ddof=0))
    LOGGER.info("min   %8.3f s", durations.min())
    for quantile in (0.05, 0.25, 0.50, 0.75, 0.95):
        LOGGER.info("%3d%%  %8.3f s", int(quantile * 100), durations.quantile(quantile))
    LOGGER.info("max   %8.3f s", durations.max())
    LOGGER.info("total %8.1f s (%.2f h)", durations.sum(), durations.sum() / 3600.0)

    off_nominal = good.loc[
        (durations - config.EXPECTED_CLIP_SECONDS).abs()
        > config.CLIP_DURATION_TOLERANCE
    ]
    LOGGER.info(
        "\nClips within %.2f s +/- %.2f s: %d / %d",
        config.EXPECTED_CLIP_SECONDS,
        config.CLIP_DURATION_TOLERANCE,
        len(good) - len(off_nominal),
        len(good),
    )
    if len(off_nominal):
        issues.append(
            f"{len(off_nominal)} file(s) deviate from the nominal "
            f"{config.EXPECTED_CLIP_SECONDS} s clip length"
        )
        LOGGER.info("Off-nominal examples:")
        for _, row in off_nominal.head(10).iterrows():
            LOGGER.info("  %-45s %7.3f s", row["filename"], row["duration_sec"])

    too_short = good.loc[good["resampled_samples"] < config.WINDOW_SAMPLES]
    if len(too_short):
        issues.append(
            f"{len(too_short)} file(s) are shorter than one {config.WINDOW_SECONDS} s "
            "window and will yield zero training samples"
        )
        LOGGER.info(
            "\nWARNING: %d file(s) shorter than %.3f s -> 0 windows.",
            len(too_short),
            config.WINDOW_SECONDS,
        )

    if deep:
        section("1c. SIGNAL QUALITY")
        clipped = good.loc[good["clipping_ratio"] > config.CLIPPING_RATIO_WARN]
        silent = good.loc[good["is_silent"]]
        dc_biased = good.loc[good["dc_offset"].abs() > config.DC_OFFSET_WARN]

        finite_rms = good.loc[np.isfinite(good["rms_dbfs"]), "rms_dbfs"]
        LOGGER.info(
            "RMS level ......... mean %.1f dBFS   min %.1f   max %.1f",
            finite_rms.mean(),
            finite_rms.min(),
            finite_rms.max(),
        )
        LOGGER.info("Peak amplitude .... max %.4f", good["peak_amplitude"].max())
        LOGGER.info("Clipped files ..... %d (>%.2f%% clipped samples)",
                    len(clipped), config.CLIPPING_RATIO_WARN * 100)
        LOGGER.info("Silent files ...... %d (RMS < %.0f dBFS)",
                    len(silent), config.SILENCE_DBFS_THRESHOLD)
        LOGGER.info("DC-biased files ... %d (|mean| > %.3f)",
                    len(dc_biased), config.DC_OFFSET_WARN)

        if len(silent):
            issues.append(f"{len(silent)} silent or near-silent file(s)")
            for _, row in silent.head(10).iterrows():
                LOGGER.info("  silent: %-40s %.1f dBFS",
                            row["filename"], row["rms_dbfs"])
        if len(clipped):
            issues.append(f"{len(clipped)} clipped file(s)")

    return issues


def report_naming(frame: pd.DataFrame) -> list[str]:
    section("2. FILENAME SCHEMA  (AA-BBB-CC-DDD-FF.wav)")
    issues: list[str] = []

    non_conforming = frame.loc[~frame["filename_conforms"]]
    LOGGER.info(
        "Conforming .................. %d / %d",
        int(frame["filename_conforms"].sum()),
        len(frame),
    )
    if len(non_conforming):
        issues.append(
            f"{len(non_conforming)} filename(s) do not match the expected schema"
        )
        LOGGER.info("Non-conforming examples:")
        for name in non_conforming["filename"].head(15):
            LOGGER.info("  %s", name)

    conforming = frame.loc[frame["filename_conforms"]]
    if conforming.empty:
        return issues

    LOGGER.info(
        "\nObserved value set per field (confirm the schema mapping against this):"
    )
    for field in config.FILENAME_FIELDS:
        values = sorted(conforming[field].unique())
        preview = ", ".join(values[:12]) + (" ..." if len(values) > 12 else "")
        LOGGER.info("  %-12s %3d distinct  [%s]", field, len(values), preview)

    unknown = conforming.loc[conforming["label"] == "UNKNOWN"]
    if len(unknown):
        codes = sorted(unknown["class_code"].unique())
        issues.append(f"unmapped class code(s): {codes}")
        LOGGER.info(
            "\nWARNING: class code(s) %s are not in CLASS_CODE_TO_LABEL "
            "(%d file(s)).",
            codes,
            len(unknown),
        )
    return issues


def report_distribution(frame: pd.DataFrame) -> list[str]:
    section("3. CLASS & FOLD DISTRIBUTION")
    issues: list[str] = []

    usable = frame.loc[frame["readable"] & frame["filename_conforms"]]
    if usable.empty:
        LOGGER.info("No usable files to summarise.")
        return ["no usable files"]

    counts = usable.groupby("label", dropna=False).agg(
        files=("filename", "count"),
        total_sec=("duration_sec", "sum"),
        windows=("n_windows", "sum"),
    )
    counts["files_pct"] = 100.0 * counts["files"] / counts["files"].sum()
    counts["windows_pct"] = 100.0 * counts["windows"] / max(counts["windows"].sum(), 1)
    LOGGER.info("Per class:")
    LOGGER.info(indent(counts.round(2).to_string()))

    # Imbalance is measured over the mapped classes only; a handful of files
    # with an unmapped class code would otherwise dominate the ratio.
    known = counts.loc[counts.index.isin(config.LABEL_TO_INDEX)]
    if len(known) >= 2:
        ratio = known["files"].max() / max(known["files"].min(), 1)
        LOGGER.info("\nImbalance ratio (fall vs non_fall, max:min) ... %.2f : 1", ratio)
        if ratio > 1.5:
            issues.append(
                f"class imbalance {ratio:.2f}:1 - use class weights in Step 4"
            )
    else:
        issues.append(
            f"only {len(known)} mapped class(es) present - expected fall and non_fall"
        )

    LOGGER.info("\nPer fold:")
    fold_counts = usable.groupby("fold_id").agg(
        files=("filename", "count"),
        windows=("n_windows", "sum"),
    )
    LOGGER.info(indent(fold_counts.to_string()))

    LOGGER.info("\nClass x fold (file counts):")
    crosstab = pd.crosstab(usable["fold_id"], usable["label"], margins=True)
    LOGGER.info(indent(crosstab.to_string()))

    n_folds = usable["fold_id"].nunique()
    LOGGER.info("\nDistinct folds .............. %d", n_folds)
    if n_folds < 3:
        issues.append(
            f"only {n_folds} fold(s) - too few to build a 70/15/15 split from "
            "whole folds; Step 3 will have to split at file level instead"
        )

    LOGGER.info("\nEvent code x class:")
    LOGGER.info(indent(pd.crosstab(usable["event_code"], usable["label"]).to_string()))

    # Invariant: non-fall clips carry event code 00 and fall clips never do.
    # A violation means a filename's label and event code disagree.
    mislabelled = usable.loc[
        (usable["event_code"] == config.NON_FALL_EVENT_CODE)
        != (usable["label"] == "non_fall")
    ]
    if len(mislabelled):
        issues.append(
            f"{len(mislabelled)} file(s) whose event code contradicts their class "
            "code - the label mapping may be wrong"
        )
        LOGGER.info(
            "\nWARNING: %d file(s) violate the event-code/class invariant.",
            len(mislabelled),
        )
        for name in mislabelled["filename"].head(10):
            LOGGER.info("  %s", name)
    else:
        LOGGER.info(
            "\nEvent-code/class invariant holds for all %d file(s).", len(usable)
        )
    return issues


def report_windows(frame: pd.DataFrame) -> list[str]:
    section("4. WINDOW SLICING PREVIEW  (YAMNet contract)")
    issues: list[str] = []

    LOGGER.info("Target sample rate ............ %d Hz, mono", config.SAMPLE_RATE)
    LOGGER.info(
        "Window ........................ %.3f s = %d samples",
        config.WINDOW_SECONDS,
        config.WINDOW_SAMPLES,
    )
    LOGGER.info(
        "Hop (%.0f%% overlap) ............. %.3f s = %d samples",
        config.WINDOW_OVERLAP * 100,
        config.HOP_SECONDS,
        config.HOP_SAMPLES,
    )
    LOGGER.info(
        "Patch produced per window ..... %d frames x %d mel bins -> %d-d embedding",
        config.PATCH_FRAMES,
        config.MEL_BANDS,
        config.EMBEDDING_DIM,
    )

    nominal_samples = int(config.EXPECTED_CLIP_SECONDS * config.SAMPLE_RATE)
    offsets = audio_utils.window_offsets(nominal_samples)
    covered = audio_utils.covered_samples(nominal_samples)
    LOGGER.info(
        "\nWorked example - a nominal %.1f s clip (%d samples):",
        config.EXPECTED_CLIP_SECONDS,
        nominal_samples,
    )
    LOGGER.info("  complete windows ............ %d", len(offsets))
    for index, start in enumerate(offsets):
        LOGGER.info(
            "    window %d: samples [%6d, %6d)  = [%.3f s, %.3f s)",
            index,
            start,
            start + config.WINDOW_SAMPLES,
            start / config.SAMPLE_RATE,
            (start + config.WINDOW_SAMPLES) / config.SAMPLE_RATE,
        )
    LOGGER.info(
        "  covered ..................... %d samples (%.3f s)",
        covered,
        covered / config.SAMPLE_RATE,
    )
    LOGGER.info(
        "  discarded tail .............. %d samples (%.3f s)",
        nominal_samples - covered,
        (nominal_samples - covered) / config.SAMPLE_RATE,
    )

    usable = frame.loc[frame["readable"] & frame["filename_conforms"]]
    if usable.empty:
        return issues

    section("4b. PROJECTED TRAINING SET")
    total_windows = int(usable["n_windows"].sum())
    zero_yield = int((usable["n_windows"] == 0).sum())
    LOGGER.info("Source files .................. %d", len(usable))
    LOGGER.info("Files yielding 0 windows ...... %d", zero_yield)
    LOGGER.info("Total windows ................. %d", total_windows)
    LOGGER.info(
        "Audio retained ................ %.1f s of %.1f s (%.1f%%)",
        usable["covered_samples"].sum() / config.SAMPLE_RATE,
        usable["resampled_samples"].sum() / config.SAMPLE_RATE,
        100.0
        * usable["covered_samples"].sum()
        / max(int(usable["resampled_samples"].sum()), 1),
    )
    LOGGER.info(
        "Windows per file .............. min %d, median %.1f, max %d",
        usable["n_windows"].min(),
        usable["n_windows"].median(),
        usable["n_windows"].max(),
    )

    LOGGER.info("\nWindows per class:")
    LOGGER.info(
        indent(
            usable.groupby("label")["n_windows"]
            .agg(["sum", "mean", "min", "max"])
            .round(2)
            .to_string()
        )
    )

    float_mb = total_windows * config.WINDOW_SAMPLES * 4 / 1e6
    int16_mb = total_windows * config.WINDOW_SAMPLES * 2 / 1e6
    embed_mb = total_windows * config.EMBEDDING_DIM * 4 / 1e6
    LOGGER.info("\nProjected on-disk footprint after Step 2:")
    LOGGER.info("  waveform chunks, int16 ...... %8.1f MB", int16_mb)
    LOGGER.info("  waveform chunks, float32 .... %8.1f MB", float_mb)
    LOGGER.info("  cached YAMNet embeddings .... %8.1f MB", embed_mb)

    if total_windows < 1000:
        issues.append(
            f"only {total_windows} windows projected - thin for training a "
            "classification head; augmentation in Step 2 will matter"
        )
    return issues


def indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def write_figures(frame: pd.DataFrame, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    usable = frame.loc[frame["readable"] & frame["filename_conforms"]]
    if usable.empty:
        return []

    written: list[Path] = []

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(usable["duration_sec"], bins=40, color="#3b7dd8", edgecolor="white")
    axes[0].axvline(
        config.WINDOW_SECONDS, color="#d1495b", linestyle="--",
        label=f"{config.WINDOW_SECONDS} s window",
    )
    axes[0].set_xlabel("duration (s)")
    axes[0].set_ylabel("files")
    axes[0].set_title("Clip duration distribution")
    axes[0].legend()

    label_counts = usable["label"].value_counts()
    axes[1].bar(label_counts.index, label_counts.to_numpy(), color="#3b7dd8")
    axes[1].set_ylabel("files")
    axes[1].set_title("Class distribution")
    figure.tight_layout()
    path = out_dir / "audit_durations_classes.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    written.append(path)

    crosstab = pd.crosstab(usable["fold_id"], usable["label"])
    figure, axis = plt.subplots(figsize=(7, 0.5 * len(crosstab) + 2.5))
    image = axis.imshow(crosstab.to_numpy(), aspect="auto", cmap="Blues")
    axis.set_xticks(range(len(crosstab.columns)), crosstab.columns)
    axis.set_yticks(range(len(crosstab.index)), crosstab.index)
    axis.set_xlabel("class")
    axis.set_ylabel("fold")
    axis.set_title("Files per class x fold")
    for row in range(crosstab.shape[0]):
        for column in range(crosstab.shape[1]):
            axis.text(column, row, crosstab.iat[row, column],
                      ha="center", va="center", fontsize=9)
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    path = out_dir / "audit_class_fold_matrix.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    written.append(path)

    return written


# ---------------------------------------------------------------------------
# Missing-dataset guidance
# ---------------------------------------------------------------------------


def report_missing_dataset(raw_dir: Path) -> None:
    section("SAFE DATASET NOT FOUND")
    LOGGER.info("Searched: %s", raw_dir)
    LOGGER.info("Exists ....... %s", raw_dir.exists())
    if raw_dir.exists():
        entries = sorted(item.name for item in raw_dir.iterdir())
        LOGGER.info("Contents ..... %s", entries if entries else "(empty)")
    LOGGER.info("Accepted extensions: %s", ", ".join(config.AUDIO_EXTENSIONS))
    LOGGER.info(
        "\nPlace the SAFE corpus under data/raw/ (nested folders are fine -- the\n"
        "scan is recursive) using the documented naming scheme, e.g.:\n"
        "\n"
        "    data/raw/01-001-01-001-01.wav      <- Fall,     fold 01\n"
        "    data/raw/02-014-03-002-04.wav      <- Non-Fall, fold 04\n"
        "\n"
        "then re-run:  python src/1_dataset_audit.py"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the raw SAFE dataset.")
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=config.OUTPUTS_DIR)
    parser.add_argument(
        "--no-deep",
        dest="deep",
        action="store_false",
        help="skip full decode; read container headers only",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="inspect only the first N files"
    )
    parser.add_argument("--no-figures", dest="figures", action="store_false")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config.ensure_dirs()
    setup_logging(config.LOGS_DIR / "dataset_audit.log")

    section("SAFE DATASET AUDIT  -  Step 1")
    LOGGER.info("Project root .... %s", config.PROJECT_ROOT)
    LOGGER.info("Raw data ........ %s", args.raw_dir)
    LOGGER.info("Deep scan ....... %s", "on (full decode)" if args.deep else "off")

    paths = audio_utils.find_audio_files(args.raw_dir)
    if not paths:
        report_missing_dataset(args.raw_dir)
        section("RESULT: BLOCKED - no audio to audit")
        return 2

    if args.limit > 0:
        LOGGER.info("Limiting to first %d of %d file(s).", args.limit, len(paths))
        paths = paths[: args.limit]

    frame = build_manifest(paths, deep=args.deep)

    issues: list[str] = []
    issues += report_integrity(frame, deep=args.deep)
    issues += report_naming(frame)
    issues += report_distribution(frame)
    issues += report_windows(frame)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "dataset_audit_manifest.csv"
    frame.to_csv(manifest_path, index=False)

    usable = frame.loc[frame["readable"] & frame["filename_conforms"]]
    summary = {
        "raw_dir": str(args.raw_dir),
        "deep_scan": args.deep,
        "files_found": len(frame),
        "files_readable": int(frame["readable"].sum()),
        "files_conforming": int(frame["filename_conforms"].sum()),
        "files_usable": len(usable),
        "total_audio_sec": float(frame.loc[frame["readable"], "duration_sec"].sum()),
        "class_file_counts": usable["label"].value_counts().to_dict(),
        "class_window_counts": usable.groupby("label")["n_windows"].sum().to_dict(),
        "fold_file_counts": usable["fold_id"].value_counts().sort_index().to_dict(),
        "total_windows": int(usable["n_windows"].sum()),
        "window_samples": config.WINDOW_SAMPLES,
        "hop_samples": config.HOP_SAMPLES,
        "sample_rate": config.SAMPLE_RATE,
        "issues": issues,
    }
    summary_path = args.out_dir / "dataset_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figures_dir = (
        config.FIGURES_DIR if args.out_dir == config.OUTPUTS_DIR
        else args.out_dir / "figures"
    )
    figure_paths = write_figures(frame, figures_dir) if args.figures else []

    section("ARTEFACTS")
    LOGGER.info("manifest ... %s", manifest_path)
    LOGGER.info("summary .... %s", summary_path)
    for path in figure_paths:
        LOGGER.info("figure ..... %s", path)
    LOGGER.info("log ........ %s", config.LOGS_DIR / "dataset_audit.log")

    section("RESULT")
    if issues:
        LOGGER.info("%d issue(s) require attention before Step 2:", len(issues))
        for index, issue in enumerate(issues, start=1):
            LOGGER.info("  %d. %s", index, issue)
        return 1

    LOGGER.info("PASS - corpus is ready for Step 2 (preprocessing).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
