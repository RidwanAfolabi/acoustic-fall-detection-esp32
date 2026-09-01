"""
Step 6e — FSD50K Corpus Audit: filter to CC0 / CC-BY clips only.

Reads the FSD50K metadata JSON files (included with the Zenodo download) and
produces a filtered manifest CSV that lists only commercially-safe clips
mapping to Phase 2 emergency or normal labels.

FSD50K must already be downloaded from:
  https://zenodo.org/record/4060432  (~21 GB dev + ~5 GB eval)

Unpack structure expected:
  data/external/fsd50k/
    dev/audio/           ← 40,966 WAV files
    eval/audio/          ← 10,231 WAV files
    FSD50K.ground_truth/
      dev.csv
      eval.csv
    FSD50K.metadata/
      dev_clips_info_FSD50K.json
      eval_clips_info_FSD50K.json
      vocabulary.csv

Output:
  data/external/fsd50k/commercial_clips.csv
  Columns: clip_id, split, wav_path, label, fsd50k_classes, license

Usage:
    C:\\aesv\\Scripts\\python.exe src\\6e_audit_fsd50k.py [--stats]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_phase2 as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("audit_fsd50k")

# ---------------------------------------------------------------------------
# License constants (as they appear in FSD50K metadata JSON)
# ---------------------------------------------------------------------------

COMMERCIAL_LICENSES = frozenset({
    "Creative Commons 0",
    "CC0",
    "Attribution",
    "Attribution 4.0",
    "Creative Commons Attribution 4.0",
    "CC BY",
    "CC BY 4.0",
    "CC-BY",
})

NON_COMMERCIAL_LICENSES = frozenset({
    "Attribution NonCommercial",
    "Attribution NonCommercial 4.0",
    "Attribution-NonCommercial",
    "CC BY-NC",
    "CC BY-NC 4.0",
    "Sampling+",
    "CC Sampling+",
})


def is_commercial_safe(license_str: str) -> bool:
    """Return True if license allows commercial use."""
    lic = license_str.strip()
    # Check for non-commercial first (more specific)
    if any(nc in lic for nc in ["NonCommercial", "NC", "Sampling+"]):
        return False
    # Then check for known commercial licenses
    if any(c in lic for c in ["CC0", "CC BY", "Attribution", "Creative Commons 0"]):
        return True
    # Unknown: be conservative and exclude
    log.debug("  Unknown license string, excluding: %r", lic)
    return False


# ---------------------------------------------------------------------------
# FSD50K metadata parsing
# ---------------------------------------------------------------------------


def load_clip_info(json_path: Path) -> dict[str, dict]:
    """Load clip metadata from FSD50K JSON info file.

    Returns a dict keyed by clip_id (str) with keys: license, title, etc.
    """
    if not json_path.exists():
        log.warning("Metadata file not found: %s", json_path)
        return {}

    with open(json_path, encoding="utf-8") as fh:
        data = json.load(fh)

    # FSD50K JSON structure: { "<clip_id>": { "license": "...", ... }, ... }
    return {str(k): v for k, v in data.items()}


def load_ground_truth(csv_path: Path) -> dict[str, list[str]]:
    """Load ground truth CSV from FSD50K.

    Returns dict: clip_id → list of class names.
    The FSD50K ground truth CSV has columns: fname, labels, mids, split
    where labels is a comma-separated list of AudioSet class names.
    """
    if not csv_path.exists():
        log.warning("Ground truth file not found: %s", csv_path)
        return {}

    gt: dict[str, list[str]] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fname = Path(row.get("fname", "")).stem  # clip_id is the filename stem
            labels_raw = row.get("labels", "")
            classes = [c.strip() for c in labels_raw.split(",") if c.strip()]
            if fname:
                gt[fname] = classes
    return gt


def assign_p2_label(class_names: list[str]) -> str | None:
    """Map a list of FSD50K class names to a Phase 2 binary label.

    Returns 'emergency', 'normal', or None (if ambiguous / both classes present).
    """
    has_emergency = any(c in cfg.FSD50K_EMERGENCY_CLASSES for c in class_names)
    has_normal = any(c in cfg.FSD50K_NORMAL_CLASSES for c in class_names)

    if has_emergency and has_normal:
        return None  # Exclude ambiguous multi-label clips
    if has_emergency:
        return "emergency"
    if has_normal:
        return "normal"
    return None  # Not relevant to this project


# ---------------------------------------------------------------------------
# Audit one split (dev or eval)
# ---------------------------------------------------------------------------


def audit_split(
    split: str,
    audio_dir: Path,
    gt_csv: Path,
    info_json: Path,
) -> list[dict]:
    """Return a list of accepted clip records for this split."""
    log.info("\n--- Auditing split: %s ---", split)

    gt = load_ground_truth(gt_csv)
    info = load_clip_info(info_json)

    log.info("  Ground truth entries: %d", len(gt))
    log.info("  Clip info entries:    %d", len(info))

    records = []
    stats = {
        "total": 0,
        "non_commercial": 0,
        "no_label_match": 0,
        "ambiguous": 0,
        "missing_wav": 0,
        "accepted": 0,
    }

    for clip_id, class_names in gt.items():
        stats["total"] += 1

        # 1. License check
        clip_meta = info.get(clip_id, {})
        license_str = clip_meta.get("license", clip_meta.get("clip_license", ""))
        if not is_commercial_safe(license_str):
            stats["non_commercial"] += 1
            continue

        # 2. Label mapping
        p2_label = assign_p2_label(class_names)
        if p2_label is None:
            if any(c in cfg.FSD50K_EMERGENCY_CLASSES for c in class_names) and \
               any(c in cfg.FSD50K_NORMAL_CLASSES for c in class_names):
                stats["ambiguous"] += 1
            else:
                stats["no_label_match"] += 1
            continue

        # 3. WAV file existence
        # FSD50K uses the numeric clip ID as the filename (e.g., 12345.wav)
        wav_path = audio_dir / f"{clip_id}.wav"
        if not wav_path.exists():
            # Try the split subdirectory structure
            alt_paths = list(audio_dir.rglob(f"{clip_id}.wav"))
            if alt_paths:
                wav_path = alt_paths[0]
            else:
                stats["missing_wav"] += 1
                continue

        records.append({
            "clip_id": clip_id,
            "split": split,
            "wav_path": str(wav_path),
            "label": p2_label,
            "fsd50k_classes": "|".join(class_names),
            "license": license_str,
        })
        stats["accepted"] += 1

    log.info("  %-18s %6d", "Total clips:", stats["total"])
    log.info("  %-18s %6d  (%.1f%%)", "Non-commercial:", stats["non_commercial"],
             100 * stats["non_commercial"] / max(1, stats["total"]))
    log.info("  %-18s %6d", "No label match:", stats["no_label_match"])
    log.info("  %-18s %6d", "Ambiguous:", stats["ambiguous"])
    log.info("  %-18s %6d", "Missing WAV:", stats["missing_wav"])
    log.info("  %-18s %6d  (%.1f%%)", "Accepted:", stats["accepted"],
             100 * stats["accepted"] / max(1, stats["total"]))

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stats", action="store_true", help="Print detailed class distribution after audit")
    p.add_argument("--fsd50k-dir", type=Path, default=cfg.FSD50K_DIR,
                   help="Path to FSD50K root directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg.ensure_p2_dirs()

    fsd_root = args.fsd50k_dir
    log.info("=" * 60)
    log.info("Step 6e — FSD50K Commercial Clip Audit")
    log.info("  FSD50K root: %s", fsd_root)
    log.info("=" * 60)

    if not fsd_root.exists():
        log.error("FSD50K directory not found: %s", fsd_root)
        log.error("Please download FSD50K from: https://zenodo.org/record/4060432")
        log.error("Expected structure:")
        log.error("  %s/dev/audio/", fsd_root)
        log.error("  %s/eval/audio/", fsd_root)
        log.error("  %s/FSD50K.ground_truth/dev.csv", fsd_root)
        log.error("  %s/FSD50K.metadata/dev_clips_info_FSD50K.json", fsd_root)
        sys.exit(1)

    # Paths inside FSD50K directory
    dev_audio   = fsd_root / "dev" / "audio"
    eval_audio  = fsd_root / "eval" / "audio"
    dev_gt      = fsd_root / "FSD50K.ground_truth" / "dev.csv"
    eval_gt     = fsd_root / "FSD50K.ground_truth" / "eval.csv"
    dev_info    = fsd_root / "FSD50K.metadata" / "dev_clips_info_FSD50K.json"
    eval_info   = fsd_root / "FSD50K.metadata" / "eval_clips_info_FSD50K.json"

    all_records: list[dict] = []

    if dev_gt.exists() and dev_info.exists():
        all_records.extend(audit_split("dev", dev_audio, dev_gt, dev_info))
    else:
        log.warning("Dev split metadata not found — skipping")

    if eval_gt.exists() and eval_info.exists():
        all_records.extend(audit_split("eval", eval_audio, eval_gt, eval_info))
    else:
        log.warning("Eval split metadata not found — skipping")

    if not all_records:
        log.error("No records accepted. Check FSD50K directory structure.")
        sys.exit(1)

    # Write manifest
    manifest_path = cfg.FSD50K_MANIFEST
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["clip_id", "split", "wav_path", "label",
                                                 "fsd50k_classes", "license"])
        writer.writeheader()
        writer.writerows(all_records)

    # Summary
    emergency_count = sum(1 for r in all_records if r["label"] == "emergency")
    normal_count    = sum(1 for r in all_records if r["label"] == "normal")

    log.info("\n" + "=" * 60)
    log.info("FSD50K audit complete.")
    log.info("  Total accepted:  %d clips", len(all_records))
    log.info("  Emergency label: %d clips", emergency_count)
    log.info("  Normal label:    %d clips", normal_count)
    log.info("  Manifest saved:  %s", manifest_path)

    if args.stats:
        from collections import Counter
        classes: list[str] = []
        for r in all_records:
            classes.extend(r["fsd50k_classes"].split("|"))
        top = Counter(classes).most_common(30)
        log.info("\n  Top 30 FSD50K classes in accepted clips:")
        for cls, cnt in top:
            log.info("    %-45s %5d", cls, cnt)

    log.info("\n  Next: run 7_preprocess_external.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
