"""
Step 6c — Download MUSAN Noise Subset.

MUSAN (Music, Speech, and Noise) is a large corpus by D. Snyder et al.
We only need the NOISE portion (~500 MB) for SNR-controlled augmentation.

Full MUSAN: ~11 GB (openslr.org/17/)
Noise-only subset: downloaded individually from the known subdirectory structure.

License: CC-BY 4.0 + US Public Domain — fully commercial-safe.
Format:  16 kHz, 16-bit PCM WAV — perfect direct match for YAMNet pipeline.

Usage:
    C:\\aesv\\Scripts\\python.exe src\\6c_download_musan.py
"""

from __future__ import annotations

import logging
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_phase2 as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("download_musan")

# ---------------------------------------------------------------------------
# Download targets
# ---------------------------------------------------------------------------
# MUSAN full archive (noise + music + speech = ~11 GB)
# We download only the noise tarball if a partial URL exists,
# otherwise we stream-extract only the noise/ subdirectory from the full archive.

MUSAN_FULL_URL  = "https://www.openslr.org/resources/17/musan.tar.gz"
MUSAN_FULL_TAR  = cfg.MUSAN_DIR / "musan.tar.gz"

# Alternative: a smaller noise-only mirror sometimes available
MUSAN_NOISE_ALT = "https://www.openslr.org/resources/17/musan.tar.gz"

# DCASE DESED (domestic environment sounds, key bathroom confusers)
# CC-BY 4.0 — 16 kHz WAV — strongly/weakly labeled 10-sec clips
DESED_REPO_URL   = "https://zenodo.org/record/6026743/files/DESED_public_eval.zip"
DESED_SYNTH_URL  = "https://zenodo.org/record/4560095/files/DESED_synth_dcase20_train.zip"
DESED_EVAL_ZIP   = cfg.DESED_DIR / "DESED_public_eval.zip"
DESED_SYNTH_ZIP  = cfg.DESED_DIR / "DESED_synth.zip"

# DESED target classes we care about (domestic bathroom confusers)
DESED_TARGET_CLASSES = {
    "Alarm_bell_ringing",
    "Running_water",
    "Dishes",
    "Electric_shaver_toothbrush",
    "Speech",
    "Vacuum_cleaner",
    "Blender",
    "Frying",
    "Cat",
    "Dog",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _progress_hook(label: str):
    def _hook(count, block, total):
        if total > 0:
            pct = min(100, int(count * block * 100 / total))
            mb_done = count * block / 1e6
            print(f"\r    [{label}] {pct}%  {mb_done:.1f} MB", end="", flush=True)
    return _hook


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download url → dest. Returns True on success."""
    if dest.exists() and dest.stat().st_size > 1_000:
        log.info("  ✓ Already present: %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    label = desc or dest.name
    log.info("  ↓ %s", label)
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress_hook(desc[:20]))
        print()
        log.info("    Saved → %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
        return True
    except Exception as exc:
        print()
        log.error("    FAILED: %s — %s", label, exc)
        if dest.exists():
            dest.unlink(missing_ok=True)
        return False


# ---------------------------------------------------------------------------
# MUSAN noise extraction
# ---------------------------------------------------------------------------


def extract_musan_noise_only(tar_path: Path) -> int:
    """Extract only the noise/ subdirectory from the full MUSAN tarball."""
    noise_dest = cfg.MUSAN_NOISE_DIR
    noise_dest.mkdir(parents=True, exist_ok=True)

    existing = list(noise_dest.rglob("*.wav"))
    if len(existing) > 100:
        log.info("  MUSAN noise already extracted: %d WAV files", len(existing))
        return len(existing)

    log.info("  Extracting noise/ subset from MUSAN tarball …")
    count = 0
    try:
        with tarfile.open(tar_path, "r:gz") as tf:
            members = [m for m in tf.getmembers() if "musan/noise/" in m.name]
            log.info("  Found %d noise entries in tarball", len(members))
            for member in members:
                # Strip leading path to put files directly under MUSAN_NOISE_DIR
                rel = member.name.replace("musan/noise/", "")
                if not rel:
                    continue
                dest_path = noise_dest / rel
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                if dest_path.exists():
                    continue
                member.name = rel
                tf.extract(member, noise_dest)
                count += 1
                if count % 50 == 0:
                    log.info("    … %d files extracted", count)
    except Exception as exc:
        log.error("  Extraction error: %s", exc)
        return 0

    log.info("  Extracted %d MUSAN noise files → %s", count, noise_dest)
    return count


# ---------------------------------------------------------------------------
# DESED download
# ---------------------------------------------------------------------------


def download_desed() -> bool:
    """Download DESED public eval set (domestic environment sounds)."""
    log.info("\n[2/2] DESED domestic environment sound event dataset")
    log.info("  Target: bathroom confounders (running water, shaver, alarm, speech)")

    # The DESED synth train set is the most useful because it's strongly labeled
    # and available as direct WAV files (not requiring YouTube download).
    # It's ~500 MB and contains 10-second 16 kHz clips.

    if any(cfg.DESED_AUDIO_DIR.rglob("*.wav")):
        count = len(list(cfg.DESED_AUDIO_DIR.rglob("*.wav")))
        log.info("  ✓ DESED already present: %d WAV files", count)
        return True

    # Try synth training set first (no YouTube dependency)
    desed_urls = [
        ("DESED synth train (strongly labeled, ~500 MB)", DESED_SYNTH_URL, DESED_SYNTH_ZIP),
        ("DESED public eval (~200 MB)", DESED_REPO_URL, DESED_EVAL_ZIP),
    ]

    success = False
    for desc, url, dest_zip in desed_urls:
        log.info("  Downloading %s …", desc)
        if download_file(url, dest_zip, desc):
            import zipfile
            log.info("  Unpacking %s …", dest_zip.name)
            try:
                with zipfile.ZipFile(dest_zip, "r") as zf:
                    zf.extractall(cfg.DESED_DIR)
                log.info("  Unpacked → %s", cfg.DESED_DIR)
                success = True
            except Exception as exc:
                log.warning("  Unpack error: %s", exc)

    if not success:
        log.warning("  DESED download failed. Will skip DESED in preprocessing.")
        log.warning("  You can manually download DESED from: https://zenodo.org/record/4560095")

    return success


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    cfg.ensure_p2_dirs()
    log.info("=" * 60)
    log.info("Step 6c — MUSAN Noise + DESED Domestic Sounds")
    log.info("=" * 60)

    # ── MUSAN noise ────────────────────────────────────────────────────────
    log.info("\n[1/2] MUSAN noise subset (CC-BY 4.0, 16 kHz, ~500 MB extracted)")

    existing_noise = list(cfg.MUSAN_NOISE_DIR.rglob("*.wav"))
    if len(existing_noise) > 100:
        log.info("  ✓ MUSAN noise already present: %d WAV files", len(existing_noise))
    else:
        log.info("  Downloading full MUSAN archive to extract noise (~11 GB full, ~500 MB noise)")
        log.info("  This may take 15–30 minutes depending on connection speed.")
        log.info("  (Only noise/ will be extracted — music/speech portions are skipped)")

        ok = download_file(MUSAN_FULL_URL, MUSAN_FULL_TAR, "MUSAN (full archive)")
        if ok:
            n = extract_musan_noise_only(MUSAN_FULL_TAR)
            log.info("  MUSAN noise extraction complete: %d files", n)
        else:
            log.error("  MUSAN download failed. Augmentation will proceed without MUSAN noise.")

    # ── DESED ──────────────────────────────────────────────────────────────
    download_desed()

    # ── Summary ────────────────────────────────────────────────────────────
    noise_count = len(list(cfg.MUSAN_NOISE_DIR.rglob("*.wav")))
    desed_count = len(list(cfg.DESED_DIR.rglob("*.wav")))

    log.info("\n" + "=" * 60)
    log.info("Done.")
    log.info("  MUSAN noise files : %d  → %s", noise_count, cfg.MUSAN_NOISE_DIR)
    log.info("  DESED audio files : %d  → %s", desed_count, cfg.DESED_DIR)
    log.info("  Next: run 6d_download_freesound.py (needs API key)")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
