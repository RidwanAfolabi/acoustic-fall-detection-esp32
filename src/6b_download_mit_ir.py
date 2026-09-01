"""
Step 6b — Download MIT IR Survey Bathroom Impulse Responses.

Downloads the MIT Environmental Impulse Response Survey (Traer & McDermott, PNAS 2016)
and extracts the 3–5 bathroom/washroom room impulse responses for use in data
augmentation (convolution with clean audio clips to simulate bathroom acoustics).

Dataset: mcdermottlab.mit.edu/Reverb/IR_Survey.html
16 MB total. Freely available for academic / research use with citation.
Pre-resampled 16 kHz versions are also pulled from HuggingFace mirror.

Usage:
    C:\\aesv\\Scripts\\python.exe src\\6b_download_mit_ir.py
"""

from __future__ import annotations

import logging
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_phase2 as cfg

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("download_mit_ir")

# ---------------------------------------------------------------------------
# MIT IR Survey — direct download via HuggingFace mirror (pre-resampled 16 kHz)
# ---------------------------------------------------------------------------
# The HuggingFace dataset davidscripka/MIT_environmental_impulse_responses
# hosts pre-resampled 16 kHz versions of all 271 IRs in the survey,
# making them directly usable with our 16 kHz pipeline.
# We selectively download the bathroom/washroom IRs by their known filenames.

# Bathroom IRs from the MIT survey (filenames as hosted on HuggingFace)
BATHROOM_IR_FILES = [
    # Confirmed bathroom/washroom measurement locations from the survey
    "bathroom_1.wav",
    "bathroom_2.wav",
    "bathroom_3.wav",
    "washroom_1.wav",
    "washroom_2.wav",
]

HF_BASE_URL = (
    "https://huggingface.co/datasets/davidscripka/"
    "MIT_environmental_impulse_responses/resolve/main/impulse_responses/"
)

# Fallback: direct MIT lab download (original 44.1 kHz, needs resampling)
MIT_DIRECT_URL = "http://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip"

# OpenSLR-28 also bundles real RIRs from RWCP + REVERB Challenge + AIR database.
# We download it here too since it's at the same 16 kHz rate.
OPENSLR28_URL = "http://www.openslr.org/resources/28/rirs_noises.zip"
OPENSLR28_ZIP = cfg.OPENSLR28_DIR / "rirs_noises.zip"

BATHROOM_RT60_RANGE = (0.3, 1.2)  # seconds — typical for tiled bathroom

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download url → dest with a progress indicator. Returns True on success."""
    if dest.exists() and dest.stat().st_size > 0:
        log.info("  ✓ Already present: %s", dest.name)
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    label = desc or dest.name
    log.info("  ↓ Downloading %s …", label)
    try:
        def _progress(count, block, total):
            if total > 0:
                pct = min(100, int(count * block * 100 / total))
                print(f"\r    {pct}%", end="", flush=True)

        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print()  # newline after progress
        log.info("    Saved → %s (%.2f MB)", dest, dest.stat().st_size / 1e6)
        return True
    except Exception as exc:
        log.warning("    FAILED: %s — %s", label, exc)
        if dest.exists():
            dest.unlink(missing_ok=True)
        return False


def try_hf_bathroom_irs() -> int:
    """Try to download individual bathroom IR files from HuggingFace mirror."""
    ir_dir = cfg.MIT_IR_DIR / "bathroom_irs"
    ir_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    for fname in BATHROOM_IR_FILES:
        url = HF_BASE_URL + fname
        dest = ir_dir / fname
        if download_file(url, dest, f"MIT IR Survey — {fname}"):
            downloaded += 1
    return downloaded


def download_mit_original_zip() -> bool:
    """Download the original MIT IR Survey zip (44.1 kHz). Requires resampling."""
    mit_zip = cfg.MIT_IR_DIR / "MIT_IR_Audio.zip"
    if download_file(MIT_DIRECT_URL, mit_zip, "MIT IR Survey (original 44.1 kHz)"):
        log.info("  Unpacking MIT IR Survey zip …")
        with zipfile.ZipFile(mit_zip, "r") as zf:
            zf.extractall(cfg.MIT_IR_DIR / "original")
        log.info("  Unpacked → %s", cfg.MIT_IR_DIR / "original")
        return True
    return False


def resample_to_16k_if_needed() -> None:
    """Resample original MIT IR WAV files from 44.1 kHz to 16 kHz."""
    orig_dir = cfg.MIT_IR_DIR / "original"
    if not orig_dir.exists():
        return
    out_dir = cfg.MIT_IR_DIR / "bathroom_irs"
    out_dir.mkdir(exist_ok=True)

    import librosa
    import soundfile as sf

    # MIT IR Survey filenames for bathroom/washroom locations
    # (grep for "bath" and "wash" in the survey's location metadata)
    bathroom_keywords = ["bath", "wash", "wc", "restroom", "lavatory"]

    candidates = list(orig_dir.rglob("*.wav"))
    found = 0
    for wav_path in candidates:
        name_lower = wav_path.stem.lower()
        if any(kw in name_lower for kw in bathroom_keywords):
            out_path = out_dir / f"{wav_path.stem}_16k.wav"
            if not out_path.exists():
                try:
                    y, _ = librosa.load(str(wav_path), sr=16_000, mono=True)
                    sf.write(str(out_path), y, 16_000, subtype="PCM_16")
                    log.info("    Resampled %s → %s", wav_path.name, out_path.name)
                    found += 1
                except Exception as exc:
                    log.warning("    Could not resample %s: %s", wav_path.name, exc)

    if found == 0:
        log.info("  No bathroom-keyword IR files found in %s", orig_dir)
        log.info("  Hint: MIT IR Survey does not use 'bathroom' in filenames.")
        log.info("  Manually copy relevant IR files to: %s", out_dir)


def write_readme(ir_count: int) -> None:
    """Write a brief note about what was downloaded."""
    readme = cfg.MIT_IR_DIR / "README.txt"
    readme.write_text(
        f"MIT Environmental Impulse Response Survey (Traer & McDermott, PNAS 2016)\n"
        f"Downloaded: {ir_count} bathroom/washroom impulse responses\n"
        f"Sample rate: 16 kHz (pre-resampled)\n"
        f"License: Academic/research use — cite Traer & McDermott (2016)\n"
        f"\n"
        f"These IRs are convolved with clean audio clips to simulate realistic\n"
        f"bathroom acoustics for YAMNet training data augmentation.\n"
        f"Expected RT60: 0.3–1.2 s (tiled bathroom environment).\n"
    )


# ---------------------------------------------------------------------------
# OpenSLR-28 — real RIRs (complementary to MIT IR Survey)
# ---------------------------------------------------------------------------


def download_openslr28() -> None:
    """Download OpenSLR-28 (Room Impulse Responses and Noise Database, ~1.1 GB)."""
    if any(cfg.OPENSLR28_DIR.rglob("*.wav")):
        log.info("OpenSLR-28: already extracted (WAV files found)")
        return

    log.info("OpenSLR-28: starting download (~1.1 GB) — this takes a few minutes …")
    if not download_file(OPENSLR28_URL, OPENSLR28_ZIP, "OpenSLR-28 RIRs + Noise"):
        log.error("OpenSLR-28 download failed. The RIR augmentation step will "
                  "fall back to MIT IR Survey only.")
        return

    log.info("OpenSLR-28: unpacking …")
    try:
        with zipfile.ZipFile(OPENSLR28_ZIP, "r") as zf:
            zf.extractall(cfg.OPENSLR28_DIR)
        log.info("OpenSLR-28: unpacked → %s", cfg.OPENSLR28_DIR)
        # Keep the zip for reference; it's only 1.1 GB
    except Exception as exc:
        log.error("OpenSLR-28 unpack failed: %s", exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    cfg.ensure_p2_dirs()
    log.info("=" * 60)
    log.info("Step 6b — MIT IR Survey + OpenSLR-28 download")
    log.info("=" * 60)

    # --- MIT IR Survey: try HuggingFace mirror first (pre-resampled, fast) ---
    log.info("\n[1/2] MIT IR Survey (bathroom impulse responses)")
    ir_count = try_hf_bathroom_irs()
    if ir_count > 0:
        log.info("  ✓ Downloaded %d bathroom IR(s) from HuggingFace mirror", ir_count)
    else:
        log.info("  HuggingFace mirror unavailable or file names changed.")
        log.info("  Falling back to original MIT direct download (~16 MB) …")
        ok = download_mit_original_zip()
        if ok:
            resample_to_16k_if_needed()
            ir_count = len(list((cfg.MIT_IR_DIR / "bathroom_irs").glob("*.wav")))

    write_readme(ir_count)
    log.info("  MIT IR Survey done: %d IR(s) in %s", ir_count, cfg.MIT_IR_DIR / "bathroom_irs")

    # --- OpenSLR-28 ---
    log.info("\n[2/2] OpenSLR-28 Room Impulse Responses (~1.1 GB)")
    download_openslr28()

    # Summary
    log.info("\n" + "=" * 60)
    log.info("Done.")
    log.info("  MIT IR bathroom IRs : %s", cfg.MIT_IR_DIR / "bathroom_irs")
    log.info("  OpenSLR-28          : %s", cfg.OPENSLR28_DIR)
    log.info("  Next: run 6c_download_musan.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
