"""
Step 6d — Freesound API Targeted Download (CC0 and CC-BY only).

Searches Freesound.org for specific sound categories and downloads clips
that are licensed CC0 ("Creative Commons 0") or CC-BY ("Attribution") only.
These are safe for commercial use.

Target sounds:
  Emergency class: screams, groans, yells, cries, thuds, body impacts
  Normal class:    toilet flush, running water, door slam, object drops

Setup:
    1. Register at https://freesound.org and apply for an API key:
       https://freesound.org/apiv2/apply
    2. Set your API key:
       Option A (recommended): set environment variable FREESOUND_API_KEY
       Option B: pass --api-key <key> on the command line
       Option C: store it in a file .freesound_key in the project root

Rate limits:
    60 requests / minute, 2000 requests / day.
    This script rate-limits itself automatically.

Download format:
    MP3 high-quality previews (128 kbps, 44.1 kHz) using API key only —
    no OAuth2 needed for previews. Previews are then resampled to 16 kHz WAV.

    For full original WAV quality, OAuth2 is required (add --oauth2 flag
    and follow the browser authentication flow). Previews are sufficient
    for training data.

Usage:
    C:\\aesv\\Scripts\\python.exe src\\6d_download_freesound.py
    C:\\aesv\\Scripts\\python.exe src\\6d_download_freesound.py --api-key YOUR_KEY
    C:\\aesv\\Scripts\\python.exe src\\6d_download_freesound.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_phase2 as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("freesound_download")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FREESOUND_API_BASE = "https://freesound.org/apiv2"
PAGE_SIZE = 150            # max allowed by Freesound API
REQUESTS_PER_MINUTE = 55  # slightly under limit for safety
REQUEST_INTERVAL = 60.0 / REQUESTS_PER_MINUTE   # ~1.09 s between requests
RESAMPLE_SR = 16_000


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def get_api_key(args_key: str | None) -> str:
    """Resolve API key: args → env var → .freesound_key file → prompt."""
    if args_key:
        return args_key

    env_key = os.environ.get("FREESOUND_API_KEY", "").strip()
    if env_key:
        log.info("Using FREESOUND_API_KEY from environment variable")
        return env_key

    key_file = Path(__file__).resolve().parent.parent / ".freesound_key"
    if key_file.exists():
        key = key_file.read_text().strip()
        if key:
            log.info("Using API key from %s", key_file)
            return key

    log.warning("No Freesound API key found!")
    log.warning("Please set FREESOUND_API_KEY environment variable, or")
    log.warning("pass --api-key <key>, or save key to: %s", key_file)
    log.warning("Register for a free key at: https://freesound.org/apiv2/apply")
    sys.exit(1)


def freesound_search(
    query: str,
    api_key: str,
    page: int = 1,
    page_size: int = PAGE_SIZE,
) -> dict:
    """Perform a Freesound text search with CC0/CC-BY license filter."""
    params = {
        "query": query,
        "filter": cfg.FREESOUND_LICENSE_FILTER,
        "fields": cfg.FREESOUND_FIELDS,
        "page": page,
        "page_size": page_size,
        "sort": "score",
        "token": api_key,
    }
    url = f"{FREESOUND_API_BASE}/search/text/?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        log.warning("  Search request failed: %s", exc)
        return {"results": [], "count": 0}


def download_preview(sound_info: dict, dest_dir: Path) -> Path | None:
    """Download the HQ MP3 preview for a Freesound clip."""
    previews = sound_info.get("previews", {})
    preview_url = previews.get("preview-hq-mp3") or previews.get("preview-lq-mp3")
    if not preview_url:
        return None

    sound_id = sound_info["id"]
    sound_name = sound_info.get("name", str(sound_id))
    # Sanitize filename
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in sound_name)
    safe_name = safe_name[:60]
    dest_mp3 = dest_dir / f"fs_{sound_id}_{safe_name}.mp3"

    if dest_mp3.exists() and dest_mp3.stat().st_size > 1000:
        return dest_mp3

    try:
        urllib.request.urlretrieve(preview_url, dest_mp3)
        return dest_mp3
    except Exception as exc:
        log.debug("  Preview download failed for %s: %s", sound_id, exc)
        return None


def mp3_to_wav_16k(mp3_path: Path) -> Path | None:
    """Convert MP3 preview to 16 kHz mono WAV using librosa + soundfile."""
    wav_path = mp3_path.with_suffix(".wav")
    if wav_path.exists() and wav_path.stat().st_size > 1000:
        return wav_path
    try:
        import librosa
        import soundfile as sf
        y, _ = librosa.load(str(mp3_path), sr=RESAMPLE_SR, mono=True)
        sf.write(str(wav_path), y, RESAMPLE_SR, subtype="PCM_16")
        mp3_path.unlink(missing_ok=True)  # Remove MP3 after conversion
        return wav_path
    except Exception as exc:
        log.debug("  MP3→WAV conversion failed for %s: %s", mp3_path.name, exc)
        return None


# ---------------------------------------------------------------------------
# Per-query download loop
# ---------------------------------------------------------------------------


def download_query(
    query_spec: dict,
    api_key: str,
    dry_run: bool = False,
) -> int:
    """Download up to max_clips for a single query spec. Returns count of WAVs saved."""
    query = query_spec["query"]
    label = query_spec["label"]
    max_clips = query_spec["max_clips"]

    out_dir = cfg.FREESOUND_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)

    # Count already-downloaded WAVs matching this query's label dir
    existing = len(list(out_dir.glob("*.wav")))
    log.info("  Query: %-40s label=%-10s max=%d  existing=%d",
             f'"{query}"', label, max_clips, existing)

    if dry_run:
        return 0

    total_needed = max_clips
    downloaded = 0
    page = 1
    _last_req_time = 0.0

    while downloaded < total_needed:
        # Rate limiting
        elapsed = time.monotonic() - _last_req_time
        if elapsed < REQUEST_INTERVAL:
            time.sleep(REQUEST_INTERVAL - elapsed)

        results = freesound_search(query, api_key, page=page)
        _last_req_time = time.monotonic()

        sounds = results.get("results", [])
        if not sounds:
            break

        for sound in sounds:
            if downloaded >= total_needed:
                break

            # Skip out-of-range durations
            duration = float(sound.get("duration", 0))
            if duration < cfg.FREESOUND_MIN_DURATION_SEC:
                continue
            if duration > cfg.FREESOUND_MAX_DURATION_SEC:
                continue

            # Skip if already downloaded
            sound_id = sound["id"]
            if list(out_dir.glob(f"fs_{sound_id}_*.wav")):
                downloaded += 1
                continue

            # Download preview and convert
            mp3 = download_preview(sound, out_dir)
            if mp3:
                wav = mp3_to_wav_16k(mp3)
                if wav:
                    downloaded += 1
                    if downloaded % 20 == 0:
                        log.info("    … %d/%d clips downloaded", downloaded, total_needed)

            # Rate limit between individual downloads
            time.sleep(0.3)

        # Check if there are more pages
        total_available = results.get("count", 0)
        if page * PAGE_SIZE >= total_available:
            break
        page += 1

    return downloaded


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api-key", default=None, help="Freesound API key")
    p.add_argument("--dry-run", action="store_true",
                   help="Show queries and counts without downloading")
    p.add_argument("--query-filter", default=None,
                   help="Only run queries containing this string")
    p.add_argument("--label-filter", default=None, choices=["emergency", "normal"],
                   help="Only download clips for this label class")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg.ensure_p2_dirs()

    log.info("=" * 60)
    log.info("Step 6d — Freesound API targeted download (CC0 / CC-BY)")
    log.info("=" * 60)

    if args.dry_run:
        log.info("DRY RUN — no downloads will be performed\n")

    api_key = get_api_key(args.api_key)

    # Filter queries if requested
    queries = cfg.FREESOUND_QUERIES
    if args.query_filter:
        queries = [q for q in queries if args.query_filter.lower() in q["query"].lower()]
    if args.label_filter:
        queries = [q for q in queries if q["label"] == args.label_filter]

    log.info("Running %d query/queries …\n", len(queries))

    total_downloaded = 0
    for i, q_spec in enumerate(queries, 1):
        log.info("[%d/%d] %s", i, len(queries), q_spec["query"])
        n = download_query(q_spec, api_key, dry_run=args.dry_run)
        total_downloaded += n

    # Final count
    emergency_count = len(list((cfg.FREESOUND_DIR / "emergency").glob("*.wav")))
    normal_count = len(list((cfg.FREESOUND_DIR / "normal").glob("*.wav")))

    log.info("\n" + "=" * 60)
    log.info("Freesound download complete.")
    log.info("  Emergency (scream/groan/impact) : %d WAV clips", emergency_count)
    log.info("  Normal (toilet/water/door)      : %d WAV clips", normal_count)
    log.info("  Output: %s", cfg.FREESOUND_DIR)
    log.info("  Next: run 6e_audit_fsd50k.py (after FSD50K is downloaded)")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
