"""
Shared audio helpers: filename parsing and the YAMNet window-slicing contract.

Step 1 (audit) uses these to *predict* how many windows the corpus will yield;
Step 2 (preprocess) uses the identical functions to actually cut them. Keeping
one implementation is what guarantees the audit's window count matches reality.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

import numpy as np

import config

_FILENAME_PATTERN = re.compile(config.FILENAME_REGEX)


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedName:
    """The five metadata fields encoded in a SAFE filename stem."""

    fold_id: str
    clip_uid: str
    event_code: str
    take_id: str
    class_code: str

    @property
    def label(self) -> str | None:
        """Human-readable class label, or None if the code is unknown."""
        return config.CLASS_CODE_TO_LABEL.get(self.class_code)

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def parse_filename(path: Path | str) -> ParsedName | None:
    """Parse an ``AA-BBB-CC-DDD-FF`` stem. Returns None if it does not conform."""
    stem = Path(path).stem
    match = _FILENAME_PATTERN.match(stem)
    if match is None:
        return None
    return ParsedName(*match.groups())


def find_audio_files(root: Path) -> list[Path]:
    """Recursively collect audio files under ``root``, sorted for determinism."""
    if not root.exists():
        return []
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in config.AUDIO_EXTENSIONS
    ]
    return sorted(files)


# ---------------------------------------------------------------------------
# Window slicing (the YAMNet 0.975 s / 50 % overlap contract)
# ---------------------------------------------------------------------------


def num_windows(
    num_samples: int,
    window: int = config.WINDOW_SAMPLES,
    hop: int = config.HOP_SAMPLES,
) -> int:
    """Count complete windows extractable from ``num_samples``.

    Partial trailing windows are dropped rather than zero-padded: a padded
    window would train the head on silence that the device never sees, since the
    firmware only ever runs inference on a fully-populated ring buffer.
    """
    if num_samples < window:
        return 0
    return 1 + (num_samples - window) // hop


def window_offsets(
    num_samples: int,
    window: int = config.WINDOW_SAMPLES,
    hop: int = config.HOP_SAMPLES,
) -> list[int]:
    """Start sample index of every complete window."""
    return [index * hop for index in range(num_windows(num_samples, window, hop))]


def covered_samples(
    num_samples: int,
    window: int = config.WINDOW_SAMPLES,
    hop: int = config.HOP_SAMPLES,
) -> int:
    """How many leading samples the complete windows span (0 if none fit)."""
    count = num_windows(num_samples, window, hop)
    if count == 0:
        return 0
    return (count - 1) * hop + window


def iter_windows(
    waveform: np.ndarray,
    window: int = config.WINDOW_SAMPLES,
    hop: int = config.HOP_SAMPLES,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(start_sample, window_view)`` for each complete window."""
    for start in window_offsets(waveform.shape[-1], window, hop):
        yield start, waveform[..., start : start + window]


# ---------------------------------------------------------------------------
# Level measurements
# ---------------------------------------------------------------------------


def rms_dbfs(waveform: np.ndarray) -> float:
    """Full-scale RMS in dB. Returns -inf for digital silence."""
    if waveform.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
    if rms <= 0.0:
        return float("-inf")
    return float(20.0 * np.log10(rms))


def clipping_ratio(
    waveform: np.ndarray,
    threshold: float = config.CLIPPING_SAMPLE_THRESHOLD,
) -> float:
    """Fraction of samples at or beyond the clipping threshold."""
    if waveform.size == 0:
        return 0.0
    return float(np.mean(np.abs(waveform) >= threshold))
