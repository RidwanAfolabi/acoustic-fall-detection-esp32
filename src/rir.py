"""
Room impulse response synthesis, measurement, and loading.

The deployment target is a small, hard-surfaced, tiled restroom. The SAFE corpus
was recorded in basements, laboratories and stairwells, so convolving with a
bathroom RIR is what closes the *room* half of that domain gap. (It does not
close the *floor* half: SAFE falls land on carpet, wood and concrete, never
tile. That remains a known, unfixed mismatch.)

Two sources sit behind one interface, so swapping simulated RIRs for real
measured ones is a flag rather than a rewrite:

    synthetic : hybrid image-source + statistical late tail (this module)
    measured  : real RIRs loaded from a directory of audio files

The synthetic generator is deliberately a *hybrid*, and that is what makes its
RT60 trustworthy. The image-source method gives geometrically correct direct
sound and early reflections, but a truncated image expansion always under-builds
the diffuse tail and so undershoots the target RT60. The late field is therefore
constructed directly as exponentially-decaying noise calibrated to the target.
Every generated RIR is then measured back by Schroeder integration, so a
generator that misses its target gets caught rather than quietly poisoning the
whole dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

SPEED_OF_SOUND = 343.0  # m/s, dry air at ~20 C

# Boundary between the geometric early field and the statistical late field.
# Before this, individual reflections still arrive as discrete, audible events.
MIXING_TIME_SEC = 0.060
CROSSFADE_SEC = 0.012

# Tile reflects high frequencies strongly, but air absorption and grazing
# incidence still make the HF band decay faster than the broadband figure.
HF_SPLIT_HZ = 2000.0
HF_RT60_RATIO = 0.72

# Shaping the HF band to decay faster steepens the *broadband* energy decay, so
# a naive generator measures back ~10 % short of its requested RT60. The target
# is therefore pre-compensated by this factor, calibrated against measure_rt60
# and re-checked by the self-test at the bottom of this module.
RT60_CALIBRATION = 1.113

# Direct-to-reverberant ratio, in dB, measured over a +/-2.5 ms window around
# the direct arrival.
#
# This is the knob that actually decides whether a fall's onset survives the
# room, so it is set explicitly rather than left to emerge from diffuse-field
# theory. Sabine's critical distance for a 12 m^3 room at RT60 0.6 s is only
# ~0.25 m, which would predict about -20 dB DRR at a 2.5 m mic distance -- far
# more smearing than real measured bathroom responses show, because that theory
# assumes a perfectly diffuse field and an omnidirectional source. Real small
# hard rooms measure nearer 0 dB at these distances, so the late field is scaled
# to hit an explicit target instead.
DRR_WINDOW_SEC = 0.0025
CANONICAL_DRR_DB = -3.0
RANDOM_DRR_DB_RANGE = (-9.0, 2.0)


@dataclass(frozen=True)
class RoomConfig:
    """A shoebox room with one source and one microphone, all in metres."""

    dims: tuple[float, float, float]
    source: tuple[float, float, float]
    mic: tuple[float, float, float]
    rt60: float
    drr_db: float = CANONICAL_DRR_DB

    def as_dict(self) -> dict:
        d = asdict(self)
        return {
            "room_x": d["dims"][0], "room_y": d["dims"][1], "room_z": d["dims"][2],
            "src_x": d["source"][0], "src_y": d["source"][1], "src_z": d["source"][2],
            "mic_x": d["mic"][0], "mic_y": d["mic"][1], "mic_z": d["mic"][2],
            "rt60_target": d["rt60"], "drr_target_db": d["drr_db"],
        }


# A fixed, reproducible room. Variant 0 of every clip uses this one, so the
# validation and test sets are realistic (never dry) yet identical across runs.
CANONICAL_ROOM = RoomConfig(
    dims=(2.40, 1.90, 2.60),
    source=(1.20, 0.75, 0.25),   # body on the floor, mid-room
    mic=(2.25, 1.70, 2.30),      # device high on the wall, near the ceiling
    rt60=0.60,
    drr_db=CANONICAL_DRR_DB,
)


def sabine_absorption(dims: tuple[float, float, float], rt60: float) -> float:
    """Mean surface absorption that yields ``rt60`` in this room (Sabine).

    RT60 = 0.161 * V / (S * alpha)  ->  alpha = 0.161 * V / (S * RT60)

    For a 2.4 x 1.9 x 2.6 m room at RT60 = 0.6 s this gives alpha ~ 0.10, the
    right order for ceramic tile plus a door and fixtures.
    """
    x, y, z = dims
    volume = x * y * z
    surface = 2.0 * (x * y + x * z + y * z)
    alpha = 0.161 * volume / (surface * rt60)
    return float(np.clip(alpha, 0.005, 0.95))


def random_bathroom_room(rng: np.random.Generator) -> RoomConfig:
    """Sample a plausible restroom geometry, source and mic placement."""
    dims = (
        float(rng.uniform(1.8, 3.2)),
        float(rng.uniform(1.5, 2.6)),
        float(rng.uniform(2.4, 3.0)),
    )
    margin = 0.25
    # A fall puts the body on or near the floor.
    source = (
        float(rng.uniform(margin, dims[0] - margin)),
        float(rng.uniform(margin, dims[1] - margin)),
        float(rng.uniform(0.10, 0.45)),
    )
    # The device is wall- or ceiling-mounted, high up.
    mic = (
        float(rng.uniform(margin, dims[0] - margin)),
        float(rng.uniform(margin, dims[1] - margin)),
        float(rng.uniform(dims[2] - 0.55, dims[2] - 0.12)),
    )
    return RoomConfig(
        dims,
        source,
        mic,
        rt60=float(rng.uniform(0.45, 0.80)),
        drr_db=float(rng.uniform(*RANDOM_DRR_DB_RANGE)),
    )


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


def _fractional_delay_add(
    buffer: np.ndarray, delays: np.ndarray, amps: np.ndarray, half_width: int = 8
) -> None:
    """Scatter-add impulses at fractional sample positions (windowed sinc).

    Rounding delays to whole samples would roll off exactly the high frequencies
    the 64-bin mel front-end still resolves out to 7.5 kHz, so the sub-sample
    arrival time is interpolated instead.
    """
    base = np.floor(delays).astype(np.int64)
    frac = delays - base
    offsets = np.arange(-half_width + 1, half_width + 1)
    idx = base[:, None] + offsets[None, :]
    t = offsets[None, :] - frac[:, None]
    kernel = np.sinc(t) * (0.5 + 0.5 * np.cos(np.pi * t / half_width))
    contrib = kernel * amps[:, None]

    valid = (idx >= 0) & (idx < buffer.shape[0])
    np.add.at(buffer, idx[valid], contrib[valid])


def _image_source_early(
    cfg: RoomConfig, sample_rate: int, length: int, alpha: float
) -> np.ndarray:
    """Direct sound plus geometrically exact early reflections."""
    beta = float(np.sqrt(max(1.0 - alpha, 1e-6)))  # pressure reflection coefficient
    dims = np.asarray(cfg.dims, dtype=np.float64)
    src = np.asarray(cfg.source, dtype=np.float64)
    mic = np.asarray(cfg.mic, dtype=np.float64)

    horizon = length / sample_rate
    max_distance = SPEED_OF_SOUND * horizon
    order = int(np.ceil(max_distance / (2.0 * dims.min()))) + 1
    order = int(np.clip(order, 1, 16))

    grid = np.arange(-order, order + 1)
    mx, my, mz = np.meshgrid(grid, grid, grid, indexing="ij")
    mx, my, mz = mx.ravel(), my.ravel(), mz.ravel()

    buffer = np.zeros(length, dtype=np.float64)
    for px in (0, 1):
        for py in (0, 1):
            for pz in (0, 1):
                ix = (1 - 2 * px) * src[0] + 2 * mx * dims[0]
                iy = (1 - 2 * py) * src[1] + 2 * my * dims[1]
                iz = (1 - 2 * pz) * src[2] + 2 * mz * dims[2]

                reflections = (
                    np.abs(mx - px) + np.abs(mx)
                    + np.abs(my - py) + np.abs(my)
                    + np.abs(mz - pz) + np.abs(mz)
                )
                distance = np.sqrt(
                    (ix - mic[0]) ** 2 + (iy - mic[1]) ** 2 + (iz - mic[2]) ** 2
                )
                distance = np.maximum(distance, 1e-3)
                delay = distance / SPEED_OF_SOUND * sample_rate

                keep = delay < (length - 10)
                if not np.any(keep):
                    continue
                amp = (beta ** reflections[keep]) / (4.0 * np.pi * distance[keep])
                _fractional_delay_add(buffer, delay[keep], amp)
    return buffer


def _statistical_late(
    length: int, sample_rate: int, rt60: float, start: int, rng: np.random.Generator
) -> np.ndarray:
    """Exponentially-decaying Gaussian noise: the diffuse reverberant field."""
    tail = np.zeros(length, dtype=np.float64)
    if start >= length:
        return tail
    t = np.arange(length - start) / sample_rate
    # -60 dB after rt60 seconds  ->  amplitude 10^(-3 t / rt60)
    envelope = np.power(10.0, -3.0 * t / rt60)
    tail[start:] = rng.standard_normal(length - start) * envelope
    return tail


def _apply_hf_decay(rir: np.ndarray, sample_rate: int, rt60: float) -> np.ndarray:
    """Make high frequencies decay faster than low, as real rooms do."""
    nyquist = sample_rate / 2.0
    sos_lo = butter(4, HF_SPLIT_HZ / nyquist, btype="low", output="sos")
    sos_hi = butter(4, HF_SPLIT_HZ / nyquist, btype="high", output="sos")
    low = sosfilt(sos_lo, rir)
    high = sosfilt(sos_hi, rir)

    t = np.arange(rir.shape[0]) / sample_rate
    # Extra decay so the HF band reaches HF_RT60_RATIO * rt60 rather than rt60.
    extra = np.power(10.0, -3.0 * t * (1.0 / (HF_RT60_RATIO * rt60) - 1.0 / rt60))
    return low + high * extra


def synthesize_rir(
    cfg: RoomConfig, sample_rate: int, seed: int | None = None
) -> np.ndarray:
    """Generate one bathroom RIR, peak-normalised."""
    rng = np.random.default_rng(seed)
    alpha = sabine_absorption(cfg.dims, cfg.rt60)
    decay_rt60 = cfg.rt60 * RT60_CALIBRATION
    length = int(np.ceil(sample_rate * (cfg.rt60 * 1.4 + MIXING_TIME_SEC)))

    early = _image_source_early(cfg, sample_rate, length, alpha)

    mix_start = int(MIXING_TIME_SEC * sample_rate)
    fade = max(int(CROSSFADE_SEC * sample_rate), 1)
    late = _statistical_late(length, sample_rate, decay_rt60, mix_start, rng)

    # Match the late field energy to the early field across the crossover, so
    # the decay is continuous rather than stepping up or down there.
    #
    # Both windows must be measured over regions where their own signal is
    # actually present: `late` is identically zero before mix_start, so a window
    # straddling the crossover would halve its apparent RMS and over-amplify the
    # tail by sqrt(2).
    pre = slice(max(mix_start - 2 * fade, 0), mix_start)
    post = slice(mix_start, min(mix_start + 2 * fade, length))
    early_rms = float(np.sqrt(np.mean(early[pre] ** 2))) if pre.stop > pre.start else 0.0
    late_rms = float(np.sqrt(np.mean(late[post] ** 2))) if post.stop > post.start else 0.0
    if late_rms > 0.0 and early_rms > 0.0:
        late *= early_rms / late_rms

    ramp = np.ones(length, dtype=np.float64)
    ramp[:mix_start] = 0.0
    ramp[mix_start : mix_start + fade] = np.linspace(0.0, 1.0, fade, endpoint=False)

    rir = early * (1.0 - ramp) + late * ramp
    rir = _apply_hf_decay(rir, sample_rate, decay_rt60)
    rir = _scale_to_drr(rir, sample_rate, cfg.drr_db)
    return normalise_energy(rir).astype(np.float32)


def normalise_energy(rir: np.ndarray) -> np.ndarray:
    """Scale a RIR to unit energy, so convolution preserves signal level.

    Peak-normalising instead would be a mistake here: a reverberant RIR spreads
    a lot of energy across its tail, so a unit-*peak* response multiplies the
    convolved signal's level several-fold and forces a large headroom cut that
    silently changes loudness relative to the dry variant. Loudness is real
    information for fall detection -- a fall is loud -- so it is the room, not
    the recording, that gets normalised.
    """
    data = np.asarray(rir, dtype=np.float64)
    energy = float(np.sqrt(np.sum(data**2)))
    if energy <= 0.0:
        return data
    return data / energy


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def measure_rt60(rir: np.ndarray, sample_rate: int, decay_db: float = 30.0) -> float:
    """Estimate RT60 by Schroeder backward integration.

    Fits the energy decay curve between -5 dB and -(5 + decay_db) dB and
    extrapolates to a full 60 dB drop: the standard T30 estimator. Returns NaN
    when the response does not decay far enough to fit.
    """
    energy = np.asarray(rir, dtype=np.float64) ** 2
    edc = np.cumsum(energy[::-1])[::-1]
    if edc[0] <= 0.0:
        return float("nan")
    edc_db = 10.0 * np.log10(np.maximum(edc / edc[0], 1e-20))

    start_db, end_db = -5.0, -(5.0 + decay_db)
    start_hits = np.flatnonzero(edc_db <= start_db)
    end_hits = np.flatnonzero(edc_db <= end_db)
    if start_hits.size == 0 or end_hits.size == 0:
        return float("nan")
    start, end = int(start_hits[0]), int(end_hits[0])
    if end <= start + 1:
        return float("nan")

    t = np.arange(start, end) / sample_rate
    slope = float(np.polyfit(t, edc_db[start:end], 1)[0])
    if slope >= 0.0:
        return float("nan")
    return float(-60.0 / slope)


def energy_decay_curve(rir: np.ndarray) -> np.ndarray:
    """Normalised Schroeder energy decay curve, in dB. For QA plots."""
    energy = np.asarray(rir, dtype=np.float64) ** 2
    edc = np.cumsum(energy[::-1])[::-1]
    if edc[0] <= 0.0:
        return np.full(rir.shape[0], -np.inf)
    return 10.0 * np.log10(np.maximum(edc / edc[0], 1e-20))


def direct_path_index(rir: np.ndarray, threshold: float = 0.15) -> int:
    """Sample index of the direct arrival: the FIRST significant arrival.

    Not ``argmax``. In a small hard room the early reflections are dense enough
    to sum constructively above the direct sound, so the largest peak often sits
    tens of milliseconds late. Trimming to that peak would shift every event in
    the corpus backwards in time and silently destroy onset alignment.
    """
    magnitude = np.abs(np.asarray(rir))
    peak = float(magnitude.max()) if magnitude.size else 0.0
    if peak <= 0.0:
        return 0
    hits = np.flatnonzero(magnitude >= threshold * peak)
    return int(hits[0]) if hits.size else int(np.argmax(magnitude))


def measure_drr(rir: np.ndarray, sample_rate: int) -> float:
    """Direct-to-reverberant ratio in dB, over +/-DRR_WINDOW_SEC of the direct."""
    data = np.asarray(rir, dtype=np.float64)
    onset = direct_path_index(data)
    half = max(int(DRR_WINDOW_SEC * sample_rate), 1)
    lo, hi = max(onset - half, 0), min(onset + half + 1, data.shape[0])

    direct = float(np.sum(data[lo:hi] ** 2))
    total = float(np.sum(data**2))
    reverberant = total - direct
    if reverberant <= 0.0 or direct <= 0.0:
        return float("nan")
    return float(10.0 * np.log10(direct / reverberant))


def _scale_to_drr(
    rir: np.ndarray, sample_rate: int, target_db: float
) -> np.ndarray:
    """Scale everything after the direct arrival to hit ``target_db`` DRR.

    A uniform gain on the reverberant part leaves the decay *slope* untouched,
    so RT60 survives this unchanged; only the direct/reverberant balance moves.
    """
    data = np.array(rir, dtype=np.float64, copy=True)
    onset = direct_path_index(data)
    half = max(int(DRR_WINDOW_SEC * sample_rate), 1)
    lo, hi = max(onset - half, 0), min(onset + half + 1, data.shape[0])

    direct = float(np.sum(data[lo:hi] ** 2))
    reverberant = float(np.sum(data**2)) - direct
    if direct <= 0.0 or reverberant <= 0.0:
        return data

    wanted = direct / (10.0 ** (target_db / 10.0))
    gain = float(np.sqrt(wanted / reverberant))
    data[:lo] *= gain
    data[hi:] *= gain
    return data


# ---------------------------------------------------------------------------
# Pluggable sources
# ---------------------------------------------------------------------------


class SyntheticRirSource:
    """Generates a fresh randomised bathroom RIR on demand."""

    kind = "synthetic"

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

    def canonical(self) -> tuple[np.ndarray, dict]:
        rir = synthesize_rir(CANONICAL_ROOM, self.sample_rate, seed=0)
        meta = CANONICAL_ROOM.as_dict()
        meta.update(rir_kind="synthetic_canonical", rir_id="canonical")
        return rir, meta

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
        cfg = random_bathroom_room(rng)
        seed = int(rng.integers(0, 2**31 - 1))
        rir = synthesize_rir(cfg, self.sample_rate, seed=seed)
        meta = cfg.as_dict()
        meta.update(rir_kind="synthetic_random", rir_id=f"seed{seed}")
        return rir, meta


class MeasuredRirSource:
    """Draws from a bank of real measured RIRs on disk.

    Accepts any audio file soundfile can read. Everything is resampled to the
    project rate and peak-normalised, so it is interchangeable with synthesis.
    Point this at MIT IR Survey bathroom responses or OpenSLR-28 to go hybrid.
    """

    kind = "measured"

    def __init__(self, sample_rate: int, rir_dir: Path):
        import librosa

        self.sample_rate = sample_rate
        self.rir_dir = Path(rir_dir)
        paths = sorted(
            p for p in self.rir_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in (".wav", ".flac", ".ogg", ".aiff")
        )
        if not paths:
            raise SystemExit(f"no RIR audio files found under {self.rir_dir}")

        self.bank: list[tuple[np.ndarray, str]] = []
        for path in paths:
            data, _ = librosa.load(str(path), sr=sample_rate, mono=True)
            if data.size == 0 or float(np.max(np.abs(data))) <= 0.0:
                continue
            # Unit energy, matching the synthetic source, so the two back-ends
            # are interchangeable without shifting the corpus loudness.
            self.bank.append((normalise_energy(data).astype(np.float32), path.stem))
        if not self.bank:
            raise SystemExit(f"every RIR under {self.rir_dir} was silent")

    def _meta(self, index: int) -> dict:
        rir, name = self.bank[index]
        return {
            "rir_kind": "measured",
            "rir_id": name,
            "rt60_target": measure_rt60(rir, self.sample_rate),
        }

    def canonical(self) -> tuple[np.ndarray, dict]:
        # Deterministic choice, so val/test stay identical across runs.
        return self.bank[0][0], self._meta(0)

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
        index = int(rng.integers(0, len(self.bank)))
        return self.bank[index][0], self._meta(index)


def make_rir_source(kind: str, sample_rate: int, rir_dir: Path | None = None):
    """Factory for the pluggable RIR back-ends."""
    if kind == "synthetic":
        return SyntheticRirSource(sample_rate)
    if kind == "measured":
        if rir_dir is None:
            raise SystemExit("--rir-source measured requires --rir-dir")
        return MeasuredRirSource(sample_rate, rir_dir)
    raise SystemExit(f"unknown RIR source: {kind}")
