"""
Try the trained detector yourself. Utility script, not part of the 1-5 pipeline.

Three modes:

    --test-set N     Run N random clips from the held-out test set and show
                     predictions against ground truth. Start here: it needs no
                     recording and tells you immediately whether the model works.

    --wav FILE ...   Classify your own audio. Any sample rate or channel count;
                     it is resampled to 16 kHz mono the same way Step 2 does.

    --mic            Live detection from the microphone, using the exact ring
                     buffer, hop and smoothing the ESP32 firmware should use.

Aggregation matters as much as the model. A single 0.975 s window is a weak
piece of evidence -- the impact of a fall occupies a fraction of a second, so
windows either side of it look like background. Step 4 measured this: window
accuracy 0.80 but clip accuracy 0.92 when averaging a clip's windows. Both
numbers are shown, and the smoothed one is the decision.

Usage
    python src/predict.py --test-set 20
    python src/predict.py --wav recording.wav
    python src/predict.py --wav recording.wav --simulate-bathroom
    python src/predict.py --mic
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audio_utils
import config

FALL = config.LABEL_TO_INDEX["fall"]
BAR_WIDTH = 34


def bar(value: float, width: int = BAR_WIDTH) -> str:
    filled = int(round(max(0.0, min(1.0, value)) * width))
    return "#" * filled + "." * (width - filled)


def verdict(probability: float, threshold: float) -> str:
    return "FALL" if probability >= threshold else "no fall"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def load_waveform_model(model_path: Path):
    """Composite model wrapped so it accepts raw 16 kHz waveform windows.

    The saved model takes 96 x 64 log-mel patches, because that is what the
    ESP32 will feed it once the front-end lives in C++. For desktop testing the
    same front-end is prepended in Python, so this script and the device compute
    identical features.
    """
    import tensorflow as tf

    import yamnet_model as ym

    composite = tf.keras.models.load_model(model_path)
    inputs = tf.keras.Input(shape=(config.WINDOW_SAMPLES,), dtype=tf.float32)
    outputs = composite(ym.LogMelLayer(name="log_mel")(inputs))
    return tf.keras.Model(inputs, outputs, name="waveform_detector")


def predict_windows(model, windows: np.ndarray) -> np.ndarray:
    """P(fall) for a batch of float32 windows in [-1, 1]."""
    if windows.size == 0:
        return np.zeros(0, dtype=np.float32)
    return model.predict(windows, verbose=0)[:, FALL]


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------


def load_wav(path: Path, simulate_bathroom: bool) -> np.ndarray:
    import librosa

    wave, _ = librosa.load(str(path), sr=config.SAMPLE_RATE, mono=True,
                           res_type="soxr_hq")
    if simulate_bathroom:
        from scipy.signal import fftconvolve

        import rir as rir_lib

        impulse, _ = rir_lib.SyntheticRirSource(config.SAMPLE_RATE).canonical()
        onset = rir_lib.direct_path_index(impulse)
        wave = fftconvolve(wave, impulse, mode="full")[onset : onset + len(wave)]
    peak = float(np.max(np.abs(wave))) if wave.size else 0.0
    if peak > 0.999:
        wave = wave * (0.999 / peak)
    return wave.astype(np.float32)


def to_windows(wave: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Slice into the same 15,600-sample windows the model was trained on.

    Audio shorter than one window is zero-padded rather than rejected, so a
    short recording still produces one usable prediction.
    """
    if wave.shape[0] < config.WINDOW_SAMPLES:
        wave = np.pad(wave, (0, config.WINDOW_SAMPLES - wave.shape[0]))
    offsets = audio_utils.window_offsets(wave.shape[0])
    windows = np.stack([wave[o : o + config.WINDOW_SAMPLES] for o in offsets])
    return windows, offsets


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def run_files(model, paths: list[Path], args) -> int:
    for path in paths:
        if not path.exists():
            print(f"\n{path}  -- not found")
            continue
        wave = load_wav(path, args.simulate_bathroom)
        windows, offsets = to_windows(wave)
        probs = predict_windows(model, windows)

        room = " [bathroom RIR applied]" if args.simulate_bathroom else ""
        print(f"\n{path.name}  {wave.shape[0]/config.SAMPLE_RATE:.2f} s  "
              f"-> {len(probs)} window(s){room}")
        for index, (offset, prob) in enumerate(zip(offsets, probs)):
            start = offset / config.SAMPLE_RATE
            print(f"   w{index}  {start:5.2f}-{start + config.WINDOW_SECONDS:5.2f}s  "
                  f"{bar(float(prob))}  P(fall)={prob:.3f}")
        mean = float(np.mean(probs))
        print(f"   {'CLIP':>4}  mean P(fall) = {mean:.3f}  ->  "
              f"{verdict(mean, args.threshold)}")
    return 0


def run_test_set(model, args) -> int:
    """Sanity check against clips the model has never seen, with known labels."""
    import pandas as pd

    chunks_path = config.PROCESSED_DIR / "chunks_int16.npy"
    indices_path = config.SPLITS_DIR / "test_indices.npy"
    manifest_path = config.SPLITS_DIR / "split_manifest.csv"
    for required in (chunks_path, indices_path, manifest_path):
        if not required.exists():
            print(f"missing {required} - run Steps 2 and 3 first")
            return 2

    chunks = np.load(chunks_path, mmap_mode="r")
    test_idx = np.load(indices_path)
    manifest = pd.read_csv(manifest_path, dtype={"clip_uid": str, "fold_id": str})
    rows = manifest.set_index("chunk_index").loc[test_idx]

    rng = np.random.default_rng(args.seed)
    clips = rows["clip_uid"].unique()
    chosen = rng.choice(clips, size=min(args.test_set, len(clips)), replace=False)

    print(f"\nHeld-out test set: {len(clips)} clips, showing {len(chosen)}.")
    print("These were never seen in training and carry the canonical bathroom RIR.\n")
    print(f"{'clip':>6} {'truth':>9} {'P(fall)':>8}  {'predicted':>9}  result")
    print("-" * 62)

    correct = 0
    for uid in sorted(chosen):
        sub = rows[rows["clip_uid"] == uid]
        windows = np.stack([
            np.asarray(chunks[i], dtype=np.float32) / 32768.0 for i in sub.index
        ])
        mean = float(np.mean(predict_windows(model, windows)))
        truth = sub["label"].iloc[0]
        predicted = "fall" if mean >= args.threshold else "non_fall"
        hit = predicted == truth
        correct += hit
        print(f"{uid:>6} {truth:>9} {mean:8.3f}  {predicted:>9}  "
              f"{'OK' if hit else 'WRONG'}")

    print("-" * 62)
    print(f"{correct}/{len(chosen)} correct ({100.0 * correct / len(chosen):.1f}%)")
    print("\nStep 4 measured 0.916 clip accuracy with recall 1.000 over all 95 "
          "test clips;\na small sample will bounce around that.")
    return 0


def run_mic(model, args) -> int:
    """Live detection mirroring the firmware's ring buffer and smoothing."""
    try:
        import sounddevice as sd
    except ImportError:
        print("sounddevice is not installed:  pip install sounddevice")
        return 2

    ring = np.zeros(config.WINDOW_SAMPLES, dtype=np.float32)
    recent: deque[float] = deque(maxlen=args.smooth)
    filled = 0

    print(f"\nListening at {config.SAMPLE_RATE} Hz. Ctrl+C to stop.")
    print(f"Window {config.WINDOW_SECONDS} s, hop {config.HOP_SECONDS:.4f} s, "
          f"decision = mean of last {args.smooth} windows "
          f"(~{args.smooth * config.HOP_SECONDS:.1f} s), threshold {args.threshold}.")
    print("Try clapping, dropping a book, or slamming a door.\n")

    try:
        with sd.InputStream(
            samplerate=config.SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=config.HOP_SAMPLES, device=args.device,
        ) as stream:
            while True:
                block, overflowed = stream.read(config.HOP_SAMPLES)
                if overflowed:
                    pass  # dropped audio is not worth aborting a live demo over
                block = block[:, 0]
                # Shift the ring buffer by one hop and append the new audio,
                # exactly as the ESP32 I2S callback will.
                ring = np.concatenate([ring[config.HOP_SAMPLES :], block])
                filled = min(filled + config.HOP_SAMPLES, config.WINDOW_SAMPLES)
                if filled < config.WINDOW_SAMPLES:
                    continue

                prob = float(predict_windows(model, ring[None, :])[0])
                recent.append(prob)
                smoothed = float(np.mean(recent))
                level = float(np.sqrt(np.mean(ring**2)))
                flag = "  <<< FALL" if smoothed >= args.threshold else ""
                print(f"\r{bar(smoothed)} smoothed={smoothed:.3f} "
                      f"now={prob:.3f} rms={level:.4f}{flag}   ", end="", flush=True)
    except KeyboardInterrupt:
        print("\nstopped.")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Test the trained fall detector.")
    p.add_argument("--model", type=Path,
                   default=config.MODELS_DIR / "yamnet_fall_detector.keras")
    # --wav kept as an alias; the flag accepts any format librosa reads.
    p.add_argument("--audio", "--wav", type=Path, nargs="+", dest="audio")
    p.add_argument("--test-set", type=int, metavar="N")
    p.add_argument("--mic", action="store_true")
    p.add_argument("--simulate-bathroom", action="store_true",
                   help="convolve input with the canonical bathroom RIR first")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--smooth", type=int, default=3,
                   help="mic mode: windows to average before deciding")
    p.add_argument("--device", type=int, default=None, help="mic mode: input device")
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = p.parse_args(argv)

    if not any((args.audio, args.test_set, args.mic)):
        p.error("choose one of --test-set N, --audio FILE, or --mic")
    if not args.model.exists():
        print(f"no model at {args.model} - run Step 4 first")
        return 2

    print(f"loading {args.model.name} ...")
    model = load_waveform_model(args.model)

    if args.test_set:
        return run_test_set(model, args)
    if args.audio:
        return run_files(model, args.audio, args)
    return run_mic(model, args)


if __name__ == "__main__":
    raise SystemExit(main())
