"""
YAMNet rebuilt as a patch-input Keras model, plus its log-mel front-end.

Why rebuild rather than just use the TF-Hub SavedModel:

TF-Hub's YAMNet is a sealed waveform-in graph. Its front-end uses tf.signal.stft
and a mel filterbank matmul, and **TFLite Micro supports neither**. So the model
that ships to the ESP32-S3 cannot take a waveform -- it must take the finished
96 x 64 log-mel patch, with the front-end written in C++ on the device:

    I2S mic -> 16 kHz ring buffer (15,600 samples)
            -> C++ log-mel front-end -> 96 x 64 patch      <- firmware
            -> TFLite Micro: MobileNetV1 + head             <- Step 5 exports
            -> softmax [non_fall, fall]

This module provides both halves of that boundary:

  * ``build_log_mel`` is the reference front-end. It is the exact computation
    the C++ code must reproduce, written in the fewest possible operations so it
    can be read as a specification.
  * ``build_backbone`` is the MobileNetV1 core with a fixed (96, 64, 1) input,
    loading the official ``yamnet.h5`` weights.

``verify_against_hub`` checks the rebuild numerically against TF-Hub on real
audio. Nothing downstream should be trusted until that passes: a silently
mismatched rebuild would train a head on one feature space and deploy it on
another.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import tensorflow as tf

import config

WEIGHTS_URL = "https://storage.googleapis.com/audioset/yamnet.h5"
WEIGHTS_NAME = "yamnet.h5"

# YAMNet's batch norm keeps its offset but not its scale, so each BN layer
# carries exactly [beta, moving_mean, moving_variance].
BN_CENTER = True
BN_SCALE = False
BN_EPSILON = 1e-4

# (kind, kernel, stride, filters) for layer1 .. layer14.
LAYER_DEFS: tuple[tuple[str, int, int, int], ...] = (
    ("conv", 3, 2, 32),
    ("sep", 3, 1, 64),
    ("sep", 3, 2, 128),
    ("sep", 3, 1, 128),
    ("sep", 3, 2, 256),
    ("sep", 3, 1, 256),
    ("sep", 3, 2, 512),
    ("sep", 3, 1, 512),
    ("sep", 3, 1, 512),
    ("sep", 3, 1, 512),
    ("sep", 3, 1, 512),
    ("sep", 3, 1, 512),
    ("sep", 3, 2, 1024),
    ("sep", 3, 1, 1024),
)

NUM_AUDIOSET_CLASSES = 521
# yamnet.h5 holds 3,751,369 parameters in total; the 521-class AudioSet layer
# accounts for 1024 * 521 + 521 = 534,025 of them, which we discard.
EXPECTED_BACKBONE_PARAMS = 3_217_344


# ---------------------------------------------------------------------------
# Log-mel front-end  (the specification for the ESP32 C++ port)
# ---------------------------------------------------------------------------


def build_log_mel(waveform: tf.Tensor) -> tf.Tensor:
    """Waveform -> log-mel patch. ``waveform`` is (batch, WINDOW_SAMPLES) float32.

    Returns (batch, 96, 64, 1).

    Every constant here is load-bearing for the firmware:
      frame length 400 (25 ms), hop 160 (10 ms), FFT 512, periodic Hann window,
      magnitude (not power) spectrum, 64 mel bins spanning 125 Hz - 7500 Hz,
      then log(mel + 0.001).

    A 15,600-sample window yields exactly 1 + (15600 - 400) // 160 = 96 frames,
    which is precisely one YAMNet patch.
    """
    fft_length = 512  # next power of two at or above the 400-sample window

    spectrogram = tf.abs(
        tf.signal.stft(
            signals=waveform,
            frame_length=config.STFT_WINDOW_SAMPLES,
            frame_step=config.STFT_HOP_SAMPLES,
            fft_length=fft_length,
        )
    )
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=config.MEL_BANDS,
        num_spectrogram_bins=fft_length // 2 + 1,
        sample_rate=config.SAMPLE_RATE,
        lower_edge_hertz=config.MEL_MIN_HZ,
        upper_edge_hertz=config.MEL_MAX_HZ,
    )
    mel = tf.matmul(spectrogram, mel_matrix)
    log_mel = tf.math.log(mel + config.LOG_OFFSET)
    return tf.expand_dims(log_mel, axis=-1)


class LogMelLayer(tf.keras.layers.Layer):
    """Keras 3 layer wrapper for :func:`build_log_mel`.

    Keras 3 refuses raw ``tf.signal`` calls on symbolic tensors, so the
    front-end has to be wrapped rather than inlined into the functional graph.
    """

    def call(self, waveform):
        return build_log_mel(waveform)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], config.PATCH_FRAMES, config.MEL_BANDS, 1)


def build_features_model() -> tf.keras.Model:
    """Keras wrapper around :func:`build_log_mel`, for batched extraction."""
    inputs = tf.keras.Input(shape=(config.WINDOW_SAMPLES,), dtype=tf.float32,
                            name="waveform")
    return tf.keras.Model(
        inputs, LogMelLayer(name="log_mel")(inputs), name="yamnet_frontend"
    )


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------


def _batch_norm(name: str):
    return tf.keras.layers.BatchNormalization(
        name=name, center=BN_CENTER, scale=BN_SCALE, epsilon=BN_EPSILON
    )


def build_backbone(include_logits: bool = False) -> tf.keras.Model:
    """MobileNetV1 core taking a fixed 96 x 64 x 1 log-mel patch.

    Output is the 1024-d global-average-pooled embedding, which is what the
    classification head consumes. The original 521-class AudioSet layer is only
    built when ``include_logits`` is set, and then only so the rebuild can be
    verified against TF-Hub.
    """
    inputs = tf.keras.Input(
        shape=(config.PATCH_FRAMES, config.MEL_BANDS, 1), dtype=tf.float32,
        name="log_mel_patch",
    )
    # Keras 3 forbids '/' in layer names, but the official checkpoint uses it
    # throughout. Layers are therefore named with underscores and this map
    # records the checkpoint path each one loads from.
    h5_map: dict[str, str] = {}

    def named(layer_cls, keras_name: str, h5_name: str, **kwargs):
        h5_map[keras_name] = h5_name
        return layer_cls(name=keras_name, **kwargs)

    net = inputs
    for index, (kind, kernel, stride, filters) in enumerate(LAYER_DEFS, start=1):
        tag = f"layer{index}"
        if kind == "conv":
            net = named(
                tf.keras.layers.Conv2D, f"{tag}_conv", f"{tag}/conv",
                filters=filters, kernel_size=kernel, strides=stride,
                padding="same", use_bias=False,
            )(net)
            net = named(
                tf.keras.layers.BatchNormalization,
                f"{tag}_conv_bn", f"{tag}/conv/bn",
                center=BN_CENTER, scale=BN_SCALE, epsilon=BN_EPSILON,
            )(net)
            net = tf.keras.layers.ReLU(name=f"{tag}_relu")(net)
        else:
            net = named(
                tf.keras.layers.DepthwiseConv2D,
                f"{tag}_depthwise_conv", f"{tag}/depthwise_conv",
                kernel_size=kernel, strides=stride, depth_multiplier=1,
                padding="same", use_bias=False,
            )(net)
            net = named(
                tf.keras.layers.BatchNormalization,
                f"{tag}_depthwise_conv_bn", f"{tag}/depthwise_conv/bn",
                center=BN_CENTER, scale=BN_SCALE, epsilon=BN_EPSILON,
            )(net)
            net = tf.keras.layers.ReLU(name=f"{tag}_depthwise_relu")(net)
            net = named(
                tf.keras.layers.Conv2D,
                f"{tag}_pointwise_conv", f"{tag}/pointwise_conv",
                filters=filters, kernel_size=1, strides=1, padding="same",
                use_bias=False,
            )(net)
            net = named(
                tf.keras.layers.BatchNormalization,
                f"{tag}_pointwise_conv_bn", f"{tag}/pointwise_conv/bn",
                center=BN_CENTER, scale=BN_SCALE, epsilon=BN_EPSILON,
            )(net)
            net = tf.keras.layers.ReLU(name=f"{tag}_pointwise_relu")(net)

    embeddings = tf.keras.layers.GlobalAveragePooling2D(name="embedding")(net)
    outputs = embeddings
    if include_logits:
        h5_map["logits"] = "logits"
        outputs = tf.keras.layers.Dense(
            NUM_AUDIOSET_CLASSES, name="logits"
        )(embeddings)

    model = tf.keras.Model(inputs, outputs, name="yamnet_backbone")
    model.yamnet_h5_map = h5_map
    return model


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


def download_weights(models_dir: Path | None = None) -> Path:
    """Fetch the official yamnet.h5 if it is not already on disk."""
    target = (models_dir or config.MODELS_DIR) / WEIGHTS_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        urllib.request.urlretrieve(WEIGHTS_URL, target)
    return target


def load_yamnet_weights(model: tf.keras.Model, weights_path: Path) -> int:
    """Copy official YAMNet weights into a rebuilt model, layer by layer.

    Keras' own ``load_weights`` is not used: the checkpoint predates Keras 3 and
    its legacy HDF5 layout does not round-trip reliably. Reading the arrays
    explicitly is version-proof, and it fails loudly on any shape mismatch
    instead of quietly leaving a layer at its random initialisation.
    """
    import h5py

    h5_map = getattr(model, "yamnet_h5_map", {})
    applied = 0
    with h5py.File(weights_path, "r") as handle:
        for layer in model.layers:
            if not layer.weights:
                continue
            name = h5_map.get(layer.name)
            if name is None:
                raise KeyError(
                    f"layer '{layer.name}' has weights but no checkpoint mapping"
                )
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                keys = ["beta", "moving_mean", "moving_variance"]
            elif isinstance(layer, tf.keras.layers.DepthwiseConv2D):
                keys = ["depthwise_kernel"]
            elif isinstance(layer, tf.keras.layers.Conv2D):
                keys = ["kernel"]
            elif isinstance(layer, tf.keras.layers.Dense):
                keys = ["kernel", "bias"]
            else:
                continue

            values = []
            for key in keys:
                path = f"{name}/{name}/{key}:0"
                if path not in handle:
                    raise KeyError(f"{path} missing from {weights_path}")
                values.append(np.asarray(handle[path]))

            expected = [tuple(w.shape) for w in layer.weights]
            got = [tuple(v.shape) for v in values]
            if expected != got:
                raise ValueError(
                    f"shape mismatch for '{name}': model {expected} vs file {got}"
                )
            layer.set_weights(values)
            applied += 1
    return applied


def load_backbone(
    weights_path: Path | None = None, include_logits: bool = False
) -> tf.keras.Model:
    """Build the backbone and populate it with the official weights."""
    model = build_backbone(include_logits=include_logits)
    load_yamnet_weights(model, weights_path or download_weights())
    model.trainable = False
    return model


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def build_head(
    hidden_units: int = 128, dropout: float = 0.4, name: str = "fall_head"
) -> tf.keras.Model:
    """Dense(128, relu) -> Dropout(0.4) -> Dense(2, softmax)."""
    inputs = tf.keras.Input(shape=(config.EMBEDDING_DIM,), name="embedding_in")
    net = tf.keras.layers.Dense(hidden_units, activation="relu", name="head_dense")(
        inputs
    )
    net = tf.keras.layers.Dropout(dropout, name="head_dropout")(net)
    outputs = tf.keras.layers.Dense(
        config.NUM_CLASSES, activation="softmax", name="head_output"
    )(net)
    return tf.keras.Model(inputs, outputs, name=name)


def build_composite(backbone: tf.keras.Model, head: tf.keras.Model) -> tf.keras.Model:
    """Patch -> embedding -> class probabilities, as one deployable graph.

    This is exactly what Step 5 quantises: fixed 96 x 64 x 1 input, no front-end
    inside the graph, softmax over [non_fall, fall].
    """
    inputs = tf.keras.Input(
        shape=(config.PATCH_FRAMES, config.MEL_BANDS, 1), dtype=tf.float32,
        name="log_mel_patch",
    )
    backbone.trainable = False
    return tf.keras.Model(
        inputs, head(backbone(inputs, training=False)), name="yamnet_fall_detector"
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_against_hub(
    waveforms: np.ndarray, backbone: tf.keras.Model
) -> dict[str, float]:
    """Compare the rebuilt front-end + backbone against TF-Hub YAMNet.

    ``waveforms`` is (n, WINDOW_SAMPLES). TF-Hub is fed one window at a time
    because its signature takes a single 1-D waveform; each 15,600-sample window
    yields exactly one patch, one embedding, and one 96 x 64 spectrogram.

    Returns the worst-case absolute differences. Small non-zero values are
    expected -- the two graphs order their floating-point operations
    differently -- but anything beyond ~1e-3 means the rebuild is wrong.
    """
    import tensorflow_hub as hub

    hub_model = hub.load("https://tfhub.dev/google/yamnet/1")

    mel_diffs, embed_diffs = [], []
    ours_mel = build_features_model().predict(waveforms, verbose=0)
    ours_embed = backbone.predict(ours_mel, verbose=0)

    for index, wave in enumerate(waveforms):
        _, hub_embed, hub_mel = hub_model(wave.astype(np.float32))
        hub_mel = hub_mel.numpy()
        hub_embed = hub_embed.numpy()
        mel_diffs.append(
            float(np.max(np.abs(hub_mel - ours_mel[index, :, :, 0])))
        )
        embed_diffs.append(
            float(np.max(np.abs(hub_embed[0] - ours_embed[index])))
        )

    reference = float(np.max(np.abs(ours_embed)))
    return {
        "max_log_mel_diff": float(np.max(mel_diffs)),
        "max_embedding_diff": float(np.max(embed_diffs)),
        "embedding_scale": reference,
        "relative_embedding_diff": float(np.max(embed_diffs)) / max(reference, 1e-9),
    }
