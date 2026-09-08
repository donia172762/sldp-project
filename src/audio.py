"""Shared audio front end.

Every later stage of the project - MFCC, LFCC, spectral descriptors, the
spectrogram CNN - reads its waveform through this module and nothing else.
Keeping a single front end is what makes the feature comparison in the report
fair: if two feature sets disagree, it is because of the features, not because
one of them silently used a different sample rate or window length.

The pipeline is::

    load_audio -> [trim_silence] -> [peak_normalize] -> pre_emphasis -> fix_length

and the two bracketed steps are **off by default**. See :func:`trim_silence`
and :func:`peak_normalize` for why - both defaults are deliberate experimental
decisions, not oversights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import librosa
import numpy as np
import soundfile as sf

# Type aliases keep the signatures below readable.
PadMode = Literal["repeat", "zeros"]
CropMode = Literal["head", "random"]

# Sample values are kept in float32: it halves memory over float64 across
# ~120k utterances and is the precision every downstream library expects.
DTYPE = np.float32


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_audio(
    path: str | Path,
    sample_rate: int = 16000,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """Read an audio file and return a float32 waveform at ``sample_rate``.

    Uses :mod:`soundfile` (libsndfile) rather than an audio backend guess, so
    both corpora load the same way: ASVspoof 2019 LA ships FLAC, ASVspoof 2021
    DF ships FLAC, WaveFake ships WAV.

    Args:
        path: File to read.
        sample_rate: Target rate in Hz. The file is resampled if it differs.
        mono: If True, multi-channel input is averaged down to one channel.

    Returns:
        ``(waveform, sample_rate)`` where ``waveform`` is 1-D float32 (or 2-D
        ``(channels, samples)`` when ``mono`` is False).

    Raises:
        RuntimeError: if the file cannot be decoded, with the path attached so
            the QC pass can report exactly which file is broken.
    """
    try:
        # always_2d gives a consistent (frames, channels) shape to reason about.
        data, native_sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:  # soundfile raises several distinct error types
        raise RuntimeError(f"Could not read audio file: {path} ({exc})") from exc

    # soundfile returns (frames, channels); the rest of the world wants
    # (channels, frames), and librosa's resampler expects time on the last axis.
    data = data.T

    if mono:
        # Downmix by averaging. Mean rather than "take channel 0" so that a
        # true stereo recording does not lose half its energy.
        data = data.mean(axis=0)

    if native_sr != sample_rate:
        data = librosa.resample(
            data, orig_sr=native_sr, target_sr=sample_rate, axis=-1
        )

    return np.ascontiguousarray(data, dtype=DTYPE), sample_rate


# ---------------------------------------------------------------------------
# Individual preprocessing steps
# ---------------------------------------------------------------------------
def pre_emphasis(x: np.ndarray, coefficient: float = 0.97) -> np.ndarray:
    """Apply the standard first-order pre-emphasis filter.

    .. math:: y[n] = x[n] - a \\cdot x[n-1], \\quad y[0] = x[0]

    This is a mild high-pass that flattens the roughly -6 dB/octave tilt of
    voiced speech, so the upper spectrum is not swamped by the first formant.
    That matters here specifically: the artefacts left by neural vocoders tend
    to sit in the *high* frequencies, which is exactly the region pre-emphasis
    lifts.

    Args:
        x: 1-D waveform.
        coefficient: Filter coefficient ``a``; ``0.0`` disables the filter.

    Returns:
        The filtered signal, same shape and dtype as ``x``.
    """
    if coefficient == 0.0:
        return x.astype(DTYPE, copy=True)

    y = np.empty_like(x, dtype=DTYPE)
    y[0] = x[0]                                   # nothing precedes the first sample
    y[1:] = x[1:] - coefficient * x[:-1]
    return y


def peak_normalize(x: np.ndarray, target_peak: float = 1.0, eps: float = 1e-9) -> np.ndarray:
    """Scale the waveform so its largest absolute sample equals ``target_peak``.

    **Off by default** (``audio.peak_normalize: false``).

    Absolute recording level in ASVspoof is largely a property of the source
    database and the post-processing chain, not of the synthesis system. A
    classifier that keys on loudness would score well on the development set
    and collapse on any recording made under different conditions. We therefore
    keep the natural level and let the cepstral features - which are already
    level-robust once the 0th coefficient is handled - do the work. The
    function stays available so the effect can be measured as an ablation.

    Args:
        x: 1-D waveform.
        target_peak: Desired maximum absolute amplitude.
        eps: Guard so an all-zero signal does not divide by zero.

    Returns:
        The scaled signal (an all-zero input is returned unchanged).
    """
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak < eps:
        return x.astype(DTYPE, copy=True)
    return (x * (target_peak / peak)).astype(DTYPE)


def trim_silence(x: np.ndarray, top_db: float = 30.0) -> np.ndarray:
    """Remove leading and trailing near-silence.

    **Off by default** (``audio.remove_silence: false``).

    This is the single most important preprocessing decision in the project.
    In ASVspoof, the duration and character of the non-speech margins differ
    systematically between bona fide and spoofed utterances - the synthesis
    pipelines produce their own, unnaturally clean, silence. A detector can
    reach a very low error rate by reading nothing but that margin, and such a
    detector learns nothing about synthesis artefacts and fails on any data
    trimmed differently. Reported in the literature as a leading cause of
    inflated ASVspoof results.

    We therefore keep the silence in the default configuration, so the
    classifier is forced to use the speech itself, and expose trimming as a
    switch (``audio.remove_silence: true``) for a deliberate ablation.

    Args:
        x: 1-D waveform.
        top_db: Anything more than this many dB below the peak counts as
            silence.

    Returns:
        The trimmed signal. If the whole signal is below threshold, the input
        is returned unchanged rather than an empty array.
    """
    trimmed, _ = librosa.effects.trim(x, top_db=top_db)
    if trimmed.size == 0:
        return x.astype(DTYPE, copy=True)
    return trimmed.astype(DTYPE)


def fix_length(
    x: np.ndarray,
    length: int,
    pad_mode: PadMode = "repeat",
    crop_mode: CropMode = "head",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Force the waveform to exactly ``length`` samples.

    Classical classifiers (SVM, Random Forest) need a fixed-size input vector,
    and a CNN needs a fixed-size spectrogram, so every utterance is squeezed to
    the same duration (4 s = 64000 samples at 16 kHz).

    Short signals are extended:
      * ``repeat`` - tile the signal until it is long enough. Preferred,
        because zero padding would add artificial silence, i.e. exactly the
        cue we are trying not to give the model.
      * ``zeros``  - append zeros (available for comparison).

    Long signals are cropped:
      * ``head``   - take the first ``length`` samples. Deterministic, so
        validation and evaluation are repeatable.
      * ``random`` - take a random window. Only for training, as light
        augmentation; requires ``rng`` for reproducibility.

    Args:
        x: 1-D waveform.
        length: Target number of samples.
        pad_mode: How to extend a short signal.
        crop_mode: How to shorten a long signal.
        rng: Generator used by ``crop_mode='random'``. Defaults to a fresh
            unseeded generator, so pass one explicitly in real runs.

    Returns:
        A 1-D float32 array of exactly ``length`` samples.

    Raises:
        ValueError: on a non-positive length, an empty input, or an unknown mode.
    """
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if x.ndim != 1:
        raise ValueError(f"fix_length expects a 1-D signal, got shape {x.shape}")
    if x.size == 0:
        raise ValueError("fix_length received an empty signal")

    if x.size < length:
        if pad_mode == "repeat":
            repeats = int(np.ceil(length / x.size))
            x = np.tile(x, repeats)[:length]
        elif pad_mode == "zeros":
            x = np.pad(x, (0, length - x.size), mode="constant")
        else:
            raise ValueError(f"Unknown pad_mode: {pad_mode!r}")
        return x.astype(DTYPE, copy=False)

    if x.size > length:
        if crop_mode == "head":
            start = 0
        elif crop_mode == "random":
            generator = rng if rng is not None else np.random.default_rng()
            start = int(generator.integers(0, x.size - length + 1))
        else:
            raise ValueError(f"Unknown crop_mode: {crop_mode!r}")
        x = x[start : start + length]

    return x.astype(DTYPE, copy=False)


# ---------------------------------------------------------------------------
# Short-time analysis
# ---------------------------------------------------------------------------
def next_power_of_two(n: int) -> int:
    """Smallest power of two that is >= ``n`` (e.g. 400 -> 512).

    The FFT is fastest at powers of two, and 512 bins over a 400-sample window
    also gives a little extra frequency interpolation for free.
    """
    if n <= 1:
        return 1
    return 1 << (int(n) - 1).bit_length()


def ms_to_samples(milliseconds: float, sample_rate: int) -> int:
    """Convert a duration in milliseconds to whole samples (25 ms -> 400)."""
    return int(round(milliseconds * sample_rate / 1000.0))


def num_frames(
    signal_length: int, frame_length: int, hop_length: int
) -> int:
    """Frames produced by non-centred framing: ``1 + (N - L) // H``.

    Returns 0 when the signal is shorter than one frame. Written out as its own
    function because the same formula has to hold for :func:`frame_signal` and
    :func:`magnitude_spectrogram`, and the tests check that it does.
    """
    if signal_length < frame_length:
        return 0
    return 1 + (signal_length - frame_length) // hop_length


def get_window(name: str, frame_length: int) -> np.ndarray:
    """Return an analysis window of ``frame_length`` samples.

    Hamming is the project default: its side lobes fall off fast enough to keep
    the harmonic structure of voiced speech readable, which is what the vocoder
    artefacts show up in.
    """
    return librosa.filters.get_window(name, frame_length, fftbins=True).astype(DTYPE)


def frame_signal(
    x: np.ndarray,
    frame_length: int,
    hop_length: int,
    window: str | None = "hamming",
) -> np.ndarray:
    """Split a signal into overlapping, windowed frames.

    Speech is only quasi-stationary, so all analysis happens on short frames.
    With the project defaults (25 ms window, 10 ms hop at 16 kHz) that is a
    400-sample frame advancing 160 samples at a time, i.e. 60% overlap.

    Frame ``k`` covers samples ``[k*hop_length, k*hop_length + frame_length)``.
    No centring or reflection padding is used, so the frame count is exactly
    :func:`num_frames` and every frame contains real signal only.

    Args:
        x: 1-D waveform.
        frame_length: Frame size in samples.
        hop_length: Advance between consecutive frames, in samples.
        window: Window name for :func:`get_window`, or ``None`` for a
            rectangular window.

    Returns:
        Array of shape ``(n_frames, frame_length)``. Empty
        ``(0, frame_length)`` if the signal is shorter than one frame.

    Raises:
        ValueError: if the input is not 1-D or the sizes are non-positive.
    """
    if x.ndim != 1:
        raise ValueError(f"frame_signal expects a 1-D signal, got shape {x.shape}")
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be positive")

    n = num_frames(x.size, frame_length, hop_length)
    if n == 0:
        return np.zeros((0, frame_length), dtype=DTYPE)

    # A strided view costs no memory; the multiplication below makes the copy.
    frames = np.lib.stride_tricks.sliding_window_view(x, frame_length)[
        :: hop_length
    ][:n]

    if window is None:
        return np.ascontiguousarray(frames, dtype=DTYPE)
    return (frames * get_window(window, frame_length)).astype(DTYPE)


def magnitude_spectrogram(
    x: np.ndarray,
    frame_length: int,
    hop_length: int,
    n_fft: int | None = None,
    window: str = "hamming",
) -> np.ndarray:
    """Magnitude STFT, frame-aligned with :func:`frame_signal`.

    Computed with :func:`librosa.stft` using ``n_fft`` = the next power of two
    at or above ``frame_length`` (400 -> 512).

    One alignment detail worth understanding, because it is the reason for the
    padding below. ``librosa`` centres the ``win_length`` window inside the
    larger ``n_fft`` frame, adding ``(n_fft - win_length) // 2`` zeros on the
    left. With ``center=False`` its frame ``k`` would therefore analyse samples
    starting at ``k*hop + 56`` rather than ``k*hop``, and it would also produce
    one frame fewer than :func:`frame_signal`. Padding the *signal* by the same
    amounts cancels both effects exactly: frame ``k`` again covers
    ``[k*hop, k*hop + frame_length)`` and the frame counts match, so magnitude
    columns line up one-to-one with :func:`frame_signal` rows.

    Args:
        x: 1-D waveform.
        frame_length: Analysis window length in samples (the ``win_length``).
        hop_length: Frame advance in samples.
        n_fft: FFT size; defaults to :func:`next_power_of_two` of
            ``frame_length``.
        window: Window name.

    Returns:
        Array of shape ``(1 + n_fft // 2, n_frames)`` holding ``|X[k, m]|``.

    Raises:
        ValueError: if ``n_fft`` is smaller than ``frame_length``.
    """
    if n_fft is None:
        n_fft = next_power_of_two(frame_length)
    if n_fft < frame_length:
        raise ValueError(f"n_fft ({n_fft}) must be >= frame_length ({frame_length})")

    pad_left = (n_fft - frame_length) // 2
    pad_right = (n_fft - frame_length) - pad_left
    padded = np.pad(x, (pad_left, pad_right), mode="constant")

    stft = librosa.stft(
        y=padded,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=frame_length,
        window=window,
        center=False,          # framing stays explicit; see docstring
    )
    return np.abs(stft).astype(DTYPE)


# ---------------------------------------------------------------------------
# The two entry points the rest of the project should call
# ---------------------------------------------------------------------------
def preprocess(
    x: np.ndarray,
    cfg,
    training: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Run the configured preprocessing chain on an already-loaded waveform.

    Order: optional silence trimming, optional peak normalisation,
    pre-emphasis, fixed-length crop/pad. Trimming comes first so that the
    fixed-length window is filled with speech rather than with margin;
    pre-emphasis comes after level handling so its output is not rescaled.

    Args:
        x: 1-D waveform, already at ``cfg.audio.sample_rate``.
        cfg: Loaded project config (``src.config.Config``); only the ``audio``
            section is read.
        training: When True, ``crop_mode='random'`` is honoured. Evaluation
            passes always crop from the head, so scores are deterministic.
        rng: Generator for random cropping.

    Returns:
        A 1-D float32 array of ``cfg.audio.duration * sample_rate`` samples.
    """
    audio_cfg = cfg.audio

    if audio_cfg.remove_silence:
        x = trim_silence(x, top_db=float(audio_cfg.trim_top_db))

    if audio_cfg.peak_normalize:
        x = peak_normalize(x)

    x = pre_emphasis(x, coefficient=float(audio_cfg.pre_emphasis))

    target_length = int(round(float(audio_cfg.duration) * int(audio_cfg.sample_rate)))
    crop_mode: CropMode = audio_cfg.crop_mode if training else "head"

    return fix_length(
        x,
        length=target_length,
        pad_mode=audio_cfg.pad_mode,
        crop_mode=crop_mode,
        rng=rng,
    )


def load_and_preprocess(
    path: str | Path,
    cfg,
    training: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Load a file and apply :func:`preprocess`. The standard way in.

    Args:
        path: Audio file to read.
        cfg: Loaded project config.
        training: See :func:`preprocess`.
        rng: See :func:`preprocess`.

    Returns:
        The preprocessed fixed-length waveform.
    """
    x, _ = load_audio(
        path,
        sample_rate=int(cfg.audio.sample_rate),
        mono=bool(cfg.audio.mono),
    )
    return preprocess(x, cfg, training=training, rng=rng)


def frame_params(cfg) -> tuple[int, int, int]:
    """Resolve ``(frame_length, hop_length, n_fft)`` in samples from the config.

    A single place to turn the millisecond settings into sample counts, so
    every feature extractor added in Phase 2 is guaranteed to use identical
    framing.
    """
    sample_rate = int(cfg.audio.sample_rate)
    frame_length = ms_to_samples(float(cfg.audio.frame_length_ms), sample_rate)
    hop_length = ms_to_samples(float(cfg.audio.frame_shift_ms), sample_rate)
    configured = cfg.audio.get("n_fft")
    n_fft = int(configured) if configured else next_power_of_two(frame_length)
    return frame_length, hop_length, n_fft
