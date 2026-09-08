"""Speech feature extraction for deepfake speech detection."""

import numpy as np
import librosa
from scipy.fftpack import dct


def extract_mfcc(
    signal: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 20,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
) -> np.ndarray:
    """
    Extract Mel-Frequency Cepstral Coefficients (MFCCs).
    """

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )

    return mfcc


def extract_lfcc(
    signal: np.ndarray,
    sr: int = 16000,
    n_lfcc: int = 20,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    n_filters: int = 40,
) -> np.ndarray:
    """
    Extract Linear-Frequency Cepstral Coefficients (LFCCs).
    """

    stft = librosa.stft(
        signal,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )

    power_spectrum = np.abs(stft) ** 2

    frequencies = np.linspace(
        0,
        sr / 2,
        power_spectrum.shape[0]
    )

    filter_edges = np.linspace(
        0,
        sr / 2,
        n_filters + 2
    )

    filter_bank = np.zeros(
        (n_filters, len(frequencies))
    )

    for i in range(n_filters):
        left = filter_edges[i]
        center = filter_edges[i + 1]
        right = filter_edges[i + 2]

        left_slope = (
            frequencies - left
        ) / (center - left)

        right_slope = (
            right - frequencies
        ) / (right - center)

        filter_bank[i] = np.maximum(
            0,
            np.minimum(left_slope, right_slope)
        )

    energies = np.dot(
        filter_bank,
        power_spectrum
    )

    energies = np.maximum(
        energies,
        1e-10
    )

    log_energies = np.log(energies)

    lfcc = dct(
        log_energies,
        type=2,
        axis=0,
        norm="ortho"
    )[:n_lfcc]

    return lfcc
def extract_spectral_features(
    signal: np.ndarray,
    sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
) -> np.ndarray:
    """
    Extract spectral features from an audio signal.

    Features:
    - Spectral centroid
    - Spectral bandwidth
    - Spectral roll-off
    - Spectral flux

    Returns
    -------
    np.ndarray
        Matrix with shape (4, number_of_frames).
    """

    magnitude = np.abs(
        librosa.stft(
            signal,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )
    )

    centroid = librosa.feature.spectral_centroid(
        S=magnitude,
        sr=sr
    )

    bandwidth = librosa.feature.spectral_bandwidth(
        S=magnitude,
        sr=sr
    )

    rolloff = librosa.feature.spectral_rolloff(
        S=magnitude,
        sr=sr,
        roll_percent=0.85
    )

    # Spectral flux = change in spectrum between consecutive frames
    normalized = magnitude / (
        np.sum(magnitude, axis=0, keepdims=True) + 1e-10
    )

    flux = np.sqrt(
        np.sum(
            np.diff(normalized, axis=1) ** 2,
            axis=0
        )
    )

    # First frame has no previous frame for comparison
    flux = np.concatenate(([0.0], flux))[np.newaxis, :]

    spectral_features = np.vstack(
        (centroid, bandwidth, rolloff, flux)
    )

    return spectral_features
def extract_time_features(
    signal: np.ndarray,
    hop_length: int = 160,
    win_length: int = 400,
) -> np.ndarray:
    """
    Extract time-domain features.

    Features:
    - Short-time energy
    - Zero-crossing rate (ZCR)

    Returns
    -------
    np.ndarray
        Matrix with shape (2, number_of_frames).
    """

    # Short-time RMS energy
    rms = librosa.feature.rms(
        y=signal,
        frame_length=win_length,
        hop_length=hop_length,
        center=True,
    )

    # Convert RMS to energy
    energy = rms ** 2

    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(
        y=signal,
        frame_length=win_length,
        hop_length=hop_length,
        center=True,
    )

    time_features = np.vstack((energy, zcr))

    return time_features
def extract_all_features(
    signal: np.ndarray,
    sr: int = 16000,
) -> np.ndarray:
    """
    Extract and combine all speech features.

    Combined features:
    - 20 MFCC
    - 20 LFCC
    - 4 spectral features
    - 2 time-domain features

    Returns
    -------
    np.ndarray
        Combined feature matrix with shape
        (46, number_of_frames).
    """

    mfcc = extract_mfcc(signal, sr=sr)
    lfcc = extract_lfcc(signal, sr=sr)
    spectral = extract_spectral_features(signal, sr=sr)
    time_features = extract_time_features(signal)

    # Ensure all feature groups have the same number of frames
    min_frames = min(
        mfcc.shape[1],
        lfcc.shape[1],
        spectral.shape[1],
        time_features.shape[1],
    )

    combined = np.vstack([
        mfcc[:, :min_frames],
        lfcc[:, :min_frames],
        spectral[:, :min_frames],
        time_features[:, :min_frames],
    ])

    return combined
def pool_features(
    features: np.ndarray,
) -> np.ndarray:
    """
    Convert frame-level features into a fixed-length vector.

    For each feature, compute:
    - Mean across all frames
    - Standard deviation across all frames

    Parameters
    ----------
    features : np.ndarray
        Feature matrix with shape
        (number_of_features, number_of_frames).

    Returns
    -------
    np.ndarray
        Fixed-length feature vector.
        For 46 input features, the output has 92 values.
    """

    mean = np.mean(features, axis=1)
    std = np.std(features, axis=1)

    pooled = np.concatenate([mean, std])

    return pooled
def extract_feature_vector(
    signal: np.ndarray,
    sr: int = 16000,
) -> np.ndarray:
    """
    Extract all features and return one fixed-length
    feature vector for a single audio recording.
    """

    features = extract_all_features(
        signal,
        sr=sr
    )

    return pool_features(features)
def get_feature_names() -> list[str]:
    """
    Return names for the 92 pooled features.

    The first 46 values are means.
    The next 46 values are standard deviations.
    """

    base_names = []

    # 20 MFCCs
    for i in range(1, 21):
        base_names.append(f"mfcc_{i:02d}")

    # 20 LFCCs
    for i in range(1, 21):
        base_names.append(f"lfcc_{i:02d}")

    # 4 spectral features
    base_names.extend([
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_rolloff",
        "spectral_flux",
    ])

    # 2 time-domain features
    base_names.extend([
        "short_time_energy",
        "zero_crossing_rate",
    ])

    mean_names = [
        f"{name}_mean"
        for name in base_names
    ]

    std_names = [
        f"{name}_std"
        for name in base_names
    ]

    return mean_names + std_names