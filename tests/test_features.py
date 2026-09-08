import numpy as np

from src.features import (
    extract_mfcc,
    extract_lfcc,
    extract_spectral_features,
    extract_time_features,
    extract_all_features,
    pool_features,
    extract_feature_vector,
    get_feature_names,
)


def create_test_signal():
    """Create a synthetic audio signal for testing."""
    sr = 16000
    duration = 1.0

    t = np.linspace(
        0,
        duration,
        int(sr * duration),
        endpoint=False
    )

    # Synthetic 440 Hz sine wave
    signal = 0.5 * np.sin(
        2 * np.pi * 440 * t
    )

    return signal, sr


def test_extract_mfcc():
    signal, sr = create_test_signal()

    mfcc = extract_mfcc(
        signal,
        sr=sr
    )

    print("MFCC shape:", mfcc.shape)

    assert mfcc.ndim == 2
    assert mfcc.shape[0] == 20
    assert np.isfinite(mfcc).all()


def test_extract_lfcc():
    signal, sr = create_test_signal()

    lfcc = extract_lfcc(
        signal,
        sr=sr
    )

    print("LFCC shape:", lfcc.shape)

    assert lfcc.ndim == 2
    assert lfcc.shape[0] == 20
    assert np.isfinite(lfcc).all()


def test_extract_spectral_features():
    signal, sr = create_test_signal()

    spectral = extract_spectral_features(
        signal,
        sr=sr
    )

    print("Spectral features shape:", spectral.shape)

    assert spectral.ndim == 2
    assert spectral.shape[0] == 4
    assert np.isfinite(spectral).all()


def test_extract_time_features():
    signal, sr = create_test_signal()

    time_features = extract_time_features(
        signal
    )

    print("Time features shape:", time_features.shape)

    assert time_features.ndim == 2
    assert time_features.shape[0] == 2
    assert np.isfinite(time_features).all()


def test_extract_all_features():
    signal, sr = create_test_signal()

    features = extract_all_features(
        signal,
        sr=sr
    )

    print("Combined features shape:", features.shape)

    assert features.ndim == 2
    assert features.shape[0] == 46
    assert np.isfinite(features).all()


def test_pool_features():
    signal, sr = create_test_signal()

    features = extract_all_features(
        signal,
        sr=sr
    )

    pooled = pool_features(features)

    print("Pooled features shape:", pooled.shape)

    assert pooled.ndim == 1
    assert pooled.shape[0] == 92
    assert np.isfinite(pooled).all()


def test_extract_feature_vector():
    signal, sr = create_test_signal()

    vector = extract_feature_vector(
        signal,
        sr=sr
    )

    print("Feature vector shape:", vector.shape)

    assert vector.ndim == 1
    assert vector.shape[0] == 92
    assert np.isfinite(vector).all()


def test_feature_names():
    names = get_feature_names()

    print("Number of feature names:", len(names))

    assert len(names) == 92
    assert len(set(names)) == 92

    assert "mfcc_01_mean" in names
    assert "lfcc_01_mean" in names
    assert "spectral_centroid_mean" in names
    assert "short_time_energy_mean" in names
    assert "zero_crossing_rate_std" in names