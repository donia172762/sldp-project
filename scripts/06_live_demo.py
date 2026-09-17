from pathlib import Path
import argparse

import joblib
import pandas as pd

from src.audio import load_and_preprocess
from src.config import load_config
from src.features import (
    extract_feature_vector,
    get_feature_names,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_svm_combined.joblib"
)


def predict_audio(audio_path: Path):

    print("=" * 60)
    print("AI-GENERATED SPEECH DETECTOR")
    print("=" * 60)

    # ---------------------------------
    # Check files
    # ---------------------------------

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    print(f"Audio: {audio_path.name}")
    print()

    # ---------------------------------
    # Load saved model
    # ---------------------------------

    saved = joblib.load(
        MODEL_PATH
    )

    model = saved["model"]
    feature_columns = saved["feature_columns"]

    # ---------------------------------
    # Load project configuration
    # ---------------------------------

    cfg = load_config()

    # ---------------------------------
    # Preprocess audio
    # Same pipeline used during training
    # ---------------------------------

    print("1. Loading and preprocessing audio...")

    signal = load_and_preprocess(
        audio_path,
        cfg,
        training=False,
    )

    # ---------------------------------
    # Extract 92 features
    # ---------------------------------

    print("2. Extracting speech features...")

    vector = extract_feature_vector(
        signal,
        sr=int(cfg.audio.sample_rate),
    )

    feature_names = get_feature_names()

    feature_df = pd.DataFrame(
        [vector],
        columns=feature_names,
    )

    # Make sure the order exactly matches
    # the order used during model training.
    x = feature_df[
        feature_columns
    ]

    # ---------------------------------
    # Prediction
    # ---------------------------------

    print("3. Running classifier...")

    prediction = int(
        model.predict(x)[0]
    )

    score = float(
        model.decision_function(x)[0]
    )

    if prediction == 1:
        label = "SPOOF / AI-GENERATED"
    else:
        label = "BONAFIDE / REAL HUMAN"

    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)

    print(
        f"Prediction : {label}"
    )

    print(
        f"SVM score  : {score:.4f}"
    )

    print()

    if score >= 0:
        print(
            "Positive score -> model leans toward SPOOF"
        )
    else:
        print(
            "Negative score -> model leans toward BONAFIDE"
        )

    print("=" * 60)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Classify one speech recording "
            "as bonafide or AI-generated."
        )
    )

    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="Path to WAV or FLAC audio file.",
    )

    args = parser.parse_args()

    audio_path = args.audio

    # If a relative path is given,
    # interpret it from project root.
    if not audio_path.is_absolute():
        audio_path = (
            PROJECT_ROOT
            / audio_path
        )

    predict_audio(
        audio_path
    )


if __name__ == "__main__":
    main()