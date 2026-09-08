from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.audio import load_and_preprocess
from src.config import load_config
from src.features import (
    extract_feature_vector,
    get_feature_names,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def extract_features_from_manifest(
    manifest_path: Path,
    output_path: Path,
) -> None:
    """
    Extract fixed-length speech features for every recording
    listed in the input manifest.

    Each audio recording is converted into a 92-dimensional
    feature vector consisting of:
    - MFCC
    - LFCC
    - Spectral features
    - Time-domain features

    Mean and standard deviation pooling are used to create
    a fixed-length vector for machine-learning classifiers.
    """

    # Load the same project configuration used in Phase 1
    cfg = load_config()

    # Read the dataset manifest
    df = pd.read_csv(manifest_path)

    rows = []
    total = len(df)

    # Get meaningful names for all 92 features
    feature_names = get_feature_names()

    for index, row in df.iterrows():
        audio_path = PROJECT_ROOT / row["path"]

        print(
            f"[{index + 1}/{total}] "
            f"Processing {row['utt_id']}"
        )

        try:
            # -----------------------------------------
            # Phase 1: preprocessing
            # -----------------------------------------
            signal = load_and_preprocess(
                audio_path,
                cfg,
                training=False,
            )

            sr = int(cfg.audio.sample_rate)

            # -----------------------------------------
            # Phase 2: feature extraction
            # -----------------------------------------
            vector = extract_feature_vector(
                signal,
                sr=sr,
            )

            # Keep important metadata from the manifest
            result = {
                "utt_id": row["utt_id"],
                "label": row["label"],
                "label_id": row["label_id"],
                "speaker": row["speaker"],
                "attack": row["attack"],
                "source_corpus": row["source_corpus"],
                "split": row["split"],
            }

            # Add the 92 extracted feature values
            for name, value in zip(
                feature_names,
                vector,
            ):
                result[name] = float(value)

            rows.append(result)

        except Exception as exc:
            print(
                f"WARNING: Failed to process "
                f"{row['utt_id']}: {exc}"
            )

    # Convert extracted features to DataFrame
    feature_df = pd.DataFrame(rows)

    # Create the output directory if necessary
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save features to CSV
    feature_df.to_csv(
        output_path,
        index=False,
    )

    print()
    print("Feature extraction complete.")
    print(f"Processed: {len(feature_df)} / {total}")
    print(f"Feature file: {output_path}")
    print(f"Feature columns: {len(feature_names)}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract Phase 2 speech features "
            "from an audio manifest."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/train.csv"),
        help="Input manifest CSV file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "features/train_features.csv"
        ),
        help="Output feature CSV file.",
    )

    args = parser.parse_args()

    manifest_path = (
        PROJECT_ROOT / args.manifest
    )

    output_path = (
        PROJECT_ROOT / args.output
    )

    # Make sure the manifest exists
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: "
            f"{manifest_path}"
        )

    extract_features_from_manifest(
        manifest_path,
        output_path,
    )


if __name__ == "__main__":
    main()