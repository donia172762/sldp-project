from pathlib import Path

import pandas as pd

from src.config import load_config
from src.manifest import (
    parse_protocol_file,
    LABEL_TO_ID,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

AUDIO_DIR = (
    PROJECT_ROOT
    / "data"
    / "DF"
    / "ASVspoof2021_DF_eval_part00"
    / "flac"
)

METADATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "DF"
    / "keys"
    / "DF"
    / "CM"
    / "trial_metadata.txt"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "manifests"
    / "df_eval_part00.csv"
)


def main():

    print("=" * 70)
    print("ASVspoof2021 DF part00 manifest")
    print("=" * 70)

    if not AUDIO_DIR.exists():
        raise FileNotFoundError(
            f"Audio folder not found: {AUDIO_DIR}"
        )

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_FILE}"
        )

    print("Reading metadata...")

    metadata = parse_protocol_file(
        METADATA_FILE
    )

    print(
        f"Metadata entries: {len(metadata)}"
    )

    # Find only audio actually downloaded in part00
    audio_files = {
        path.stem: path
        for path in AUDIO_DIR.glob("*.flac")
    }

    print(
        f"FLAC files in part00: {len(audio_files)}"
    )

    # Keep only metadata rows whose audio exists
    subset = metadata[
        metadata["utt_id"].isin(
            audio_files.keys()
        )
    ].copy()

    subset["path"] = subset["utt_id"].map(
        lambda utt: (
            audio_files[utt]
            .relative_to(PROJECT_ROOT)
            .as_posix()
        )
    )

    subset["label"] = (
        subset["label"]
        .str.lower()
    )

    subset["label_id"] = (
        subset["label"]
        .map(LABEL_TO_ID)
    )

    subset["source_corpus"] = (
        "asvspoof2021_df"
    )

    subset["split"] = (
        "df_eval_part00"
    )

    subset = subset[
        [
            "utt_id",
            "path",
            "label",
            "label_id",
            "speaker",
            "attack",
            "source_corpus",
            "split",
        ]
    ]

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    subset.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 70)
    print("DF PART00 SUMMARY")
    print("=" * 70)

    print(
        subset["label"]
        .value_counts()
        .to_string()
    )

    print()

    print(
        f"Total usable files: {len(subset)}"
    )

    print(
        f"Manifest saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()