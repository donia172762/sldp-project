from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "manifests"
    / "df_eval_part00.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "manifests"
    / "df_eval_part00_balanced.csv"
)

RANDOM_SEED = 1337


def main():

    print("=" * 70)
    print("CREATE BALANCED ASVspoof2021 DF SUBSET")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    bonafide = df[
        df["label"] == "bonafide"
    ].copy()

    spoof = df[
        df["label"] == "spoof"
    ].copy()

    print(
        f"Original bonafide: {len(bonafide)}"
    )

    print(
        f"Original spoof    : {len(spoof)}"
    )

    # Use all bonafide files and
    # sample the same number of spoof files.
    spoof_sample = spoof.sample(
        n=len(bonafide),
        random_state=RANDOM_SEED
    )

    balanced = pd.concat(
        [
            bonafide,
            spoof_sample
        ],
        ignore_index=True
    )

    # Shuffle the final subset.
    balanced = balanced.sample(
        frac=1.0,
        random_state=RANDOM_SEED
    ).reset_index(
        drop=True
    )

    balanced.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("=" * 70)
    print("BALANCED SUBSET")
    print("=" * 70)

    print(
        balanced["label"]
        .value_counts()
        .to_string()
    )

    print()

    print(
        f"Total files: {len(balanced)}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()