from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = PROJECT_ROOT / "features" / "train_features.csv"
VAL_FILE = PROJECT_ROOT / "features" / "val_features.csv"

DF_FILE = (
    PROJECT_ROOT
    / "features"
    / "df_part00_balanced_features.csv"
)

LA_RESULTS_FILE = (
    PROJECT_ROOT
    / "reports"
    / "eval_all_models.csv"
)

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def calculate_eer(y_true, scores):

    fpr, tpr, thresholds = roc_curve(
        y_true,
        scores,
        pos_label=1,
    )

    fnr = 1.0 - tpr

    index = np.nanargmin(
        np.abs(fpr - fnr)
    )

    eer = (
        fpr[index]
        + fnr[index]
    ) / 2.0

    threshold = thresholds[index]

    return float(eer), float(threshold)


def evaluate_model(
    model_name,
    feature_name,
    model,
    x_train,
    y_train,
    x_test,
    y_test,
):

    print()
    print("=" * 72)
    print(f"{model_name} | {feature_name}")
    print("=" * 72)

    print("Training on ASVspoof2019 LA...")

    model.fit(
        x_train,
        y_train
    )

    print(
        "Testing on ASVspoof2021 DF..."
    )

    predictions = model.predict(
        x_test
    )

    if hasattr(
        model,
        "decision_function"
    ):

        scores = model.decision_function(
            x_test
        )

    else:

        scores = model.predict_proba(
            x_test
        )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        scores
    )

    eer, eer_threshold = calculate_eer(
        y_test.to_numpy(),
        np.asarray(scores)
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1-score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    print(
        f"EER      : {eer:.4f}"
    )

    # ----------------------------------
    # Confusion matrix
    # ----------------------------------

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Bonafide",
            "Spoof"
        ]
    )

    display.plot(
        ax=ax,
        values_format="d",
        colorbar=False
    )

    ax.set_title(
        f"DF Cross-Corpus\n"
        f"{model_name} - {feature_name}"
    )

    filename = (
        f"df_cm_{model_name}_{feature_name}"
        .lower()
        .replace(" ", "_")
        .replace("+", "_plus_")
        + ".png"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / filename,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    return {
        "model": model_name,
        "features": feature_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "eer": eer,
        "eer_threshold": eer_threshold,
    }


def save_results(results):

    output = (
        REPORTS_DIR
        / "df_cross_corpus_results.csv"
    )

    pd.DataFrame(
        results
    ).to_csv(
        output,
        index=False
    )


def main():

    print("=" * 72)
    print("ASVSPOOF2021 DF CROSS-CORPUS EVALUATION")
    print("=" * 72)

    print(
        "Loading feature files..."
    )

    train_df = pd.read_csv(
        TRAIN_FILE
    )

    val_df = pd.read_csv(
        VAL_FILE
    )

    df_test = pd.read_csv(
        DF_FILE
    )

    development_df = pd.concat(
        [
            train_df,
            val_df
        ],
        ignore_index=True
    )

    print(
        f"LA training samples : "
        f"{len(development_df)}"
    )

    print(
        f"DF test samples     : "
        f"{len(df_test)}"
    )

    print(
        f"DF bonafide         : "
        f"{(df_test['label'] == 'bonafide').sum()}"
    )

    print(
        f"DF spoof            : "
        f"{(df_test['label'] == 'spoof').sum()}"
    )

    y_train = (
        development_df["label_id"]
    )

    y_test = (
        df_test["label_id"]
    )

    # ----------------------------------
    # Feature groups
    # ----------------------------------

    mfcc_columns = [
        column
        for column in development_df.columns
        if column.startswith("mfcc_")
    ]

    lfcc_columns = [
        column
        for column in development_df.columns
        if column.startswith("lfcc_")
    ]

    spectral_columns = [
        column
        for column in development_df.columns
        if column.startswith("spectral_")
    ]

    time_columns = [
        column
        for column in development_df.columns
        if (
            column.startswith(
                "short_time_energy"
            )
            or column.startswith(
                "zero_crossing_rate"
            )
        )
    ]

    feature_sets = {

        "MFCC":
            mfcc_columns,

        "LFCC":
            lfcc_columns,

        "MFCC+Spectral":
            (
                mfcc_columns
                + spectral_columns
                + time_columns
            ),

        "Combined":
            (
                mfcc_columns
                + lfcc_columns
                + spectral_columns
                + time_columns
            ),
    }

    print()
    print("Feature dimensions:")

    for name, columns in feature_sets.items():

        print(
            f"{name}: {len(columns)}"
        )

    results = []

    # ----------------------------------
    # Run all experiments
    # ----------------------------------

    for feature_name, columns in feature_sets.items():

        x_train = development_df[
            columns
        ]

        x_test = df_test[
            columns
        ]

        # SVM
        svm = Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LinearSVC(
                    C=1.0,
                    class_weight="balanced",
                    random_state=1337,
                    max_iter=10000,
                )
            ),
        ])

        result = evaluate_model(
            "SVM",
            feature_name,
            svm,
            x_train,
            y_train,
            x_test,
            y_test,
        )

        results.append(
            result
        )

        save_results(
            results
        )

        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=1337,
            n_jobs=-1,
        )

        result = evaluate_model(
            "Random Forest",
            feature_name,
            rf,
            x_train,
            y_train,
            x_test,
            y_test,
        )

        results.append(
            result
        )

        save_results(
            results
        )

    # ----------------------------------
    # DF final table
    # ----------------------------------

    df_results = pd.DataFrame(
        results
    )

    output_file = (
        REPORTS_DIR
        / "df_cross_corpus_results.csv"
    )

    print()
    print("=" * 72)
    print("DF CROSS-CORPUS RESULTS")
    print("=" * 72)

    print(
        df_results.to_string(
            index=False
        )
    )

    # ----------------------------------
    # Compare LA eval vs DF
    # ----------------------------------

    if LA_RESULTS_FILE.exists():

        la_results = pd.read_csv(
            LA_RESULTS_FILE
        )

        metrics = [
            "accuracy",
            "f1",
            "roc_auc",
            "eer",
        ]

        la_small = la_results[
            [
                "model",
                "features"
            ]
            + metrics
        ].copy()

        df_small = df_results[
            [
                "model",
                "features"
            ]
            + metrics
        ].copy()

        la_small = la_small.rename(
            columns={
                metric:
                f"la_{metric}"
                for metric in metrics
            }
        )

        df_small = df_small.rename(
            columns={
                metric:
                f"df_{metric}"
                for metric in metrics
            }
        )

        comparison = pd.merge(
            la_small,
            df_small,
            on=[
                "model",
                "features"
            ],
            how="inner"
        )

        comparison[
            "auc_change"
        ] = (
            comparison["df_roc_auc"]
            - comparison["la_roc_auc"]
        )

        comparison[
            "eer_change"
        ] = (
            comparison["df_eer"]
            - comparison["la_eer"]
        )

        comparison_file = (
            REPORTS_DIR
            / "la_vs_df_generalization.csv"
        )

        comparison.to_csv(
            comparison_file,
            index=False
        )

        print()
        print("=" * 72)
        print("LA EVAL VS DF CROSS-CORPUS")
        print("=" * 72)

        print(
            comparison.to_string(
                index=False
            )
        )

        print()
        print(
            f"Comparison saved: "
            f"{comparison_file}"
        )

    print()
    print(
        f"DF results saved: {output_file}"
    )

    print(
        f"Figures saved: {FIGURES_DIR}"
    )


if __name__ == "__main__":
    main()