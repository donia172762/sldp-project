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
EVAL_FILE = PROJECT_ROOT / "features" / "eval_features.csv"

VALIDATION_RESULTS_FILE = (
    PROJECT_ROOT
    / "reports"
    / "classical_results.csv"
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
    """
    Calculate Equal Error Rate.
    """

    fpr, tpr, thresholds = roc_curve(
        y_true,
        scores,
        pos_label=1
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


def evaluate(
    model_name,
    feature_name,
    model,
    x_train,
    y_train,
    x_eval,
    y_eval,
):
    print()
    print("=" * 72)
    print(
        f"{model_name} | "
        f"{feature_name}"
    )
    print("=" * 72)

    print("Training on train + validation...")

    model.fit(
        x_train,
        y_train
    )

    print("Evaluating on final eval set...")

    predictions = model.predict(
        x_eval
    )

    if hasattr(
        model,
        "decision_function"
    ):
        scores = model.decision_function(
            x_eval
        )

    else:
        scores = model.predict_proba(
            x_eval
        )[:, 1]

    accuracy = accuracy_score(
        y_eval,
        predictions
    )

    precision = precision_score(
        y_eval,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_eval,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_eval,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_eval,
        scores
    )

    eer, eer_threshold = calculate_eer(
        y_eval.to_numpy(),
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

    # ---------------------------------
    # Save confusion matrix
    # ---------------------------------

    cm = confusion_matrix(
        y_eval,
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
        f"Eval - {model_name} - "
        f"{feature_name}"
    )

    filename = (
        f"eval_cm_"
        f"{model_name}_"
        f"{feature_name}"
    )

    filename = (
        filename
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


def save_partial(results):
    """
    Save completed experiments immediately.
    """

    path = (
        REPORTS_DIR
        / "eval_all_models.csv"
    )

    pd.DataFrame(
        results
    ).to_csv(
        path,
        index=False
    )


def main():

    print("=" * 72)
    print("EVALUATION OF ALL PREDEFINED MODELS")
    print("=" * 72)

    print("Loading feature files...")

    train_df = pd.read_csv(
        TRAIN_FILE
    )

    val_df = pd.read_csv(
        VAL_FILE
    )

    eval_df = pd.read_csv(
        EVAL_FILE
    )

    development_df = pd.concat(
        [
            train_df,
            val_df
        ],
        ignore_index=True
    )

    print(
        f"Train samples      : "
        f"{len(train_df)}"
    )

    print(
        f"Validation samples : "
        f"{len(val_df)}"
    )

    print(
        f"Development total  : "
        f"{len(development_df)}"
    )

    print(
        f"Evaluation samples : "
        f"{len(eval_df)}"
    )

    y_development = (
        development_df["label_id"]
    )

    y_eval = eval_df["label_id"]

    # ---------------------------------
    # Feature groups
    # ---------------------------------

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

    combined_columns = (
        mfcc_columns
        + lfcc_columns
        + spectral_columns
        + time_columns
    )

    feature_sets = {
        "MFCC": mfcc_columns,

        "LFCC": lfcc_columns,

        "MFCC+Spectral": (
            mfcc_columns
            + spectral_columns
            + time_columns
        ),

        "Combined": combined_columns,
    }

    print()
    print("Feature dimensions:")

    for name, columns in feature_sets.items():
        print(
            f"{name}: {len(columns)}"
        )

    results = []

    # ---------------------------------
    # Evaluate all predefined setups
    # ---------------------------------

    for feature_name, columns in feature_sets.items():

        x_development = (
            development_df[columns]
        )

        x_eval = eval_df[columns]

        # =============================
        # Linear SVM
        # =============================

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

        result = evaluate(
            "SVM",
            feature_name,
            svm,
            x_development,
            y_development,
            x_eval,
            y_eval,
        )

        results.append(result)

        save_partial(results)

        # =============================
        # Random Forest
        # =============================

        rf = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=1337,
            n_jobs=-1,
        )

        result = evaluate(
            "Random Forest",
            feature_name,
            rf,
            x_development,
            y_development,
            x_eval,
            y_eval,
        )

        results.append(result)

        save_partial(results)

    # ---------------------------------
    # Final eval results
    # ---------------------------------

    eval_results = pd.DataFrame(
        results
    )

    eval_path = (
        REPORTS_DIR
        / "eval_all_models.csv"
    )

    eval_results.to_csv(
        eval_path,
        index=False
    )

    print()
    print("=" * 72)
    print("FINAL EVAL COMPARISON")
    print("=" * 72)

    print(
        eval_results.to_string(
            index=False
        )
    )

    # ---------------------------------
    # Validation vs Eval comparison
    # ---------------------------------

    if VALIDATION_RESULTS_FILE.exists():

        validation = pd.read_csv(
            VALIDATION_RESULTS_FILE
        )

        metrics = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "eer",
        ]

        validation_small = validation[
            [
                "model",
                "features"
            ]
            + metrics
        ].copy()

        eval_small = eval_results[
            [
                "model",
                "features"
            ]
            + metrics
        ].copy()

        validation_small = (
            validation_small.rename(
                columns={
                    metric:
                    f"val_{metric}"
                    for metric in metrics
                }
            )
        )

        eval_small = (
            eval_small.rename(
                columns={
                    metric:
                    f"eval_{metric}"
                    for metric in metrics
                }
            )
        )

        comparison = pd.merge(
            validation_small,
            eval_small,
            on=[
                "model",
                "features"
            ],
            how="inner"
        )

        comparison[
            "roc_auc_drop"
        ] = (
            comparison["val_roc_auc"]
            - comparison["eval_roc_auc"]
        )

        comparison[
            "eer_increase"
        ] = (
            comparison["eval_eer"]
            - comparison["val_eer"]
        )

        comparison_path = (
            REPORTS_DIR
            / "validation_vs_eval.csv"
        )

        comparison.to_csv(
            comparison_path,
            index=False
        )

        print()
        print("=" * 72)
        print("VALIDATION VS EVAL")
        print("=" * 72)

        print(
            comparison.to_string(
                index=False
            )
        )

        print()
        print(
            "Saved:"
        )

        print(
            comparison_path
        )

    print()
    print(
        "Eval results saved:"
    )

    print(
        eval_path
    )

    print()
    print(
        "Eval confusion matrices:"
    )

    print(
        FIGURES_DIR
    )


if __name__ == "__main__":
    main()