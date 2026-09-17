from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

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

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def calculate_eer(y_true, scores):
    """
    Calculate Equal Error Rate (EER).

    Class 0 = bonafide
    Class 1 = spoof
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


def main():

    print("=" * 72)
    print("FINAL EVALUATION")
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

    print(
        f"Train samples      : {len(train_df)}"
    )

    print(
        f"Validation samples : {len(val_df)}"
    )

    print(
        f"Evaluation samples : {len(eval_df)}"
    )

    # --------------------------------------------------
    # Combine train + validation after model selection
    # --------------------------------------------------

    development_df = pd.concat(
        [
            train_df,
            val_df
        ],
        ignore_index=True
    )

    print(
        f"Final training pool : {len(development_df)}"
    )

    # --------------------------------------------------
    # Use all 92 features
    # --------------------------------------------------

    metadata_columns = {
        "utt_id",
        "label",
        "label_id",
        "speaker",
        "attack",
        "source_corpus",
        "split",
    }

    feature_columns = [
        column
        for column in development_df.columns
        if column not in metadata_columns
    ]

    print(
        f"Combined features   : {len(feature_columns)}"
    )

    if len(feature_columns) != 92:
        raise ValueError(
            f"Expected 92 features, "
            f"found {len(feature_columns)}"
        )

    x_train = development_df[
        feature_columns
    ]

    y_train = development_df[
        "label_id"
    ]

    x_eval = eval_df[
        feature_columns
    ]

    y_eval = eval_df[
        "label_id"
    ]

    # --------------------------------------------------
    # Final model:
    # Combined features + Linear SVM
    # --------------------------------------------------

    model = Pipeline([
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

    print()
    print("Training final Combined SVM...")

    model.fit(
        x_train,
        y_train
    )

    print("Training complete.")

    # --------------------------------------------------
    # Evaluate on untouched eval set
    # --------------------------------------------------

    print()
    print("Evaluating on eval set...")

    predictions = model.predict(
        x_eval
    )

    scores = model.decision_function(
        x_eval
    )

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
        scores
    )

    print()
    print("=" * 72)
    print("FINAL TEST RESULTS")
    print("=" * 72)

    print(
        f"Accuracy     : {accuracy:.4f}"
    )

    print(
        f"Precision    : {precision:.4f}"
    )

    print(
        f"Recall       : {recall:.4f}"
    )

    print(
        f"F1-score     : {f1:.4f}"
    )

    print(
        f"ROC-AUC      : {roc_auc:.4f}"
    )

    print(
        f"EER          : {eer:.4f}"
    )

    print(
        f"EER threshold: {eer_threshold:.4f}"
    )

    # --------------------------------------------------
    # Save metrics
    # --------------------------------------------------

    metrics_df = pd.DataFrame([
        {
            "model": "Linear SVM",
            "features": "Combined 92",
            "training_samples": len(
                development_df
            ),
            "evaluation_samples": len(
                eval_df
            ),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "eer": eer,
            "eer_threshold": eer_threshold,
        }
    ])

    metrics_path = (
        REPORTS_DIR
        / "final_eval_results.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False
    )

    # --------------------------------------------------
    # Save predictions
    # --------------------------------------------------

    predictions_df = pd.DataFrame({
        "utt_id": eval_df[
            "utt_id"
        ],
        "true_label": eval_df[
            "label"
        ],
        "true_label_id": y_eval,
        "predicted_label_id": predictions,
        "predicted_label": np.where(
            predictions == 1,
            "spoof",
            "bonafide"
        ),
        "svm_score": scores,
    })

    predictions_path = (
        REPORTS_DIR
        / "final_eval_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path,
        index=False
    )

    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------

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
        "Final Combined SVM - Eval Set"
    )

    fig.tight_layout()

    cm_path = (
        FIGURES_DIR
        / "final_confusion_matrix.png"
    )

    fig.savefig(
        cm_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    # --------------------------------------------------
    # ROC curve
    # --------------------------------------------------

    fpr, tpr, _ = roc_curve(
        y_eval,
        scores,
        pos_label=1
    )

    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    ax.plot(
        fpr,
        tpr,
        label=f"AUC = {roc_auc:.4f}"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "ROC Curve - Final Combined SVM"
    )

    ax.legend()

    fig.tight_layout()

    roc_path = (
        FIGURES_DIR
        / "final_roc_curve.png"
    )

    fig.savefig(
        roc_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    # --------------------------------------------------
    # Save trained model for live demo
    # --------------------------------------------------

    model_path = (
        MODELS_DIR
        / "final_svm_combined.joblib"
    )

    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
        },
        model_path
    )

    print()
    print("=" * 72)
    print("FILES SAVED")
    print("=" * 72)

    print(
        metrics_path
    )

    print(
        predictions_path
    )

    print(
        cm_path
    )

    print(
        roc_path
    )

    print(
        model_path
    )


if __name__ == "__main__":
    main()