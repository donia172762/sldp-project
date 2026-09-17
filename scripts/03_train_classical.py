from pathlib import Path

import numpy as np
import pandas as pd

# Use a non-GUI backend.
# This avoids Tkinter/thread crashes on Windows.
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
    Calculate Equal Error Rate (EER).

    Bona fide = 0
    Spoof = 1
    """

    fpr, tpr, thresholds = roc_curve(
        y_true,
        scores,
        pos_label=1,
    )

    fnr = 1.0 - tpr

    # Find the point where FPR and FNR
    # are closest to each other.
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
    x_val,
    y_val,
):
    print()
    print("=" * 70)
    print(f"{model_name} | {feature_name}")
    print("=" * 70)

    print("Training...")

    model.fit(
        x_train,
        y_train
    )

    predictions = model.predict(
        x_val
    )

    # Continuous scores are needed
    # for ROC-AUC and EER.
    if hasattr(model, "decision_function"):
        scores = model.decision_function(
            x_val
        )

    elif hasattr(model, "predict_proba"):
        scores = model.predict_proba(
            x_val
        )[:, 1]

    else:
        scores = predictions

    accuracy = accuracy_score(
        y_val,
        predictions
    )

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_val,
        scores
    )

    eer, eer_threshold = calculate_eer(
        y_val.to_numpy(),
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

    print(
        f"EER threshold: "
        f"{eer_threshold:.4f}"
    )

    # ---------------------------------
    # Confusion matrix
    # ---------------------------------

    cm = confusion_matrix(
        y_val,
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
        colorbar=False,
    )

    ax.set_title(
        f"{model_name} - {feature_name}"
    )

    figure_name = (
        f"cm_{model_name}_{feature_name}"
        .lower()
        .replace(" ", "_")
        .replace("+", "_plus_")
        + ".png"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / figure_name,
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
    """
    Save results after every experiment.

    This means that even if one later experiment
    fails, completed results are not lost.
    """

    results_df = pd.DataFrame(
        results
    )

    results_path = (
        REPORTS_DIR
        / "classical_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )


def main():

    print("Loading feature files...")

    train_df = pd.read_csv(
        TRAIN_FILE
    )

    val_df = pd.read_csv(
        VAL_FILE
    )

    print(
        f"Train samples: {len(train_df)}"
    )

    print(
        f"Validation samples: {len(val_df)}"
    )

    y_train = train_df["label_id"]
    y_val = val_df["label_id"]

    # ---------------------------------
    # Feature groups
    # ---------------------------------

    mfcc_columns = [
        column
        for column in train_df.columns
        if column.startswith("mfcc_")
    ]

    lfcc_columns = [
        column
        for column in train_df.columns
        if column.startswith("lfcc_")
    ]

    spectral_columns = [
        column
        for column in train_df.columns
        if column.startswith("spectral_")
    ]

    time_columns = [
        column
        for column in train_df.columns
        if (
            column.startswith(
                "short_time_energy"
            )
            or column.startswith(
                "zero_crossing_rate"
            )
        )
    ]

    all_columns = (
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

        "Combined": all_columns,
    }

    print()
    print("Feature dimensions:")

    for name, columns in feature_sets.items():
        print(
            f"{name}: {len(columns)}"
        )

    results = []

    # ---------------------------------
    # Run all experiments
    # ---------------------------------

    for feature_name, columns in feature_sets.items():

        x_train = train_df[
            columns
        ]

        x_val = val_df[
            columns
        ]

        # =============================
        # SVM
        # =============================

        svm_model = Pipeline([
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
            svm_model,
            x_train,
            y_train,
            x_val,
            y_val,
        )

        results.append(
            result
        )

        save_results(
            results
        )

        # =============================
        # Random Forest
        # =============================

        rf_model = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=1337,
            n_jobs=-1,
        )

        result = evaluate_model(
            "Random Forest",
            feature_name,
            rf_model,
            x_train,
            y_train,
            x_val,
            y_val,
        )

        results.append(
            result
        )

        save_results(
            results
        )

    # ---------------------------------
    # Final comparison
    # ---------------------------------

    results_df = pd.DataFrame(
        results
    )

    print()
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        REPORTS_DIR
        / "classical_results.csv"
    )

    print()
    print(
        "Confusion matrices saved to:"
    )

    print(
        FIGURES_DIR
    )


if __name__ == "__main__":
    main()