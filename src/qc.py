"""Quality control over a manifest.

Before extracting a single MFCC we need to know what the audio actually looks
like. This module walks a capped random sample of a manifest and measures, per
file: duration, native sample rate, mean (DC) offset, RMS level, peak level,
clipping rate and silence ratio - and records the files that will not decode at
all.

Why each measurement earns its place in the report:

* **duration** - drives the 4 s fixed-length choice; if most utterances are
  shorter, repeat-padding is doing most of the work and we should say so.
* **sample rate** - anything other than 16 kHz means a resampling step is
  active, and resampling has its own high-frequency signature that a deepfake
  detector can mistake for a synthesis artefact.
* **DC offset** - a non-zero mean biases short-time energy and the zero-crossing
  rate, two of the time-domain features on the project's feature list.
* **RMS / peak / clipping** - tells us whether level differs systematically
  between classes, which is precisely the database artefact we refuse to
  normalise away (see :func:`src.audio.peak_normalize`) and must therefore be
  able to quantify.
* **silence ratio** - measures the ASVspoof silence shortcut directly. If bona
  fide and spoof differ sharply here, that is the confound to discuss.

The pass is read-only and never modifies the corpora.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

# Use the non-interactive Agg backend: the QC script writes PNG files and must
# run the same way in a terminal, in CI, and over SSH with no display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from . import audio as audio_module
from .manifest import resolve_path

# Columns produced by measure_file(), fixed so the summary CSV is stable.
QC_COLUMNS: tuple[str, ...] = (
    "utt_id",
    "path",
    "label",
    "split",
    "source_corpus",
    "ok",
    "error",
    "sample_rate",
    "n_channels",
    "n_samples",
    "duration_s",
    "dc_offset",
    "rms",
    "rms_dbfs",
    "peak",
    "clipping_rate",
    "silence_ratio",
)


def measure_file(
    path: Path,
    clipping_threshold: float = 0.99,
    silence_top_db: float = 30.0,
) -> dict[str, Any]:
    """Measure one audio file.

    The file is read at its **native** sample rate and without any
    preprocessing: the point of QC is to describe the data as it arrives, not
    as the front end will reshape it.

    Args:
        path: Audio file.
        clipping_threshold: A sample with ``|x| >= threshold`` counts as clipped.
        silence_top_db: Frames more than this many dB below the file's peak
            count as silence.

    Returns:
        A dict of measurements. On failure, ``ok`` is False and ``error``
        carries the message; the numeric fields are NaN.
    """
    failure: dict[str, Any] = {
        key: np.nan
        for key in QC_COLUMNS
        if key not in {"utt_id", "path", "label", "split", "source_corpus", "ok", "error"}
    }

    try:
        data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:  # unreadable, truncated, or not audio at all
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", **failure}

    n_channels = int(data.shape[1])
    mono = data.mean(axis=1)

    if mono.size == 0:
        return {"ok": False, "error": "empty file (0 samples)", **failure}

    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(mono.astype(np.float64) ** 2)))
    # -inf would break the histograms, so digital silence is floored at -120 dB.
    rms_dbfs = float(20.0 * np.log10(rms)) if rms > 1e-12 else -120.0

    clipping_rate = float(np.mean(np.abs(mono) >= clipping_threshold))

    return {
        "ok": True,
        "error": "",
        "sample_rate": int(sample_rate),
        "n_channels": n_channels,
        "n_samples": int(mono.size),
        "duration_s": float(mono.size / sample_rate),
        # Mean of the waveform: for a well-recorded file this is ~0.
        "dc_offset": float(np.mean(mono.astype(np.float64))),
        "rms": rms,
        "rms_dbfs": rms_dbfs,
        "peak": peak,
        "clipping_rate": clipping_rate,
        "silence_ratio": silence_ratio(mono, sample_rate, top_db=silence_top_db),
    }


def silence_ratio(
    x: np.ndarray, sample_rate: int, top_db: float = 30.0
) -> float:
    """Fraction of the signal that is near-silent.

    Measured as ``1 - (trimmed length / original length)`` using the same
    :func:`librosa.effects.trim` criterion the optional preprocessing step
    would use. It therefore reports exactly how much audio the
    ``remove_silence`` ablation would discard, which is the number we want when
    arguing about the ASVspoof silence shortcut.

    Args:
        x: 1-D waveform.
        sample_rate: Unused numerically, kept in the signature so the meaning
            of the returned ratio (a proportion of duration) is explicit.
        top_db: Silence threshold below the peak, in dB.

    Returns:
        A value in ``[0, 1]``. Returns 1.0 for an all-silent file.
    """
    del sample_rate  # documented above; trim works in samples
    if x.size == 0:
        return 1.0
    # trim_silence deliberately returns an all-silent input unchanged (an empty
    # waveform is useless downstream), so that case is handled here first
    # rather than being reported as "nothing was silent".
    if float(np.max(np.abs(x))) < 1e-6:
        return 1.0
    trimmed = audio_module.trim_silence(x, top_db=top_db)
    return float(1.0 - trimmed.size / x.size)


def run_qc(
    frame: pd.DataFrame,
    project_root: Path,
    max_files: int = 2000,
    seed: int = 1337,
    clipping_threshold: float = 0.99,
    silence_top_db: float = 30.0,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Measure a capped, stratified-by-split random sample of a manifest.

    Sampling is capped because a full pass over ASVspoof 2019 LA plus 2021 DF
    is hundreds of thousands of files; a few thousand is more than enough to
    characterise the distributions. The sample is drawn per split so that a
    small split is not swamped by a large one, and it is seeded so the QC
    figures in the report are reproducible.

    Args:
        frame: A manifest.
        project_root: Root for resolving relative manifest paths.
        max_files: Maximum number of files to measure in total.
        seed: Sampling seed.
        clipping_threshold: Passed to :func:`measure_file`.
        silence_top_db: Passed to :func:`measure_file`.
        show_progress: Display a tqdm bar.

    Returns:
        A DataFrame with :data:`QC_COLUMNS`, one row per measured file.
    """
    if frame.empty:
        return pd.DataFrame(columns=list(QC_COLUMNS))

    sample = _sample_per_split(frame, max_files=max_files, seed=seed)

    records: list[dict[str, Any]] = []
    iterator = sample.itertuples(index=False)
    if show_progress:
        iterator = tqdm(iterator, total=len(sample), desc="QC", unit="file")

    for row in iterator:
        measurement = measure_file(
            resolve_path(str(row.path), project_root),
            clipping_threshold=clipping_threshold,
            silence_top_db=silence_top_db,
        )
        records.append(
            {
                "utt_id": row.utt_id,
                "path": row.path,
                "label": row.label,
                "split": row.split,
                "source_corpus": row.source_corpus,
                **measurement,
            }
        )

    return pd.DataFrame(records)[list(QC_COLUMNS)]


def _sample_per_split(
    frame: pd.DataFrame, max_files: int, seed: int
) -> pd.DataFrame:
    """Draw at most ``max_files`` rows, spread evenly over the splits."""
    if len(frame) <= max_files:
        return frame

    splits = frame["split"].unique()
    per_split = max(1, max_files // len(splits))

    chunks = [
        group.sample(n=min(per_split, len(group)), random_state=seed)
        for _, group in frame.groupby("split", sort=True)
    ]
    return pd.concat(chunks, ignore_index=True)


def summarize_qc(qc_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-file measurements into a per-(split, label) summary table.

    This table is what goes into the report: it is small enough to print and it
    answers the questions the rubric asks about preprocessing and dataset
    design.

    Args:
        qc_frame: Output of :func:`run_qc`.

    Returns:
        One row per ``(split, label)`` with counts and mean/median statistics.
    """
    if qc_frame.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    for (split, label), group in qc_frame.groupby(["split", "label"], sort=True):
        usable = group[group["ok"]]
        rows.append(
            {
                "split": split,
                "label": label,
                "n_files": len(group),
                "n_corrupt": int((~group["ok"]).sum()),
                "sample_rates": ",".join(
                    sorted(
                        str(int(r)) for r in usable["sample_rate"].dropna().unique()
                    )
                )
                or "-",
                "dur_mean_s": round(float(usable["duration_s"].mean()), 3)
                if len(usable)
                else np.nan,
                "dur_median_s": round(float(usable["duration_s"].median()), 3)
                if len(usable)
                else np.nan,
                "dur_min_s": round(float(usable["duration_s"].min()), 3)
                if len(usable)
                else np.nan,
                "dur_max_s": round(float(usable["duration_s"].max()), 3)
                if len(usable)
                else np.nan,
                "rms_dbfs_mean": round(float(usable["rms_dbfs"].mean()), 2)
                if len(usable)
                else np.nan,
                "peak_mean": round(float(usable["peak"].mean()), 4)
                if len(usable)
                else np.nan,
                "dc_offset_mean": float(usable["dc_offset"].mean())
                if len(usable)
                else np.nan,
                "dc_offset_absmax": float(usable["dc_offset"].abs().max())
                if len(usable)
                else np.nan,
                "clipping_rate_mean": float(usable["clipping_rate"].mean())
                if len(usable)
                else np.nan,
                "silence_ratio_mean": round(float(usable["silence_ratio"].mean()), 4)
                if len(usable)
                else np.nan,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
# Colour-blind-safe pair used consistently for the two classes across all
# figures, so the report's plots read as one set.
CLASS_COLORS: dict[str, str] = {"bonafide": "#0072B2", "spoof": "#D55E00"}


def _class_color(label: str) -> str:
    return CLASS_COLORS.get(label, "#666666")


def plot_duration_histogram(
    qc_frame: pd.DataFrame, output_path: Path, dpi: int = 150
) -> Path:
    """Overlaid duration histograms, one per class.

    Justifies the fixed-length window: the 4 s line shows at a glance what
    fraction of the corpus is padded versus cropped.
    """
    good = qc_frame[qc_frame["ok"]]
    figure, axis = plt.subplots(figsize=(6.5, 3.6))

    if len(good):
        bins = np.histogram_bin_edges(good["duration_s"], bins=40)
        for label, group in good.groupby("label", sort=True):
            axis.hist(
                group["duration_s"],
                bins=bins,
                alpha=0.65,
                label=f"{label} (n={len(group)})",
                color=_class_color(str(label)),
            )
        axis.axvline(
            4.0, color="black", linestyle="--", linewidth=1.2,
            label="fixed length (4 s)",
        )

    axis.set_xlabel("Duration (s)")
    axis.set_ylabel("Number of utterances")
    axis.set_title("Utterance duration by class")
    axis.legend(frameon=False, fontsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)
    return output_path


def plot_level_and_silence(
    qc_frame: pd.DataFrame, output_path: Path, dpi: int = 150
) -> Path:
    """Two panels: RMS level in dBFS, and silence ratio, both split by class.

    The left panel is the evidence behind leaving ``peak_normalize`` off; the
    right panel quantifies the ASVspoof silence shortcut. If the right panel
    shows the two classes separating, that is a confound to report, not a
    result to celebrate.
    """
    good = qc_frame[qc_frame["ok"]]
    figure, (left, right) = plt.subplots(1, 2, figsize=(9.5, 3.6))

    if len(good):
        level_bins = np.histogram_bin_edges(good["rms_dbfs"], bins=35)
        silence_bins = np.linspace(0.0, 1.0, 36)
        for label, group in good.groupby("label", sort=True):
            color = _class_color(str(label))
            left.hist(group["rms_dbfs"], bins=level_bins, alpha=0.65,
                      label=str(label), color=color)
            right.hist(group["silence_ratio"], bins=silence_bins, alpha=0.65,
                       label=str(label), color=color)

    left.set_xlabel("RMS level (dBFS)")
    left.set_ylabel("Number of utterances")
    left.set_title("Signal level by class")
    left.legend(frameon=False, fontsize=8)

    right.set_xlabel("Silence ratio (fraction trimmed at 30 dB)")
    right.set_ylabel("Number of utterances")
    right.set_title("Leading/trailing silence by class")
    right.legend(frameon=False, fontsize=8)

    for axis in (left, right):
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)
    return output_path


def plot_class_balance(
    manifest_frame: pd.DataFrame, output_path: Path, dpi: int = 150
) -> Path:
    """Stacked bar of bona fide vs spoof counts per split.

    ASVspoof is heavily spoof-weighted (roughly 1:9), which is exactly why the
    evaluation section must report balanced metrics and EER rather than plain
    accuracy. This figure is the justification for that choice.
    """
    figure, axis = plt.subplots(figsize=(6.5, 3.6))

    if not manifest_frame.empty:
        counts = (
            manifest_frame.pivot_table(
                index="split", columns="label", values="utt_id",
                aggfunc="count", fill_value=0,
            )
            .sort_index()
        )
        bottom = np.zeros(len(counts))
        positions = np.arange(len(counts))
        for label in [c for c in ("bonafide", "spoof") if c in counts.columns]:
            values = counts[label].to_numpy(dtype=float)
            axis.bar(positions, values, bottom=bottom, label=label,
                     color=_class_color(label), width=0.6)
            for x, (value, base) in enumerate(zip(values, bottom)):
                if value > 0:
                    axis.text(x, base + value / 2, f"{int(value)}",
                              ha="center", va="center", fontsize=8, color="white")
            bottom += values
        axis.set_xticks(positions)
        axis.set_xticklabels(counts.index, rotation=0)

    axis.set_ylabel("Number of utterances")
    axis.set_title("Class balance per split")
    axis.legend(frameon=False, fontsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)
    return output_path


def make_figures(
    qc_frame: pd.DataFrame,
    manifest_frame: pd.DataFrame,
    figures_dir: Path,
    dpi: int = 150,
) -> list[Path]:
    """Write all QC figures and return the paths written."""
    return [
        plot_duration_histogram(qc_frame, figures_dir / "qc_durations.png", dpi),
        plot_level_and_silence(qc_frame, figures_dir / "qc_level_silence.png", dpi),
        plot_class_balance(manifest_frame, figures_dir / "qc_class_balance.png", dpi),
    ]
