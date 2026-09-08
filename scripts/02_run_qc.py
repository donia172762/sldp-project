"""Run the audio quality-control pass over a manifest.

    python scripts/02_run_qc.py
    python scripts/02_run_qc.py --manifest manifests/train.csv --max-files 500

Reads a manifest, measures a capped random sample of the audio, writes
``reports/qc_per_file.csv`` and ``reports/qc_summary.csv``, and saves the
figures used in the report to ``reports/figures/``.

The pass is read-only: it never modifies the corpora and never applies the
preprocessing chain, because its job is to describe the audio as it arrives.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import manifest as manifest_module  # noqa: E402
from src import qc as qc_module  # noqa: E402
from src.config import load_config, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Quality-control pass over the audio listed in a manifest.",
    )
    parser.add_argument(
        "--manifest", type=str, default=None,
        help="Manifest to inspect. Defaults to manifests/all.<format>.",
    )
    parser.add_argument(
        "--max-files", type=int, default=None,
        help="Cap on files measured. Defaults to qc.max_files in config.yaml.",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to an alternative config.yaml.",
    )
    return parser.parse_args()


def main() -> int:
    """Measure, summarise, plot. Returns a process exit code."""
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    fmt = str(cfg.manifest.format)
    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else cfg.path("manifests", f"all.{fmt}")
    )

    if not manifest_path.is_file():
        print(f"Manifest not found: {manifest_path}")
        print("Run scripts/01_build_manifest.py first.")
        return 1

    frame = manifest_module.load_manifest(manifest_path)
    max_files = args.max_files if args.max_files is not None else int(cfg.qc.max_files)

    print("=" * 72)
    print("AUDIO QUALITY CONTROL")
    print("=" * 72)
    print(f"  manifest   {manifest_path.name}  ({len(frame)} utterances)")
    print(f"  measuring  up to {max_files} files, sampled per split, seed "
          f"{int(cfg.qc.sample_seed)}")
    print()

    qc_frame = qc_module.run_qc(
        frame,
        project_root=cfg.project_root,
        max_files=max_files,
        seed=int(cfg.qc.sample_seed),
        clipping_threshold=float(cfg.qc.clipping_threshold),
        silence_top_db=float(cfg.qc.silence_top_db),
    )

    if qc_frame.empty:
        print("Nothing was measured (the manifest is empty).")
        return 1

    # --- Corrupt / unreadable files ----------------------------------------
    corrupt = qc_frame[~qc_frame["ok"]]
    print()
    print(f"Measured {len(qc_frame)} files; {len(corrupt)} unreadable.")
    if len(corrupt):
        print("\nUnreadable files (first 10):")
        for row in corrupt.head(10).itertuples(index=False):
            print(f"  {row.utt_id}: {row.error}")

    good = qc_frame[qc_frame["ok"]]

    # --- Headline numbers ---------------------------------------------------
    rates = sorted(int(r) for r in good["sample_rate"].unique())
    print()
    print("Sample rates encountered:", ", ".join(f"{r} Hz" for r in rates))
    if rates != [int(cfg.audio.sample_rate)]:
        print(f"  NOTE: not all files are at {int(cfg.audio.sample_rate)} Hz, so the "
              "front end is resampling. Resampling leaves its own high-frequency")
        print("  signature - keep it in mind when interpreting high-band artefacts.")

    n_clipped = int((good["clipping_rate"] > 0).sum())
    max_dc = float(good["dc_offset"].abs().max())
    print(f"Files containing clipped samples: {n_clipped} / {len(good)}")
    print(f"Largest absolute DC offset:       {max_dc:.6f}")

    # --- Per-(split, label) summary ----------------------------------------
    summary = qc_module.summarize_qc(qc_frame)
    print()
    print("=" * 72)
    print("SUMMARY BY SPLIT AND CLASS")
    print("=" * 72)
    with_pandas_width = summary.to_string(index=False)
    print(with_pandas_width)

    # --- Write outputs ------------------------------------------------------
    reports_dir = cfg.ensure_dir("reports")
    figures_dir = cfg.ensure_dir("figures")

    per_file_path = reports_dir / "qc_per_file.csv"
    summary_path = reports_dir / "qc_summary.csv"
    qc_frame.to_csv(per_file_path, index=False)
    summary.to_csv(summary_path, index=False)

    figure_paths = qc_module.make_figures(
        qc_frame, frame, figures_dir, dpi=int(cfg.qc.figure_dpi)
    )

    print()
    print("=" * 72)
    print("WROTE")
    print("=" * 72)
    for path in [per_file_path, summary_path, *figure_paths]:
        print(f"  {path.relative_to(cfg.project_root)}")
    print("\nThe three figures are report-ready: durations justify the 4 s fixed")
    print("length, level/silence justify the preprocessing defaults, and class")
    print("balance justifies reporting EER rather than plain accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
