"""Build the project manifests and verify that no split leaks into another.

    python scripts/01_build_manifest.py --synthetic          # smoke test
    python scripts/01_build_manifest.py --asvspoof2019       # real training data
    python scripts/01_build_manifest.py --all                # everything present

The script writes one combined manifest plus one file per split, prints the
class balance and attack-type counts, and then runs :func:`check_leakage`.
A leakage failure is a hard error: it exits non-zero and writes nothing, so a
compromised manifest can never be picked up silently by a later stage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import manifest as manifest_module  # noqa: E402
from src.config import load_config, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Build train/val/eval manifests from the available corpora.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Include the synthetic smoke-test dataset.",
    )
    parser.add_argument(
        "--asvspoof2019", action="store_true",
        help="Include ASVspoof 2019 LA (train/dev/eval).",
    )
    parser.add_argument(
        "--asvspoof2021", action="store_true",
        help="Include ASVspoof 2021 DF (evaluation only).",
    )
    parser.add_argument(
        "--wavefake", action="store_true",
        help="Include WaveFake (cross-generator evaluation only).",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Include every source that is present on disk.",
    )
    parser.add_argument(
        "--no-hash-check", action="store_true",
        help="Skip the duplicate-audio hash check (faster on large corpora).",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to an alternative config.yaml.",
    )
    return parser.parse_args()


def main() -> int:
    """Build, validate and write the manifests. Returns a process exit code."""
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)

    use_synthetic = args.synthetic or args.all
    use_2019 = args.asvspoof2019 or args.all
    use_2021 = args.asvspoof2021 or args.all
    use_wavefake = args.wavefake or args.all

    if not any((use_synthetic, use_2019, use_2021, use_wavefake)):
        print("Nothing selected. Pass --synthetic, --asvspoof2019, ... or --all.")
        return 2

    frames: list[pd.DataFrame] = []

    # --- Training pool -----------------------------------------------------
    # Exactly one source may supply the 'train' split, because the validation
    # split is carved out of it by speaker. Mixing two would make "speaker"
    # ambiguous across corpora that number their speakers independently, and
    # the disjointness guarantee would become meaningless.
    training_pool = manifest_module.empty_manifest()

    if use_2019:
        print("\n[1/4] ASVspoof 2019 LA ...")
        training_pool = manifest_module.build_asvspoof2019_la(cfg)

    if training_pool.empty and use_synthetic:
        if use_2019:
            print("      ASVspoof 2019 LA is not available; falling back to the "
                  "synthetic demo.")
        print("\n[1/4] Synthetic smoke-test dataset ...")
        print("      NOTE: SMOKE TEST ONLY. Nothing measured here is a result.")
        training_pool = manifest_module.build_synthetic(cfg)
    elif use_2019 and use_synthetic and not training_pool.empty:
        print("      Ignoring the synthetic demo: ASVspoof 2019 LA is present and")
        print("      supplies the training split. The demo is only a stand-in.")

    frames.append(training_pool)

    # --- Evaluation-only sources -------------------------------------------
    if use_2021:
        print("\n[2/4] ASVspoof 2021 DF (evaluation only) ...")
        frames.append(manifest_module.build_asvspoof2021_df(cfg))

    if use_wavefake:
        print("\n[3/4] WaveFake (cross-generator evaluation only) ...")
        frames.append(manifest_module.build_wavefake(cfg))

    combined = manifest_module.concat_manifests(frames)
    if combined.empty:
        print("\nNo data found. Check the paths in config.yaml, or run:")
        print("    python scripts/make_synthetic_demo.py")
        return 1

    # --- Speaker-disjoint validation split ---------------------------------
    print("\n[4/4] Carving a speaker-disjoint validation split ...")
    if (combined["split"] == "train").any():
        combined = manifest_module.make_speaker_disjoint_split(
            combined,
            val_fraction=float(cfg.manifest.val_fraction),
            seed=int(cfg.manifest.split_seed),
        )
        train_speakers = set(combined.loc[combined["split"] == "train", "speaker"])
        val_speakers = set(combined.loc[combined["split"] == "val", "speaker"])
        print(f"      train: {len(train_speakers)} speakers, "
              f"{int((combined['split'] == 'train').sum())} utterances")
        print(f"      val:   {len(val_speakers)} speakers, "
              f"{int((combined['split'] == 'val').sum())} utterances")
        print(f"      speaker overlap: {len(train_speakers & val_speakers)} "
              f"(must be 0)")
    else:
        print("      No 'train' split present; nothing to carve.")

    # --- Summary -----------------------------------------------------------
    print("\n" + "=" * 72)
    print("MANIFEST SUMMARY")
    print("=" * 72)
    summary_text = manifest_module.summarize(combined)
    print(summary_text)

    # --- Leakage check -----------------------------------------------------
    print("\n" + "=" * 72)
    print("LEAKAGE CHECK")
    print("=" * 72)
    report = manifest_module.check_leakage(
        combined,
        project_root=cfg.project_root,
        check_hashes=not args.no_hash_check,
        hash_bytes=int(cfg.manifest.hash_bytes),
        raise_on_error=False,
    )

    for note in report["notes"]:
        print(f"  note:  {note}")

    if report["errors"]:
        for error in report["errors"]:
            print(f"  ERROR: {error}")
        print("\nLeakage detected. No manifest was written.")
        return 1

    checks = [
        "no utt_id appears in two splits",
        "no speaker appears in a training split and elsewhere",
        "no identical audio file appears in two splits"
        if not args.no_hash_check
        else "duplicate-audio check SKIPPED (--no-hash-check)",
    ]
    for check in checks:
        print(f"  OK:    {check}")

    # --- Write -------------------------------------------------------------
    manifest_dir = cfg.ensure_dir("manifests")
    fmt = str(cfg.manifest.format)

    written = [
        manifest_module.save_manifest(combined, manifest_dir / "all", fmt=fmt)
    ]
    for split_name, group in combined.groupby("split", sort=True):
        written.append(
            manifest_module.save_manifest(group, manifest_dir / str(split_name), fmt=fmt)
        )

    reports_dir = cfg.ensure_dir("reports")
    summary_path = reports_dir / "manifest_summary.txt"
    summary_path.write_text(summary_text + "\n", encoding="utf-8")

    print("\n" + "=" * 72)
    print("WROTE")
    print("=" * 72)
    for path in written:
        print(f"  {path.relative_to(cfg.project_root)}  ({len(manifest_module.load_manifest(path))} rows)")
    print(f"  {summary_path.relative_to(cfg.project_root)}")
    print("\nNext: python scripts/02_run_qc.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
