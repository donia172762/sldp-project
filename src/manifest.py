"""Manifest construction: turn three very different corpora into one table.

A *manifest* is a tidy table with one row per utterance and always the same
columns::

    utt_id, path, label, label_id, speaker, attack, source_corpus, split

Everything downstream (feature extraction, training, evaluation) reads a
manifest and never touches a corpus directory directly. That gives us one
place to enforce the property the rubric cares about most: no data leakage
between the training material and anything we report numbers on.

Three sources are supported:

* **ASVspoof 2019 LA** - training and validation material. Ships protocol
  files listing speaker, utterance, attack and key.
* **ASVspoof 2021 DF** - evaluation only. Ships ``trial_metadata.txt``, whose
  columns are *not* the same as the 2019 protocol's.
* **WaveFake** - cross-generator generalisation. No protocol at all; the
  directory layout is the label.

Because the two ASVspoof formats differ in both column count and column order,
the parser here never indexes a fixed column. It identifies each field by what
it *looks like* (see :func:`parse_protocol_line`), which makes it robust to the
protocol variants that ship with different releases of the corpora.
"""

from __future__ import annotations

import hashlib
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# The canonical column order. Kept as a constant so tests and later phases can
# assert against it rather than against a literal list.
MANIFEST_COLUMNS: tuple[str, ...] = (
    "utt_id",
    "path",
    "label",
    "label_id",
    "speaker",
    "attack",
    "source_corpus",
    "split",
)

# Label convention for the whole project: bona fide = 0, spoof = 1.
# Spoof is the positive class, so "recall" reads as "fraction of deepfakes
# caught", which is the operationally meaningful direction.
LABEL_TO_ID: dict[str, int] = {"bonafide": 0, "spoof": 1}

# Placeholders used when a field genuinely does not exist for a row, e.g. a
# bona fide utterance has no attack, WaveFake has no speaker labels.
UNKNOWN_SPEAKER = "unknown"
NO_ATTACK = "bonafide"

# --- Token patterns used to identify protocol fields by shape ---------------
# ASVspoof utterance ids:  LA_T_1138215, LA_D_1047731, LA_E_2482298, DF_E_2000011
RE_UTT_ID = re.compile(r"^(LA|PA|DF)_[TDE]_\d+$")
# ASVspoof speaker ids:    LA_0079, PA_0021   (four digits, no T/D/E part)
RE_SPEAKER = re.compile(r"^(LA|PA)_\d{4}$")
# Attack / system ids:     A01 .. A19
RE_ATTACK = re.compile(r"^A\d{2}$")
# Class key
RE_LABEL = re.compile(r"^(bonafide|spoof)$", re.IGNORECASE)

# WaveFake ships one directory per vocoder. The directory name *is* the attack
# label. Names in the wild look like "ljspeech_melgan" or just "melgan"; we
# strip a leading corpus prefix so the attack label is comparable across the
# LJSpeech and JSUT halves.
WAVEFAKE_CORPUS_PREFIXES: tuple[str, ...] = ("ljspeech_", "jsut_")
# Substrings that identify a directory as a *generator's* output. Matching on
# the architecture name is safer than matching on the corpus name, because the
# genuine JSUT directory is itself called "jsut_ver1.1" - a corpus-prefix rule
# alone would misfile it as a generator.
WAVEFAKE_GENERATOR_KEYWORDS: tuple[str, ...] = (
    "melgan",
    "wavegan",
    "hifigan",
    "waveglow",
    "wavernn",
    "waveflow",
    "avocodo",
    "bigvgan",
    "diffwave",
    "vocoder",
    "tts",
    "gan",
)
# Substrings that identify a directory as genuine recordings.
WAVEFAKE_REAL_DIR_HINTS: tuple[str, ...] = ("ljspeech", "jsut", "real", "bonafide")


class LeakageError(AssertionError):
    """Raised by :func:`check_leakage` when a split boundary has been violated."""


# ---------------------------------------------------------------------------
# Protocol parsing
# ---------------------------------------------------------------------------
def parse_protocol_line(line: str) -> dict[str, str] | None:
    """Extract the fields of one protocol line by token *shape*, not position.

    Handles both formats we need without a per-format branch:

    ASVspoof 2019 LA (5 columns)::

        LA_0079 LA_T_1138215 - - bonafide
        LA_0079 LA_T_1272637 - A01 spoof

    ASVspoof 2021 DF ``trial_metadata.txt`` (8+ columns, different order and
    extra fields)::

        LA_0023 DF_E_2000001 nocodec asvspoof A07 spoof notrim eval

    The identification rules are:

    ===========  =====================================================
    field        rule
    ===========  =====================================================
    ``utt_id``   matches ``^(LA|PA|DF)_[TDE]_\\d+$``
    ``speaker``  matches ``^(LA|PA)_\\d{4}$``
    ``attack``   matches ``^A\\d\\d$``
    ``label``    the token ``bonafide`` or ``spoof``
    ===========  =====================================================

    Args:
        line: One raw line from a protocol file.

    Returns:
        A dict with keys ``utt_id``, ``speaker``, ``attack``, ``label``, or
        ``None`` for blank/comment lines and for lines with no identifiable
        utterance id or key (the caller warns about those).
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    tokens = stripped.split()

    utt_id: str | None = None
    speaker: str | None = None
    attack: str | None = None
    label: str | None = None

    for token in tokens:
        if utt_id is None and RE_UTT_ID.match(token):
            utt_id = token
        elif speaker is None and RE_SPEAKER.match(token):
            speaker = token
        elif attack is None and RE_ATTACK.match(token):
            attack = token
        elif label is None and RE_LABEL.match(token):
            label = token.lower()

    # An utterance id and a key are the minimum needed to build a row.
    if utt_id is None or label is None:
        return None

    return {
        "utt_id": utt_id,
        "speaker": speaker if speaker is not None else UNKNOWN_SPEAKER,
        # Bona fide rows carry no attack id; label them explicitly rather than
        # leaving a NaN, so `groupby('attack')` in the report is complete.
        "attack": attack if attack is not None else NO_ATTACK,
        "label": label,
    }


def parse_protocol_file(protocol_path: str | Path) -> pd.DataFrame:
    """Parse a whole ASVspoof protocol / metadata file.

    Lines that cannot be parsed are counted and reported once as a warning
    rather than raising, so a single malformed trailing line does not stop a
    build over 100k utterances.

    Args:
        protocol_path: Path to ``ASVspoof2019.LA.cm.train.trn.txt``,
            ``trial_metadata.txt``, or the synthetic demo's protocol.

    Returns:
        DataFrame with columns ``utt_id, speaker, attack, label``.

    Raises:
        FileNotFoundError: if the protocol file does not exist.
        ValueError: if not a single line could be parsed.
    """
    path = Path(protocol_path)
    if not path.is_file():
        raise FileNotFoundError(f"Protocol file not found: {path}")

    rows: list[dict[str, str]] = []
    skipped = 0

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parsed = parse_protocol_line(line)
            if parsed is None:
                if line.strip() and not line.strip().startswith("#"):
                    skipped += 1
            else:
                rows.append(parsed)

    if not rows:
        raise ValueError(
            f"No usable lines in protocol {path}. "
            "Expected each line to contain an utterance id such as LA_T_1138215 "
            "and the key 'bonafide' or 'spoof'."
        )

    if skipped:
        warnings.warn(
            f"{path.name}: skipped {skipped} unparsable line(s) out of "
            f"{skipped + len(rows)}.",
            stacklevel=2,
        )

    frame = pd.DataFrame(rows)
    n_missing_speaker = int((frame["speaker"] == UNKNOWN_SPEAKER).sum())
    if n_missing_speaker:
        warnings.warn(
            f"{path.name}: {n_missing_speaker} row(s) had no recognisable speaker id; "
            f"they were labelled '{UNKNOWN_SPEAKER}'. Speaker-disjoint splitting "
            "cannot protect those rows.",
            stacklevel=2,
        )
    return frame


# ---------------------------------------------------------------------------
# Path handling
# ---------------------------------------------------------------------------
def find_audio_dir(corpus_root: str | Path, extension: str = ".flac") -> Path:
    """Locate the directory holding the audio for a corpus.

    ASVspoof releases nest audio at slightly different depths
    (``LA/ASVspoof2019_LA_train/flac``, ``ASVspoof2021_DF_eval/flac``, ...), so
    rather than hardcoding a layout we look for the first directory that
    actually contains files with the expected extension, preferring one named
    ``flac`` or ``wav``.

    Args:
        corpus_root: Directory to search under.
        extension: File extension to look for, including the dot.

    Returns:
        The directory containing the audio.

    Raises:
        FileNotFoundError: if no such directory exists.
    """
    root = Path(corpus_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {root}")

    if any(root.glob(f"*{extension}")):
        return root

    candidates = sorted(
        (d for d in root.rglob("*") if d.is_dir() and any(d.glob(f"*{extension}"))),
        # Prefer conventionally named directories, then the shallowest path.
        key=lambda d: (d.name.lower() not in {"flac", "wav"}, len(d.parts)),
    )
    if not candidates:
        raise FileNotFoundError(
            f"No directory containing '*{extension}' files was found under {root}."
        )
    return candidates[0]


def relativize(path: Path, root: Path) -> str:
    """Store paths relative to the project root when possible.

    A manifest that contains ``data/LA/.../LA_T_1138215.flac`` can be shared
    between the three of us and with the instructor; one containing
    ``C:/Users/<name>/...`` cannot. Paths outside the project (e.g. corpora on
    an external drive) are kept absolute.
    """
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def resolve_path(path_value: str, root: Path) -> Path:
    """Inverse of :func:`relativize`: manifest string -> usable absolute path."""
    candidate = Path(path_value)
    return candidate if candidate.is_absolute() else (root / candidate)


# ---------------------------------------------------------------------------
# Manifest builders
# ---------------------------------------------------------------------------
def _finalize(frame: pd.DataFrame, source_corpus: str, split: str) -> pd.DataFrame:
    """Add the derived columns and enforce the canonical column order."""
    frame = frame.copy()
    frame["label"] = frame["label"].str.lower()
    frame["label_id"] = frame["label"].map(LABEL_TO_ID).astype("int64")
    frame["source_corpus"] = source_corpus
    frame["split"] = split
    return frame[list(MANIFEST_COLUMNS)].reset_index(drop=True)


def build_manifest_from_protocol(
    protocol_path: str | Path,
    audio_dir: str | Path,
    source_corpus: str,
    split: str,
    project_root: Path,
    extension: str = ".flac",
    require_audio: bool = True,
) -> pd.DataFrame:
    """Build a manifest from an ASVspoof-style protocol plus an audio directory.

    Used for ASVspoof 2019 LA, ASVspoof 2021 DF and the synthetic demo alike -
    the format differences are absorbed by :func:`parse_protocol_line`.

    Args:
        protocol_path: Protocol / trial-metadata file.
        audio_dir: Directory holding ``<utt_id><extension>`` files.
        source_corpus: Value for the ``source_corpus`` column, e.g.
            ``"asvspoof2019_la"``.
        split: Value for the ``split`` column, e.g. ``"train"``.
        project_root: Used to write portable relative paths.
        extension: Audio file extension.
        require_audio: If True, rows whose audio file is missing are dropped
            (with a warning). If False, they are kept, which is useful for
            parser tests that have no audio at all.

    Returns:
        A manifest DataFrame with :data:`MANIFEST_COLUMNS`.
    """
    frame = parse_protocol_file(protocol_path)
    audio_root = Path(audio_dir)

    paths = [audio_root / f"{utt}{extension}" for utt in frame["utt_id"]]

    if require_audio:
        exists = np.fromiter((p.is_file() for p in paths), dtype=bool, count=len(paths))
        missing = int((~exists).sum())
        if missing:
            warnings.warn(
                f"{source_corpus}/{split}: {missing} of {len(paths)} files listed in "
                f"the protocol were not found under {audio_root}; they were dropped. "
                "This is expected if you downloaded only part of the corpus.",
                stacklevel=2,
            )
        frame = frame.loc[exists].reset_index(drop=True)
        paths = [p for p, keep in zip(paths, exists) if keep]

    frame["path"] = [relativize(p, project_root) for p in paths]
    return _finalize(frame, source_corpus, split)


def build_asvspoof2019_la(
    cfg,
    splits: Sequence[str] = ("train", "dev", "eval"),
) -> pd.DataFrame:
    """Build manifests for the requested ASVspoof 2019 LA partitions.

    Expects the official layout::

        <asvspoof2019_la>/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt
        <asvspoof2019_la>/ASVspoof2019_LA_train/flac/*.flac

    Args:
        cfg: Loaded project config.
        splits: Which partitions to include.

    Returns:
        The concatenated manifest, or an empty manifest if the corpus is absent.
    """
    root = cfg.path("asvspoof2019_la")
    if not root.is_dir():
        warnings.warn(
            f"ASVspoof 2019 LA not found at {root}; skipping. See README for "
            "download instructions.",
            stacklevel=2,
        )
        return empty_manifest()

    protocol_dir = root / "ASVspoof2019_LA_cm_protocols"
    # train uses the .trn suffix, dev/eval use .trl - hence the lookup table.
    protocol_names = {
        "train": "ASVspoof2019.LA.cm.train.trn.txt",
        "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
        "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
    }

    frames: list[pd.DataFrame] = []
    for split in splits:
        protocol = protocol_dir / protocol_names[split]
        audio_root = root / f"ASVspoof2019_LA_{split}"
        if not protocol.is_file() or not audio_root.is_dir():
            warnings.warn(
                f"ASVspoof 2019 LA '{split}' partition incomplete "
                f"(protocol={protocol.is_file()}, audio={audio_root.is_dir()}); skipping.",
                stacklevel=2,
            )
            continue
        frames.append(
            build_manifest_from_protocol(
                protocol_path=protocol,
                audio_dir=find_audio_dir(audio_root, ".flac"),
                source_corpus="asvspoof2019_la",
                # 2019 dev/eval are held out; only 'train' is later re-split.
                split=split,
                project_root=cfg.project_root,
            )
        )

    return concat_manifests(frames)


def build_asvspoof2021_df(cfg) -> pd.DataFrame:
    """Build the ASVspoof 2021 DF manifest. **Evaluation only.**

    The keys ship separately from the audio, so the metadata file is searched
    for by name anywhere under the corpus root.

    Args:
        cfg: Loaded project config.

    Returns:
        A manifest whose ``split`` is ``"eval2021df"``, or an empty manifest.
    """
    root = cfg.path("asvspoof2021_df")
    if not root.is_dir():
        warnings.warn(
            f"ASVspoof 2021 DF not found at {root}; skipping. See README.",
            stacklevel=2,
        )
        return empty_manifest()

    metadata_files = sorted(root.rglob("trial_metadata.txt"))
    if not metadata_files:
        warnings.warn(
            f"No trial_metadata.txt under {root}. The DF keys are a separate "
            "download from the audio; see README.",
            stacklevel=2,
        )
        return empty_manifest()

    return build_manifest_from_protocol(
        protocol_path=metadata_files[0],
        audio_dir=find_audio_dir(root, ".flac"),
        source_corpus="asvspoof2021_df",
        split="eval2021df",
        project_root=cfg.project_root,
    )


def _wavefake_attack_label(directory_name: str) -> str:
    """Turn a WaveFake directory name into a clean generator label.

    ``ljspeech_melgan`` -> ``melgan``, ``jsut_multi_band_melgan`` ->
    ``multi_band_melgan``. Keeping the corpus prefix out of the label means a
    detector's per-generator scores can be compared across the two halves.
    """
    name = directory_name.strip().lower()
    for prefix in WAVEFAKE_CORPUS_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def build_wavefake(cfg, extension: str = ".wav") -> pd.DataFrame:
    """Build the WaveFake manifest by directory scan.

    WaveFake has no protocol file: the layout carries the labels. Expected::

        <wavefake>/LJSpeech-1.1/wavs/*.wav        <- bona fide
        <wavefake>/ljspeech_melgan/*.wav          <- spoof, attack='melgan'
        <wavefake>/ljspeech_hifiGAN/*.wav         <- spoof, attack='hifigan'
        ...

    A top-level directory is treated as bona fide if its name looks like a
    source corpus (LJSpeech / JSUT / real / bonafide) and as a generator
    otherwise, with the directory name used as the attack label.

    Speaker labels are not available: LJSpeech is a single speaker and JSUT is
    a single speaker, so we record the corpus name as the speaker. That is
    honest about what we know and keeps the leakage check meaningful.

    Args:
        cfg: Loaded project config.
        extension: Audio extension to collect.

    Returns:
        A manifest whose ``split`` is ``"evalwavefake"``, or an empty manifest.
    """
    root = cfg.path("wavefake")
    if not root.is_dir():
        warnings.warn(
            f"WaveFake not found at {root}; skipping. See README.", stacklevel=2
        )
        return empty_manifest()

    rows: list[dict[str, object]] = []
    for directory in sorted(d for d in root.iterdir() if d.is_dir()):
        files = sorted(directory.rglob(f"*{extension}"))
        if not files:
            continue

        name = directory.name.lower()
        # Generator keywords win over corpus hints, because a generator
        # directory is named after both ("ljspeech_melgan"). Only a directory
        # with no architecture name in it can be genuine audio.
        is_generator = any(k in name for k in WAVEFAKE_GENERATOR_KEYWORDS)
        is_real = not is_generator and any(
            hint in name for hint in WAVEFAKE_REAL_DIR_HINTS
        )
        if not is_generator and not is_real:
            warnings.warn(
                f"WaveFake: directory '{directory.name}' matches neither a known "
                f"generator nor a source corpus; treating it as spoof. Add it to "
                f"WAVEFAKE_GENERATOR_KEYWORDS or WAVEFAKE_REAL_DIR_HINTS if wrong.",
                stacklevel=2,
            )

        label = "bonafide" if is_real else "spoof"
        attack = NO_ATTACK if is_real else _wavefake_attack_label(directory.name)
        speaker = "ljspeech" if "ljspeech" in name else ("jsut" if "jsut" in name else name)

        for audio_path in files:
            rows.append(
                {
                    # Directory + stem, because the same LJSpeech utterance id
                    # is reused by every generator.
                    "utt_id": f"{directory.name}/{audio_path.stem}",
                    "path": relativize(audio_path, cfg.project_root),
                    "label": label,
                    "speaker": speaker,
                    "attack": attack,
                }
            )

    if not rows:
        warnings.warn(f"No '*{extension}' files found under {root}.", stacklevel=2)
        return empty_manifest()

    return _finalize(pd.DataFrame(rows), "wavefake", "evalwavefake")


def build_synthetic(cfg) -> pd.DataFrame:
    """Build the manifest for the synthetic smoke-test dataset.

    The demo writes an ASVspoof-2019-LA-format protocol on purpose, so this is
    just :func:`build_manifest_from_protocol` pointed at it - which means the
    demo genuinely exercises the real protocol parser rather than a shortcut.

    Args:
        cfg: Loaded project config.

    Returns:
        A manifest whose ``split`` is ``"train"`` (it stands in for the 2019 LA
        train partition and is re-split by :func:`make_speaker_disjoint_split`),
        or an empty manifest if the demo has not been generated.
    """
    root = cfg.path("synthetic")
    protocol = root / "protocol.txt"
    audio_dir = root / "flac"

    if not protocol.is_file() or not audio_dir.is_dir():
        warnings.warn(
            f"Synthetic demo not found at {root}. Run "
            "`python scripts/make_synthetic_demo.py` first.",
            stacklevel=2,
        )
        return empty_manifest()

    return build_manifest_from_protocol(
        protocol_path=protocol,
        audio_dir=audio_dir,
        source_corpus="synthetic_demo",
        split="train",
        project_root=cfg.project_root,
    )


def empty_manifest() -> pd.DataFrame:
    """An empty DataFrame with the canonical columns and dtypes."""
    frame = pd.DataFrame({column: pd.Series(dtype="object") for column in MANIFEST_COLUMNS})
    frame["label_id"] = frame["label_id"].astype("int64")
    return frame


def concat_manifests(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate manifests, dropping empty ones, preserving column order."""
    non_empty = [f for f in frames if f is not None and len(f) > 0]
    if not non_empty:
        return empty_manifest()
    return pd.concat(non_empty, ignore_index=True)[list(MANIFEST_COLUMNS)]


# ---------------------------------------------------------------------------
# Speaker-disjoint validation split
# ---------------------------------------------------------------------------
def make_speaker_disjoint_split(
    frame: pd.DataFrame,
    val_fraction: float = 0.15,
    seed: int = 1337,
    from_split: str = "train",
    val_split_name: str = "val",
) -> pd.DataFrame:
    """Carve a speaker-disjoint validation set out of a training split.

    Why speaker-disjoint and not a plain random split: several utterances share
    each speaker, and a random split would put the *same voice* on both sides.
    The model could then key on speaker identity, and the validation score
    would measure memorisation rather than deepfake detection - the score would
    look excellent and mean nothing. Grouping by speaker forces validation onto
    voices the model has never heard, which is the same condition the 2021 DF
    evaluation imposes.

    Implemented with :class:`sklearn.model_selection.GroupShuffleSplit`, which
    splits on the group labels; the seed makes the partition reproducible.

    Args:
        frame: A manifest containing rows whose ``split`` equals ``from_split``.
        val_fraction: Approximate fraction of *speakers* (not utterances) to
            move to validation.
        seed: RNG seed for the group split.
        from_split: Split to carve from.
        val_split_name: Name given to the new split.

    Returns:
        A copy of ``frame`` with the ``split`` column updated in place for the
        selected rows. Rows in other splits are untouched.

    Raises:
        ValueError: if ``from_split`` is absent, if the fraction is out of
            range, or if there are too few distinct speakers to split on.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    result = frame.copy()
    mask = result["split"] == from_split
    if not mask.any():
        raise ValueError(f"No rows with split == {from_split!r} to split.")

    subset = result.loc[mask]
    speakers = subset["speaker"].to_numpy()
    n_speakers = len(np.unique(speakers))

    if n_speakers < 2:
        raise ValueError(
            f"Need at least 2 distinct speakers for a speaker-disjoint split, "
            f"found {n_speakers}. (If the speaker column is all "
            f"'{UNKNOWN_SPEAKER}', the protocol had no speaker ids.)"
        )

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    # GroupShuffleSplit ignores X and y; only `groups` matters here.
    train_idx, val_idx = next(splitter.split(subset, groups=speakers))

    val_positions = subset.index[val_idx]
    result.loc[val_positions, "split"] = val_split_name
    return result


# ---------------------------------------------------------------------------
# Leakage checking
# ---------------------------------------------------------------------------
def file_hash(path: str | Path, max_bytes: int = 262_144) -> str:
    """SHA-1 of the first ``max_bytes`` of a file, used for duplicate detection.

    Hashing a prefix rather than the whole file keeps a pass over ~120k
    utterances to seconds. For distinct recordings the first 256 kB (header
    plus roughly the first few seconds of encoded audio) already differ; a true
    byte-for-byte duplicate is what we are hunting, and it collides here too.

    Args:
        path: File to hash.
        max_bytes: Number of leading bytes to read.

    Returns:
        Hex digest, or ``""`` if the file cannot be read (the QC pass reports
        unreadable files separately).
    """
    digest = hashlib.sha1()
    try:
        with Path(path).open("rb") as handle:
            digest.update(handle.read(max_bytes))
    except OSError:
        return ""
    return digest.hexdigest()


def check_leakage(
    frame: pd.DataFrame,
    project_root: Path | None = None,
    check_hashes: bool = True,
    hash_bytes: int = 262_144,
    train_splits: Sequence[str] = ("train",),
    raise_on_error: bool = True,
) -> dict[str, list[str]]:
    """Assert that the split boundaries hold.

    Three checks, in increasing cost:

    1. **Utterance overlap.** No ``utt_id`` may appear in two splits. Utterance
       ids are also required to be unique overall.
    2. **Speaker overlap.** No speaker may appear in both a training split and
       any other split, and no speaker may appear in two splits of the same
       corpus. Cross-corpus overlap between two *evaluation* splits is reported
       as a note rather than an error: ASVspoof 2021 DF is derived from the
       2019 LA evaluation partition, so those two legitimately share speakers -
       what matters is that neither shares with train.
    3. **Duplicate audio.** Byte-prefix hashes must not repeat across splits.
       This catches the case where the same file was copied into two corpus
       directories, which no id-based check would notice.

    Args:
        frame: The full manifest.
        project_root: Root for resolving relative paths; required when
            ``check_hashes`` is True.
        check_hashes: Whether to run the (I/O-bound) duplicate-audio check.
        hash_bytes: Bytes per file to hash.
        train_splits: Splits treated as training material.
        raise_on_error: Raise :class:`LeakageError` when problems are found.
            Set to False to inspect the report instead.

    Returns:
        A dict with keys ``errors`` and ``notes``, each a list of messages.

    Raises:
        LeakageError: if problems were found and ``raise_on_error`` is True.
    """
    errors: list[str] = []
    notes: list[str] = []

    if frame.empty:
        return {"errors": errors, "notes": notes}

    # --- 1. utterance ids ---------------------------------------------------
    per_utt_splits = frame.groupby("utt_id")["split"].nunique()
    shared_utts = per_utt_splits[per_utt_splits > 1]
    if len(shared_utts):
        errors.append(
            f"{len(shared_utts)} utt_id(s) appear in more than one split, e.g. "
            f"{list(shared_utts.index[:5])}"
        )

    duplicate_ids = frame["utt_id"].duplicated().sum()
    if duplicate_ids:
        errors.append(f"{duplicate_ids} duplicated utt_id(s) in the manifest.")

    # --- 2. speakers --------------------------------------------------------
    # 'unknown' is a placeholder shared by many rows, so it is not evidence of
    # overlap and is excluded from this check.
    known = frame[frame["speaker"] != UNKNOWN_SPEAKER]
    speaker_splits: dict[str, set[str]] = defaultdict(set)
    speaker_corpora: dict[str, set[str]] = defaultdict(set)
    for speaker, split, corpus in zip(
        known["speaker"], known["split"], known["source_corpus"]
    ):
        speaker_splits[speaker].add(split)
        speaker_corpora[speaker].add(corpus)

    train_set = set(train_splits)
    train_conflicts: list[str] = []
    within_corpus_conflicts: list[str] = []
    cross_corpus_notes: list[str] = []

    for speaker, splits in speaker_splits.items():
        if len(splits) < 2:
            continue
        if splits & train_set:
            train_conflicts.append(f"{speaker}: {sorted(splits)}")
        elif len(speaker_corpora[speaker]) == 1:
            within_corpus_conflicts.append(f"{speaker}: {sorted(splits)}")
        else:
            cross_corpus_notes.append(f"{speaker}: {sorted(splits)}")

    if train_conflicts:
        errors.append(
            f"{len(train_conflicts)} speaker(s) appear in a training split and "
            f"elsewhere, e.g. {train_conflicts[:5]}"
        )
    if within_corpus_conflicts:
        errors.append(
            f"{len(within_corpus_conflicts)} speaker(s) span two splits of the same "
            f"corpus, e.g. {within_corpus_conflicts[:5]}"
        )
    if cross_corpus_notes:
        notes.append(
            f"{len(cross_corpus_notes)} speaker(s) appear in two evaluation splits from "
            f"different corpora (expected: ASVspoof 2021 DF is built from the 2019 LA "
            f"evaluation partition), e.g. {cross_corpus_notes[:3]}"
        )

    # --- 3. duplicate audio -------------------------------------------------
    if check_hashes:
        if project_root is None:
            raise ValueError("project_root is required when check_hashes is True.")

        hashes = [
            file_hash(resolve_path(str(p), project_root), max_bytes=hash_bytes)
            for p in frame["path"]
        ]
        hashed = frame.assign(_hash=hashes)
        unreadable = int((hashed["_hash"] == "").sum())
        if unreadable:
            notes.append(
                f"{unreadable} file(s) could not be hashed (missing or unreadable); "
                "they were excluded from the duplicate check. Run the QC script."
            )
        hashed = hashed[hashed["_hash"] != ""]

        if len(hashed):
            grouped = hashed.groupby("_hash")
            group_size = grouped.size()
            splits_per_hash = grouped["split"].nunique()

            cross_split = splits_per_hash[splits_per_hash > 1]
            if len(cross_split):
                example = hashed[hashed["_hash"] == cross_split.index[0]]
                errors.append(
                    f"{len(cross_split)} identical audio file(s) appear in more "
                    f"than one split, e.g. {list(example['utt_id'][:3])}"
                )

            # Repeats confined to one split are not leakage, but they do skew
            # that split's class balance, so they are worth reporting.
            within_only = group_size[(group_size > 1) & (splits_per_hash == 1)]
            n_within = int((within_only - 1).sum())
            if n_within:
                notes.append(
                    f"{n_within} duplicated audio file(s) within a single split - "
                    "not leakage, but it inflates that split's size."
                )

    if errors and raise_on_error:
        raise LeakageError(
            "Data leakage detected:\n  - " + "\n  - ".join(errors)
        )

    return {"errors": errors, "notes": notes}


# ---------------------------------------------------------------------------
# Reporting and I/O
# ---------------------------------------------------------------------------
def summarize(frame: pd.DataFrame) -> str:
    """Render class balance and attack counts per split as text.

    Returned as a string (rather than printed) so scripts can both show it and
    write it to ``reports/``.

    Args:
        frame: A manifest.

    Returns:
        A formatted multi-line summary.
    """
    if frame.empty:
        return "Manifest is empty - nothing to summarise."

    lines: list[str] = []
    lines.append(f"Total utterances: {len(frame)}")
    lines.append("")

    lines.append("Class balance per split")
    lines.append("-" * 64)
    balance = (
        frame.pivot_table(
            index=["source_corpus", "split"],
            columns="label",
            values="utt_id",
            aggfunc="count",
            fill_value=0,
        )
        .assign(total=lambda d: d.sum(axis=1))
    )
    if "spoof" in balance.columns:
        balance["spoof_%"] = (100 * balance["spoof"] / balance["total"]).round(1)
    lines.append(balance.to_string())
    lines.append("")

    lines.append("Attack types per split")
    lines.append("-" * 64)
    attacks = (
        frame.groupby(["split", "attack"])["utt_id"]
        .count()
        .rename("n")
        .reset_index()
        .pivot(index="attack", columns="split", values="n")
        .fillna(0)
        .astype(int)
    )
    lines.append(attacks.to_string())
    lines.append("")

    lines.append("Speakers per split")
    lines.append("-" * 64)
    speakers = frame.groupby("split")["speaker"].nunique().rename("n_speakers")
    lines.append(speakers.to_string())

    return "\n".join(lines)


def save_manifest(frame: pd.DataFrame, path: str | Path, fmt: str = "csv") -> Path:
    """Write a manifest to disk as CSV or Parquet.

    Args:
        frame: The manifest.
        path: Destination path *without* relying on the extension; the correct
            suffix for ``fmt`` is applied.
        fmt: ``"csv"`` or ``"parquet"``.

    Returns:
        The path actually written.

    Raises:
        ValueError: on an unknown format.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        destination = destination.with_suffix(".csv")
        frame.to_csv(destination, index=False)
    elif fmt == "parquet":
        destination = destination.with_suffix(".parquet")
        frame.to_parquet(destination, index=False)
    else:
        raise ValueError(f"Unknown manifest format: {fmt!r} (use 'csv' or 'parquet')")

    return destination


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Read a manifest written by :func:`save_manifest`.

    Args:
        path: CSV or Parquet manifest.

    Returns:
        The manifest DataFrame.

    Raises:
        FileNotFoundError: if the file is missing.
        ValueError: on an unknown extension or missing columns.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Manifest not found: {source}")

    if source.suffix == ".csv":
        frame = pd.read_csv(source)
    elif source.suffix == ".parquet":
        frame = pd.read_parquet(source)
    else:
        raise ValueError(f"Unknown manifest extension: {source.suffix}")

    missing = set(MANIFEST_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest {source} is missing columns: {sorted(missing)}")
    return frame
