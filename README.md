# Detecting AI-Generated Speech with Traditional Speech Features

Course project for **Spoken Language Processing**, Birzeit University, Summer 2026.

A machine-learning system that distinguishes real human speech from AI-generated
(deepfake) speech using classical speech-processing features — MFCC, LFCC,
spectral descriptors and time-domain features — rather than end-to-end deep
learning. The question the project asks is whether these traditional features
capture the acoustic artefacts that modern TTS and voice-conversion systems
leave behind.

> **Status: Phase 1 complete.** This repository currently contains the *data
> foundation* only: configuration, the shared audio front end, manifest
> construction with leakage checking, and a quality-control pass. Feature
> extraction, classifiers and evaluation are Phase 2 and Phase 3. See
> [Project phases](#project-phases).

---

## Quick start (no downloads needed)

The whole pipeline runs today on a generated stand-in dataset, so you can
verify the code before the real corpora finish downloading.

```bash
python -m pip install -r requirements.txt

python scripts/make_synthetic_demo.py       # writes 120 synthetic FLAC files
python scripts/01_build_manifest.py --synthetic
python scripts/02_run_qc.py
python -m pytest tests/ -v
```

Expected: 120 files generated, a 90/30 speaker-disjoint train/val split, a
clean leakage report, three figures in `reports/figures/`, and 50 passing tests.

> ⚠️ **The synthetic dataset is a smoke test, never a result.** No human spoke
> the "bonafide" files and no TTS system produced the "spoof" files; both are
> additive-synthesis constructions that differ by a deliberate 7 kHz low-pass
> and a deliberate lack of pitch/phase irregularity. Any accuracy, EER or
> ROC-AUC measured on it demonstrates that the code runs, and nothing else. It
> must not appear in the report as an experimental result.

---

## Repository layout

```
sldp-project/
├── config.yaml               every tunable number, one place
├── requirements.txt          pinned dependencies
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.py             YAML loader, path resolution, seeding
│   ├── audio.py              the shared audio front end
│   ├── manifest.py           corpus parsing, splits, leakage checks
│   └── qc.py                 quality pass + report figures
├── scripts/
│   ├── make_synthetic_demo.py  generate the smoke-test dataset
│   ├── 01_build_manifest.py    build + validate manifests
│   └── 02_run_qc.py            audio quality control
├── tests/
│   └── test_phase1.py        50 tests, no corpus required
├── data/                     corpora go here (gitignored)
├── manifests/                generated CSVs (tracked — they are small)
├── features/                 cached features, Phase 2 (gitignored)
└── reports/
    └── figures/              QC figures for the report (tracked)
```

`data/` and `features/` are gitignored: the corpora are tens of gigabytes and
carry their own licences, and cached features are fully reproducible from the
corpora plus `config.yaml`. `manifests/` and `reports/` **are** tracked, because
they are small and they are what makes an experiment reproducible.

---

## Obtaining the datasets

All three are free but require registration or acceptance of a licence. Place
them under `data/` exactly as shown, or point `paths.*` in `config.yaml`
somewhere else (an external drive is fine — absolute paths are honoured).

### 1. ASVspoof 2019 LA — training and validation

Source: <https://datashare.ed.ac.uk/handle/10283/3336> (Edinburgh DataShare,
~23 GB). Download `LA.zip` and unpack it so you get:

```
data/LA/
├── ASVspoof2019_LA_cm_protocols/
│   ├── ASVspoof2019.LA.cm.train.trn.txt
│   ├── ASVspoof2019.LA.cm.dev.trl.txt
│   └── ASVspoof2019.LA.cm.eval.trl.txt
├── ASVspoof2019_LA_train/flac/*.flac      (25 380 utterances)
├── ASVspoof2019_LA_dev/flac/*.flac        (24 844)
└── ASVspoof2019_LA_eval/flac/*.flac       (71 237)
```

Protocol format (5 space-separated columns):

```
LA_0079 LA_T_1138215 - - bonafide
LA_0079 LA_T_1272637 - A01 spoof
```

### 2. ASVspoof 2021 DF — evaluation only

Two separate downloads.

* **Audio** (~90 GB): <https://zenodo.org/record/4835108>
* **Keys and metadata**: <https://www.asvspoof.org/index2021.html> →
  `ASVspoof2021_DF_eval_keys`

Unpack so that both end up under one root:

```
data/ASVspoof2021_DF/
├── ASVspoof2021_DF_eval/flac/*.flac
└── keys/DF/CM/trial_metadata.txt
```

Metadata format (8+ columns, **a different order from 2019**):

```
LA_0023 DF_E_2000001 nocodec asvspoof A07 spoof notrim eval
```

The build script searches recursively for `trial_metadata.txt`, so a slightly
different nesting is fine.

### 3. WaveFake — cross-generator generalisation

Source: <https://zenodo.org/record/5642694> (~30 GB). You also need the genuine
audio it was generated from: LJSpeech-1.1
(<https://keithito.com/LJ-Speech-Dataset/>) and, optionally, JSUT.

```
data/WaveFake/
├── LJSpeech-1.1/wavs/*.wav          bona fide
├── ljspeech_melgan/*.wav            spoof, attack = melgan
├── ljspeech_parallel_wavegan/*.wav  spoof, attack = parallel_wavegan
├── ljspeech_multi_band_melgan/*.wav spoof, attack = multi_band_melgan
├── ljspeech_full_band_melgan/*.wav  spoof, attack = full_band_melgan
├── ljspeech_hifiGAN/*.wav           spoof, attack = hifigan
└── ljspeech_waveglow/*.wav          spoof, attack = waveglow
```

There is no protocol file: the directory layout carries the labels. A directory
is treated as a generator if its name contains a known architecture keyword
(`melgan`, `wavegan`, `hifigan`, `waveglow`, …) and as genuine audio otherwise;
the corpus prefix is stripped so `ljspeech_melgan` and `jsut_melgan` both get
the attack label `melgan`.

**Nothing breaks if a corpus is missing.** Each builder warns and returns an
empty manifest, so you can work with whatever has finished downloading.

---

## Commands

### `scripts/make_synthetic_demo.py`

```bash
python scripts/make_synthetic_demo.py
```

Writes 120 FLAC files (8 pseudo-speakers × 15 utterances, ~2 s each, 16 kHz) to
`data/synthetic_demo/flac/`, plus `protocol.txt` in ASVspoof 2019 LA format —
so the demo is parsed by the *real* protocol parser, not a shortcut.

The two classes are built from the same source-filter model (harmonic source,
speaker-specific formant envelope, syllable-rate amplitude envelope). The spoof
class then gets two alterations chosen because real neural vocoders exhibit
them:

1. **Band limitation above 7 kHz** (8th-order Butterworth). Vocoders reconstruct
   from a mel-spectrogram whose upper bands are wide and sparse, so the top of
   the band comes back poorly. Measured on the generated files, the spoof class
   sits **~16 dB lower** above 7.2 kHz relative to the 0.3–3 kHz band.
2. **Over-regular harmonic phase, no jitter or shimmer.** Real phonation wanders
   cycle to cycle; a vocoder reconstructing from magnitude alone tends to be
   *too* regular.

Both classes get the **same** distribution of leading/trailing silence, on
purpose — see [Design decisions](#design-decisions).

### `scripts/01_build_manifest.py`

```bash
python scripts/01_build_manifest.py --synthetic       # smoke test
python scripts/01_build_manifest.py --asvspoof2019    # real training data
python scripts/01_build_manifest.py --all             # everything present
python scripts/01_build_manifest.py --all --no-hash-check   # skip duplicate scan
```

Builds `manifests/all.csv` plus one CSV per split, prints class balance and
attack-type counts, and runs the leakage check. **If leakage is detected the
script exits non-zero and writes nothing**, so a compromised manifest can never
be picked up silently by a later stage.

Manifest columns:

| column | meaning |
|---|---|
| `utt_id` | utterance id, e.g. `LA_T_1138215` |
| `path` | relative to the project root, forward slashes, portable between machines |
| `label` | `bonafide` or `spoof` |
| `label_id` | `0` = bonafide, `1` = spoof (spoof is the positive class) |
| `speaker` | e.g. `LA_0079`, or `unknown` when the protocol gives none |
| `attack` | `A01`…`A19`, a WaveFake generator name, or `bonafide` |
| `source_corpus` | `asvspoof2019_la`, `asvspoof2021_df`, `wavefake`, `synthetic_demo` |
| `split` | `train`, `val`, `dev`, `eval`, `eval2021df`, `evalwavefake` |

### `scripts/02_run_qc.py`

```bash
python scripts/02_run_qc.py
python scripts/02_run_qc.py --manifest manifests/train.csv --max-files 500
```

Measures a capped, per-split random sample (default 2000 files, seeded) and
reports duration, sample rates, RMS/peak level, DC offset, clipping rate,
silence ratio, and unreadable files. Writes `reports/qc_per_file.csv`,
`reports/qc_summary.csv`, and three report-ready figures:

* `qc_durations.png` — duration histogram per class, with the 4 s fixed-length
  line, justifying the crop/pad choice.
* `qc_level_silence.png` — RMS level and silence ratio per class; the evidence
  behind the two preprocessing defaults below.
* `qc_class_balance.png` — bona fide vs spoof per split; the justification for
  reporting EER and balanced metrics rather than plain accuracy.

### `tests/test_phase1.py`

```bash
python -m pytest tests/ -v
```

50 tests, no corpus required. They cover resampling correctness (spectral, not
just length), `fix_length` for shorter and longer inputs, the pre-emphasis
equation, the 25 ms/10 ms frame count, STFT/framing alignment, both ASVspoof
protocol formats, speaker-disjointness, and — importantly — that the leakage
checks actually *fire* on deliberately corrupted splits.

---

## Design decisions

These are the choices an examiner is most likely to ask about. Each is
implemented as a config switch, so the alternative can be run as an ablation.

### Silence is **not** trimmed by default (`remove_silence: false`)

This is the single most consequential preprocessing decision in the project.

In ASVspoof, the duration and character of the non-speech margins differ
systematically between bona fide and spoofed utterances: the synthesis
pipelines produce their own, unnaturally clean, silence. A classifier can reach
a very low error rate by reading nothing but that margin. Such a detector has
learned nothing about synthesis artefacts, and it collapses on any data that
was trimmed differently — which is precisely the generalisation the project is
supposed to measure.

We therefore keep the silence, forcing the classifier onto the speech itself.
`trim_silence()` remains available and `qc_level_silence.png` quantifies how
much audio the ablation would remove, so we can report the effect rather than
merely assert it.

### Level is **not** normalised by default (`peak_normalize: false`)

Absolute recording level is largely a property of the source database and its
post-processing chain, not of the synthesis system. Normalising every file to
the same peak would erase a measurable difference that we want to *report* as a
database artefact — and a classifier that keys on loudness would look excellent
in development and fail on anything recorded under different conditions.

The cepstral features are already largely level-robust once the 0th coefficient
is handled, so we lose little by leaving level alone.

### Short signals are repeat-padded, not zero-padded (`pad_mode: repeat`)

Zero-padding to reach 4 s would append artificial silence — reintroducing, at
the padding stage, exactly the cue we refused to create at the trimming stage,
and correlating it with utterance duration. Tiling the signal keeps every frame
filled with speech.

### ASVspoof 2021 DF is **evaluation only**

The 2021 DF evaluation set is derived from the ASVspoof 2019 LA *evaluation*
partition (plus additional sources), passed through varied codecs and
compression. Training on any part of it would destroy the property that makes
it valuable: it is our measurement of how the detector behaves on unseen
attacks, unseen speakers and unseen channel conditions.

Concretely: **train and validate on ASVspoof 2019 LA train only**; report on
2019 LA dev/eval, on 2021 DF, and on WaveFake. Because 2021 DF and 2019 LA eval
share speakers by construction, `check_leakage()` reports that overlap as an
informational *note* rather than an error — the check that matters is that
neither shares a speaker with `train`.

### How leakage is prevented

Four mechanisms, in order of when they act:

1. **A speaker-disjoint validation split.** The internal validation set is
   carved out of ASVspoof 2019 LA train with
   `sklearn.model_selection.GroupShuffleSplit` grouped by speaker (~15%, seeded
   at 1337). A plain random split would put the *same voice* on both sides;
   the model could then key on speaker identity and the validation score would
   measure memorisation, not deepfake detection. Grouping by speaker forces
   validation onto voices the model has never heard — the same condition the
   2021 DF evaluation imposes.
2. **`check_leakage()` runs on every build** and asserts three things:
   * no `utt_id` appears in two splits (and no `utt_id` is duplicated at all);
   * no speaker appears in a training split and anywhere else, and no speaker
     spans two splits of the same corpus;
   * no byte-identical audio file appears in two splits. This last one is
     hash-based (SHA-1 of the first 256 kB) and catches what no id-based check
     can: the same recording copied under two different names.
3. **A failed check is fatal.** `01_build_manifest.py` exits non-zero and writes
   no manifest at all, so a later stage cannot pick up a compromised split.
4. **The tests verify the checks fire**, not just that they pass — each check
   has a test that deliberately corrupts a split and asserts a `LeakageError`.

### One shared audio front end

Every stage reads audio through `src.audio.load_and_preprocess()` and nothing
else. This is what makes the feature comparison in the report *fair*: if MFCC
and LFCC disagree, it is because of the features, not because one of them
silently used a different sample rate, window length or normalisation.

The framing is standard for speech: 25 ms Hamming window, 10 ms hop, at 16 kHz
that is a 400-sample frame advancing 160 samples, with `n_fft = 512` (the next
power of two). No centring or reflection padding, so frame *k* covers exactly
samples `[k·160, k·160 + 400)` and the frame count is
`1 + (N − 400) // 160` — 398 frames for a 4 s utterance. `magnitude_spectrogram()`
pads the signal so that `librosa.stft`'s internally centred window lines its
columns up one-to-one with `frame_signal()`'s rows; a test asserts this.

### Reproducibility

`config.yaml` holds every tunable number; `set_seed()` seeds `random`, NumPy and
(if installed) PyTorch; the split, the QC sample and the synthetic dataset all
take explicit seeds. Dependency versions are pinned in `requirements.txt`, since
resampler and FFT implementations change between releases.

---

## Project phases

| Phase | Contents | Status |
|---|---|---|
| **1. Data foundation** | config, audio front end, manifests, leakage checks, QC | ✅ complete |
| **2. Features + classical models** | `src/features.py` (MFCC, LFCC, optional CQCC, spectral centroid / bandwidth / roll-off / flux, short-time energy, ZCR), `src/models.py` (SVM, Random Forest), `src/evaluate.py` (accuracy, precision, recall, F1, confusion matrix, ROC-AUC, EER) | planned |
| **3. CNN + generalisation** | spectrogram CNN; cross-corpus evaluation on 2021 DF; cross-generator evaluation on WaveFake | planned |

Phase 2 modules consume manifests and call `load_and_preprocess()`; nothing in
Phase 1 needs to change to accommodate them.

---

## Requirements

Python 3.11+ (developed and tested on 3.12.5, Windows 11). Install with
`pip install -r requirements.txt`. PyTorch is listed but commented out — it is
needed only for the Phase 3 CNN and is not imported by any Phase 1 code.

---

## References

To be completed for the final report. Consulted for Phase 1:

* Todisco et al., "ASVspoof 2019: Future Horizons in Spoofed and Fake Audio
  Detection", *Interspeech 2019*.
* Yamagishi et al., "ASVspoof 2021: accelerating progress in spoofed and deepfake
  speech detection", *ASVspoof 2021 Workshop*.
* Frank & Schönherr, "WaveFake: A Data Set to Facilitate Audio Deepfake
  Detection", *NeurIPS 2021 Datasets and Benchmarks*.
* Müller et al., "Does Audio Deepfake Detection Generalize?", *Interspeech 2022*
  — the source of the silence-shortcut finding that motivates
  `remove_silence: false`.
* McFee et al., "librosa: Audio and Music Signal Analysis in Python", *SciPy 2015*.
* Pedregosa et al., "Scikit-learn: Machine Learning in Python", *JMLR* 12, 2011.

**AI tool disclosure** (required by the project brief): Claude (Anthropic) was
used to assist with Phase 1 code generation, docstring and README drafting, and
test design. All code was reviewed, executed and verified by the team; the
design decisions documented above are the team's own and are defended in the
report.

---

## Team

| Member | Phase 1 responsibility |
|---|---|
| _TBD_ | audio front end (`src/audio.py`), framing/STFT alignment, tests |
| _TBD_ | manifests and splits (`src/manifest.py`), leakage checking |
| _TBD_ | QC pass and figures (`src/qc.py`), synthetic demo, documentation |

Fill this in before submission — Section 7 of the report requires a detailed
per-member contribution statement.
