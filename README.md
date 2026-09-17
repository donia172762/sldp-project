# Detecting AI-Generated Speech with Traditional Speech Features

Course project for **Spoken Language Processing**, Birzeit University, Summer 2026.

A machine-learning system that distinguishes real human speech from AI-generated
(deepfake) speech using classical speech-processing features — MFCC, LFCC,
spectral descriptors and time-domain features — rather than end-to-end deep
learning. The question the project asks is whether these traditional features
capture the acoustic artefacts that modern TTS and voice-conversion systems
leave behind.

> **Status: Phase 1 complete; Phase 2 feature extraction complete.**
> The repository currently contains the complete data foundation and the
> traditional speech-feature extraction pipeline. Implemented features include
> MFCC, LFCC, spectral descriptors, short-time energy, and zero-crossing rate.
> Frame-level features are combined and statistically pooled into a fixed-length
> 92-dimensional vector for each recording, ready for classical machine-learning
> classifiers.

---

## Quick start

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

The complete Phase 1 smoke-test pipeline can be run using:

```bash
python scripts/make_synthetic_demo.py
python scripts/01_build_manifest.py --synthetic
python scripts/02_run_qc.py
python -m pytest tests/ -v
```

The synthetic pipeline generates 120 audio files and creates a 90/30
speaker-disjoint train/validation split.

Phase 2 features can then be extracted using:

```bash
python -m scripts.02_extract_features \
    --manifest manifests/train.csv \
    --output features/train_features.csv
```

and:

```bash
python -m scripts.02_extract_features \
    --manifest manifests/val.csv \
    --output features/val_features.csv
```

> ⚠️ **The synthetic dataset is a smoke test, never a final experimental result.**
> No human spoke the "bonafide" files and no real TTS system produced the
> "spoof" files. Any accuracy, EER, ROC-AUC, or other performance measured on
> this synthetic dataset only demonstrates that the pipeline runs correctly.
> It must not be presented in the final report as a real experimental result.

---

## Repository layout

```text
sldp-project/
├── config.yaml
├── requirements.txt
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── audio.py
│   ├── manifest.py
│   ├── qc.py
│   └── features.py
│
├── scripts/
│   ├── make_synthetic_demo.py
│   ├── 01_build_manifest.py
│   ├── 02_run_qc.py
│   └── 02_extract_features.py
│
├── tests/
│   ├── test_phase1.py
│   └── test_features.py
│
├── data/
├── manifests/
├── features/
│   ├── train_features.csv
│   └── val_features.csv
│
└── reports/
    └── figures/
```

### Main modules

- `src/config.py` — configuration loading, path resolution and seeding.
- `src/audio.py` — shared audio loading and preprocessing front end.
- `src/manifest.py` — dataset parsing, splitting and leakage checking.
- `src/qc.py` — audio quality-control analysis.
- `src/features.py` — Phase 2 MFCC, LFCC, spectral and time-domain
  feature extraction, feature combination and statistical pooling.
- `scripts/02_extract_features.py` — processes a manifest and writes
  fixed-length feature vectors to CSV.
- `tests/test_features.py` — automated tests for the Phase 2 feature pipeline.

---

## Obtaining the datasets

### 1. ASVspoof 2019 LA — training and validation

Source: Edinburgh DataShare.

Download the ASVspoof 2019 LA dataset and unpack it so the project contains:

```text
data/LA/
├── ASVspoof2019_LA_cm_protocols/
│   ├── ASVspoof2019.LA.cm.train.trn.txt
│   ├── ASVspoof2019.LA.cm.dev.trl.txt
│   └── ASVspoof2019.LA.cm.eval.trl.txt
│
├── ASVspoof2019_LA_train/flac/*.flac
├── ASVspoof2019_LA_dev/flac/*.flac
└── ASVspoof2019_LA_eval/flac/*.flac
```

Protocol examples:

```text
LA_0079 LA_T_1138215 - - bonafide
LA_0079 LA_T_1272637 - A01 spoof
```

### 2. ASVspoof 2021 DF — evaluation

The ASVspoof 2021 DF dataset is used for evaluation and
generalisation experiments.

Expected structure:

```text
data/ASVspoof2021_DF/
├── ASVspoof2021_DF_eval/flac/*.flac
└── keys/DF/CM/trial_metadata.txt
```

Example metadata:

```text
LA_0023 DF_E_2000001 nocodec asvspoof A07 spoof notrim eval
```

### 3. WaveFake — cross-generator generalisation

WaveFake can be used to investigate whether the detector generalises to
different speech-generation architectures.

Expected structure:

```text
data/WaveFake/
├── LJSpeech-1.1/wavs/*.wav
├── ljspeech_melgan/*.wav
├── ljspeech_parallel_wavegan/*.wav
├── ljspeech_multi_band_melgan/*.wav
├── ljspeech_full_band_melgan/*.wav
├── ljspeech_hifiGAN/*.wav
└── ljspeech_waveglow/*.wav
```

The directory structure identifies the synthesis method, allowing the
generator name to be retained as the `attack` metadata field.

---

# Phase 1 — Data Foundation

## Manifest construction

The manifest builder creates a common representation of all recordings.

Run:

```bash
python scripts/01_build_manifest.py --synthetic
```

or, when the real datasets are available:

```bash
python scripts/01_build_manifest.py --asvspoof2019
```

or:

```bash
python scripts/01_build_manifest.py --all
```

Manifest columns:

| Column | Meaning |
|---|---|
| `utt_id` | Unique utterance identifier |
| `path` | Audio path relative to the project root |
| `label` | `bonafide` or `spoof` |
| `label_id` | `0` = bonafide, `1` = spoof |
| `speaker` | Speaker identifier |
| `attack` | Synthesis/attack identifier |
| `source_corpus` | Source dataset |
| `split` | Dataset split |

---

## Leakage prevention

The project includes several mechanisms to prevent data leakage.

### Speaker-disjoint validation

The internal validation split is separated by speaker. This prevents the same
speaker from appearing in both training and validation data and reduces the
chance that a classifier simply memorises speaker identity.

### Duplicate detection

The manifest pipeline checks for:

- duplicated utterance IDs;
- speakers appearing in incompatible splits;
- byte-identical audio appearing across splits.

A failed leakage check prevents the compromised manifest from being used by
later stages.

---

## Quality control

Run:

```bash
python scripts/02_run_qc.py
```

The QC stage examines properties including:

- duration;
- sample rate;
- RMS and peak level;
- DC offset;
- clipping;
- silence ratio;
- unreadable audio files.

It can also generate report-ready figures for duration, level/silence and
class balance.

---

## Audio preprocessing

Every project stage uses the same audio front end through:

```python
src.audio.load_and_preprocess()
```

Using one shared preprocessing pipeline ensures that feature comparisons are
fair.

The standard speech-analysis settings are based on:

- Sample rate: **16 kHz**
- Window length: **25 ms**
- Hop length: **10 ms**
- Window size: **400 samples**
- Hop size: **160 samples**
- FFT size: **512**

This common preprocessing stage feeds directly into Phase 2.

---

# Phase 2 — Speech Feature Extraction

Phase 2 implements traditional speech and signal-processing features for
AI-generated speech detection.

Every recording first passes through the shared Phase 1 preprocessing
pipeline. The same preprocessed signal is therefore used for MFCC, LFCC,
spectral and time-domain feature extraction.

---

## MFCC

**Mel-Frequency Cepstral Coefficients (MFCCs)** represent the short-term
spectral envelope of speech using the perceptually motivated Mel-frequency
scale.

The implementation extracts:

```text
20 MFCC coefficients per frame
```

The output for an audio recording therefore has the form:

```text
(20, number_of_frames)
```

---

## LFCC

**Linear-Frequency Cepstral Coefficients (LFCCs)** use a linear-frequency
filter bank rather than the Mel-frequency spacing used by MFCC.

The implementation performs:

```text
STFT
 ↓
Power spectrum
 ↓
Linear-frequency triangular filter bank
 ↓
Log filter-bank energies
 ↓
Discrete Cosine Transform (DCT)
 ↓
20 LFCC coefficients
```

The resulting matrix has:

```text
(20, number_of_frames)
```

LFCC is particularly useful for comparison with MFCC because it retains a
different representation of high-frequency spectral information.

---

## Spectral features

Four additional spectral descriptors are extracted.

### Spectral centroid

Represents the centre of mass of the spectrum and provides an indication of
spectral brightness.

### Spectral bandwidth

Measures how widely the spectral energy is distributed around the spectral
centroid.

### Spectral roll-off

Measures the frequency below which a specified percentage of the spectral
energy is contained.

The implementation uses an 85% roll-off threshold.

### Spectral flux

Measures frame-to-frame changes in the normalised magnitude spectrum.

Together, these produce:

```text
4 spectral features per frame
```

---

## Time-domain features

Two time-domain descriptors are also extracted.

### Short-time energy

Measures the signal energy within each short analysis frame.

### Zero-crossing rate

Measures how frequently the waveform changes sign within a frame.

Together:

```text
2 time-domain features per frame
```

---

## Feature combination

All feature families are combined at frame level.

| Feature family | Features |
|---|---:|
| MFCC | 20 |
| LFCC | 20 |
| Spectral descriptors | 4 |
| Time-domain descriptors | 2 |
| **Total** | **46** |

Therefore, the combined frame-level representation has the shape:

```text
(46, number_of_frames)
```

---

## Statistical pooling

Different recordings may contain different numbers of frames.

Classical machine-learning classifiers such as SVM and Random Forest require
every training example to contain the same number of input values.

For this reason, the frame-level representation is converted into a
fixed-length recording-level vector.

For each of the 46 features, the pipeline calculates:

1. Mean across frames.
2. Standard deviation across frames.

Therefore:

```text
46 × 2 = 92
```

Each audio recording is finally represented by a:

```text
92-dimensional feature vector
```

This representation is ready to be consumed by the classical
machine-learning stage.

---

## Feature names

The feature CSV files use meaningful names rather than anonymous feature
indices.

Examples include:

```text
mfcc_01_mean
mfcc_02_mean
...
mfcc_20_mean

lfcc_01_mean
...
lfcc_20_mean

spectral_centroid_mean
spectral_bandwidth_mean
spectral_rolloff_mean
spectral_flux_mean

short_time_energy_mean
zero_crossing_rate_mean

mfcc_01_std
...
lfcc_20_std

spectral_centroid_std
spectral_bandwidth_std
spectral_rolloff_std
spectral_flux_std

short_time_energy_std
zero_crossing_rate_std
```

This makes the feature representation easier to interpret during modelling
and comparative analysis.

---

## Running Phase 2

### Training features

Run:

```bash
python -m scripts.02_extract_features \
    --manifest manifests/train.csv \
    --output features/train_features.csv
```

Current smoke-test result:

```text
Processed: 90 / 90
Feature columns: 92
```

### Validation features

Run:

```bash
python -m scripts.02_extract_features \
    --manifest manifests/val.csv \
    --output features/val_features.csv
```

Current smoke-test result:

```text
Processed: 30 / 30
Feature columns: 92
```

---

## Feature CSV format

Seven metadata columns from the manifest are retained:

```text
utt_id
label
label_id
speaker
attack
source_corpus
split
```

They are followed by the 92 extracted features.

Therefore, the current synthetic smoke-test files have the following shapes:

```text
train_features.csv → (90, 99)
val_features.csv   → (30, 99)
```

because:

```text
7 metadata columns + 92 feature columns = 99 columns
```

The `label_id` column can later be used as the target variable for the
machine-learning classifiers.

---

## Phase 2 tests

The feature extraction pipeline has an independent automated test suite.

Run:

```bash
python -m pytest tests/test_features.py -s
```

The tests cover:

1. MFCC extraction.
2. LFCC extraction.
3. Spectral feature extraction.
4. Time-domain feature extraction.
5. Combined frame-level features.
6. Statistical pooling.
7. Fixed-length feature-vector extraction.
8. Feature-name generation.

Current result:

```text
8 passed
```

The tests check feature dimensions and verify that the extracted values are
finite.

Expected feature dimensions for the one-second synthetic unit-test signal are:

```text
MFCC shape:              (20, 101)
LFCC shape:              (20, 101)
Spectral features shape: (4, 101)
Time features shape:     (2, 101)
Combined features shape: (46, 101)
Pooled features shape:   (92,)
Feature vector shape:    (92,)
Number of feature names: 92
```

---

# Design Decisions

## Silence is not trimmed by default

Silence can potentially become a shortcut feature if bona fide and spoof
datasets contain systematically different non-speech margins.

The pipeline therefore does not trim silence by default. This helps prevent
the classifier from learning an artificial dataset-specific cue instead of
speech-synthesis artefacts.

---

## Level is not normalised by default

Absolute recording level may reflect properties of the source database and
post-processing pipeline.

Peak normalisation is therefore disabled by default so its effect can be
studied explicitly rather than silently changing the recordings.

---

## Short signals are repeat-padded

Zero-padding short signals would introduce artificial silence.

Instead, short recordings are repeat-padded so that the fixed-length input
does not receive a duration-dependent artificial silence cue.

---

## ASVspoof 2021 DF is evaluation only

ASVspoof 2021 DF is reserved for evaluation/generalisation experiments.

Training on it would undermine its purpose as a test of performance on unseen
attacks and recording conditions.

---

## One shared audio front end

All feature families use the same Phase 1 preprocessing function.

This is important for fair comparisons:

```text
Audio
  ↓
Shared preprocessing
  ↓
┌───────────────┬──────────────┬───────────────┐
↓               ↓              ↓
MFCC            LFCC           Other features
└───────────────┴──────────────┴───────────────┘
                ↓
        Feature combination
```

Therefore, if one feature family performs better than another, the difference
cannot be attributed to inconsistent preprocessing.

---

# Current Feature Pipeline

The implemented pipeline is:

```text
Real / AI-generated speech
            ↓
      Audio preprocessing
            ↓
      Short-time analysis
            ↓
 ┌──────────┼───────────┬──────────────┐
 ↓          ↓           ↓              ↓
MFCC       LFCC      Spectral       Time-domain
20         20           4              2
 └──────────┴───────────┴──────────────┘
            ↓
     Feature combination
            ↓
 46 frame-level features
            ↓
   Mean + Std pooling
            ↓
 92-dimensional vector
            ↓
 Classical ML classifier
       (next stage)
            ↓
 REAL vs AI-GENERATED
```

---

# Project Phases

| Phase | Contents | Status |
|---|---|---|
| **1. Data foundation** | Configuration, shared audio front end, manifests, leakage checks and quality control | ✅ Complete |
| **2A. Feature extraction** | MFCC, LFCC, spectral features, time-domain features, feature combination and statistical pooling | ✅ Complete |
| **2B. Classical classification & evaluation** | SVM, Random Forest, accuracy, precision, recall, F1, confusion matrix, ROC-AUC/EER | ✅ Complete |
| **3. CNN + generalisation** | Spectrogram CNN, cross-corpus and cross-generator evaluation | In Progress |

Phase 2 feature extraction consumes the Phase 1 manifests and uses the shared
`load_and_preprocess()` audio front end.

No Phase 1 implementation needs to be changed to train the classical models.

---

# Reproducibility

`config.yaml` contains the project's configurable parameters.

The project uses explicit random seeds for operations such as dataset
splitting and synthetic data generation.

Dependency versions are specified in:

```text
requirements.txt
```

The project is designed for Python 3.11+ and has been developed and tested
using Python 3.12.

---

# Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Phase 1 and Phase 2 use libraries including:

- NumPy
- Pandas
- SciPy
- librosa
- scikit-learn
- PyYAML
- pytest

PyTorch is only required if the optional CNN stage is implemented.

---

# References

References consulted for the project include:

1. Todisco et al., **"ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection,"** Interspeech 2019.

2. Yamagishi et al., **"ASVspoof 2021: Accelerating Progress in Spoofed and Deepfake Speech Detection,"** ASVspoof 2021 Workshop.

3. Frank and Schönherr, **"WaveFake: A Data Set to Facilitate Audio Deepfake Detection,"** NeurIPS 2021 Datasets and Benchmarks.

4. Müller et al., **"Does Audio Deepfake Detection Generalize?"** Interspeech 2022.

5. McFee et al., **"librosa: Audio and Music Signal Analysis in Python,"** SciPy 2015.

6. Pedregosa et al., **"Scikit-learn: Machine Learning in Python,"** Journal of Machine Learning Research, 2011.

---

# AI Tool Disclosure

The use of AI tools is permitted by the project specification and is
documented here for transparency.

**Claude (Anthropic)** was used to assist with Phase 1 code generation,
docstring and README drafting, and test design.

**ChatGPT (OpenAI)** was used during Phase 2 to assist with implementation
and debugging of the traditional speech-feature extraction pipeline,
development of automated tests, integration with the Phase 1 preprocessing
pipeline, and documentation.

All generated or suggested code was reviewed, executed, tested and verified
by the team.

---

# Team

| Member | Responsibility |
|---|---|
| _TBD_ | Phase 1 — audio preprocessing / manifests / QC |
| _TBD_ | Phase 2A — MFCC, LFCC, spectral and time-domain feature extraction, feature combination, pooling and testing |
| _TBD_ | Phase 2B / Phase 3 — classical classification, evaluation, CNN and/or generalisation experiments |

The final report must include a detailed per-member contribution statement
describing implementation, testing, documentation and presentation work.
