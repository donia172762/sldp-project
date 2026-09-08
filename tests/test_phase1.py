"""Phase 1 test suite.

Run with::

    python -m pytest tests/ -v

The tests deliberately avoid any real corpus: everything is either a signal
constructed in memory or the synthetic protocol format written into a
``tmp_path``. That is what makes the suite runnable on any machine, including
the instructor's, with no downloads.

Coverage maps onto the parts of Phase 1 that would silently corrupt every later
result if they were wrong:

* resampling actually preserves frequency content;
* the fixed-length step behaves correctly when the input is too short and when
  it is too long;
* pre-emphasis matches its defining equation;
* the frame count matches the 25 ms / 10 ms specification, and the STFT agrees
  with it;
* both ASVspoof protocol formats parse to the same schema;
* the validation split is speaker-disjoint;
* the leakage check actually fires when a split is deliberately corrupted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import audio, manifest  # noqa: E402
from src.config import Config, load_config, new_rng, set_seed  # noqa: E402

SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cfg() -> Config:
    """The real project config, so the tests check the shipped settings."""
    return load_config()


@pytest.fixture
def sine() -> np.ndarray:
    """A 1 s, 1 kHz sine at 16 kHz - a signal with a known spectrum."""
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    return np.sin(2 * np.pi * 1000.0 * t).astype(np.float32)


def dominant_frequency(x: np.ndarray, sample_rate: int) -> float:
    """Frequency of the largest FFT magnitude bin, in Hz."""
    spectrum = np.abs(np.fft.rfft(x))
    frequencies = np.fft.rfftfreq(x.size, d=1.0 / sample_rate)
    return float(frequencies[int(np.argmax(spectrum))])


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def test_config_attribute_access_and_paths(cfg: Config) -> None:
    """Nested attribute access works and paths resolve under the project root."""
    assert cfg.audio.sample_rate == SAMPLE_RATE
    assert cfg.audio.frame_length_ms == 25.0
    assert cfg.audio.frame_shift_ms == 10.0

    manifests_dir = cfg.path("manifests")
    assert manifests_dir.is_absolute()
    assert manifests_dir.parent == cfg.project_root


def test_config_documented_defaults_are_off(cfg: Config) -> None:
    """The two deliberate defaults must stay off; flipping them is an ablation.

    This test exists to make an accidental change visible in CI rather than in
    a suspiciously good result three weeks later.
    """
    assert cfg.audio.remove_silence is False, (
        "Trimming silence by default reintroduces the ASVspoof silence shortcut."
    )
    assert cfg.audio.peak_normalize is False, (
        "Peak normalisation by default erases a level cue we want to measure."
    )


def test_set_seed_is_reproducible() -> None:
    """The same seed produces the same draws from every seeded generator."""
    set_seed(1337)
    first = np.random.rand(5)
    set_seed(1337)
    assert np.allclose(first, np.random.rand(5))


# ---------------------------------------------------------------------------
# audio: loading and resampling
# ---------------------------------------------------------------------------
def test_load_audio_resamples_and_preserves_frequency(tmp_path: Path) -> None:
    """Resampling 44.1 kHz -> 16 kHz keeps a 1 kHz tone at 1 kHz.

    Checks both the sample count (within one sample of the exact ratio) and,
    more importantly, the spectral content: a resampler that silently changed
    playback speed would pass a length check but corrupt every feature.
    """
    native_rate = 44100
    t = np.arange(native_rate) / native_rate
    tone = np.sin(2 * np.pi * 1000.0 * t).astype(np.float32)

    path = tmp_path / "tone.wav"
    sf.write(path, tone, native_rate)

    loaded, rate = audio.load_audio(path, sample_rate=SAMPLE_RATE)

    assert rate == SAMPLE_RATE
    assert abs(loaded.size - SAMPLE_RATE) <= 1
    assert loaded.dtype == np.float32
    # 1 Hz tolerance: the rFFT bin spacing here is 1 Hz.
    assert abs(dominant_frequency(loaded, SAMPLE_RATE) - 1000.0) < 2.0


def test_load_audio_downmixes_to_mono(tmp_path: Path) -> None:
    """A stereo file becomes 1-D, and the downmix is the channel mean."""
    left = np.zeros(1000, dtype=np.float32)
    right = np.ones(1000, dtype=np.float32) * 0.5
    stereo = np.stack([left, right], axis=1)

    path = tmp_path / "stereo.wav"
    sf.write(path, stereo, SAMPLE_RATE)

    loaded, _ = audio.load_audio(path, sample_rate=SAMPLE_RATE, mono=True)
    assert loaded.ndim == 1
    assert np.allclose(loaded, 0.25, atol=1e-4)


def test_load_audio_reads_flac(tmp_path: Path) -> None:
    """FLAC must work: it is the format both ASVspoof releases ship in."""
    signal = (np.random.default_rng(0).standard_normal(4000) * 0.1).astype(np.float32)
    path = tmp_path / "clip.flac"
    sf.write(path, signal, SAMPLE_RATE, format="FLAC", subtype="PCM_16")

    loaded, rate = audio.load_audio(path, sample_rate=SAMPLE_RATE)
    assert rate == SAMPLE_RATE
    assert loaded.size == signal.size
    # PCM_16 quantisation is ~3e-5, so an exact comparison would be wrong.
    assert np.max(np.abs(loaded - signal)) < 1e-3


def test_load_audio_raises_on_unreadable_file(tmp_path: Path) -> None:
    """A corrupt file raises RuntimeError naming the path, for the QC report."""
    bad = tmp_path / "corrupt.flac"
    bad.write_bytes(b"this is definitely not audio")

    with pytest.raises(RuntimeError, match="corrupt.flac"):
        audio.load_audio(bad)


# ---------------------------------------------------------------------------
# audio: pre-emphasis
# ---------------------------------------------------------------------------
def test_pre_emphasis_matches_its_equation() -> None:
    """y[n] = x[n] - 0.97*x[n-1], with y[0] = x[0]."""
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y = audio.pre_emphasis(x, coefficient=0.97)

    assert y[0] == pytest.approx(1.0)
    assert y[1] == pytest.approx(2.0 - 0.97 * 1.0)
    assert y[2] == pytest.approx(3.0 - 0.97 * 2.0)
    assert y[3] == pytest.approx(4.0 - 0.97 * 3.0)
    assert y.shape == x.shape


def test_pre_emphasis_zero_coefficient_is_identity() -> None:
    """coefficient=0 disables the filter, so ablations can turn it off."""
    x = np.array([1.0, -2.0, 3.5], dtype=np.float32)
    assert np.allclose(audio.pre_emphasis(x, coefficient=0.0), x)


def test_pre_emphasis_boosts_high_frequencies() -> None:
    """The filter is a high-pass: a 7 kHz tone survives better than a 100 Hz one.

    This is the property the project actually relies on - vocoder artefacts sit
    high in the band - so it is worth asserting rather than assuming.
    """
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    low = np.sin(2 * np.pi * 100.0 * t).astype(np.float32)
    high = np.sin(2 * np.pi * 7000.0 * t).astype(np.float32)

    low_gain = np.std(audio.pre_emphasis(low)) / np.std(low)
    high_gain = np.std(audio.pre_emphasis(high)) / np.std(high)
    assert high_gain > low_gain


# ---------------------------------------------------------------------------
# audio: fixed length
# ---------------------------------------------------------------------------
def test_fix_length_repeat_pads_a_short_signal() -> None:
    """A short signal is tiled, not zero-padded, and reaches the exact length."""
    x = np.arange(10, dtype=np.float32)
    y = audio.fix_length(x, length=25, pad_mode="repeat")

    assert y.size == 25
    assert np.allclose(y[:10], x)
    assert np.allclose(y[10:20], x)
    assert np.allclose(y[20:], x[:5])
    # The point of repeat-padding: no artificial silence is introduced.
    assert np.count_nonzero(y == 0.0) == 3  # only the genuine zeros from x


def test_fix_length_zero_pads_when_asked() -> None:
    """pad_mode='zeros' appends silence, available for the ablation."""
    x = np.ones(10, dtype=np.float32)
    y = audio.fix_length(x, length=25, pad_mode="zeros")

    assert y.size == 25
    assert np.allclose(y[:10], 1.0)
    assert np.allclose(y[10:], 0.0)


def test_fix_length_head_crops_a_long_signal() -> None:
    """A long signal is cropped deterministically from the head."""
    x = np.arange(100, dtype=np.float32)
    y = audio.fix_length(x, length=40, crop_mode="head")

    assert y.size == 40
    assert np.allclose(y, x[:40])


def test_fix_length_random_crop_is_seeded_and_in_bounds() -> None:
    """Random cropping returns a genuine contiguous window and is reproducible."""
    x = np.arange(100, dtype=np.float32)

    first = audio.fix_length(x, 40, crop_mode="random", rng=new_rng(7))
    second = audio.fix_length(x, 40, crop_mode="random", rng=new_rng(7))

    assert first.size == 40
    assert np.allclose(first, second)                 # same seed, same window
    assert np.allclose(np.diff(first), 1.0)           # contiguous
    assert 0 <= first[0] <= 60                        # a legal start offset


def test_fix_length_exact_length_is_unchanged() -> None:
    """A signal already at the target length passes through untouched."""
    x = np.arange(64000, dtype=np.float32)
    assert np.allclose(audio.fix_length(x, 64000), x)


def test_fix_length_rejects_bad_input() -> None:
    """Empty signals and unknown modes fail loudly rather than silently."""
    with pytest.raises(ValueError):
        audio.fix_length(np.array([], dtype=np.float32), 10)
    with pytest.raises(ValueError):
        audio.fix_length(np.ones(10, dtype=np.float32), 20, pad_mode="nonsense")
    with pytest.raises(ValueError):
        audio.fix_length(np.ones(30, dtype=np.float32), 20, crop_mode="nonsense")


# ---------------------------------------------------------------------------
# audio: framing and spectrogram
# ---------------------------------------------------------------------------
def test_frame_params_match_the_25_10_ms_spec(cfg: Config) -> None:
    """25 ms -> 400 samples, 10 ms -> 160 samples, n_fft -> 512."""
    frame_length, hop_length, n_fft = audio.frame_params(cfg)
    assert frame_length == 400
    assert hop_length == 160
    assert n_fft == 512
    assert audio.next_power_of_two(400) == 512


def test_frame_count_matches_the_formula() -> None:
    """Framing a 4 s signal gives exactly 1 + (64000 - 400) // 160 = 398 frames."""
    x = np.random.default_rng(0).standard_normal(64000).astype(np.float32)
    frames = audio.frame_signal(x, frame_length=400, hop_length=160)

    expected = 1 + (64000 - 400) // 160
    assert expected == 398
    assert frames.shape == (expected, 400)
    assert audio.num_frames(64000, 400, 160) == expected


def test_frames_start_at_multiples_of_the_hop() -> None:
    """Frame k covers samples [k*hop, k*hop + frame_length) - no centring."""
    x = np.arange(1000, dtype=np.float32)
    frames = audio.frame_signal(x, frame_length=400, hop_length=160, window=None)

    assert frames[0][0] == pytest.approx(0.0)
    assert frames[1][0] == pytest.approx(160.0)
    assert frames[2][0] == pytest.approx(320.0)


def test_frame_signal_applies_the_window() -> None:
    """A Hamming window is applied, and it is the standard one."""
    x = np.ones(400, dtype=np.float32)
    frames = audio.frame_signal(x, frame_length=400, hop_length=160, window="hamming")

    window = audio.get_window("hamming", 400)
    assert frames.shape == (1, 400)
    assert np.allclose(frames[0], window, atol=1e-6)
    # Hamming does not reach zero at the edges (unlike Hann) - a quick check
    # that we really got Hamming.
    assert 0.05 < window[0] < 0.1


def test_frame_signal_handles_a_too_short_signal() -> None:
    """Shorter than one frame yields an empty (0, frame_length) array, not a crash."""
    frames = audio.frame_signal(np.ones(100, dtype=np.float32), 400, 160)
    assert frames.shape == (0, 400)


def test_spectrogram_shape_agrees_with_framing() -> None:
    """The STFT produces one column per frame_signal row, with 1 + n_fft/2 bins."""
    x = np.random.default_rng(1).standard_normal(64000).astype(np.float32)

    frames = audio.frame_signal(x, 400, 160)
    spectrogram = audio.magnitude_spectrogram(x, frame_length=400, hop_length=160)

    assert spectrogram.shape == (1 + 512 // 2, frames.shape[0])
    assert np.all(spectrogram >= 0.0)


def test_spectrogram_frames_align_with_frame_signal() -> None:
    """Column m of the STFT is the DFT of row m of frame_signal.

    This is the alignment that the padding in magnitude_spectrogram exists to
    guarantee; if it ever breaks, spectrogram features and frame-level features
    would silently describe different moments in time.
    """
    x = np.random.default_rng(2).standard_normal(8000).astype(np.float32)

    frames = audio.frame_signal(x, 400, 160, window="hamming")
    spectrogram = audio.magnitude_spectrogram(x, 400, 160, n_fft=512)

    for index in (0, 5, frames.shape[0] - 1):
        reference = np.abs(np.fft.rfft(frames[index], n=512))
        assert np.allclose(spectrogram[:, index], reference, atol=1e-4)


def test_spectrogram_locates_a_known_tone() -> None:
    """A 2 kHz tone peaks in the bin nearest 2 kHz (bin 64 at n_fft=512)."""
    t = np.arange(16000) / SAMPLE_RATE
    tone = np.sin(2 * np.pi * 2000.0 * t).astype(np.float32)

    spectrogram = audio.magnitude_spectrogram(tone, 400, 160, n_fft=512)
    peak_bin = int(np.argmax(spectrogram.mean(axis=1)))
    expected_bin = int(round(2000.0 / (SAMPLE_RATE / 512)))

    assert abs(peak_bin - expected_bin) <= 1


def test_spectrogram_rejects_too_small_n_fft() -> None:
    """n_fft < frame_length would truncate the window; refuse it."""
    with pytest.raises(ValueError):
        audio.magnitude_spectrogram(np.ones(1000, dtype=np.float32), 400, 160, n_fft=256)


# ---------------------------------------------------------------------------
# audio: the end-to-end front end
# ---------------------------------------------------------------------------
def test_preprocess_produces_the_configured_length(cfg: Config) -> None:
    """preprocess always returns duration * sample_rate samples."""
    expected = int(cfg.audio.duration * cfg.audio.sample_rate)
    assert expected == 64000

    short = np.random.default_rng(3).standard_normal(8000).astype(np.float32)
    long = np.random.default_rng(4).standard_normal(200000).astype(np.float32)

    assert audio.preprocess(short, cfg).size == expected
    assert audio.preprocess(long, cfg).size == expected


def test_load_and_preprocess_round_trip(tmp_path: Path, cfg: Config) -> None:
    """A file on disk goes in, a fixed-length float32 vector comes out."""
    signal = (np.random.default_rng(5).standard_normal(20000) * 0.2).astype(np.float32)
    path = tmp_path / "utt.flac"
    sf.write(path, signal, SAMPLE_RATE, format="FLAC", subtype="PCM_16")

    result = audio.load_and_preprocess(path, cfg)
    assert result.shape == (64000,)
    assert result.dtype == np.float32
    assert np.all(np.isfinite(result))


def test_peak_normalize_is_a_no_op_on_silence() -> None:
    """An all-zero signal must not divide by zero."""
    silence = np.zeros(100, dtype=np.float32)
    assert np.allclose(audio.peak_normalize(silence), 0.0)


def test_trim_silence_removes_margins_but_never_everything() -> None:
    """Silence is trimmed from both ends; an all-silent input survives intact."""
    speech = (np.random.default_rng(6).standard_normal(8000) * 0.5).astype(np.float32)
    padded = np.concatenate(
        [np.zeros(4000, dtype=np.float32), speech, np.zeros(4000, dtype=np.float32)]
    )

    trimmed = audio.trim_silence(padded, top_db=30.0)
    assert trimmed.size < padded.size
    assert trimmed.size >= speech.size * 0.8

    silence = np.zeros(1000, dtype=np.float32)
    assert audio.trim_silence(silence).size == 1000


# ---------------------------------------------------------------------------
# manifest: protocol parsing
# ---------------------------------------------------------------------------
# One line of each real format. The column *order* differs (attack is 4th in
# 2019 and 5th in 2021) and so does the column count, which is exactly why the
# parser identifies fields by shape.
ASVSPOOF2019_LINES = """\
LA_0079 LA_T_1138215 - - bonafide
LA_0079 LA_T_1272637 - A01 spoof
LA_0080 LA_T_1000137 - A04 spoof
LA_0081 LA_T_1005579 - - bonafide
"""

ASVSPOOF2021_LINES = """\
LA_0023 DF_E_2000001 nocodec asvspoof A07 spoof notrim eval
LA_0024 DF_E_2000002 nocodec asvspoof - bonafide notrim eval
LA_0025 DF_E_2000003 low_mp3 vcc2018 A11 spoof notrim progress
"""


def test_parse_2019_la_format() -> None:
    """The 5-column 2019 LA protocol parses to the canonical fields."""
    rows = [manifest.parse_protocol_line(line) for line in ASVSPOOF2019_LINES.splitlines()]
    assert all(row is not None for row in rows)

    assert rows[0] == {
        "utt_id": "LA_T_1138215",
        "speaker": "LA_0079",
        "attack": "bonafide",       # no attack id on a bona fide row
        "label": "bonafide",
    }
    assert rows[1]["attack"] == "A01"
    assert rows[1]["label"] == "spoof"
    assert rows[2]["speaker"] == "LA_0080"


def test_parse_2021_df_format() -> None:
    """The 8-column 2021 DF metadata parses to the *same* fields.

    Note the different column positions: the attack id is column 5 here and
    column 4 in the 2019 protocol. A positional parser would silently read
    'asvspoof' as the attack.
    """
    rows = [manifest.parse_protocol_line(line) for line in ASVSPOOF2021_LINES.splitlines()]
    assert all(row is not None for row in rows)

    assert rows[0]["utt_id"] == "DF_E_2000001"
    assert rows[0]["speaker"] == "LA_0023"
    assert rows[0]["attack"] == "A07"
    assert rows[0]["label"] == "spoof"

    assert rows[1]["label"] == "bonafide"
    assert rows[1]["attack"] == "bonafide"
    # 'vcc2018' and 'low_mp3' must not be mistaken for an attack id.
    assert rows[2]["attack"] == "A11"


def test_parse_protocol_line_ignores_blanks_and_junk() -> None:
    """Blank lines, comments and lines with no key return None."""
    assert manifest.parse_protocol_line("") is None
    assert manifest.parse_protocol_line("   \n") is None
    assert manifest.parse_protocol_line("# a comment") is None
    assert manifest.parse_protocol_line("LA_0079 LA_T_1138215 - -") is None  # no key
    assert manifest.parse_protocol_line("garbage tokens here") is None


def test_parse_protocol_file_warns_on_missing_speaker(tmp_path: Path) -> None:
    """A protocol without speaker ids parses but warns - it cannot be split."""
    path = tmp_path / "nospeaker.txt"
    path.write_text("LA_T_0000001 - - bonafide\nLA_T_0000002 - A01 spoof\n", encoding="utf-8")

    with pytest.warns(UserWarning, match="speaker"):
        frame = manifest.parse_protocol_file(path)

    assert len(frame) == 2
    assert set(frame["speaker"]) == {manifest.UNKNOWN_SPEAKER}


def test_parse_protocol_file_rejects_an_unusable_file(tmp_path: Path) -> None:
    """A file with nothing parsable is an error, not an empty manifest."""
    path = tmp_path / "junk.txt"
    path.write_text("nothing here\nor here\n", encoding="utf-8")

    with pytest.raises(ValueError, match="No usable lines"):
        manifest.parse_protocol_file(path)


def test_build_manifest_from_protocol_schema(tmp_path: Path) -> None:
    """The builder emits the canonical columns with the right label ids."""
    protocol = tmp_path / "protocol.txt"
    protocol.write_text(ASVSPOOF2019_LINES, encoding="utf-8")

    frame = manifest.build_manifest_from_protocol(
        protocol_path=protocol,
        audio_dir=tmp_path / "flac",
        source_corpus="test_corpus",
        split="train",
        project_root=tmp_path,
        require_audio=False,   # no audio in this test; the parser is the subject
    )

    assert list(frame.columns) == list(manifest.MANIFEST_COLUMNS)
    assert len(frame) == 4
    assert set(frame["source_corpus"]) == {"test_corpus"}
    assert set(frame["split"]) == {"train"}
    # bonafide -> 0, spoof -> 1, consistently.
    assert frame.loc[frame["label"] == "bonafide", "label_id"].eq(0).all()
    assert frame.loc[frame["label"] == "spoof", "label_id"].eq(1).all()


# ---------------------------------------------------------------------------
# manifest: the speaker-disjoint split
# ---------------------------------------------------------------------------
def make_frame(n_speakers: int = 10, n_per_speaker: int = 10) -> pd.DataFrame:
    """A manifest-shaped frame with known speakers and a balanced class mix."""
    rows = []
    counter = 0
    for speaker_index in range(n_speakers):
        for utterance_index in range(n_per_speaker):
            counter += 1
            is_spoof = utterance_index % 2 == 0
            rows.append(
                {
                    "utt_id": f"LA_T_{counter:07d}",
                    "path": f"data/fake/LA_T_{counter:07d}.flac",
                    "label": "spoof" if is_spoof else "bonafide",
                    "label_id": 1 if is_spoof else 0,
                    "speaker": f"LA_{speaker_index:04d}",
                    "attack": "A01" if is_spoof else "bonafide",
                    "source_corpus": "test_corpus",
                    "split": "train",
                }
            )
    return pd.DataFrame(rows)[list(manifest.MANIFEST_COLUMNS)]


def test_split_is_speaker_disjoint() -> None:
    """No speaker may appear in both train and val. This is the core guarantee."""
    frame = make_frame()
    split = manifest.make_speaker_disjoint_split(frame, val_fraction=0.3, seed=1337)

    train_speakers = set(split.loc[split["split"] == "train", "speaker"])
    val_speakers = set(split.loc[split["split"] == "val", "speaker"])

    assert train_speakers, "train must not be empty"
    assert val_speakers, "val must not be empty"
    assert train_speakers.isdisjoint(val_speakers)
    # Every original row survives, in one split or the other.
    assert len(split) == len(frame)
    assert set(split["split"]) == {"train", "val"}


def test_split_is_reproducible_and_roughly_the_right_size() -> None:
    """The same seed gives the same split; the size is near the requested fraction."""
    frame = make_frame(n_speakers=20, n_per_speaker=10)

    first = manifest.make_speaker_disjoint_split(frame, val_fraction=0.15, seed=1337)
    second = manifest.make_speaker_disjoint_split(frame, val_fraction=0.15, seed=1337)
    different = manifest.make_speaker_disjoint_split(frame, val_fraction=0.15, seed=99)

    assert first["split"].equals(second["split"])
    assert not first["split"].equals(different["split"])

    val_fraction = (first["split"] == "val").mean()
    # Groups are whole speakers, so the utterance fraction only approximates
    # the requested speaker fraction.
    assert 0.05 < val_fraction < 0.30


def test_split_keeps_both_classes_on_both_sides() -> None:
    """Splitting by speaker must not accidentally empty a class."""
    frame = make_frame()
    split = manifest.make_speaker_disjoint_split(frame, val_fraction=0.3, seed=1337)

    for name in ("train", "val"):
        labels = set(split.loc[split["split"] == name, "label"])
        assert labels == {"bonafide", "spoof"}


def test_split_refuses_when_there_are_too_few_speakers() -> None:
    """One speaker cannot be split disjointly; say so instead of returning junk."""
    frame = make_frame(n_speakers=1, n_per_speaker=10)
    with pytest.raises(ValueError, match="at least 2 distinct speakers"):
        manifest.make_speaker_disjoint_split(frame, val_fraction=0.3, seed=1337)


# ---------------------------------------------------------------------------
# manifest: leakage detection
# ---------------------------------------------------------------------------
def test_check_leakage_passes_on_a_clean_split() -> None:
    """A correctly built split produces no errors."""
    frame = manifest.make_speaker_disjoint_split(make_frame(), val_fraction=0.3, seed=1337)
    report = manifest.check_leakage(frame, check_hashes=False, raise_on_error=True)
    assert report["errors"] == []


def test_check_leakage_detects_a_shared_speaker() -> None:
    """Moving one speaker's utterance into val must be caught."""
    frame = manifest.make_speaker_disjoint_split(make_frame(), val_fraction=0.3, seed=1337)

    # Deliberate corruption: take a row belonging to a training speaker and
    # relabel it as validation. This is what a naive random split would do
    # dozens of times over.
    train_row = frame.index[frame["split"] == "train"][0]
    frame.loc[train_row, "split"] = "val"

    with pytest.raises(manifest.LeakageError, match="speaker"):
        manifest.check_leakage(frame, check_hashes=False, raise_on_error=True)


def test_check_leakage_detects_a_shared_utterance_id() -> None:
    """The same utt_id in two splits must be caught."""
    frame = manifest.make_speaker_disjoint_split(make_frame(), val_fraction=0.3, seed=1337)

    duplicate = frame[frame["split"] == "train"].iloc[[0]].copy()
    duplicate["split"] = "val"
    duplicate["speaker"] = "LA_9999"          # isolate the utt_id check
    corrupted = pd.concat([frame, duplicate], ignore_index=True)

    report = manifest.check_leakage(corrupted, check_hashes=False, raise_on_error=False)
    assert any("utt_id" in error for error in report["errors"])

    with pytest.raises(manifest.LeakageError):
        manifest.check_leakage(corrupted, check_hashes=False, raise_on_error=True)


def test_check_leakage_detects_duplicate_audio(tmp_path: Path) -> None:
    """Byte-identical files in two splits are caught even with different ids.

    This is the check that a purely id-based test cannot make: the same
    recording copied under two names would otherwise pass unnoticed.
    """
    audio_dir = tmp_path / "flac"
    audio_dir.mkdir()

    signal = (np.random.default_rng(8).standard_normal(4000) * 0.2).astype(np.float32)
    unique = (np.random.default_rng(9).standard_normal(4000) * 0.2).astype(np.float32)

    for name, data in (("a", signal), ("b", signal), ("c", unique)):
        sf.write(audio_dir / f"{name}.flac", data, SAMPLE_RATE,
                 format="FLAC", subtype="PCM_16")

    frame = pd.DataFrame(
        [
            {"utt_id": "LA_T_0000001", "path": "flac/a.flac", "label": "bonafide",
             "label_id": 0, "speaker": "LA_0001", "attack": "bonafide",
             "source_corpus": "t", "split": "train"},
            # Same audio, different id, different speaker - only the hash catches it.
            {"utt_id": "LA_T_0000002", "path": "flac/b.flac", "label": "bonafide",
             "label_id": 0, "speaker": "LA_0002", "attack": "bonafide",
             "source_corpus": "t", "split": "val"},
            {"utt_id": "LA_T_0000003", "path": "flac/c.flac", "label": "spoof",
             "label_id": 1, "speaker": "LA_0003", "attack": "A01",
             "source_corpus": "t", "split": "val"},
        ]
    )[list(manifest.MANIFEST_COLUMNS)]

    with pytest.raises(manifest.LeakageError, match="identical audio"):
        manifest.check_leakage(frame, project_root=tmp_path, check_hashes=True,
                               raise_on_error=True)


def test_check_leakage_allows_cross_corpus_eval_overlap() -> None:
    """A speaker shared by two *evaluation* corpora is a note, not an error.

    ASVspoof 2021 DF is built from the 2019 LA evaluation partition, so they
    legitimately share speakers. What matters is that neither touches train.
    """
    frame = pd.DataFrame(
        [
            {"utt_id": "LA_T_0000001", "path": "a.flac", "label": "bonafide",
             "label_id": 0, "speaker": "LA_0001", "attack": "bonafide",
             "source_corpus": "asvspoof2019_la", "split": "train"},
            {"utt_id": "LA_E_0000002", "path": "b.flac", "label": "spoof",
             "label_id": 1, "speaker": "LA_0055", "attack": "A10",
             "source_corpus": "asvspoof2019_la", "split": "eval"},
            {"utt_id": "DF_E_0000003", "path": "c.flac", "label": "spoof",
             "label_id": 1, "speaker": "LA_0055", "attack": "A10",
             "source_corpus": "asvspoof2021_df", "split": "eval2021df"},
        ]
    )[list(manifest.MANIFEST_COLUMNS)]

    report = manifest.check_leakage(frame, check_hashes=False, raise_on_error=False)
    assert report["errors"] == []
    assert any("evaluation splits" in note for note in report["notes"])


def test_check_leakage_ignores_the_unknown_speaker_placeholder() -> None:
    """'unknown' is a placeholder, not a speaker, so it is not overlap."""
    frame = make_frame(n_speakers=4, n_per_speaker=4)
    frame["speaker"] = manifest.UNKNOWN_SPEAKER
    frame.loc[frame.index[:8], "split"] = "eval"

    report = manifest.check_leakage(frame, check_hashes=False, raise_on_error=False)
    assert report["errors"] == []


# ---------------------------------------------------------------------------
# manifest: I/O and summary
# ---------------------------------------------------------------------------
def test_manifest_round_trips_through_csv(tmp_path: Path) -> None:
    """save_manifest / load_manifest preserve the schema and the rows."""
    frame = make_frame(n_speakers=3, n_per_speaker=4)
    path = manifest.save_manifest(frame, tmp_path / "m", fmt="csv")

    assert path.suffix == ".csv"
    reloaded = manifest.load_manifest(path)
    assert list(reloaded.columns) == list(manifest.MANIFEST_COLUMNS)
    pd.testing.assert_frame_equal(frame, reloaded)


def test_load_manifest_rejects_a_missing_column(tmp_path: Path) -> None:
    """A manifest missing a required column fails at load, not three stages later."""
    path = tmp_path / "broken.csv"
    pd.DataFrame({"utt_id": ["a"], "path": ["b"]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing columns"):
        manifest.load_manifest(path)


def test_summarize_reports_classes_and_attacks() -> None:
    """The summary mentions both classes, the attack ids and the splits."""
    frame = manifest.make_speaker_disjoint_split(make_frame(), val_fraction=0.3, seed=1337)
    text = manifest.summarize(frame)

    for expected in ("bonafide", "spoof", "A01", "train", "val", "Speakers per split"):
        assert expected in text


def test_relativize_and_resolve_round_trip(tmp_path: Path) -> None:
    """Paths stored in a manifest are portable and resolve back correctly."""
    target = tmp_path / "data" / "x.flac"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"")

    stored = manifest.relativize(target, tmp_path)
    assert stored == "data/x.flac"        # forward slashes, even on Windows
    assert manifest.resolve_path(stored, tmp_path) == target


# ---------------------------------------------------------------------------
# End-to-end: the synthetic protocol format really is parseable
# ---------------------------------------------------------------------------
def test_synthetic_style_dataset_builds_and_splits(tmp_path: Path) -> None:
    """A miniature stand-in for the demo: write FLAC + protocol, build, split, check.

    This is the whole Phase 1 contract in one test, without touching the real
    ``data/`` directory.
    """
    # A config pointing entirely inside tmp_path.
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"paths": {"synthetic": "data/synthetic_demo"}}),
        encoding="utf-8",
    )
    local_cfg = load_config(config_path)

    root = tmp_path / "data" / "synthetic_demo"
    audio_dir = root / "flac"
    audio_dir.mkdir(parents=True)

    rng = new_rng(11)
    lines: list[str] = []
    counter = 0
    for speaker_index in range(4):
        for utterance_index in range(6):
            counter += 1
            utt_id = f"LA_T_{counter:07d}"
            # Distinct noise per file, so no two files hash alike.
            data = (rng.standard_normal(SAMPLE_RATE) * 0.2).astype(np.float32)
            sf.write(audio_dir / f"{utt_id}.flac", data, SAMPLE_RATE,
                     format="FLAC", subtype="PCM_16")

            speaker = f"LA_{speaker_index:04d}"
            if utterance_index % 2 == 0:
                lines.append(f"{speaker} {utt_id} - A01 spoof")
            else:
                lines.append(f"{speaker} {utt_id} - - bonafide")

    (root / "protocol.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    frame = manifest.build_synthetic(local_cfg)
    assert len(frame) == 24
    assert list(frame.columns) == list(manifest.MANIFEST_COLUMNS)

    split = manifest.make_speaker_disjoint_split(frame, val_fraction=0.25, seed=1337)
    report = manifest.check_leakage(
        split, project_root=local_cfg.project_root, check_hashes=True,
        raise_on_error=True,
    )
    assert report["errors"] == []

    # And every listed file actually loads through the shared front end.
    sample_path = manifest.resolve_path(str(split["path"].iloc[0]), local_cfg.project_root)
    waveform, rate = audio.load_audio(sample_path)
    assert rate == SAMPLE_RATE
    assert waveform.ndim == 1
