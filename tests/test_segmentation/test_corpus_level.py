# Licence: BSD-3-Clause
import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from canapy.corpus import Corpus
from canapy.segmentation import AlreadySegmented, to_syllable_level

SR = 44100
TONE_HZ = 4000.0
BURST_S = 0.040
GAP_S = 0.020
SILENCE_S = 0.300
PHRASES = [("A", 4), ("B", 6)]


@pytest.fixture()
def phrase_corpus(tmp_path, config):
    """One file holding two phrases of tone bursts, plus a silence between."""
    rng = np.random.default_rng(0)
    chunks, rows, cursor = [], [], 0.0

    def silence(duration):
        nonlocal cursor
        chunks.append(rng.normal(0, 1e-3, int(duration * SR)))
        cursor += duration

    silence(SILENCE_S)
    for label, n_bursts in PHRASES:
        onset = cursor
        for i in range(n_bursts):
            if i:
                silence(GAP_S)
            n = int(BURST_S * SR)
            t = np.arange(n) / SR
            chunks.append(np.sin(2 * np.pi * TONE_HZ * t) + rng.normal(0, 1e-3, n))
            cursor += BURST_S
        rows.append({"label": label, "onset_s": onset, "offset_s": cursor})
        silence(SILENCE_S)

    path = tmp_path / "song.wav"
    sf.write(str(path), np.concatenate(chunks), SR)

    df = pd.DataFrame(rows)
    df["notated_path"] = str(path)
    df["annot_path"] = str(tmp_path / "song.csv")

    return Corpus.from_df(df, config=config)


def test_phrases_become_syllables(phrase_corpus):
    result = to_syllable_level(phrase_corpus)
    assert len(result.dataset) == sum(n for _, n in PHRASES)


def test_labels_are_inherited(phrase_corpus):
    result = to_syllable_level(phrase_corpus)
    counts = result.dataset.groupby("label").size().to_dict()
    assert counts == {label: n for label, n in PHRASES}


def test_syllables_stay_inside_their_phrase(phrase_corpus):
    phrases = phrase_corpus.dataset
    result = to_syllable_level(phrase_corpus).dataset

    for row in result.itertuples(index=False):
        parent = phrases[phrases.label == row.label].iloc[0]
        assert row.onset_s >= parent.onset_s - 1e-9
        assert row.offset_s <= parent.offset_s + 1e-9
        assert row.offset_s > row.onset_s


def test_silence_rows_pass_through_untouched(phrase_corpus, config):
    silence_tag = config.transforms.annots.silence_tag
    df = phrase_corpus.dataset.copy()
    df.loc[len(df)] = {**df.iloc[0].to_dict(), "label": silence_tag,
                       "onset_s": 0.0, "offset_s": SILENCE_S}
    corpus = Corpus.from_df(df, config=config)

    result = to_syllable_level(corpus).dataset
    silences = result[result.label == silence_tag]

    assert len(silences) == 1
    assert silences.iloc[0].offset_s == pytest.approx(SILENCE_S)


def test_audio_resources_survive(phrase_corpus):
    phrase_corpus.register_data_resource("marker", pd.DataFrame(
        {"notated_path": phrase_corpus.dataset.notated_path.unique()}
    ))
    result = to_syllable_level(phrase_corpus)
    assert "marker" in result.data_resources


def test_an_unreadable_file_leaves_its_phrases_alone(phrase_corpus, config, tmp_path):
    df = phrase_corpus.dataset.copy()
    missing = df.iloc[0].to_dict()
    missing["notated_path"] = str(tmp_path / "missing.wav")
    df.loc[len(df)] = missing
    corpus = Corpus.from_df(df, config=config)

    result = to_syllable_level(corpus).dataset

    assert len(result[result.notated_path == missing["notated_path"]]) == 1
    assert len(result) == sum(n for _, n in PHRASES) + 1


def test_second_application_is_refused(phrase_corpus, config):
    config.data["transforms"]["annots"]["min_label_duration"] = (
        config.segmentation.min_syllable_duration
    )
    with pytest.raises(AlreadySegmented):
        to_syllable_level(phrase_corpus)


def test_the_result_is_refused_a_second_time(phrase_corpus):
    with pytest.raises(AlreadySegmented):
        to_syllable_level(to_syllable_level(phrase_corpus))


def test_a_long_minimum_syllable_is_not_mistaken_for_a_segmented_corpus(
    phrase_corpus, config
):
    # the panel lets this be raised well past min_label_duration; doing so must
    # not make a plain phrase-level corpus look already segmented
    config.data["segmentation"]["min_syllable_duration"] = 0.030

    result = to_syllable_level(phrase_corpus)

    assert len(result.dataset) == sum(n for _, n in PHRASES)
