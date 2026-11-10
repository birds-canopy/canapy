# Licence: BSD-3-Clause
"""Correcting the bounds of an annotation, and checkpointing that correction."""
import pandas as pd
import pytest
import toml

from canapy.corpus import Corpus
from canapy.correction.base import Corrector, correct_bounds, make_annot_key

PATH = "/song.wav"


@pytest.fixture()
def corpus(config):
    df = pd.DataFrame({
        "label": ["a", "b", "c"],
        "onset_s": [0.0, 1.0, 2.0],
        "offset_s": [0.5, 1.5, 2.5],
        "notated_path": [PATH] * 3,
    })
    return Corpus.from_df(df, config=config)


def _key(onset):
    return make_annot_key(PATH, onset)


def test_both_bounds_are_moved(corpus):
    result = correct_bounds(corpus, {_key(1.0): (0.9, 1.7)})

    row = result.dataset[result.dataset.label == "b"].iloc[0]
    assert (row.onset_s, row.offset_s) == pytest.approx((0.9, 1.7))


def test_the_other_annotations_are_left_alone(corpus):
    result = correct_bounds(corpus, {_key(1.0): (0.9, 1.7)})

    others = result.dataset[result.dataset.label != "b"]
    assert list(others.onset_s) == [0.0, 2.0]
    assert list(others.offset_s) == [0.5, 2.5]


def test_the_label_is_untouched(corpus):
    result = correct_bounds(corpus, {_key(1.0): (0.9, 1.7)})
    assert list(result.dataset.label) == ["a", "b", "c"]


def test_the_corrected_annotation_gets_the_bounds_it_asked_for(corpus):
    result = correct_bounds(corpus, {_key(1.0): (0.4, 2.1)})

    row = result.dataset[result.dataset.label == "b"].iloc[0]
    assert (row.onset_s, row.offset_s) == pytest.approx((0.4, 2.1))


def test_a_neighbour_grown_into_is_trimmed_not_left_overlapping(corpus):
    # canapy gives shared frames to whichever starts later, so an untrimmed
    # neighbour would silently undo the correction
    result = correct_bounds(corpus, {_key(1.0): (0.4, 2.1)}).dataset

    before = result[result.label == "a"].iloc[0]
    after = result[result.label == "c"].iloc[0]
    assert before.offset_s == pytest.approx(0.4)
    assert after.onset_s == pytest.approx(2.1)


def test_the_result_holds_no_overlap(corpus):
    result = correct_bounds(corpus, {_key(1.0): (0.4, 2.1)}).dataset.sort_values("onset_s")
    assert (result.onset_s.values[1:] >= result.offset_s.values[:-1] - 1e-9).all()


def test_a_neighbour_swallowed_whole_is_dropped(corpus):
    result = correct_bounds(corpus, {_key(1.0): (0.0, 2.5)}).dataset
    assert list(result.label) == ["b"]


def test_a_neighbour_left_alone_keeps_its_bounds(corpus):
    result = correct_bounds(corpus, {_key(1.0): (1.1, 1.4)}).dataset
    assert list(result.onset_s) == pytest.approx([0.0, 1.1, 2.0])
    assert list(result.offset_s) == pytest.approx([0.5, 1.4, 2.5])


def test_two_corrections_do_not_trim_each_other(corpus):
    result = correct_bounds(corpus, {
        _key(0.0): (0.0, 1.2),
        _key(1.0): (1.0, 1.5),
    }).dataset
    assert list(result[result.label == "a"].offset_s) == pytest.approx([1.2])
    assert list(result[result.label == "b"].onset_s) == pytest.approx([1.0])


def test_a_vanished_annotation_is_skipped_not_raised(corpus, caplog):
    result = correct_bounds(corpus, {make_annot_key("/gone.wav", 9.0): (0.1, 0.2)})
    assert len(result.dataset) == 3


def test_bounds_are_applied_after_the_labels(corpus, tmp_path):
    corrector = Corrector(checkpoint_directory=tmp_path)

    result = corrector.correct(
        corpus,
        annot_corrections={_key(1.0): "z"},
        bounds_corrections={_key(1.0): (0.9, 1.7)},
    )

    # both landed on the same annotation: the label correction was resolved
    # before its key was invalidated by the move
    row = result.dataset[(result.dataset.onset_s - 0.9).abs() < 1e-9].iloc[0]
    assert row.label == "z"
    assert row.offset_s == pytest.approx(1.7)


def test_a_checkpoint_carries_the_bounds(tmp_path, corpus):
    corrector = Corrector(checkpoint_directory=tmp_path)
    corrector.correct(corpus, bounds_corrections={_key(1.0): (0.9, 1.7)},
                      checkpoint=True)

    written = toml.load(tmp_path / "correction-1.toml")

    assert written["bounds"] == {_key(1.0): [0.9, 1.7]}


def test_a_checkpoint_written_before_bounds_still_loads(tmp_path):
    (tmp_path / "correction-1.toml").write_text('[class]\n\n[annot]\n')

    corrector = Corrector.from_checkpoints(tmp_path)

    assert corrector.correction_history[0]["bounds"] == {}


def test_a_checkpointed_bounds_correction_replays(tmp_path, corpus):
    Corrector(checkpoint_directory=tmp_path).correct(
        corpus, bounds_corrections={_key(1.0): (0.9, 1.7)}, checkpoint=True
    )

    replayed = Corrector.from_checkpoints(tmp_path).correct_from_history(corpus, 0)

    row = replayed.dataset[replayed.dataset.label == "b"].iloc[0]
    assert (row.onset_s, row.offset_s) == pytest.approx((0.9, 1.7))


def test_a_checkpoint_writes_every_channel_even_when_empty(tmp_path, corpus):
    # toml.dump drops a None, and from_checkpoints requires class and annot
    Corrector(checkpoint_directory=tmp_path).correct(
        corpus, bounds_corrections={_key(1.0): (0.9, 1.7)}, checkpoint=True
    )

    written = toml.load(tmp_path / "correction-1.toml")

    assert set(written) == {"class", "annot", "bounds"}


def test_a_moved_annotation_is_not_dropped_in_its_neighbour_s_place(corpus):
    # b lands on a's onset; resolving the drop key after the move would find b
    result = correct_bounds(corpus, {_key(1.0): (0.0, 2.5)}).dataset

    assert list(result.label) == ["b"]
    assert (result.iloc[0].onset_s, result.iloc[0].offset_s) == pytest.approx((0.0, 2.5))
