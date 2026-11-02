# Licence: BSD-3-Clause
"""Keeping one's place in a paged sample list across a corpus change."""
import pandas as pd
import pytest

from dashboard.view.ordering import (
    ANCHOR_EXACT, ANCHOR_FALLBACK, ANCHOR_MISSING,
    anchors_of, locate, restore_page,
)

PAGE = 5


def _frame(n=60, path="/song.wav"):
    return pd.DataFrame({
        "label": ["A"] * n,
        "onset_s": [i * 0.2 for i in range(n)],
        "offset_s": [i * 0.2 + 0.1 for i in range(n)],
        "notated_path": [path] * n,
    })


def test_a_surviving_anchor_is_found_again():
    frame = _frame()
    keys = anchors_of(frame)
    assert locate(keys[17], keys, frame) == (17, ANCHOR_EXACT)


def test_a_vanished_anchor_falls_back_to_its_neighbour():
    frame = _frame()
    keys = anchors_of(frame)
    after = frame.drop(index=[17])

    position, status = locate(keys[17], keys, after)

    assert status == ANCHOR_FALLBACK
    assert anchors_of(after)[position] == keys[18]


def test_the_fallback_prefers_the_sample_after_the_anchor():
    # the anchor opens its page, so the sample after it is still on that page
    # while the one before is on the previous one
    frame = _frame()
    keys = anchors_of(frame)
    after = frame.drop(index=[20])

    position, _ = locate(keys[20], keys, after)

    assert restore_page(keys[20], keys, after, PAGE)[0] == 20 // PAGE
    assert anchors_of(after)[position] == keys[21]


def test_an_unknown_anchor_opens_on_the_first_page():
    assert locate("nowhere::0.0", [], _frame()) == (0, ANCHOR_MISSING)
    assert locate(None, [], _frame()) == (0, ANCHOR_MISSING)


def test_the_page_follows_the_samples_that_disappeared_before_it():
    frame = _frame()
    keys = anchors_of(frame)
    anchor = keys[40]                      # page 8
    after = frame.drop(index=range(0, 10))  # ten samples vanish ahead of it

    page, status = restore_page(anchor, keys, after, PAGE)

    assert status == ANCHOR_EXACT
    assert page == 6                        # position 30, so page 6
    assert anchor in anchors_of(after.iloc[page * PAGE:(page + 1) * PAGE])


def test_editing_an_onset_still_lands_on_the_same_page():
    frame = _frame()
    keys = anchors_of(frame)
    edited = frame.copy()
    edited.loc[20, "onset_s"] -= 0.03      # the anchor's own key changes

    page, status = restore_page(keys[20], keys, edited, PAGE)

    assert status == ANCHOR_FALLBACK
    assert page == 20 // PAGE


def test_the_pages_of_two_files_do_not_collide():
    a, b = _frame(10, "/a.wav"), _frame(10, "/b.wav")
    frame = pd.concat([a, b], ignore_index=True)
    keys = anchors_of(frame)
    assert len(set(keys)) == len(keys)
    assert locate(keys[12], keys, frame) == (12, ANCHOR_EXACT)
