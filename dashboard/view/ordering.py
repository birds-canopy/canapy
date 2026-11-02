# Author: Axel Arnaud
# Licence: BSD-3-Clause
# Copyright: Axel Arnaud
"""Keeping one's place in a paged list of samples across a corpus change.

Applying a correction rebuilds the corpus, and with it the list a page was cut
from. A page number does not survive that: annotations disappear, and the ones
after them shift up. What can survive is the sample that was on screen, so
the page is found again from it rather than from its number.
"""
from canapy.correction.base import make_annot_key

ANCHOR_MISSING = "missing"
ANCHOR_EXACT = "exact"
ANCHOR_FALLBACK = "fallback"


def anchor_of(row):
    """Stable identifier of one annotation, for finding it again later.

    The same key the correction store uses, so an annotation is designated the
    same way whether it is being corrected or merely looked at.
    """
    return make_annot_key(row.notated_path, row.onset_s)


def anchors_of(frame):
    return [anchor_of(row) for row in frame.itertuples()]


def locate(anchor, previous_anchors, new_frame):
    """Where ``anchor`` sits in ``new_frame``, or the nearest survivor.

    Returns ``(position, status)``. An annotation that no longer exists — merged
    away, or whose onset an edit has moved — falls back to the closest one that
    is still there in ``previous_anchors``.
    """
    if anchor is None:
        return 0, ANCHOR_MISSING

    positions = {key: i for i, key in enumerate(anchors_of(new_frame))}
    if anchor in positions:
        return positions[anchor], ANCHOR_EXACT

    if anchor not in previous_anchors:
        return 0, ANCHOR_MISSING

    # forward before backward: the anchor opens its page, so the sample after
    # it is still on that page while the one before is on the previous one
    start = previous_anchors.index(anchor)
    for distance in range(1, len(previous_anchors)):
        for candidate in (start + distance, start - distance):
            if 0 <= candidate < len(previous_anchors):
                key = previous_anchors[candidate]
                if key in positions:
                    return positions[key], ANCHOR_FALLBACK

    return 0, ANCHOR_MISSING


def page_of(position, page_size):
    return position // page_size if page_size else 0


def restore_page(anchor, previous_anchors, new_frame, page_size):
    """The page of ``new_frame`` to reopen on, and why."""
    position, status = locate(anchor, previous_anchors, new_frame)
    return page_of(position, page_size), status
