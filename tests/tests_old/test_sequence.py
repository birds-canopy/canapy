import pytest

from canapy.sequence import group, lev_dist, lev_sim, lcs


@pytest.mark.parametrize(
    "sequence,expected,min_frame",
    [
        (
            ["A", "A", "A", "B", "B", "B", "C", "C", "D", "A", "A"],
            [("A", 3), ("B", 3), ("C", 2), ("A", 2)],
            2,
        ),
        (
            ["A", "A", "B", "A", "B", "B", "B", "C", "C", "D", "A", "A"],
            [("A", 2), ("B", 3), ("C", 2), ("A", 2)],
            2,
        ),
    ],
)
def test_group(sequence, expected, min_frame):
    assert group(sequence, min_frame_nb=min_frame) == expected


@pytest.mark.parametrize(
    "s,t,expected",
    [(list("ABCDAB"), list("ACBDB"), 3), (list("ABCDABA"), list("BCDAB"), 2)],
)
def test_levenshtein(s, t, expected):
    assert lev_dist(s, t) == expected


@pytest.mark.parametrize(
    "s,t,expected",
    [
        (list("ABCDAB"), list("ACBDB"), 8 / 11),
        (list("ABCDABA"), list("BCDAB"), 10 / 12),
    ],
)
def test_levenshtein_sim(s, t, expected):
    assert -1e-3 < lev_sim(s, t) - expected < 1e-3


@pytest.mark.parametrize(
    "s,t,expected",
    [(list("ABCDAB"), list("ACBDB"), 4), (list("ABCDABA"), list("BCDAB"), 5)],
)
def test_lcs(s, t, expected):
    assert lcs(s, t) == expected
