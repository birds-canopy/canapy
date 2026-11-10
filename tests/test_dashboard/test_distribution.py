# Licence: BSD-3-Clause
import numpy as np
import pandas as pd

from dashboard.view.ordering import anchor_of
from dashboard.view.preprocess.distribution import box_and_outliers, distribution_figure


def _frame():
    # class A: [1 .. 10] and one far-away value; class B: tight, without outlier
    values_a = list(np.arange(1.0, 11.0)) + [100.0]
    values_b = [5.0, 5.1, 5.2, 5.3]
    rows = [
        {"label": "A", "value": v, "notated_path": f"a{i}.wav", "onset_s": float(i), "offset_s": i + 1.0}
        for i, v in enumerate(values_a)
    ] + [
        {"label": "B", "value": v, "notated_path": f"b{i}.wav", "onset_s": float(i), "offset_s": i + 1.0}
        for i, v in enumerate(values_b)
    ]
    return pd.DataFrame(rows)


def test_outliers_are_the_samples_outside_the_whiskers():
    frame = _frame()
    boxes, outliers = box_and_outliers(frame, "value", ["A", "B"])

    assert boxes["label"].tolist() == ["A", "B"]
    assert boxes.set_index("label").loc["A", "upper"] == 10.0
    assert outliers["value"].tolist() == [100.0]
    assert outliers["key"].iloc[0] == anchor_of(frame[frame["value"] == 100.0].iloc[0])


def test_missing_values_are_ignored():
    frame = _frame()
    frame.loc[frame["label"] == "B", "value"] = np.nan
    boxes, outliers = box_and_outliers(frame, "value", ["A", "B", "C"])

    assert boxes["label"].tolist() == ["A"]
    assert set(outliers["label"]) == {"A"}


def test_no_value_at_all_gives_empty_tables():
    frame = _frame().assign(value=np.nan)
    boxes, outliers = box_and_outliers(frame, "value", ["A", "B"])

    assert boxes.empty and outliers.empty
    distribution_figure(frame, "value", ["A", "B"], "y", "t", lambda label, key: None)


def _points_source(fig):
    return next(r.data_source for r in fig.renderers if "key" in r.data_source.data)


def test_clicking_a_point_designates_its_sample_and_can_be_repeated():
    frame = _frame()
    selected = []
    fig = distribution_figure(frame, "value", ["A", "B"], "y", "t",
                              lambda label, key: selected.append((label, key)))
    source = _points_source(fig)
    expected = ("A", anchor_of(frame[frame["value"] == 100.0].iloc[0]))

    source.selected.indices = [0]
    assert source.selected.indices == []
    source.selected.indices = [0]

    assert selected == [expected, expected]
