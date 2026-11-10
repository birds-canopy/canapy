# Author: Axel Arnaud
# Licence: BSD-3-Clause
# Copyright: Axel Arnaud
"""Interactive per-class plots of the Sample Correction panel."""
import pathlib

import numpy as np
import pandas as pd
from bokeh.models import ColumnDataSource, FactorRange, HoverTool, Span, TapTool
from bokeh.plotting import figure
from matplotlib.cbook import boxplot_stats

from ..ordering import anchor_of

BOX_COLUMNS = ["label", "lower", "q1", "median", "q3", "upper"]
OUTLIER_COLUMNS = ["label", "value", "key", "file", "time"]
BLUE = "#3b82f6"
DARK = "#1f2937"


def box_and_outliers(frame, value_col, classes):
    """Box statistics per class, and the samples outside the whiskers.

    Returns two DataFrames with ``BOX_COLUMNS`` and ``OUTLIER_COLUMNS``. An
    outlier carries the key that designates its sample (see ``anchor_of``).
    """
    boxes, outliers = [], []
    for label in classes:
        rows = frame[(frame["label"] == label) & frame[value_col].notna()]
        if rows.empty:
            continue
        stats = boxplot_stats(rows[value_col].to_numpy(dtype=float))[0]
        boxes.append({
            "label": label,
            "lower": stats["whislo"],
            "q1": stats["q1"],
            "median": stats["med"],
            "q3": stats["q3"],
            "upper": stats["whishi"],
        })
        is_outlier = (rows[value_col] < stats["whislo"]) | (rows[value_col] > stats["whishi"])
        for row in rows[is_outlier].itertuples():
            outliers.append({
                "label": label,
                "value": float(getattr(row, value_col)),
                "key": anchor_of(row),
                "file": pathlib.Path(row.notated_path).stem,
                "time": f"{row.onset_s:.2f}s - {row.offset_s:.2f}s",
            })
    return pd.DataFrame(boxes, columns=BOX_COLUMNS), pd.DataFrame(outliers, columns=OUTLIER_COLUMNS)


def _base_figure(classes, title, y_label):
    fig = figure(
        title=title,
        x_range=FactorRange(factors=list(classes)),
        height=350,
        sizing_mode="stretch_width",
        tools="pan,box_zoom,wheel_zoom,reset",
        active_drag="box_zoom",
        toolbar_location="above",
    )
    fig.yaxis.axis_label = y_label
    fig.xaxis.major_label_orientation = np.pi / 2 if len(classes) > 10 else np.pi / 4
    fig.xgrid.grid_line_color = None
    return fig


def count_figure(classes, counts, threshold, title):
    fig = _base_figure(classes, f"{title}, minimal fitting threshold: {threshold} samples", "Count")
    source = ColumnDataSource({"label": list(classes), "count": [int(c) for c in counts]})
    fig.vbar(x="label", top="count", width=0.8, color=BLUE, source=source)
    fig.add_layout(Span(location=threshold, dimension="width", line_color="red",
                        line_dash="dashed", line_width=1.5))
    fig.add_tools(HoverTool(tooltips=[("Class", "@label"), ("Count", "@count")]))
    return fig


def distribution_figure(frame, value_col, classes, y_label, title, on_select):
    """Box plots whose outliers call ``on_select(label, key)`` when clicked."""
    fig = _base_figure(classes, title, y_label)
    boxes, outliers = box_and_outliers(frame, value_col, classes)

    box_source = ColumnDataSource(boxes)
    fig.segment(x0="label", y0="lower", x1="label", y1="upper", color=DARK, source=box_source)
    for end in ("lower", "upper"):
        fig.vbar(x="label", top=end, bottom=end, width=0.35, line_color=DARK, source=box_source)
    fig.vbar(x="label", top="q3", bottom="q1", width=0.7, fill_color=BLUE, line_color=DARK,
             source=box_source)
    fig.vbar(x="label", top="median", bottom="median", width=0.7, line_color="white",
             line_width=2, source=box_source)

    source = ColumnDataSource(outliers)
    points = fig.scatter(
        x="label", y="value", size=7, fill_color="white", line_color=DARK, source=source,
        hover_fill_color="#f59e0b", nonselection_fill_alpha=1, nonselection_line_alpha=1,
    )
    fig.add_tools(
        TapTool(renderers=[points]),
        HoverTool(renderers=[points], tooltips=[
            ("Class", "@label"), (y_label, "@value{0.[00]}"), ("File", "@file"), ("Time", "@time"),
        ]),
    )

    def on_tap(attr, old, new):
        if not new:
            return
        index = new[0]
        # emptied so that the same point can be clicked again
        source.selected.indices = []
        on_select(source.data["label"][index], source.data["key"][index])

    source.selected.on_change("indices", on_tap)
    return fig
