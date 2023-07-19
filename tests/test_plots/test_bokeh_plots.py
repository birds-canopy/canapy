# Author: Nathan Trouvain at 19/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import numpy as np
import pandas as pd

from canapy.plots import plot_bokeh_confusion_matrix, plot_bokeh_label_count


def test_plot_bokeh_confusion_matrix():

    cm = np.random.uniform(size=(10, 10))
    classes = list("abcdefghij")

    fig = plot_bokeh_confusion_matrix(cm, classes)

    assert fig.title is None

    classes = None

    fig = plot_bokeh_confusion_matrix(cm, classes, title="foo")

    assert fig.title.text == "foo"


def test_plot_bokeh_label_count():

    df = pd.DataFrame({
        "label": list("aaabbcdefghabfjkdaz")
        })

    fig, counts = plot_bokeh_label_count(df)

    assert counts.loc["a"] == 5
    assert counts.loc["z"] == 1
