# Author: Nathan Trouvain at 17/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import numpy as np
import pandas as pd


if __name__ == "__main__":
    df = pd.DataFrame({
        "a": list("aaaaaa"),
        })

    dfp = pd.DataFrame({
        "p1": list("aaaaaa"),
        "p2": list("aaaaaa"),
        "p3": list("aabbbb"),
        })

    x = dfp.apply(lambda column: (df["a"] == column).sum() / len(df), axis="rows")
    print(x)

    uniques = dfp.apply(lambda col: np.unique(col, return_counts=True), axis="rows").T
    uniques.columns = ["label", "count"]
    uniques["score"] = x
    print(uniques)

    b = {}
    for i in range(10):
        a = dfp.apply(
            lambda col: pd.Series(
                np.unique(col, return_counts=True) + (
                    (col == df["a"]).sum() / len(df),),
                index=["label", "count", "score"]
                ),
            axis="rows").T

        labels = {r.Index: r.label[np.argmax(r.count)] for r in a.itertuples()}
        b[i] = labels

    print(pd.DataFrame.from_dict(b, orient="index"))

    print((a["score"] < 1).all())

    print(a["count"].values)
