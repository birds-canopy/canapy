# Author: Nathan Trouvain at 11/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import logging

import pandas as pd

logger = logging.getLogger("canapy")


def mark_whole_corpus_as_train(corpus):
    df = corpus.dataset
    df["train"] = True
    corpus.dataset = df
    logger.info("Marking whole corpus as training data.")
    return corpus


def query_split(corpus, split="all"):
    if split != "all":
        if split == "test":
            query = "not train"
        elif split == "train":
            query = "train"
        else:
            raise ValueError(f"split should be 'all', 'train' or 'test, "
                             f"not {split}.")
        return corpus[query]
    else:
        return corpus
