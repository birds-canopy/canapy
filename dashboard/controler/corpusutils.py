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
