import tempfile

from pathlib import Path

import pytest
import numpy as np

from canapy import Trainer, SynModel, NSynModel, Ensemble
from canapy import Dataset


@pytest.fixture
def trainset():
    return Dataset("./data/test-rouge6")


@pytest.fixture
def dataset():
    vocab = Dataset("./data/test-rouge6").vocab
    return Dataset("./data/test-rouge6-noannots/", vocab=vocab)


@pytest.fixture
def trainer():
    data = Dataset("./data/test-rouge6")
    return Trainer(data)


@pytest.fixture
def models():
    data = Dataset("./data/test-rouge6")
    trainer = Trainer(data)
    trainer.train()

    return trainer.syn_esn, trainer.nsyn_esn


@pytest.mark.parametrize(
    "model,synres,nsynres",
    [
        ("all", np.ndarray, np.ndarray),
        ("syn", np.ndarray, type(None)),
        ("nsyn", type(None), np.ndarray),
    ],
)
def test_trainer(trainset, model, synres, nsynres):
    trainer = Trainer(trainset)

    trainer.train(model=model)

    assert type(trainer.nsyn_esn.esn.Wout) is nsynres
    assert type(trainer.syn_esn.esn.Wout) is synres


def test_annotate(dataset, models):
    syn_esn, nsyn_esn = models

    outs = syn_esn.annotate(dataset, to_group=True)
    assert len(outs) == len(list(dataset.audios))
    outs = nsyn_esn.annotate(dataset, to_group=True)
    assert len(outs) == len(list(dataset.audios))
