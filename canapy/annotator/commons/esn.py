# Author: Nathan Trouvain at 04/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import numpy as np

from reservoirpy.nodes import Reservoir, Ridge, ESN
from reservoirpy.mat_gen import fast_spectral_initialization

from .exceptions import NotTrainedError
from .postprocess import group_frames, maximum_a_posteriori
from .mfccs import load_mfccs_for_annotation


def init_esn_model(model_config, input_dim, audio_features, seed=None):
    scalings = []
    if "mfcc" in audio_features:
        iss = np.ones((input_dim,)) * model_config.iss
        scalings.append(iss)
    if "delta" in audio_features:
        isd = np.ones((input_dim,)) * model_config.isd
        scalings.append(isd)
    if "delta2" in audio_features:
        isd2 = np.ones((input_dim,)) * model_config.isd2
        scalings.append(isd2)

    input_scaling = np.concatenate(scalings, axis=0)
    bias_scaling = model_config.iss

    reservoir = Reservoir(
        model_config.units,
        sr=model_config.sr,
        lr=model_config.leak,
        input_scaling=input_scaling,
        bias_scaling=bias_scaling,
        W=fast_spectral_initialization,
        seed=seed,
    )

    readout = Ridge(ridge=model_config.ridge)

    return ESN(
        reservoir=reservoir,
        readout=readout,
        workers=model_config.workers,
        backend=model_config.backend,
    )


def predict_with_esn(
    annotator,
    corpus,
    return_classes=True,
    return_group=False,
    return_raw=False,
    redo_transforms=False,
):
    if not annotator.trained:
        raise NotTrainedError(
            "Call .fit on annotated data (Corpus) before calling " ".predict."
        )

    corpus = annotator.transforms(
        corpus,
        purpose="annotation",
        output_directory=annotator.spec_directory,
    )

    notated_paths, mfccs = load_mfccs_for_annotation(corpus)

    print(annotator.rpy_model.input_dim, annotator.rpy_model.output_dim)

    raw_preds = annotator.rpy_model.run(mfccs)

    cls_preds = None
    group_preds = None
    if return_classes or return_group:
        cls_preds = []
        for y in raw_preds:
            y_map = maximum_a_posteriori(y, classes=annotator.vocab)
            cls_preds.append(y_map)

        if return_group:
            group_preds = []
            for y_cls in cls_preds:
                seq = group_frames(y_cls)
                group_preds.append(seq)

    if not return_raw:
        raw_preds = None
    if not return_classes:
        cls_preds = None
    if not return_group:
        group_preds = None

    return notated_paths, group_preds, cls_preds, raw_preds
