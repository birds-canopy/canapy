# Author: Nathan Trouvain at 04/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import numpy as np

from reservoirpy.nodes import Reservoir, Ridge, ESN
from reservoirpy.mat_gen import fast_spectral_initialization


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
