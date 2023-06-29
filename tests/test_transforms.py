# Author: Nathan Trouvain at 29/06/2023 <nathan.trouvain@inria.fr>
# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>
import crowsetta
import numpy as np

from canapy.corpus import Corpus
from canapy.transforms.commons.training import encode_labels


def test_encode_labels():
    import numpy as np

    # crowsetta.register_format(Marron1CSV)
    c = Corpus.from_directory(
        audio_directory="/home/nathan/Documents/Code/canapy-test/data/",
        annots_directory="/home/nathan/Documents/Code/canapy-test/data/",
        )

    df = c.dataset

    df["train"] = False

    samples = df.sample(50).index
    df.loc[samples, "train"] = True

    print(df.query("train == True"))

    c = encode_labels(c, resource_name="encoded_labels")

    print(c)


if __name__ == "__main__":

    test_encode_labels()
