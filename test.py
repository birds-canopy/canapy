from pathlib import Path

from canapy import Corpus
from canapy.annotator import SynAnnotator


path = Path("/run/media/nathan/Nathan4T/test_data_canapy")

if __name__ == "__main__":
    annotated_corpus = Corpus.from_directory(
        audio_directory=path / "data/annotated",
        annots_directory=path / "data/annotated",
        audio_ext=".npy",
    )

    non_annotated_corpus = Corpus.from_directory(
        audio_directory=path / "data/non_annotated",
        audio_ext=".npy",
    )

    syn_annotator = SynAnnotator()

    syn_annotator.fit(annotated_corpus)

    predictions = syn_annotator.predict(non_annotated_corpus)

    print(predictions.dataset)
