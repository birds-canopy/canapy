from canapy.corpus import Corpus
from canapy.annotator.synannotator import SynAnnotator
from canapy.config import default_config

my_corpus = Corpus.from_directory(
    audio_directory="/home/vincent/Documents/data_canary/audio",  # path to audio data
    annots_directory="/home/vincent/Documents/data_canary/annotations" # path to annot data
)
my_syn_annotator = SynAnnotator(default_config, "/home/vincent/Documents/data_canary/spec")
my_syn_annotator.fit(my_corpus)