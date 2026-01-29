from canapy.corpus import Corpus
from canapy.annotator import get_annotator
from config import default_config, Config
from canapy import metrics
import pickle
import toml
import pickletools
syn_annotator = get_annotator("syn-esn")(default_config)
nsyn_annotator = get_annotator("nsyn-esn")(default_config)
ensemble_annotator = get_annotator("ensemble")(default_config)
with "output/model/syn-esn" as file:
     config_syn = Config.from_file(file)