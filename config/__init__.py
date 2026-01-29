import os

from .config import Config

_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "default",
    "default.config.toml"
)

# _path_syn = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)),
#     "model",
#     "syn-esn.toml"
# )

# _path_nsyn = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)),
#     "model",
#     "nsyn-esn.toml"
# )
default_config = Config.from_file(_path)
# model_syn = Config.from_file(_path_syn)
# model_nsyn = Config.from_file(_path_nsyn)
