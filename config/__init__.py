import os

from .config import Config

_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "default",
    "default.config.toml"
)

default_config = Config.from_file(_path)
