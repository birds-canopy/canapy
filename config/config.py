# Author: Nathan Trouvain at 27/06/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import toml
import yaml
import json
import logging
import collections.abc

import jsonschema

from pathlib import Path
from collections import UserDict


logger = logging.getLogger("config")


_FORMATTERS = {
    "json": json,
    "yaml": yaml,
    "toml": toml,
}


class _BaseConfig(UserDict):

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __setattr__(self, attr, value):
        if attr == "data":
            self.__dict__["data"] = value
            return
        self[attr] = value

    def __getattr__(self, attr):
        if attr in self.data:
            if isinstance(self.data[attr], _BaseConfig):
                return type(self.data[attr])(**self.data[attr])
            if isinstance(self.data[attr], collections.abc.Mapping):
                return type(self)(**self.data[attr])
            return self.data[attr]
        elif attr in self.__dict__:
            return self.__dict__[attr]
        else:
            raise AttributeError(attr)

    def __repr__(self):
        return type(self).__name__ + super(_BaseConfig, self).__repr__()
        
    def items(self):
        for key in self.data.keys():
            yield key, getattr(self, key)

    def solve(self, path: str):
        """Get items by key paths of the form:
        value = dict['key1/key2/key3']
        Useful for nested dictionnaries only.
        You may escape the '/' chararcter in 
        a key name using '\/'.
        """
        # Colons are a rather safe substitute
        substitute = path.replace("\/", ":")
        parts = [p.replace(":", "/") for p in substitute.split("/")]
        v = self.data
        for part in parts:
            v = v[part]
        return v
 

class ConfigSchema(_BaseConfig):
    
    def validate(self, config):
        jsonschema.validate(dict(**self), dict(**config))


class Config(_BaseConfig):

    @classmethod
    def from_file(cls, config_path):
        msg = ""
        try:
            with open(config_path, "r") as fp:
                config = yaml.load(fp, yaml.Loader)
        except Exception as e:
            msg += "\nYAML:" + str(e)
            try:
                with open(config_path, "r") as fp:
                    config = toml.load(fp)
            except Exception as e:
                msg += "\nTOML" + str(e)
                try:
                    with open(config_path, "r") as fp:
                        config = json.load(fp)
                except Exception as e:
                    msg += "\nJSON" + str(e)
                    raise IOError(
                        f"Unkown configuration file format. "
                        f"Expected a JSON, YAML or TOML file. "
                        f"Errors: {e}")
        
        return cls(**config)

    def __init__(self, schema=None, **kwargs):
        super().__init__(**kwargs)
        self._schema = None
        if schema is not None:
            self._schema = ConfigSchema(**schema)
    
    def __repr__(self):
        return "Config" + super(_BaseConfig, self).__repr__()

    @property
    def schema(self):
        return self._schema

    def to_disk(self, config_path, format="yaml"):

        data = self.data
        if self.schema is not None:
            data["schema"] = self.schema
        
        if format not in _FORMATTERS:
            logger.warn("Unkown format: {format}. Falling back on YAML.")
            format = "yaml"
        
        formatter = _FORMATTERS[format]

        with open(config_path, "w+") as fp:
            if format == "yaml":
                yaml.dump(dict(**data), fp, yaml.Dumper)
            else:
                formatter.dump(dict(**data), fp)


default_config = Config.from_file(Path(__file__).absolute().parent / "store" / "default.config.yml")


if __name__ == "__main__":
    import pickle
    
    c = pickle.loads(pickle.dumps(default_config))

    dd = set(default_config.schema.keys())
    cc = set(c.schema.keys())
    assert dd == cc

    dd = set(default_config.data.keys())
    cc = set(c.data.keys())
    assert dd == cc