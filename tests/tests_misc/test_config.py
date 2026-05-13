# Author: Nathan Trouvain at 04/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: BSD-3-Clause
# Copyright: Nathan Trouvain
from config import Config

config_stub = Config(a=1, b=[1, "a", 2.0], c=dict(e="fff"))


def test_load_config(tmp_path):
    p = str(tmp_path / "config.toml")
    config_stub.to_disk(p)
    c = Config.from_file(p)
    assert c.a == 1


def test_save_config(tmp_path):
    p = str(tmp_path / "config_2.toml")
    config_stub.to_disk(p)
    import os
    assert os.path.exists(p)


def test_config_getattr():
    assert config_stub.a == 1
    assert config_stub.b == [1, "a", 2.0]
    assert isinstance(config_stub.c, Config)
    assert config_stub.c.e == "fff"

    assert config_stub["a"] == 1


def test_config_setattr():
    c = Config()
    c.a = 1
    assert c.a == 1

    c.b = dict(a=1, b=2)
    assert isinstance(c.b, Config)


def test_load_config_preserves_list_and_nested(tmp_path):
    p = str(tmp_path / "config3.toml")
    config_stub.to_disk(p)
    c = Config.from_file(p)
    assert c.b == [1, "a", 2.0]
    assert isinstance(c.c, Config)
    assert c.c.e == "fff"


def test_config_solve_nested_path():
    c = Config(model=dict(syn=dict(units=100)))
    assert c.solve("model/syn/units") == 100
