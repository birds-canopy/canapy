# Author: Nathan Trouvain at 04/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
from canapy.config import Config

config_stub = Config(a=1, b=[1, "a", 2.0], c=dict(e="fff", g={1, 2, 3}))


def test_load_config():
    c = Config.from_file("./config.toml")
    print(c)


def test_save_config():
    c = config_stub
    c.to_disk("./config_2.toml")
    print(c)


def test_config_getattr():
    assert config_stub.a == 1
    assert config_stub.b == [1, "a", 2.0]
    assert isinstance(config_stub.c, Config)
    assert config_stub.c.e == "fff"
    assert isinstance(config_stub.c.g, set)

    assert config_stub["a"] == 1


def test_config_setattr():
    c = Config()
    c.a = 1
    assert c.a == 1

    c.b = dict(a=1, b=2)
    assert isinstance(c.b, Config)


if __name__ == "__main__":
    test_load_config()
    test_save_config()
    test_config_setattr()
    test_config_getattr()
