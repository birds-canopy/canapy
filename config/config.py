# Author: Nathan Trouvain at 27/06/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import toml
import collections.abc

from collections import UserDict


class Config(UserDict):

    @classmethod
    def from_file(cls, config_path):
        with open(config_path, "r") as fp:
            config = toml.load(fp)

        return cls(**config)

    def to_disk(self, config_path):
        with open(config_path, "w+") as fp:
            toml.dump(dict(**self.data), fp)

    def __setattr__(self, attr, value):
        if "data" not in self.__dict__:
            self.__dict__["data"] = {}
        self.data[attr] = value

    def __getattr__(self, attr):
        if attr in self.data:
            if isinstance(self.data[attr], collections.abc.Mapping):
                return Config(**self.data[attr])
            return self.data[attr]
        elif attr in self.__dict__:
            return self.__dict__[attr]
        else:
            raise AttributeError(attr)

    def __repr__(self):
        return "Config " + super(Config, self).__repr__()

# Attempt of compatibility with previous Config objects

# new_names = {"syn.N": "model.syn.units",
#              "syn.sr": "model.syn.sr",
#              "syn.leak": "model.syn.leak",
#              "syn.iss": "model.syn.iss",
#              "syn.isd": "model.syn.isd",
#              "syn.isd2": "model.syn.isd2",
#              "syn.ridge": "model.syn.ridge",
#
#              "nsyn.N": "model.nsyn.units",
#              "nsyn.sr": "model.nsyn.sr",
#              "nsyn.leak": "model.nsyn.leak",
#              "nsyn.iss": "model.nsyn.iss",
#              "nsyn.isd": "model.nsyn.isd",
#              "nsyn.isd2": "model.nsyn.isd2",
#              "nsyn.ridge": "model.nsyn.ridge",
#
#              "sampling_rate": "transforms.audio.sampling_rate",
#              "strict_window": "transforms.audio.strict_window",
#              "n_fft": "transforms.audio.n_fft",
#              "hop_length": "transforms.audio.hop_length",
#              "win_length": "transforms.audio.win_length",
#              "n_mfcc": "transforms.audio.n_mfcc",
#              "lifter": "transforms.audio.lifter",
#              "fmin": "transforms.audio.fmin",
#              "fmax": "transforms.audio.fmax",
#              "mfcc": "",  # Particular
#              "delta": "",  # Particular
#              "delta2": "",  # Particular
#              "padding": ["transforms.audio.delta.padding",
#              "transforms.audio.delta2.padding"],
#              "min_class_duration":
#              "transforms.training.balance.min_class_total_duration",
#              "min_silence_duration":
#              "transforms.training.balance.min_silence_duration",
#              # "min_analysis_window_per_sample" <no_correspondance>
#              # "min_correct_timesteps_per_sample" <no_correspondance>
#              # "min_frame_nb" <no_correspondance>
#              "keep_separate": "transforms.annots.lonely_label",
#              "test_ratio": "transforms.training.test_ratio",
#              "seed": "misc.seed"}

#
# @classmethod
# def compat(cls, old_config):
#
#     new_config = default_config
#
#     def change_attr(old_nm, new_nm):
#         old_name_list = old_nm.split('.')
#         new_name_list = new_nm.split('.')
#         setattr(get_sub_attr(new_config, new_name_list[1::-1]), new_name_list[-1],
#                 get_sub_attr(old_config, old_name_list[::-1]))
#
#     def get_sub_attr(base, attrs):
#         print(base, attrs)
#         if len(attrs) > 0:
#             next_attr = attrs.pop()
#             return get_sub_attr(base[next_attr], attrs)
#         return base
#
#     for old_name, new_name in cls.new_names.items():
#         if isinstance(new_name, list):
#             for _new_name in new_name:
#                 change_attr(old_name, _new_name)
#         elif isinstance(new_name, str):
#             change_attr(old_name, new_name)
#     new_config.transforms.audio.audio_features = [param for param in ["mfcc",
#     "delta", "delta2"] if old_config[param]]
#     return new_config
