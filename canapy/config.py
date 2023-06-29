# Author: Nathan Trouvain at 27/06/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import yaml
import random
import math


class Config(dict):
    @classmethod
    def from_file(cls, config_path):
        with open(config_path, "r") as fp:
            config = yaml.load(fp)

        return cls(**config)

    def __setattr__(self, attr, value):
        self[attr] = value

    def __getattr__(self, attr):
        try:
            if attr in ["syn", "nsyn"]:
                return Config(self[attr])
            if attr == "seed":
                if self["seed"] is None:
                    self["seed"] = random.randint(1, 9999)
            return self[attr]
        except KeyError:
            raise AttributeError(attr)

    def __repr__(self):
        return "Config " + super(Config, self).__repr__()

    def __str__(self):
        return "Config " + super(Config, self).__str__()

    def get(self, *items):
        if len(items) == 1:
            return super(Config, self).get(items[0])
        return tuple(super(Config, self).get(it) for it in items)

    def as_frames(self, attr):
        return round(self[attr] * self["sampling_rate"])

    def as_fftwindow(self, attr):
        if hasattr(self, "strict_window") and self["strict_window"] is True:
            return round(self[attr] * self["sampling_rate"])
        else:
            power = math.log(self[attr] * self["sampling_rate"]) / math.log(2)
            return 2 ** round(power)

    def as_duration(self, attr):
        return self[attr] / self["sampling_rate"]

    def duration(self, frames):
        """
        Compute the duration in seconds of a number of timesteps
        """
        return frames / self["sampling_rate"]

    def audio_steps(self, duration):
        """
        Compute the number of discreet timestep contained in a time in seconds
        """
        return round(duration * self["sampling_rate"])

    def frames(self, duration):
        """
        Compute the number of MFCC windows contained in a time in seconds.
        """
        return round(
            round(duration * self["sampling_rate"]) / self.as_fftwindow("hop_length")
        )

    def to_duration(self, mfcc_frames):
        return mfcc_frames * self.as_fftwindow("hop_length") / self["sampling_rate"]


default_config = Config()
