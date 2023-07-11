# Author: Nathan Trouvain at 10/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import logging
from pathlib import Path

import panel as pn
import param

logger = logging.getLogger("canapy-dashboard")


def _go_up(directory):
    current_directory = Path(directory.current_directory)
    current_directory = current_directory.parent
    directory.current_directory = str(current_directory)
    directory.param.directory_list.objects = [str(d) for d in current_directory.iterdir() if
                                         d.is_dir()]


def _makedir(directory):
    current_directory = Path(directory.current_directory)
    if current_directory.exists():
        pn.state.notifications.warning('Directory already exists !', duration=2000)
    else:
        try:
            current_directory.mkdir(parents=True, exist_ok=False)
        except OSError as e:
            pn.state.notifications.error(e, duration=3000)

    directory.param.directory_list.objects = [str(d) for d in current_directory.iterdir() if
                                         d.is_dir()]


class Directory(pn.viewable.Viewer):
    current_directory = param.String(str(Path.home()))
    directory_list = param.ListSelector(objects=[str(d) for d in Path.home().iterdir() if d.is_dir()], precedence=0.3)
    go_up = param.Action(default=_go_up, precedence=0.1)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._view = pn.Column(
            pn.Row(self.param.go_up),
            self.param.current_directory,
            pn.Param(self.param.directory_list)
            )

    def __repr__(self):
        return self.current_directory

    def __panel__(self):
        return self._view

    @param.depends("current_directory", watch=True)
    def _update_directory_list(self):
        current_directory = Path(self.current_directory)
        if current_directory.exists():
            self.param.directory_list.objects = [str(d) for d in current_directory.iterdir() if d.is_dir()]
        else:
            logger.warning(f"Directory {current_directory} does not exist!")

    @param.depends("directory_list", watch=True)
    def _update_current(self):
        self.current_directory = self.directory_list[0]


class NewDirectory(Directory):
    create = param.Action(default=_makedir, precedence=0.2)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.param.current_directory.label = "New directory"

        self._view = pn.Column(
            pn.Row(self.param.go_up, self.param.create),
            self.param.current_directory,
            self.param.directory_list,
            )


class ProjectFiles(pn.viewable.Viewer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.directory = Directory()

    def __panel__(self):
        return self.directory


if __name__ == "__main__":
    pn.extension(sizing_mode="stretch_width", design="material", notifications=True)
    #App().servable()
    ProjectFiles().show()
