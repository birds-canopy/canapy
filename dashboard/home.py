# Author: Nathan Trouvain at 10/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import pathlib
import time

import panel as pn
import param

HOME_PATH = str(pathlib.Path.home())


class HomeView(pn.viewable.Viewer):
    # audio_directory = param.FileSelector(path="~")
    # annots_directory = param.FileSelector(path="~")
    # # # spec_directory = param.FileSelector(path="~")
    # # # config_path = param.FileSelector(path="~")

    def __init__(self, **params):
        super().__init__(**params)

        self._open_project = pn.widgets.Button()
        self._new_project = pn.widgets.Button()

        self._view = pn.Row([self._open_project, self._new_project])

    def __panel__(self):
        return self._view


class App(pn.viewable.Viewer):
    run = param.Event(doc="Runs for click_delay seconds when clicked")
    runs = param.Integer(doc="The number of runs")
    status = param.String(default="No runs yet")

    load_delay = param.Number(default=0.5)
    run_delay = param.Number(default=0.5)

    def __init__(self, **params):
        super().__init__(**params)

        result = self._load()
        self._time = time.time()

        self._status_pane = pn.pane.Markdown(
            self.status, height=40, align="start", margin=(0, 5, 10, 5)
        )
        self._result_pane = pn.Column(result)

        button = pn.widgets.Button.from_param(self.param.run, sizing_mode="fixed")
        self._view = pn.Column(pn.Row(button, self._status_pane), self._result_pane)

    def __panel__(self):
        return self._view

    def _start_run(self):
        self.status = f"Running ..."
        self._time = time.time()

    def _stop_run(self):
        now = time.time()
        duration = round(now - self._time, 3)
        self._time = now
        self.runs += 1
        self.status = f"Finished run {self.runs} in {duration}sec"

    @param.depends("run", watch=True)
    def _run_with_status_update(self):
        self._start_run()
        self._result_pane[:] = [self._run()]
        self._stop_run()

    @param.depends("status", watch=True)
    def _update_status_pane(self):
        self._status_pane.object = self.status

    def _load(self):
        time.sleep(self.load_delay)
        return "Loaded"

    def _run(self):
        time.sleep(self.run_delay)
        return f"Result {self.runs+1}"


if __name__ == "__main__":
    pn.extension(sizing_mode="stretch_width", template="material")

    # App().servable()
    HomeView().show()
