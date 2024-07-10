# Author: Nathan Trouvain at 18/07/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
"""Python tool for canary vocalization dataset correction and
Reservoir Computing model training.

After installation, launch the dashboard using the command:
```
canapy dash [--port INTEGER]
```

You may also annotate from the command-line using:
```
canapy annotate [ARGUMENTS]
```
"""
import click
import panel as pn

from canapy.cli import annotate

from .app import CanapyDashboard

@click.group(
    name="canapy",
    help="Python tool for birdsong "
         "vocalization dataset correction and "
         "Reservoir Computing annotate models training.",
)
def cli():
    pass


@click.command("dash", help="Launch canapy dashboard.")
@click.option(
    "-p",
    "--port",
    default=9321,
    help="Port use by the Bokeh server. By default, 9321.",
)
def display_dashboard(port):
    pn.extension()
    dashboard = CanapyDashboard(port=port)
    dashboard.show()
    return 0

cli.add_command(display_dashboard)

# TODO: (for Nathan) Clearly separate canapy CLI from dashboard
# TODO: Even better: allow to annotate from inside the dashboard
cli.add_command(annotate)
