import logging

from string import punctuation

import param
import panel as pn

from . import View


pn.extension('floatpanel')

logger = logging.getLogger("canapy-dashboard")


BASE_TYPE_MAPPING = {
    "str": "String",
    "bool": "Boolean",
    "number": "Number",
    "int": "Integer",
    "array": "List"
}

SPECIAL_TYPE_MAPPING = {
    "str-enum": "Selector",
    "array-enum": "ListSelector",
    "str-path": "Path",
    "str-filename": "Filename",
    "str-foldername": "Foldername",
    "int-magnitude": "Magnitude",
    "array-number": "NumericTuple",
}


def settings_layout(schema, config, name="", main_layout=None, level=0):

    if main_layout is None:
        main_layout = pn.Column(sizing_mode='stretch_width')

    if name is None:
        name = schema.get("title", "settings")

    if "properties" in schema:
        header_level = "#" * max(level + 1, 4)
        title = f"{header_level}{name.capitalize()}"
        n_cols = min(3 - level, 1)

        main_layout.append(pn.pane.Markdown(title))

        properties = [p for p in schema.properties.keys() if "properties" not in schema.properties[p]]
        if len(properties) > 0:
            settings_factory(properties, schema, parent_name=name)

        sub_params = [p for p in schema.properties.keys() if p not in properties]

        if len(sub_params) > 0 :
            sub_layout = pn.GridBox(n_cols=n_cols, sizing_mode='stretch_width')
            for name in sub_params:
                sub_settings = settings_layout(schema.properties[name], config, name, sub_layout, level+1)
                sub_layout.append(sub_settings)
            main_layout.append(sub_layout)

    return main_layout


def settings_factory(props, schema, parent_name):
    """Build settings parameter objects based on configuration file or,
    preferentiably, configuration JSONschema-like dict."""

    params = {}
    callbacks = {}

    for name in props:
        scheme = schema[name]
        label = name.split("__")[-1]
        doc = scheme["description"]
        default = scheme["default"]
        ptype = scheme["type"]

        param_cls = getattr(param, BASE_TYPE_MAPPING.get(ptype, "Parameter"))

        params[name] = param_cls(default=default, doc=doc, label=label)

        callbacks[name] = ...

        #TODO: More elaborate type mapping.
        #TODO: Include other annotations.

    # Dark magic: param.Parameterized inspect class attributes
    # at metaclass level. We hence need to pass parameters as
    # metaclass parameters using this type() secret spell.
    cls_name = parent_name.translate(str.maketrans(punctuation + " ", "_"*(len(punctuation) + 1)))
    Settings = type(f"Settings_{cls_name}", (param.Parameterized,), params)

    return Settings()


class SettingsControl:
    def __init__(self, widget, callback):
        ...


class SettingsView(View):
    """Display configuration parameters and dashboard settings."""

    def __init__(self, parent):
        super().__init__(parent)

        schema = self.controler.config.schema
        settings = settings_factory(schema)
        self.layout = pn.panel(settings.param, loading_indicator=True)
        #TODO: create interactions
