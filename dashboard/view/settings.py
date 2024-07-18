import logging
import os
import yaml

# import param
import panel as pn

from . import View

pn.extension('floatpanel')

logger = logging.getLogger("canapy-dashboard")


# BASE_TYPE_MAPPING = {
#     "str": "String",
#     "bool": "Boolean",
#     "number": "Number",
#     "int": "Integer",
#     "array": "List"
# }

# SPECIAL_TYPE_MAPPING = {
#     "str-enum": "Selector",
#     "array-enum": "ListSelector",
#     "str-path": "Path",
#     "str-filename": "Filename",
#     "str-foldername": "Foldername",
#     "int-magnitude": "Magnitude",
#     "array-number": "NumericTuple",
# }


# def settings_factory(schema):
#     """Build settings parameter objects based on configuration file or,
#     preferentiably, configuration JSONschema-like dict."""

#     params = {}
#     for name, scheme in schema.flatten().items():
#         label = name.split("__")[-1]
#         doc = scheme["description"]
#         default = scheme["default"]
#         ptype = scheme["type"]

#         param_cls = getattr(param, BASE_TYPE_MAPPING.get(ptype, "Parameter"))

#         params[name] = param_cls(default=default, doc=doc, label=label)

#         #TODO: More elaborate type mapping.
#         #TODO: Include other annotations.

#     # Dark magic: param.Parameterized inspect class attributes
#     # at metaclass level. We hence need to pass parameters as
#     # metaclass parameters using this type() secret spell.
#     Settings = type("Settings", (param.Parameterized,), params)

#     return Settings()


class SettingsView(View):
    """Display configuration parameters and dashboard settings."""

    def __init__(self, parent):
        super().__init__(parent)

        # schema = self.controler.config.schema
        # settings = settings_factory(schema)
        # self.layout = pn.panel(settings.param, loading_indicator=True)
        # TODO: create interactions

        pn.config.raw_css.append("""
        .GreyCard {
            border-radius: 15px;
            background-color: #F7F7F7;
            padding: 10px;
        }
        """)

        self.load_settings = pn.widgets.Button(name="Load settings", button_type="primary", margin=(0, 0, 0, 470))
        self.load_settings.on_click(self.on_click_load_settings)

        self.config_selector = pn.widgets.FileInput(accept='.yml', description="default.config.yml", align='center')

        self.load_settings_panel_validate = pn.widgets.Button(name="Validate", button_type="success", align="center")
        self.load_settings_panel_validate.on_click(self.on_click_validate_load_settings)

        self.load_settings_panel = pn.layout.FloatPanel(
            pn.Row(self.config_selector, self.load_settings_panel_validate, align='center'),
            name='Select a config file (.yml)',
            config={"headerControls": {"close": "remove",
                                       "maximize": "remove"}},
            width=425, height=75,
            contained=False, position='center',
        )

        self.reset_settings = pn.widgets.Button(name="Reset", button_type="primary", margin=(0, 0, 0, 25))
        self.reset_settings.on_click(self.on_click_reset_settings)

        self.switch_mode = pn.widgets.Switch(name='Expert mode', align='center', height=5)
        self.switch_mode.param.watch(self.switch_action, 'value')

        self.save_settings = pn.widgets.Button(name="Save settings", button_type="success", align='center')
        self.save_settings.on_click(self.on_click_save_settings)

        self.apply_settings = pn.widgets.Button(name="Apply", button_type="success", align='center')
        self.apply_settings.on_click(self.on_click_apply_settings)

        self.notification_settings = pn.pane.Alert(alert_type="success", visible=False, width=200,
                                                   align='center')

        # Sampling rate
        self.sampling_rate = pn.widgets.IntInput(name='Sampling Rate', value=44100, default_value=44100, step=500,
                                                 width=255)
        self.tip_sampling_rate = pn.widgets.TooltipIcon(value="Common sampling rates (Hertz) : 44100, 48000, 96000...")

        self.output_directory = pn.widgets.TextInput(name='Output directory', value=os.getcwd() + "\output", width=255)
        self.tip_output_directory = pn.widgets.TooltipIcon(
            value="Specify the location where you want to save the trained model(s).")

        # Transforms audio
        self.fmin = pn.widgets.IntInput(name='Min. Frequency', value=500, default_value=500, step=10, width=255)
        self.tip_fmin = pn.widgets.TooltipIcon(
            value="Minimum frequency to be considered when extracting characteristics, in Hertz.")
        self.fmax = pn.widgets.IntInput(name='Max. Frequency', value=8000, default_value=8000, step=50, width=255)
        self.tip_fmax = pn.widgets.TooltipIcon(
            value="Maximum frequency to be considered when extracting characteristics, in Hertz.")
        self.n_fft = pn.widgets.IntInput(name='N_FFT', value=2048, default_value=2048, step=100, disabled=True)
        self.audio_feature_mfcc = pn.widgets.Checkbox(name='MFCC', value=True, default_value=True, disabled=True)
        self.audio_feature_delta = pn.widgets.Checkbox(name='delta', value=True, default_value=True, disabled=True)
        self.audio_feature_delta2 = pn.widgets.Checkbox(name='delta 2', value=True, default_value=True, disabled=True)
        self.hop_length = pn.widgets.FloatSlider(name='Hop Length', start=0, end=0.1, step=0.005, value=0.01,
                                                 default_value=0.01,
                                                 disabled=True)
        self.win_length = pn.widgets.FloatSlider(name='Win Length', start=0, end=0.1, step=0.005, value=0.02,
                                                 default_value=0.02,
                                                 disabled=True)
        self.n_mfcc = pn.widgets.IntInput(name='N_MFCC', value=13, default_value=13, step=1, disabled=True)
        self.lifter = pn.widgets.IntInput(name='Lifter', value=40, default_value=40, step=2, disabled=True)

        # Transforms annotations
        self.time_precision = pn.widgets.FloatInput(name='Time Precision', start=0, end=0.01, step=0.001, value=0.001,
                                                    default_value=0.001, width=255, disabled=False)
        self.tip_time_precision = pn.widgets.TooltipIcon(value="Time accuracy of annotations, in seconds.")

        self.min_label_duration = pn.widgets.FloatSlider(name='Min Label Duration', start=0, end=0.1, step=0.01,
                                                         value=0.02, default_value=0.02, width=255)
        self.tip_min_label_duration = pn.widgets.TooltipIcon(
            value="Minimum duration of a label, in seconds. Labels shorter than this value will be ignored or merged.")

        self.lonely_labels = pn.widgets.TextInput(name='Lonely Labels', value="cri,TRASH", default_value="cri,TRASH",
                                                  width=255)
        self.tip_lonely_labels = pn.widgets.TooltipIcon(
            value="List of labels considered isolated and which may require special treatment. Separated by ','")

        self.min_silence_gap = pn.widgets.FloatInput(name='Min Silence Gap', start=0, end=0.01, step=0.001, value=0.001,
                                                     default_value=0.001, width=255)
        self.tip_min_silence_gap = pn.widgets.TooltipIcon(
            value="Minimum silence interval, in seconds, to separate two audio segments.")

        self.silence_tag = pn.widgets.TextInput(name='Silence label', value="SIL", default_value="SIL", width=255)
        self.tip_silence_tag = pn.widgets.TooltipIcon(value="Tag used to mark silence segments.")

        # Transforms audio delta and delta2
        self.delta_padding = pn.widgets.TextInput(name='Delta padding', value="wrap", default_value="wrap",
                                                  disabled=True)
        self.delta2_padding = pn.widgets.TextInput(name='Delta 2 padding', value="wrap", default_value="wrap",
                                                   disabled=True)

        # Transforms training
        self.max_sequences = pn.widgets.IntInput(name='Max files', value=-1, default_value=-1, step=50, width=255)
        self.tip_max_sequences = pn.widgets.TooltipIcon(
            value="Maximum number of files for training. -1 mean that there is no limit.")

        self.test_ratio = pn.widgets.FloatSlider(name='Test ratio', start=0, end=1, step=0.1, value=0.2,
                                                 default_value=0.2, width=255, disabled=False)
        self.tip_test_ratio = pn.widgets.TooltipIcon(
            value="The test ratio indicates the fraction of the total data reserved for testing to evaluate the model's performance.")
        self.min_class_total_duration = pn.widgets.FloatInput(name="Min class duration", value=2, default_value=2,
                                                              disabled=True)
        self.min_silence_duration = pn.widgets.FloatInput(name="Min silence duration", value=0.2, default_value=0.2,
                                                          disabled=True)
        self.noise_std = pn.widgets.FloatInput(name="noise", value=0.01, default_value=0.01, disabled=True)

        # Syn model
        self.syn_units = pn.widgets.IntSlider(name='Units', start=250, end=2000, step=250, value=1000,
                                              default_value=1000, disabled=True, align='center')
        self.syn_sr = pn.widgets.FloatSlider(name='Spectral radius', start=0, end=1, step=0.1, value=0.4,
                                             default_value=0.4,
                                             disabled=True, align='center')
        self.syn_leak = pn.widgets.FloatSlider(name='Leakage', start=0, end=1, step=0.05, value=0.1, default_value=0.1,
                                               disabled=True, align='center')
        self.syn_iss = pn.widgets.FloatInput(name='MFCC input scaling', value=0.0005, default_value=0.0005,
                                             disabled=True, align='center')
        self.syn_isd = pn.widgets.FloatInput(name='MFCC derivatives input scaling', value=0.02, default_value=0.02,
                                             disabled=True, align='center')
        self.syn_isd2 = pn.widgets.FloatInput(name='MFCC second derivatives input scaling', value=0.002,
                                              default_value=0.002,
                                              disabled=True, align='center')
        self.syn_ridge = pn.widgets.FloatInput(name='Ridge regularisation', value=1e-8, default_value=1e-8,
                                               disabled=True, align='center', step=1e-8)
        self.syn_backend = pn.widgets.Select(name='Backend', options=['multiprocessing', 'choice2', 'choice3'],
                                             value='multiprocessing', default_value='multiprocessing',
                                             disabled=True, align='center')
        self.syn_workers = pn.widgets.IntInput(name='Workers for the backend', value=-1, default_value=-1,
                                               disabled=True, align='center')

        # Nsyn model
        self.nsyn_units = pn.widgets.IntSlider(name='Units', start=250, end=2000, step=250, value=1000,
                                               default_value=1000, disabled=True, align='center')
        self.nsyn_sr = pn.widgets.FloatSlider(name='Spectral radius', start=0, end=1, step=0.1, value=0.4,
                                              default_value=0.4,
                                              disabled=True, align='center')
        self.nsyn_leak = pn.widgets.FloatSlider(name='Leakage', start=0, end=1, step=0.05, value=0.1, default_value=0.1,
                                                disabled=True, align='center')
        self.nsyn_iss = pn.widgets.FloatInput(name='MFCC input scaling', value=0.0005, default_value=0.0005,
                                              disabled=True, align='center')
        self.nsyn_isd = pn.widgets.FloatInput(name='MFCC derivatives input scaling', value=0.02, default_value=0.02,
                                              disabled=True, align='center')
        self.nsyn_isd2 = pn.widgets.FloatInput(name='MFCC second derivatives input scaling', value=0.002,
                                               default_value=0.002,
                                               disabled=True, align='center')
        self.nsyn_ridge = pn.widgets.FloatInput(name='Ridge regularisation', value=1e-8, default_value=1e-8,
                                                disabled=True, align='center', step=1e-8)
        self.nsyn_backend = pn.widgets.Select(name='Backend', options=['multiprocessing', 'choice2', 'choice3'],
                                              value='multiprocessing', default_value='multiprocessing',
                                              disabled=True, align='center')
        self.nsyn_workers = pn.widgets.IntInput(name='Workers for the backend', value=-1, default_value=-1,
                                                disabled=True, align='center')

        # Correction
        self.min_segment_proportion_agreement = pn.widgets.FloatInput(name='Minimum proportion of agreement', start=0,
                                                                      end=1, step=0.01,
                                                                      value=0.66, default_value=0.66, width=255)
        self.tip_min_segment_proportion_agreement = pn.widgets.TooltipIcon(
            value="Minimum proportion of agreement to consider a segment as valid when correcting annotations.")

        self.advanced_settings_widgets = [
            self.n_fft, self.audio_feature_mfcc, self.audio_feature_delta, self.audio_feature_delta2,
            self.hop_length, self.win_length, self.n_mfcc, self.lifter,
            self.delta_padding, self.delta2_padding,
            self.min_class_total_duration, self.min_silence_duration, self.noise_std,
            self.syn_units, self.syn_sr, self.syn_leak, self.syn_iss, self.syn_isd, self.syn_isd2, self.syn_ridge,
            self.syn_backend, self.syn_workers,
            self.nsyn_units, self.nsyn_sr, self.nsyn_leak, self.nsyn_iss, self.nsyn_isd, self.nsyn_isd2,
            self.nsyn_ridge, self.nsyn_backend, self.nsyn_workers,
        ]

        self.settings_widgets = [
            self.sampling_rate, self.output_directory,
            self.fmin, self.fmax, self.n_fft,
            self.audio_feature_mfcc, self.audio_feature_delta, self.audio_feature_delta2,
            self.hop_length, self.win_length, self.n_mfcc, self.lifter,
            self.time_precision, self.min_label_duration, self.lonely_labels,
            self.min_silence_gap, self.silence_tag,
            self.delta_padding, self.delta2_padding,
            self.max_sequences, self.test_ratio,
            self.min_class_total_duration, self.min_silence_duration, self.noise_std,
            self.syn_units, self.syn_sr, self.syn_leak, self.syn_iss, self.syn_isd, self.syn_isd2, self.syn_ridge,
            self.syn_backend, self.syn_workers,
            self.nsyn_units, self.nsyn_sr, self.nsyn_leak, self.nsyn_iss, self.nsyn_isd, self.nsyn_isd2,
            self.nsyn_ridge, self.nsyn_backend, self.nsyn_workers,
            self.min_segment_proportion_agreement,
        ]

        self.layout = pn.Accordion(('Settings', pn.Column(
            pn.Row(
                pn.pane.Markdown(f"""## Settings :"""),
                self.load_settings,
                self.reset_settings,
                pn.Column(
                    self.switch_mode,
                    pn.pane.Markdown(f"""### **Expert mode**""", align='center'),
                    margin=(0, 0, 0, 25)
                )
            ),
            pn.Row(
                pn.Column(
                    pn.pane.Markdown(f"""### Audio files :""", align='center'),
                    pn.Row(
                        self.sampling_rate,
                        pn.widgets.TooltipIcon(value="Common sampling rates (Hertz) : 44100, 48000, 96000..."),
                    ),
                    pn.Row(
                        self.fmin, self.tip_fmin
                    ),
                    pn.Row(
                        self.fmax, self.tip_fmax
                    ),
                    pn.pane.Markdown(f"""### Correction :""", align='center'),
                    pn.Row(
                        self.min_segment_proportion_agreement,
                        self.tip_min_segment_proportion_agreement
                    ),
                    sizing_mode="stretch_width"
                ),
                pn.Column(
                    pn.pane.Markdown(f"""### Annotation :""", align='center'),
                    pn.Row(
                        self.time_precision,
                        self.tip_time_precision
                    ),
                    pn.Row(
                        self.min_label_duration,
                        self.tip_min_label_duration
                    ),
                    pn.Row(
                        self.lonely_labels,
                        self.tip_lonely_labels
                    ),
                    pn.Row(
                        self.min_silence_gap,
                        self.tip_min_silence_gap
                    ),
                    pn.Row(
                        self.silence_tag,
                        self.tip_silence_tag
                    ),
                    sizing_mode="stretch_width"
                ),
                pn.Column(
                    pn.pane.Markdown(f"""### Training data :""", align='center'),
                    pn.Row(
                        self.max_sequences,
                        self.tip_max_sequences
                    ),
                    pn.Row(
                        self.test_ratio,
                        self.tip_test_ratio
                    ),
                    pn.pane.Markdown(f"""### Output directory :""", align='center'),
                    pn.Row(self.output_directory, self.tip_output_directory),
                    sizing_mode="stretch_width"
                ),
            ),
            pn.pane.Markdown(f"""## Advanced settings :"""),
            pn.Row(
                pn.Column(
                    pn.pane.Markdown(f"""### Audio transform :""", align='center'),
                    self.n_fft,
                    pn.pane.Markdown(f"""Audio Features :""", height=25),
                    pn.Row(self.audio_feature_mfcc, self.audio_feature_delta, self.audio_feature_delta2
                           ),
                    self.hop_length,
                    self.win_length,
                    self.n_mfcc,
                    self.lifter,
                    sizing_mode="stretch_width"
                ),
                pn.Column(
                    pn.pane.Markdown(f"""### Advanced audio transform :""", align='center'),
                    self.delta_padding,
                    self.delta2_padding,
                    sizing_mode="stretch_width"
                ),
                pn.Column(
                    pn.pane.Markdown(f"""### Balance :""", align='center'),
                    self.min_class_total_duration,
                    self.min_silence_duration,
                    self.noise_std,
                    sizing_mode="stretch_width"
                )
            ),
            pn.Row(
                pn.Column(
                    pn.pane.Markdown(f"""### Syntactic model parameters :""", align='center'),
                    self.syn_units,
                    self.syn_sr,
                    self.syn_leak,
                    self.syn_iss,
                    self.syn_isd,
                    self.syn_isd2,
                    self.syn_ridge,
                    self.syn_backend,
                    self.syn_workers,
                    sizing_mode="stretch_width"
                ),
                pn.Column(
                    pn.pane.Markdown(f"""### Non-syntactic model parameters :""", align='center'),
                    self.nsyn_units,
                    self.nsyn_sr,
                    self.nsyn_leak,
                    self.nsyn_iss,
                    self.nsyn_isd,
                    self.nsyn_isd2,
                    self.nsyn_ridge,
                    self.nsyn_backend,
                    self.nsyn_workers,
                    sizing_mode="stretch_width"
                ),
                pn.Column(
                    pn.Spacer(width=330)
                )
            ),
            pn.Row(
                self.save_settings, self.apply_settings, self.notification_settings
            ),
            width=975,
            margin=(20, 0, 0, 0)
        )), width=975)

    def on_click_reset_settings(self, event):
        for setting in self.settings_widgets:
            if setting.value and setting.default_value:
                setting.value = setting.default_value
            if setting.value == 0 and setting.default_value:
                setting.value = setting.default_value

    def on_click_load_settings(self, event):
        self.layout.append(self.load_settings_panel)
        self.load_settings_panel.visible = True

    def on_click_validate_load_settings(self, event):
        config = self.config_selector.value
        #TODO : Transfer the config uploaded to the controler
        self.layout.remove(self.load_settings_panel)

    def on_click_save_settings(self, event):
        #TODO : Save the config as a .yaml file

        # with open('config.yml', 'w') as file:
        #     yaml.dump(config, file, default_flow_style=False, allow_unicode=True)

        self.notification_settings.visible = False
        self.notification_settings.object = "Settings saved !"
        self.notification_settings.visible = True

    def on_click_apply_settings(self, event):
        self.notification_settings.visible = False
        self.notification_settings.object = "Settings applied !"
        self.notification_settings.visible = True

    def switch_action(self, event):
        if event.new:
            for widget in self.advanced_settings_widgets:
                widget.disabled = False
        else:
            for widget in self.advanced_settings_widgets:
                widget.disabled = True
