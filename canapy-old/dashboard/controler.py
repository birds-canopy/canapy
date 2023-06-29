import itertools
import gc
from pathlib import Path

import pandas as pd
import numpy as np
import librosa as lbr

from sklearn import metrics
from tqdm import tqdm
from joblib import Parallel, delayed
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.models import LogColorMapper, LogTicker
from bokeh.models import ColorBar
from matplotlib.figure import Figure

from .. import Dataset
from .. import Trainer
from .. import Annotator
from . import del_temp, create_memmap, close_memmap


class Controler(object):
    def __init__(self, data, output, dash, audioformat, rate):
        self.dataset = Dataset(data, audioformat=audioformat, rate=rate)
        self.config = self.dataset.config
        self.trainer = Trainer(self.dataset)
        self.annotator = Annotator(
            [self.trainer.syn_esn, self.trainer.nsyn_esn], dataset=self.dataset
        )
        self.dash = dash
        self.new_corrections = dict()

        self.initialize_output(output)
        self.initialize_annots()

        self.iter = 0
        self.next_iter()

        self.step = "train"

    def stop_app(self):
        del_temp()
        self.dash.stop()
        print("Server shutdown.")

    def initialize_output(self, output):
        self.output = make_directory(Path(output))

        self.checkpoint_dir = make_directory(self.output / "checkpoints")
        self.models_out = make_directory(self.output / "models")

    def initialize_annots(self):
        self.dataset.annotations = {}

    def initialize_models(self):
        self.trainer = Trainer(self.dataset)
        self.annotator = Annotator(
            [self.trainer.syn_esn, self.trainer.nsyn_esn], dataset=self.dataset
        )

    def next_iter(self):
        self.iter += 1
        print(f"Current training iteration : {self.iter}")

        self.new_corrections = {"syll": {}, "sample": {}}

        self.curr_ckpt = make_directory(self.checkpoint_dir / str(self.iter))
        self.curr_models = make_directory(self.curr_ckpt / "models")

        print("Creating empty annotations...")
        self.initialize_annots()
        print("Initializing models...")
        self.initialize_models()
        print("Done.")

    def checkpoint(self):
        self.dataset.checkpoint(self.curr_ckpt)
        for model in self.annotator.models:
            model.save(self.curr_models / model.name)

    def next_step(self, export=False):

        if self.step == "train":
            self.step = "eval"

            print("Fetching metrics...")
            self.get_metrics()
            print('Setting current dashboard to "eval".')

        elif self.step == "eval" and not export:
            self.step = "train"

            print("Updating dataset...")
            self.dataset.update(iteration=self.iter, corrections=self.new_corrections)
            print("Saving dataset chekpoint...")
            self.checkpoint()
            print('Setting current dashboard to "train".')
            self.next_iter()

        elif export:
            self.step = "export"

            print("Updating dataset...")
            self.dataset.update(iteration=self.iter, corrections=self.new_corrections)
            print("Saving dataset chekpoint...")
            self.dataset.checkpoint(self.output)
            print('Setting current dashboard to "export".')
            self.next_iter()

        # del_temp()
        # print("All temporary files deleted.")
        # print("Switching panel...")
        self.dash.switch_panel()
        print("Done.")

    def train_syn(self):
        self.trainer.train(model="syn", save_models=self.curr_models)

    def train_syn_export(self):
        features = self.dataset.to_features(mode="syn")
        self.trainer.syn_esn.train(features, test=False)
        self.trainer.syn_esn.save(self.models_out / "syn")

    def train_nsyn(self):
        self.trainer.train(model="nsyn", save_models=self.curr_models)

    def train_nsyn_export(self):
        features = self.dataset.to_features(mode="nsyn")
        self.trainer.nsyn_esn.train(features, test=False)
        self.trainer.nsyn_esn.save(self.models_out / "nsyn")

    def annotate_syn(self):
        syn, truth, vect = self.annotator.run(
            model="syn", vectors=True, return_truths=True
        )
        self.dataset.annotations["syn"] = create_memmap(syn, "syn_annots")
        self.dataset.annotations["truth"] = create_memmap(truth, "truth_annots")
        self.syn_annots_vectors = create_memmap(vect, "syn_vect")

        del syn
        del truth
        del vect
        gc.collect()

    def annotate_nsyn(self):
        nsyn, vect = self.annotator.run(model="nsyn", vectors=True)
        self.dataset.annotations["nsyn"] = create_memmap(nsyn, "nsyn_annots")
        self.nsyn_annots_vectors = create_memmap(vect, "nsyn_vect")

        del nsyn
        del vect
        gc.collect()

    def annotate_ensemble(self):
        ens = self.annotator.run(
            models_vectors=[self.syn_annots_vectors, self.nsyn_annots_vectors],
            model="ensemble",
        )
        self.dataset.annotations["ensemble"] = create_memmap(ens, "ens_annots")

        del ens
        del self.syn_annots_vectors
        close_memmap("syn_vect")
        del self.nsyn_annots_vectors
        close_memmap("nsyn_vect")

        gc.collect()

    def get_metrics(self):
        annots = self.dataset.annotations
        flat_annots = {k: np.concatenate([*s.values()]) for k, s in annots.items()}

        truth = flat_annots["truth"]
        labels = self.dataset.vocab
        cms = {}
        reports = {}
        for model, values in tqdm(
            flat_annots.items(), "Computing metrics for annotations"
        ):

            if model != "truth":
                cms[model] = metrics.confusion_matrix(
                    truth, values, labels=labels, normalize="true"
                )

                reports[model] = metrics.classification_report(
                    truth, values, digits=3, output_dict=True, zero_division=0
                )
        self.cms = cms
        self.reports = reports

        self.fetch_misclassified_samples()
        self.misclassified_counts_plot()

    def fetch_misclassified_samples(self):

        annots = self.dataset.annotations
        df = self.dataset.df
        bad_ones = {}
        for rep in tqdm(
            df[df["syll"] != "SIL"].itertuples(), "Fetching all misclassified samples"
        ):
            song = rep.wave
            start_y = self.config.frames(rep.start)
            end_y = self.config.frames(rep.end)

            scores = []
            detected_labels = []
            for m in annots.keys():
                if m != "truth":
                    preds = np.array(annots[m][song][start_y:end_y])
                    truth = np.array(annots["truth"][song][start_y:end_y])
                    scores.append(np.sum(preds == truth) / len(truth))
                    label_freqs = np.unique(preds, return_counts=True)
                    if len(label_freqs[1]) != 0:
                        detected_labels.append(label_freqs[0][label_freqs[1].argmax()])

            if (
                np.sum(np.array(scores) > self.config.min_correct_timesteps_per_sample)
                == 0
            ):
                bad_ones[rep.Index] = detected_labels

        self.misclass_indexes = bad_ones
        self.misclass_df = self.dataset.df.iloc[list(self.misclass_indexes.keys())]
        self.misclassified_counts_plot()

    def misclassified_counts_plot(self):

        s = pd.DataFrame(self.misclass_df.groupby("syll")["syll"].count())
        s.columns = ["counts"]
        s.sort_values(by=["counts"], ascending=False, inplace=True)

        data = ColumnDataSource({"x": s.index.values.tolist(), "top": s.counts})

        p = figure(
            title="Misclassified samples count",
            tools="hover,save",
            x_range=s.index.values.tolist(),
            plot_height=350,
            tooltips=[("class", "@x"), ("", "@top")],
        )

        p.xaxis.major_label_orientation = np.pi / 3

        p.vbar(x="x", top="top", width=0.9, source=data)

        self.misclass_counts = s
        self.misclass_plot = p
        self.misclass_labels = self.misclass_counts.index.values.tolist()

    def confusion_matrix(self, model):

        labels = self.dataset.vocab
        cm = self.cms[model]

        data = ColumnDataSource(
            {
                "xsyll": [u for u, v in itertools.product(labels, labels)],
                "ysyll": [v for u, v in itertools.product(labels, labels)],
                "val": cm.T.ravel(),
            }
        )

        colormap = LogColorMapper(palette="Magma256", low=1e-3, high=1.0)

        p = figure(
            title=f"{model} confusion matrix",
            x_axis_location="above",
            tools="hover,save",
            x_range=labels,
            y_range=list(reversed(labels)),
            tooltips=[("class", "@ysyll, @xsyll"), ("", "@val")],
        )

        p.plot_width = 800
        p.plot_height = 800
        p.grid.grid_line_color = None
        p.axis.axis_line_color = None
        p.axis.major_tick_line_color = None
        p.axis.major_label_text_font_size = "12px"
        p.axis.major_label_standoff = 0
        p.xaxis.major_label_orientation = np.pi / 3

        p.rect(
            "xsyll",
            "ysyll",
            1.0,
            1.0,
            source=data,
            line_color="gray",
            color={"field": "val", "transform": colormap},
            hover_line_color="white",
        )

        color_bar = ColorBar(
            color_mapper=colormap,
            ticker=LogTicker(),
            label_standoff=12,
            border_line_color=None,
            location=(0, 0),
        )

        p.add_layout(color_bar, "right")

        return p

    def load_sample(self, sample):
        wave, start, end = sample.wave, sample.start, sample.end

        wave_file = Path(self.dataset.audiodir, wave)
        if wave_file.suffix == ".npy":
            y = np.load(wave_file)
            sr = self.config.sampling_rate
        else:
            y, sr = lbr.load(wave_file, sr=self.config.sampling_rate)

        hop_length = self.config.as_fftwindow("hop_length")

        max_length = 3 * sr
        d = end * sr - start * sr
        delta = max(0, (max_length - d)) / 2

        s_delta = start * sr - delta
        e_delta = end * sr + delta

        if s_delta < 0:
            e_delta += -s_delta
        elif e_delta > len(y):
            s_delta -= e_delta - len(y)

        s_delta = max(0, round(s_delta))
        e_delta = min(len(y), round(e_delta))

        full = y[s_delta:e_delta]
        spec = lbr.feature.melspectrogram(
            y=full,
            sr=sr,
            n_fft=self.config.as_fftwindow("n_fft"),
            win_length=self.config.as_fftwindow("win_length"),
            hop_length=self.config.as_fftwindow("hop_length"),
            fmin=self.config.fmin,
            fmax=self.config.fmax,
        )
        spec = lbr.power_to_db(spec)

        fig = Figure(figsize=(3, 3))
        ax = fig.subplots()
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        ax.imshow(spec, origin="lower", cmap="magma")
        ax.axvline(
            round((start * sr - s_delta) / hop_length),
            color="white",
            linestyle="--",
            marker=">",
            markevery=0.01,
        )
        ax.axvline(
            spec.shape[1] - round((e_delta - end * sr) / hop_length),
            color="white",
            linestyle="--",
            marker="<",
            markevery=0.01,
        )

        sample_only = y[max(0, round(start * sr)) : min(len(y), round(end * sr))]
        return (
            fig,
            np.int16(full * np.iinfo(np.int16).max),
            np.int16(sample_only * np.iinfo(np.int16).max),
            sr,
        )

    def load_repertoire(self, selected_samples):
        with Parallel() as parallel:
            specs = parallel(
                delayed(self.load_sample)(s) for s in selected_samples.itertuples()
            )
        return specs


def make_directory(directory):
    if not directory.exists():
        directory.mkdir(parents=True)
    return directory
