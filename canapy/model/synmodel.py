# Author: Nathan Trouvain at 28/06/2023 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
import logging
import numpy as np

from reservoirpy.nodes import Rerservoir, Ridge, ESN

from .base import Model
from ..transforms.synesn import SynESNTransform

logger = logging.getLogger("canapy")


class SynModel(Model):

    def __init__(self, config, transform_output_directory):
        self.config = config
        self.transform = SynESNTransform()
        self.transform_output_directory = transform_output_directory
        self.rpy_model = self.initialize()

    def initialize(self):
        scalings = []
        if "mfcc" in self.input_list:
            iss = np.ones((self.input_dim,)) * self.config.iss
            scalings.append(iss)
        if "delta" in self.input_list:
            isd = np.ones((self.input_dim,)) * self.config.isd
            scalings.append(isd)
        if "delta2" in self.input_list:
            isd2 = np.ones((self.input_dim,)) * self.config.isd2
            scalings.append(isd2)

        input_scaling = np.concatenate(scalings, axis=0)
        bias_scaling = self.config.iss

        reservoir = Reservoir(
            self.config.N,
            sr=self.config.sr,
            lr=self.config.leak,
            input_scaling=input_scaling,
            bias_scaling=bias_scaling,
            W=fast_spectral_initialization,
            seed=self.seed,
            )

        readout = Ridge(ridge=1e-8)

        return ESN(reservoir=reservoir, readout=readout, workers=-1, backend="loky")

    def fit(self, corpus):

        corpus = self.transform(
            corpus, purpose="training", output_directory=self.transform_output_directory
            )

        # load data
        df = corpus.dataset
        df["seqid"] = df["sequence"].astype(str) + df["annotation"].astype(str)

        sampling_rate = self.config.transforms.audio.sampling_rate
        hop_length = self.config.transforms.audio.mfcc.hop_length

        df["onset_spec"] = np.round(df["onset_s"] * sampling_rate / hop_length)
        df["offset_spec"] = np.round(df["offset_s"] * sampling_rate / hop_length)

        n_classes = len(df["label"].unique())

        train_mfcc = []
        train_labels = []
        for seqid in df["seqid"].unique():

            seq_annots = df.query("seqid == @seqid")
            notated_audio = seq_annots.loc[0, "notated_path"]

            seq_end = df.loc["offset_spec", -1]
            mfcc = np.load(notated_audio)

            if seq_end > mfcc.shape(1):


            repeated_labels = np.zeros((seq_end, n_classes))
            for row in seq_annots.itertuples():
                y[s:e] = syll

            if (y == "").sum() != 0:
                y[y == ""] = "SIL"

            return y


        # repeat labels along time axis

        # train

        self.initialize()


    def predict(self, corpus):
        ...





class Trainer:
    def __init__(self, corpus):
        self.dataset = dataset

        input_list = []
        for key, value in self.dataset.config.items():
            if key in ["mfcc", "delta", "delta2"] and value:
                input_list.append(key)

        self.syn_esn = SynModel(
            config=self.dataset.config.syn,
            input_dim=self.dataset.config.n_mfcc,
            input_list=input_list,
            seed=self.dataset.config.seed,
            vocab=self.dataset.vocab,
        )

        self.nsyn_esn = NSynModel(
            config=self.dataset.config.nsyn,
            input_dim=self.dataset.config.n_mfcc,
            input_list=input_list,
            seed=self.dataset.config.seed,
            vocab=self.dataset.vocab,
        )



    def to_annotations(self, y):
        return np.array([self.dataset.vocab[t] for t in y.argmax(axis=1)])

    def to_oh(self, seq):
        return self.dataset.oh_encoder.transform(seq.reshape(-1, 1))

    def test(self, y_hat, y_truth):
        print("Test results :")

        y_ahat = [self.to_annotations(y) for y in y_hat]
        y_atruth = [self.to_annotations(y) for y in y_truth]
        y_oh = [self.to_oh(y) for y in y_ahat]

        report = self.report(y_oh, y_truth)
        pprint(report, width=30)

        seq_hat = [
            group(s, min_frame_nb=self.dataset.config.min_frame_nb) for s in y_ahat
        ]
        seq_truth = [
            group(s, min_frame_nb=self.dataset.config.min_frame_nb) for s in y_atruth
        ]

        levs = [lev_sim(s, t) for s, t in zip(seq_hat, seq_truth)]
        mean_lev = np.mean(levs)
        std_lev = np.std(levs)

        report["levenshtein"] = {"mean": mean_lev, "std": std_lev}

        print(f"Levenshtein similarity (mean) : {mean_lev} \u00B1 {std_lev}")

        return report

    def report(self, y_annot, y_test):
        return metrics.classification_report(
            np.vstack(y_annot),
            np.vstack(y_test),
            target_names=self.dataset.vocab,
            zero_division=0,
            output_dict=True,
        )

    def randomize_seed(self, model="all"):
        seed = random.randint(1, 9999)
        return self.reset_seed(seed, model="all")

    def reset_seed(self, seed, model="all"):
        if model in ["nsyn", "all"]:
            self.nsyn_esn.seed = seed
            self.nsyn_esn._build_esn()
        if model in ["syn", "all"]:
            self.syn_esn.seed = seed
            self.syn_esn._build_esn()

        return seed

    def save_train_annotation(
        self, directory, model, y_val, y_train, y_annots, y_test, instance=0
    ):
        np.savez(
            Path(directory, f"{instance}-{model}-train-annots"),
            **{str(k): y_val[k] for k in range(len(y_val))},
        )
        np.savez(
            Path(directory, f"{instance}-{model}-train-truths"),
            **{str(k): y_val[k] for k in range(len(y_train))},
        )
        np.savez(
            Path(directory, f"{instance}-{model}-test-annots"),
            **{str(k): y_val[k] for k in range(len(y_annots))},
        )
        np.savez(
            Path(directory, f"{instance}-{model}-test-truths"),
            **{str(k): y_val[k] for k in range(len(y_test))},
        )

    def train(
        self,
        model="all",
        instances=1,
        folds=None,
        max_songs=None,
        randomize_seed=False,
        save_models=None,
        save_tests=None,
        save_results=None,
        seeds=None,
        data=None,
    ):
        if save_models is not None:
            save_models = Path(save_models)
            if not save_models.exists():
                save_models.mkdir(parents=True)

        if save_results is not None:
            save_results = Path(save_results)
            if not save_results.exists():
                save_results.mkdir(parents=True)

        if data is None:
            if model == "all":
                (
                    (X_train, y_train),
                    (Xn_train, yn_train),
                    (X_test, y_test),
                ) = self.dataset.to_features(split=True, mode=model)
            else:
                (X_train, y_train), (X_test, y_test) = self.dataset.to_features(
                    split=True, mode=model
                )
                if model == "nsyn":
                    Xn_train, yn_train = X_train, y_train
        else:
            (X_train, y_train), (Xn_train, yn_train), (X_test, y_test) = data

        reports = {}
        if model == "all":
            for i in range(instances):
                report = {}

                if randomize_seed:
                    report["seed"] = self.randomize_seed(model)
                elif seeds is not None:
                    report["seed"] = self.reset_seed(seeds[i], model)
                else:
                    report["seed"] = self.dataset.config.seed

                print(f"{'#'*5} Syntactic training n°{i+1} {'#'*5}")
                y_annots, y_test, y_val, y_train = self.syn_esn.train(
                    (X_train, y_train, X_test, y_test), return_validation=True
                )
                report["syn"] = self.test(y_annots, y_test)
                report["syn-val"] = self.test(y_val, y_train)

                if save_results is not None:
                    self.save_train_annotation(
                        save_results,
                        "syn",
                        y_val,
                        y_train,
                        y_annots,
                        y_test,
                        instance=i,
                    )

                print(f"{'#'*5} Non syntactic training n° {i+1} {'#'*5}")
                y_annots, y_test, y_val, yn_train = self.nsyn_esn.train(
                    (Xn_train, yn_train, X_test, y_test), return_validation=True
                )
                report["nsyn"] = self.test(y_annots, y_test)
                report["nsyn-val"] = self.test(y_val, yn_train)

                if save_results is not None:
                    self.save_train_annotation(
                        save_results,
                        "nsyn",
                        y_val,
                        yn_train,
                        y_annots,
                        y_test,
                        instance=i,
                    )

                if instances > 1:
                    reports[i] = report
                else:
                    reports = report

            if save_models is not None:
                self.nsyn_esn.save(Path(save_models) / "nsyn")
                self.syn_esn.save(Path(save_models) / "syn")

        if model == "syn":
            for i in range(instances):
                report = {}

                if randomize_seed:
                    report["seed"] = self.randomize_seed(model)
                elif seeds is not None:
                    report["seed"] = self.reset_seed(seeds[i], model)
                else:
                    report["seed"] = self.dataset.config.seed

                print(f"{'#'*5} Syntactic training n°{i+1} {'#'*5}")
                y_annots, y_test, y_val, y_train = self.syn_esn.train(
                    (X_train, y_train, X_test, y_test)
                )
                report["syn"] = self.test(y_annots, y_test)
                report["syn-val"] = self.test(y_val, y_train)

                if save_results is not None:
                    self.save_train_annotation(
                        save_results,
                        "syn",
                        y_val,
                        y_train,
                        y_annots,
                        y_test,
                        instance=i,
                    )

                if instances > 1:
                    reports[i] = report
                else:
                    reports = report

            if save_models is not None:
                self.nsyn_esn.save(Path(save_models) / "syn")

        if model == "nsyn":
            for i in range(instances):
                report = {}

                if randomize_seed:
                    report["seed"] = self.randomize_seed(model)
                elif seeds is not None:
                    report["seed"] = self.reset_seed(seeds[i], model)
                else:
                    report["seed"] = self.dataset.config.seed

                print(f"{'#'*5} Non syntactic training n°{i+1} {'#'*5}")
                y_annots, y_test, y_val, y_train = self.nsyn_esn.train(
                    (Xn_train, yn_train, X_test, y_test)
                )
                report["nsyn"] = self.test(y_annots, y_test)
                report["nsyn-val"] = self.test(y_val, y_train)

                if save_results is not None:
                    self.save_train_annotation(
                        save_results,
                        "nsyn",
                        y_val,
                        y_train,
                        y_annots,
                        y_test,
                        instance=i,
                    )

                if instances > 1:
                    reports[i] = report
                else:
                    reports = report

            if save_models is not None:
                self.nsyn_esn.save(Path(save_models) / "nsyn")

        if save_models is not None:
            self.dataset.export_config(Path(save_models) / "config.json")

        if save_tests is not None:
            with Path(save_tests).open("w+") as f:
                json.dump(reports, f)

        return reports
