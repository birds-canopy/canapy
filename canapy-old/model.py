import json
import dill as pickle
import random
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import reservoirpy as rpy
from sklearn import metrics
from joblib import delayed, Parallel
from tqdm import tqdm
from reservoirpy.mat_gen import fast_spectral_initialization
from reservoirpy.nodes import Reservoir, Ridge, ESN
from reservoirpy import activationsfunc as F

from .sequence import group, lev_sim


class NotTrainableError(Exception):
    pass


class Trainer(object):
    def __init__(self, dataset):

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


class Annotator(object):
    @staticmethod
    def load(model_dir):
        model_dir = Path(model_dir)
        models = []
        for model_file in model_dir.iterdir():
            if model_file.is_file():
                with model_file.open("rb") as f:
                    models.append(pickle.load(f))

        return models

    def __init__(self, models, dataset=None, ensemble=True):

        self.dataset = dataset

        if type(models) is str:
            self.models = self.load(models)
        else:
            self.models = models

        self.vocab = self.models[0].vocab

        if ensemble:
            self.models.append(Ensemble(self.models.copy(), vocab=self.vocab))

    def to_csv(self, annotations, directory, model_name=None, config=None):

        config = config if config is not None else self.dataset.config
        model_name = "" if model_name is None else model_name

        sr = config.sampling_rate
        hop = config.as_fftwindow("hop_length")

        songs = list(annotations.keys())

        @delayed
        def export_one(song, annotation):
            seq = group(annotation)

            durations = {"start": [], "end": [], "syll": [], "frames": []}
            onset = 0.0
            for s in seq:
                end = onset + (s[1] * (hop / sr))
                if s[0] != "SIL":
                    durations["syll"].append(s[0])
                    durations["end"].append(round(end, 3))
                    durations["start"].append(round(onset, 3))
                    durations["frames"].append(s[1])
                onset = end

            df = pd.DataFrame(durations)

            file_name = Path(directory) / Path(song)
            df.to_csv(file_name.with_suffix(".csv"), index=False)

        with Parallel(n_jobs=-1) as parallel:
            parallel(
                export_one(song, annotations[song])
                for song in tqdm(songs, f"Exporting to .csv - {model_name}")
            )

    def run(
        self,
        model="syn",
        dataset=None,
        to_group=False,
        return_truths=False,
        vectors=False,
        models_vectors=None,
        csv_directory: str = None,
    ):

        dataset = self.dataset if dataset is None else dataset

        if model == "all":
            models = self.models
            if "ensemble" in [m.name for m in models]:
                models = [m for m in models if m.name == "ensemble"]
        else:
            models = [m for m in self.models if m.name == model]

        outputs = tuple()
        for M in models:
            if M.name == "ensemble" and models_vectors is not None:
                outs = M.annotate_from_models_outputs(
                    *models_vectors, to_group=to_group
                )
                outputs = outs
                break
            else:
                outs = M.annotate(
                    dataset,
                    vectors=vectors,
                    to_group=to_group,
                    return_truths=return_truths,
                )

            if model == "all":
                if M.name == "ensemble":
                    outputs = outs
                    break
                else:
                    annots = {M.name: outs[0]}
                    outputs = (annots, *outs[1:])
            else:
                annots = {M.name: outs[0]}
                outputs = (annots, *outs[1:])

        if csv_directory is not None and not vectors:
            for model_name, annotations in outputs[0].items():
                directory = Path(csv_directory) / model_name
                if not directory.exists():
                    directory.mkdir(parents=True)

                self.to_csv(
                    annotations,
                    directory,
                    model_name=model_name,
                    config=dataset.config,
                )
        elif csv_directory and vectors:
            raise ValueError(
                "Impossible to export vectors to csv. vectors should be False."
            )

        if not hasattr(outputs, "items") and len(outputs[0]) == 1:
            outputs = (outputs[0][list(outputs[0].keys())[0]], *outputs[1:])

        return outputs


class Model(object):
    def __init__(self, config, input_dim, input_list, name=None, seed=None, vocab=None):
        self.config = config
        self.input_dim = input_dim
        self.input_list = input_list
        self.seed = seed
        self.name = name
        self.vocab = vocab

        self.esn = self._build_esn()

    def _build_esn(self):
        """
        Build an ESN instance for syn or nsyn annotation.
        """
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

    def train(self, features, return_validation=True, test=True):

        model_name = self.__class__.__name__

        if test:
            X_train, y_train, X_test, y_test = features
        else:
            X_train, y_train = features

        self.esn.fit(X_train, y_train)

        if test:
            val_outputs = None
            if return_validation is True:
                val_outputs = self.esn.run(X_train)

            outputs = self.esn.run(X_test)
            return outputs, y_test, val_outputs, y_train

    def annotate(
        self, dataset, to_group=False, vocab=None, vectors=False, return_truths=False
    ):

        features, teachers = dataset.to_features(mode="syn", return_dict=True)

        X = [f for f in features.values()]
        all_vectors = self.esn.run(X)
        vocab = self.vocab if self.vocab is not None else vocab

        if vocab is None and to_group:
            raise ValueError(
                "if group, 'vocab' should not be "
                "None. No grouping possible "
                "on float vectors."
            )
        else:
            all_sylls = [to_annotations(y, vocab) for y in all_vectors]
            if return_truths:
                teachers = {k: to_annotations(y, vocab) for k, y in teachers.items()}

        if vectors:
            all_vectors = {
                song: vect for song, vect in zip(features.keys(), all_vectors)
            }
        else:
            all_vectors = None

        if to_group:

            @delayed
            def group_one(song, sylls):
                return song, group(sylls)

            with Parallel(n_jobs=-1) as parallel:
                out = parallel(
                    group_one(song, sylls)
                    for song, sylls in tqdm(
                        zip(features.keys(), all_sylls),
                        "Grouping annotations",
                        total=len(all_sylls),
                    )
                )
            out = dict(out)
        else:
            out = {song: out for song, out in zip(features.keys(), all_sylls)}

        if return_truths:
            outs = {song: sylls for song, sylls in zip(features.keys(), all_sylls)}
            return outs, teachers, all_vectors

        return out, all_vectors

    def annotate_features(self, X):
        all_vectors = self.esn.run(X)
        return all_vectors

    def save(self, filename):

        with Path(filename).open("wb+") as f:
            pickle.dump(self, f)


class SynModel(Model):
    def __init__(self, *args, **kwargs):

        super(SynModel, self).__init__(*args, **kwargs)

        if self.name is None:
            self.name = "syn"


class NSynModel(Model):
    def __init__(self, *args, **kwargs):

        super(NSynModel, self).__init__(*args, **kwargs)

        if self.name is None:
            self.name = "nsyn"


class Ensemble(Model):
    def __init__(self, models, vocab=None):
        self.name = "ensemble"
        self.models = models
        self.vocab = vocab

    def _build_matrices(self, *args, **kwargs):
        pass

    def _build_esn(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        raise NotTrainableError("Ensemble models cannot be trained.")

    def annotate(
        self,
        dataset=None,
        vocab=None,
        to_group=False,
        vectors=False,
        return_truths=False,
    ):

        if vocab is None:
            if self.vocab is not None:
                vocab = self.vocab
            elif dataset is not None:
                vocab = dataset.vocab
            else:
                raise ValueError(
                    "vocab is mandatory when performing " "ensemble model annotations."
                )

        features, teachers = dataset.to_features(mode="syn", return_dict=True)

        models = ["ensemble"] + [m.name for m in self.models]
        annotations = {m: {} for m in models}
        vects = {m: {} for m in models}

        @delayed
        def annotate_one(song, feat):
            bests, argmaxs = [], []

            vects = {}
            annotations = {}

            for model in self.models:
                out = model.annotate_features([feat])
                bests.append(F.softmax(out).max(axis=1))
                argmaxs.append(F.softmax(out).argmax(axis=1))

                if vocab is None and to_group:
                    raise ValueError(
                        "If to_group, 'vocab' should not be "
                        "None. No grouping possible "
                        "on float vectors."
                    )
                else:
                    all_sylls = to_annotations(out, vocab)
                    if to_group:
                        all_sylls = [group(sylls) for sylls in all_sylls]
                        # all_sylls = group(all_sylls)

                annotations[model.name] = all_sylls
                vects[model.name] = out

            bests = np.asarray(bests)
            argmaxs = np.asarray(argmaxs)
            vote = [
                vocab[argmaxs[c, i]] for i, c in enumerate(np.argmax(bests, axis=0))
            ]

            if to_group:
                vote = [group(v) for v in vote]
                # vote = group(vote)

            annotations["ensemble"] = vote

            return song, annotations, vects

        with Parallel(n_jobs=-1) as parallel:
            results = parallel(
                annotate_one(song, feat)
                for song, feat in tqdm(features.items(), "Annotating")
            )

        for res in results:
            for m in res[1].keys():
                annotations[m][res[0]] = res[1][m]

        for res in results:
            for m in res[2].keys():
                vects[m][res[0]] = res[2][m]

        if not vectors:
            vects = None

        if return_truths:
            annotations["truth"] = teachers

        return annotations, vects

    def annotate_from_models_outputs(self, *models_vectors, vocab=None, to_group=False):

        vocab = self.vocab if vocab is None else vocab

        annotations = {}
        for song in models_vectors[0].keys():
            bests, argmaxs = [], []
            for m, _ in enumerate(self.models):
                bests.append(F.softmax(models_vectors[m][song]).max(axis=1))
                argmaxs.append(F.softmax(models_vectors[m][song]).argmax(axis=1))

            bests = np.asarray(bests)
            argmaxs = np.asarray(argmaxs)
            vote = [
                vocab[argmaxs[c, i]] for i, c in enumerate(np.argmax(bests, axis=0))
            ]

            annotations[song] = vote

        return annotations


def to_annotations(y, vocab):
    return [vocab[t] for t in y.argmax(axis=1)]
