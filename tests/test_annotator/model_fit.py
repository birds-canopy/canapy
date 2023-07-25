# <complete>
# <verify>


from canapy.corpus import Corpus


def print_config(config=None):
    if config is None:
        from canapy.config import default_config
        print(default_config)
    print(config)


def load_corpus(verif_print=False):
    my_corpus_audio_and_anot = Corpus.from_directory(
        audio_directory="/home/vincent/Documents/data_canary/audio",
        annots_directory="/home/vincent/Documents/data_canary/annotations",
        )
    my_corpus_audio = Corpus.from_directory(
        audio_directory="/home/vincent/Documents/data_canary/express",
    )
    if verif_print:
        print(my_corpus_audio_and_anot.dataset)
        print(len(my_corpus_audio_and_anot))

    return my_corpus_audio_and_anot, my_corpus_audio



def load_annotators(verif_print=False):

    from canapy.annotator import Annotator

    my_annotator_from_disk = Annotator.from_disk("/home/vincent/Documents/Travail/Stage_L3/canapy-master_github/data_alizee/rouge_6_data_and_model/rouge6_model/models/syn")

    #print(my_annotator_from_disk.config)



def test_corpus_query(corpus, verif_print=False):
    # 'corpus' is the original corpus
    corpus_without_cri = corpus["label != 'cri'"]
    # corpus_without_cri is a copy of corpus where every line of the dataset with the label 'cri' is erased

    corpus_first_seconds = corpus["offset_s <= 10"]
    # corpus_first_seconds is a copy of corpus where there is only line that offset_s is smaller than 10
    # so it contains the annotation that stop before the first 10 seconds

    corpus_long_phrase = corpus["offset_s - onset_s > 1"]
    # corpus_long_phrase is a copy of corpus where every phrase of the dataset that last less than a second
    # are erased

    if verif_print:
        print(corpus_without_cri.dataset.to_string()[:1000])
        print(corpus_first_seconds.dataset.to_string()[:10000])
        print(corpus_long_phrase.dataset.to_string()[:1000])



def create_annotators(corpus, verif_print=False):

    from config import default_config

    from canapy.annotator.synannotator import SynAnnotator

    my_syn_annotator = SynAnnotator(default_config, "/home/vincent/Documents/data_canary/spec")

    from canapy.annotator.nsynannotator import NSynAnnotator

    my_nsyn_annotator = NSynAnnotator(default_config, "/home/vincent/Documents/data_canary/spec")

    from canapy.annotator.ensemble import Ensemble

    my_ensemble_annotator = Ensemble(default_config, "/home/vincent/Documents/data_canary/spec")

    if verif_print:
        print(type(my_syn_annotator))
        print(type(my_nsyn_annotator))
        print(type(my_ensemble_annotator))

    return my_syn_annotator, my_nsyn_annotator, my_ensemble_annotator


def train_annotators(corpus, my_syn_annotator, my_nsyn_annotator, my_ensemble_annotator, verif_print=False):

    trained_syn = my_syn_annotator.fit(corpus)

    trained_nsyn = my_nsyn_annotator.fit(corpus)

    trained_ensemble = my_ensemble_annotator.fit(corpus)

    if verif_print:
        print(trained_syn.vocab)
        print(trained_nsyn.vocab)
        print(trained_ensemble.vocab)

    return trained_syn, trained_nsyn, trained_ensemble


def predict_annotators(corpus, trained_syn, trained_nsyn, trained_ensemble, verif_print=False):

    # Prediction of syntactic annotator
    corpus_syn_predict = trained_syn.predict(corpus)

    # Prediction of not-syntactic annotator
    corpus_nsyn_predict = trained_nsyn.predict(corpus)

    # Prediction of ensemble annotator
    corpus_syn_predict_raw = trained_syn.predict(corpus, return_raw=True)
    corpus_nsyn_predict_raw = trained_nsyn.predict(corpus, return_raw=True)
    corpus_ensemble_predict = trained_ensemble.predict([corpus_syn_predict_raw, corpus_nsyn_predict_raw])

    if verif_print:
        print(corpus_syn_predict.dataset)
        print(corpus_nsyn_predict.dataset)
        print(corpus_ensemble_predict)

    return corpus_syn_predict, corpus_nsyn_predict, corpus_ensemble_predict

def metrics(g_corpus, p_corpus, print_verif=False):

    from canapy.metrics import sklearn_classification_report, sklearn_confusion_matrix, segment_error_rate

    r1 = sklearn_classification_report(g_corpus, p_corpus)
    r2 = sklearn_confusion_matrix(g_corpus, p_corpus)
    r3 = segment_error_rate(g_corpus, p_corpus)

    if print_verif:
        print("classification_report : ", r1)
        print("confusion_matrix : ", r2)
        print("segment_error_rate : ", r3)


def jupyter():

    from canapy.corpus import Corpus
    from canapy.annotator import get_annotator
    from config import default_config

    syn_annotator = get_annotator("syn-esn")(default_config, "../../tuto/spec")
    nsyn_annotator = get_annotator("nsyn-esn")(default_config, "../../tuto/spec")
    ensemble_annotator = get_annotator("ensemble")(default_config, "../../tuto/spec")


    corpus_annotated_songs = Corpus.from_directory(audio_directory="../../tuto/annotated_songs", annots_directory="../../tuto/annotated_songs")

    corpus_non_annotated_songs = Corpus.from_directory(audio_directory="../../tuto/non_annotated_songs")


    syn_annotator.fit(corpus_annotated_songs)
    nsyn_annotator.fit(corpus_annotated_songs)
    ensemble_annotator.fit(corpus_annotated_songs)
    print(len(corpus_non_annotated_songs))

    corpus_syn_predict = syn_annotator.predict(corpus_non_annotated_songs)
    corpus_nsyn_predict = nsyn_annotator.predict(corpus_non_annotated_songs)

    corpus_syn_predict_raw = syn_annotator.predict(corpus_non_annotated_songs, return_raw=True)
    corpus_nsyn_predict_raw = nsyn_annotator.predict(corpus_non_annotated_songs, return_raw=True)

    corpus_ensemble_predict = ensemble_annotator.predict([corpus_syn_predict_raw, corpus_nsyn_predict_raw])


def dashboard():
    import panel as pn

    from dashboard.app import CanapyDashboard

    args = {'data_directory':'./tuto/annotated_songs', 'output_directory':'./tuto/results', 'config_path':'./config/template/default.config.toml', 'port':9321, 'annot_format':'marron1csv', 'audio_ext':'.wav', 'annotators':['syn-esn', 'nsyn-esn']}

    pn.extension()
    dashboard = CanapyDashboard(**vars(args))
    dashboard.show()


if __name__ == "__main__":

    #print_config()

    #load_annotators()

    #my_corpus, my_corpus_audio = load_corpus()

    #test_corpus_query(my_corpus)

    #syn, nsyn, ensemble = create_annotators(my_corpus)

    #trained_syn, trained_nsyn, trained_ensemble = train_annotators(my_corpus, syn, nsyn, ensemble)

    #c1, c2, c3 = predict_annotators(my_corpus, trained_syn, trained_nsyn, trained_ensemble, verif_print=True)

    #print("\n\n\n\n\n\n\n SYN")
    #metrics(my_corpus, c1, print_verif=True)
    #print("\n\n\n\n\n\n\n NSYN")
    #metrics(my_corpus, c2, print_verif=True)
    #print("\n\n\n\n\n\n\n ENSEMBLE")
    #metrics(my_corpus, c3, print_verif=True)

    #jupyter()

    dashboard()










