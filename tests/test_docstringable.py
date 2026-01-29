# <complete>
# <verify>


from canapy.corpus import Corpus


def open_corp():
    my_corpus_audio_and_anot = Corpus.from_directory(
        audio_directory="home/vincent/Documents/data_canary/audio",
        annots_directory="home/vincent/Documents/data_canary/annotations",
    )
    my_corpus_audio = Corpus.from_directory(
        audio_directory="home/vincent/Documents/data_canary/audio",
    )

    print(my_corpus_audio_and_anot.dataset)
    print(my_corpus_audio["label == 'SIL'"])

if __name__ == "__main__":

    open_corp()



