# Canapy

**Automatic audio annotation tools for animal vocalizations**
--------

**Summary:**

- [1. Installation](#installation)
- [2. Prepare your dataset](#prepare_data)
- [3. Run canapy dashboard](#dashboard)
- [4. Using quick commands](#quick_commands)
- [5. Using canapy Python library](#library)
- [6. Change configuration](#config)
- [7. Support](#support)

## Installation <a name="installation"></a>

Canapy dashboard and tools can be installed using _pip_ (python installation package). You can install canapy using one of these two options:

If you do not have _pip_ you can [find info here](https://pip.pypa.io/en/stable/installation/) to install it.

**1st option to install canapy (local copy)**
```bash
git clone git@github.com:birds-canopy/canapy.git
pip install -e canapy/.
```

or replace the second command line by this one if you want to install from another path (where you cloned the canapy repository):
```bash
pip install -e <path to canapy directory containing pyproject.toml>
```

**2nd option to install canapy**

```bash
pip install -e git+[https://github.com/birds-canopy/canapy.git#egg=canapy-reborn](https://github.com/birds-canopy/canapy.git#egg=canapy-reborn)
```

## Prepare your dataset <a name="prepare_data"></a>

Canapy uses supervised machine learning tools to create automatic annotators,
and thus requires some hand-made annotations to bootstrap the annotation pipeline.
Using our proposed method, we recommend to ideally have between 30 minutes and 1
hour of annotated sounds to train an automatic annotator - but from our experiments
on canary data with 10 min of songs you can already obtain nice results! This may of course vary
depending on the nature of the annotated vocalizations. Canapy was primarily
designed to annotate bird songs, in particular domestic canary songs.

Two sources of data are required to train an annotator: annotations and audio.

### Annotations

Annotations are typically segments of audio labeled using a custom code representing
different vocal units, like phonemes, syllables or words in human speech. In their
most essential form, they are defined using the triplet (onset, offset, label),
representing an annotated segment, delineated in time.

For the time being, canapy only deals with non-overlapping annotation segments,
and can thus only work on a single track of annotations.

#### The default annotation format: marron1csv

This format is inspired by the M1-spring dataset, a dataset of more than 400
hand-labeled songs of one domestic canary. It's a simple, straightforward format,
that is best expressed in a comma-separated values spreadsheet (.csv file).

Four named columns of data are needed to define an annotation:

- `wave`: the name of the audio track being annotated.
- `start`: the beginning of the annotation on the audio track, in seconds
- `end`: the end of the annotation on the audio track, in seconds
- `syll`: the annotation label

An example .csv file may look like this:
<br/>
![CSV Example](images/csv_example_placeholder.png)

#### Use another format

Audio annotations come in many different formats these days. You may have used
Audacity, Raven, or Praat to annotate your data by hand.

By default, canapy uses its own annotation format, called marron1csv, to process
annotation data. To allow using a different format, canapy was built on top of
[crowsetta](https://github.com/vocalpy/crowsetta), an audio annotation formats managing tool, which can handle many
different annotation format coming from many different annotation software.
We recommend diving into [crowsetta documentation](https://crowsetta.readthedocs.io/en/latest/index.html) to learn more about annotation
formats.

### Audio

Audio recordings handled by canapy can have any sampling frequency. They must be
mono audio recordings. If stereo audio are provided, they will be converted to mono.

Canapy currently works with two audio data formats: WAV files (.wav) and Numpy arrays
(.npy).

### Training dataset format

When creating new automatic annotators for your data, you should provide
some hand-labeled audios in order to train canapy to annotate this data.

Because canapy will try to split your dataset in two parts (one for training
and one to test its capabilities), you should provide several audio and
annotation files. Canapy will consider each audio file as one sequence
of vocalizations, and will never cut this sequence when training or
annotating. When dealing with songbirds for instance, one file should
ideally contain a single song sequence.

Your dataset should therefore look something like this:

```text
├── song_dataset
    └── annotations
        ├── song1.csv
        ├── song2.csv
        ...
        └── songN.csv
    ├── audio
        ├── song1.wav
        ├── song2.wav
        ...
        └── songN.wav
```

Here, .csv files in the annotations/ folder
contain annotations in marron1csv format (depending on
your annotation format you may have different file extension) and .wav
files in the audio/ folder are your audio recordings in WAV format.

You can also provide audio recording and annotation files all
mixed in a single directory:

```text
├── song_dataset
    └── data
        ├── song1.csv
        ├── song1.wav
        ├── song2.csv
        ├── song2.wav
        ...
        ├── songN.csv
        └── songN.wav
```

Pay attention to how your audio files are named. Audio filenames
will be used by annotation tools to link annotations with their
corresponding audio. For instance, using the marron1csv annotation
format, all values in the `wave` column in the .csv files must match one of the
audio filenames.

### Non-annotated dataset format

Once training has been performed, your dataset may consist only of audio
files. As no dataset split is required for annotating files, your dataset
may be one single file, or several smaller files. We do not recommend using
too long files however. Depending on your computer, using very long recordings
may be suboptimal, or even crash the annotator.

## Run canapy dashboard <a name="dashboard"></a>

The easiest way to train annotators, check the quality of the dataset and automatically annotate a new dataset is by using the canapy dashboard application.

### Load data
To run the dashboard and load your dataset at `song_dataset/`, you have two options:

#### Option 1: Load data in the dashboard
The dashboard has a **load data** page. Run:
```bash
canapy dash
```
Then in the home page click on the **load data** button. You can load folders where your dataset is.

**In Section 1 "Source selection":**
* If audio and annotations are placed in the same folder, select your folder path in the *data directory (-d)* field, with the **combined folders** option selected.
* If audio and annotations are in different folders, select your folder paths in the *audio directory* and *annotations directory* fields, with the **separate folders** option selected.

**In Section 2 "Configuration":**
* If you want to automatically annotate data, specify the folder where your models have been saved in the *Model directory (-c)* field. (**/!\Warning : you should have already trained models. see below**)
* If you want to specify a folder to save the output of canapy (either models or annotations depending of the pipeline used), specify it in the *Output directory (-o)* field. If you don't specify any folder, an "output" directory will be created by default.

### Option 2: Specify paths in the terminal
You can directly specify paths when launching the dashboard, hence you won't have to load data in the dashboard.

* If you want to **train models** or **edit annotations**, simply run:

```bash
canapy dash -a song_dataset/annotations -s song_dataset/audio -o output
```

or, if audio and annotations are placed in the same directory:

```bash
canapy dash -d song_dataset/data -o output
```

* If you want to **automatically annotate audio files using trained models** run:
```bash
canapy dash -s song_dataset/audio -c models_folder -o output
```
or
```bash
canapy dash -d song_dataset/audio -c models_folder -o output
```

The dashboard should open in your browser, at localhost:9321. If not, simply reach localhost:9321 in your favorite browser.
All the data produced by the dashboard (models and checkpoints or annotations) will be stored in `output/`.

The first dashboard you will see is the home page. From there you can **Load data** as described earlier, **Edit dataset** to edit annotations, **Train models** to then use these to automatically annotate unlabeled audio, or **Annotate** using trained models. Let's go through these different pipelines.

### Pipeline 1: Edit dataset
Click on the button "Edit Dataset". The **Preprocessing** page will be loaded, with 2 sections.

#### Section 1: Class merge
In the class merge section, you can listen to annotation classes (*2. audio repertoire*), compare these and merge them if they are too similar.
To merge classes, in *1. class correction*, specify the new label you want a class to be attributed to and click on the *apply* button.

#### Section 2: Sample correction
In this section, you can correct labels one by one (for example if there have been annotation errors). You can visualize mean statistics for each class (=labels).

Canapy provides powerful statistics to help you decide which classes to merge. In *1. Global analysis & Selection*, after clicking on the *calculate stats* button, you can view the distribution of:
- **Duration**: The average length of the class.
- **Centroid**: The average frequency centroid (brightness of the sound).
- **Mean Slope**: The average variation of frequency over time (Hz/s).

The **Mean Slope** is particularly useful for analyzing repetitive phrases (trills). A positive slope indicates an ascending sound, while a negative slope indicates a descending sound, helping you distinguish between syllables that may look similar but sound different.

Below these stats, you can select a class and listen/view statistics for each individual sample of the class. If they are wrongly labeled, you can correct the label. Click on the "save all" button to save the label correction.

After editing your dataset, you can export it with the *export* button, or go back to the home page. Label changes will be kept in memory for the training pipeline.

### Pipeline 2: Train models
In this pipeline, you can train models using annotated files, to then use them to automatically annotate unlabeled data.

**/!\ The dataset used for training should be similar to the one you want to automatically annotate.**

*Example: I have a large dataset of one bird annotation. I manually annotated a small part of it. I use the annotated part to train models. I then use the trained models to automatically annotate the unlabeled dataset.*

#### 1st page: Preprocessing
This page is the same as the **Edit dataset** page.
Once you have preprocessed the dataset, or if you don't want to preprocess it, you can click on the *next step* button to go to the next step.

#### 2nd page: Train
Click on the *Start Training* button to start training models.
Two models are built during the training phase. They both are based on an Echo State Network (ESN), a kind of artificial neural network, and have the same parameters. They are, however, trained on two different tasks:

- the **syn** model (syntactic model) is trained to annotate whole songs. Entire songs and annotations files are presented to the models during training. Thus, the model is trained only on the available data, meaning that imbalance in number between the categories of bird phrases is preserved. The model is also expected to rely on syntactic information to produce its annotations, being trained on the real order of the phrases in the songs.
- the **nsyn** model (non syntactic model) is trained to annotate only randomly mixed phrases, with an artificially balanced number of phrases samples. This model is expected to rely only on inner characteristics of each type of syllables to annotate the songs, without taking into account their context in the song. Imbalance in number is also *not* preserved, meaning the model has to give the same importance to all categories of syllables.

Finally, a third model, called **ensemble**, combines the outputs of the two previous models using a voting system to combine the "judgements" of the two models into a new one.

At the end of the training sequence, click on *next step* to display the **eval** dashboard (it can take some time to display, don't worry, click only **once** on the button).

#### 3rd page: Eval
The **eval** page is similar in some ways to the **preprocessing** page.

##### Section 1: Class merge
In *1. Evaluation Metrics*, you can see the performance of the trained models for each model for the train and the test phase. The spreadsheet on the right helps you assess the models' performance for each class and in general; 1 means the labels are perfectly classified. The confusion matrix on the left helps you visualize how classes can be misclassified.
If a class is strongly misclassified as another class, they might be acoustically too similar; you can merge these below with the same modules as the **preprocessing** page.

#### Section 2: Sample correction
Although the use is very similar to the **preprocessing** page, the principle differs. In this section, misclassified samples for each class are shown.
You can view which classes have the most misclassified samples, and you can listen to these samples by clicking on the desired class button.
If the misclassified sample was originally wrongly labeled, you can assign it the right label.

If you are satisfied with the performance of the models, you can click on the *export* button to export the models. Else, you can click on the *next step* button to start training models again. You should do 3-4 iterations of training-evaluating to be sure that you have fixed all the annotations.

#### 4th page: Export
This page will export the trained models. Once the models are exported (in the folder specified as output), you can go back home or quit canapy.

### 3rd Pipeline: Annotate unlabeled data

To use this pipeline, you need to have trained models loaded, see *Load data*.
If you trained models with multiple iterations, load the desired iteration: instead of loading a path like `output/model`, load `output/model/4` for example.
If you haven't closed Canapy since you trained models, you should make sure you load the correct dataset (i.e., the unlabeled one) and the exported models you just trained using **Load data**.

Select which models you want to use to automatically annotate unlabeled data by clicking on the buttons *Syn-ESN*, *NSyn-ESN* and *Ensemble*.
Click on the *Start annotation* button to automatically annotate using the desired models.
Once the annotation is finished, click on *Export Annotation*. Annotations will be exported in the folder specified as output in **Load data** or in the launching command.

## Using quick commands <a name="quick_commands"></a>

With canapy, you can run behaviors without having to run the dashboard or using the library.

### Fast training
To quickly train models, simply run:
```bash
canapy train -a song_dataset/annotations -s song_dataset/audio -o output
```
or
```bash
canapy train -d song_dataset/audio -o output
```
Canapy will then train models one time and directly export the trained models in the output folder.
The behavior is similar to using the train pipeline of the dashboard without correcting anything in the preprocessing and eval phase and by running the training phase only once.

### Fast annotation
To quickly annotate data, simply run:
```bash
canapy annotate -a song_dataset/annotations -s song_dataset/audio -c models_folder -o output
```
or
```bash
canapy annotate -d song_dataset/audio -c models_folder -o output
```
The behavior is identical to the annotation pipeline with every model selected.

## Using canapy Python library <a name="library"></a>

Canapy is primarily a Python tool to build simple and fast automatic
audio annotation pipelines, using a simple yet efficient machine learning
technique: Reservoir Computing.

An annotation pipeline can be defined using two objects: the `Corpus` and the
Annotator.

### Dealing with data: the `Corpus` object

The `Corpus` object is a representation of your dataset within canapy.
It holds reference to audio data, is in charge of loading and
formatting your annotations (when needed), and may also store some
other things like preprocessed data - spectrograms, for instance.

#### Create a `Corpus` object

To load your dataset into a `Corpus` object, simply use:

```python
from canapy import Corpus

corpus = Corpus.from_directory(
  audio_directory="song_dataset/audio/",
  annots_directory="song_dataset/annotations/"
)
```

#### Specify annotation format

By default, the annotation format is marron1csv, but you may change
to any other format provided by crowsetta, using the `annot_format`
argument. You may also change the expected audio format using the `audio_ext`
argument, and setting it to `".wav"` or `".npy"` (respectively to
provide WAV files or Numpy arrays archive files).

```python
corpus = Corpus.from_directory(
  audio_directory="song_dataset/audio/",
  annots_directory="song_dataset/annotations/",
  annot_format="aud-seq", # Audacity label track format
  audio_ext=".wav",  # Search for .wav files in the audio directory
)
```

#### Load data from a single directory or only audio data

As explained in [Prepare your data](#prepare-your-dataset), you can also provide
a link to a single directory containing both annotations and audio, or create
an audio-only `Corpus` by omitting the `annots_directory` argument:

```python
# Annotated corpus, all data in the
# same directory
corpus = Corpus.from_directory(
  audio_directory="song_dataset/data/",
  annots_directory="song_dataset/data/" # Same directory !
)

# Non-annotated corpus (only audio)
non_annotated_corpus = Corpus.from_directory(
  audio_directory="song_dataset/audio/",
)
```

#### The `.dataset` attribute

The `Corpus` object will automatically format your data into crowsetta standard
annotation format `generic-seq`. This makes data formats interchangeable to some
extent. You can access a tabular representation of annotations (as a `pandas.DataFrame`)
from the `dataset` attribute:

```python
print(corpus.dataset)
```

Output:

```
    notated_path   onset_s    offset_s    label    annotation   sequence
0      song1.wav      1.20        1.42        A             0          0
1      song1.wav      1.55        2.12        B             0          0
2      song1.wav      2.41        2.79        C             0          0
3      song1.wav      2.89        3.45        A             0          0

```

The `notated_path` column keep tracks of the attached audio file.
The `onset_s`, `offset_s`, and `label` columns respectively store
annotation segments start, end, and label. All onsets and offsets
are expressed in seconds since the beginning of audio track.

The `annotation` and `sequence` columns are special
attributes of crowsetta `generic-seq` format, which we do not
directly use in canapy.

If your corpus is not annotated (only audio), the code above
will return `None`:

```python
print(corpus.dataset)
```

Output:

```text
None
```

#### Save data to CSV

`Corpus` can be saved to disk as CSV files, one per audio file,
if they have annotations:

```python
corpus.to_directory("/save_directory")
```

### Annotate data

Annotation in canapy is performed by an Annotator.
There are several Annotators currently available,
but the simplest one and the most useful is the
SynAnnotator:

```python
from canapy.annotator import SynAnnotator

annotator = SynAnnotator()
```

This object is in charge of training a
machine learning model able to annotate
your data, based on some audio and annotations
stored in a `Corpus`, and eventually annotate
a `Corpus` with unlabelled audio recordings.

#### Train an annotator

After creating an annotated `Corpus` object,
you may `.fit` your annotator to your
dataset:

```python
annotator.fit(corpus)
```

This trains the annotator on your dataset.
You may access the labels learned by the
annotator from the `.vocab` attribute:

```python
print(annotator.vocab)
```

#### Save an annotator to disk

You can save an annotator on your computer using
the `.to_disk` method:

```python
annotator.to_disk("save_directory/annotator")
```

#### Load an annotator from disk

After having saved an annotator on your computer, you
can load it again using the `.from_disk` method of the
`Annotator` base class:

```python
from canapy.annotator import Annotator

annotator = Annotator.from_disk("saved_directory/annotator")
```

#### Annotate a `Corpus` of audio

You may now annotate unlabeled audio using the `.predict` method
of your annotator, generating a new `Corpus` with freshly
computed annotations:

```python
# Load some unlabelled data
corpus = Corpus.from_directory(audio_directory="song_data/audio")

# Annotate !
labeled_corpus = annotator.predict(corpus)

print(labeled_corpus.dataset)

# Additionally save your annotated `Corpus`
labeled_corpus.to_directory("song_data/new_annotations")
```

## Change configuration <a name="config"></a>

Canapy configuration is stored in configuration files in TOML format.
They are human readable, and it is possible to comment them for
additional clarity.

You can access canapy default configuration from `config.default_config`:

```python
from config import default_config

print(default_config)

# It's basically a big nested dictionary of values
print(default_config.transforms.annots.time_precision)
```

### Change parameters from an existing configuration

The best way to quickly change some parameters, such as the audio sampling
rate, is to change them directly from the default configuration.

First, import the default configuration, and then change the parameter you wish
to change:

```python
from copy import deepcopy
from config import default_config

# Copy the default configuration
my_config = deepcopy(default_config)

# Change the audio sampling frequency
# to 16000Hz
my_config.transforms.audio.sampling_rate = 16000
```

The objects in charge of dealing with the configuration throughout
your annotation pipeline are your `Corpus` and Annotator. To apply your
configuration, change your `Corpus` configuration files:

```python
corpus = Corpus.from_directory(audio_directory="song_dateset/audio")
# Apply your configuration
corpus.config = my_config
```

And give your configuration as parameter to your Annotator:

```python
annotator = SynAnnotator(config=my_config)
```

### Saving a configuration to disk

As configuration files are necessary to your pipelines,
we recommend to save your configuration as a TOML file
if you make any change to the default configuration,
using the `.to_disk` method:

```python
my_config.to_disk("saved_directory/my_config.toml")
```

### Create your own configuration file

To create your own configuration file, start from the existing
default configuration, make some changes, and save it somewhere,
let's say at `saved_directory/my_config.toml`.

> [!WARNING]
> Do not change default parameter names! Most of them are required
> by canapy to work.

You can now load your configuration file directly from your `Corpus`
object, using the `config_path` argument:

```python
corpus = Corpus.from_directory(
  annots_directory="song_dataset/annots",
  audio_directory="song_dataset/audio",
  config_path="saved_directory/my_config.toml")
```

You may now check that your `Corpus` `.config` is
identical to your personal configuration file:

```python
print(corpus.config)
```

You can finally inject this configuration file in your
new Annotators:

```python
annotator = SynAnnotator(config=corpus.config)
```

You may also change the dashboard configuration
by providing this file as argument using the `--config_path`
parameter.


## Support <a name="support"></a>

If you have any problems with using Canapy, don't hesitate to contact Axel Arnaud or Xavier Hinaut at Inria Mnemosyne team:
<axel.arnaud@inria.fr>
<xavier.hinaut@inria.fr>
