# Canapy

**Automatic audio annotation tools for animal vocalizations**

--------

Canapy trains automatic annotators for animal vocalizations using [Reservoir Computing](https://reservoirpy.readthedocs.io/) (Echo State Networks). It comes with an interactive dashboard to guide you through the full pipeline: dataset preparation, model training, evaluation, and annotation.

> For the full reference documentation, see [README_extended.md](README_extended.md).

## Installation

```bash
git clone git@github.com:birds-canopy/canapy.git
pip install -e canapy/.
```

## Quick start

### 1. Prepare your dataset

You need hand-labeled audio recordings. Annotations should be `.csv` files in **marron1csv** format (columns: `wave`, `start`, `end`, `syll`). Audio must be mono WAV files.

Recommended structure:

```text
song_dataset/
├── annotations/
│   ├── song1.csv
│   └── song2.csv
└── audio/
    ├── song1.wav
    └── song2.wav
```

Aim for 30 min–1 hour of annotated data (10 min can already give good results on canary songs).

### 2. Launch the dashboard

```bash
canapy dash -a song_dataset/annotations -s song_dataset/audio -o output
```

Or launch without arguments and use the **Load data** page:

```bash
canapy dash
```

The dashboard opens at [localhost:9321](http://localhost:9321).

### 3. Train → Evaluate → Annotate

The dashboard guides you through three main pipelines:

**Train models**
- *(Optional)* Clean up your dataset on the **Preprocessing** page: merge similar classes, correct mislabeled samples, trim excess silence.
- On the **Train** page, click *Start Training*. Three models are trained: `syn` (uses song context), `nsyn` (class-balanced, context-free), and `ensemble` (combines both).
- *(Optional)* Run a **hyperparameter search** before training to find better ESN parameters automatically.
- On the **Eval** page, check the confusion matrix and per-class metrics. Correct remaining errors and iterate (3–4 rounds is typical).
- Export the trained models.

**Annotate unlabeled data**
```bash
canapy dash -d song_dataset/audio -c output/model -o output/annotations
```
Load your unlabeled audio and your trained models, select the model(s) to use, and click *Start annotation*.

**Quick commands (no dashboard)**
```bash
# Train once and export
canapy train -d song_dataset/data -o output

# Annotate
canapy annotate -d song_dataset/audio -c models_folder -o output
```

## Support

Contact Axel Arnaud or Xavier Hinaut at Inria Mnemosyne:
<axel.arnaud@inria.fr> — <xavier.hinaut@inria.fr>
