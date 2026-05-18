# Canapy

**Automatic audio annotation tools for animal vocalizations**

--------

*Powered by: [ReservoirPy](https://reservoirpy.readthedocs.io/) · [Crowsetta](https://crowsetta.readthedocs.io/)*

--------

Canapy trains automatic annotators for animal vocalizations using [Reservoir Computing](https://reservoirpy.readthedocs.io/) (Echo State Networks). It comes with an interactive dashboard to guide you through the full pipeline: dataset preparation, model training, evaluation, and annotation.

> For the full reference documentation, see [README_extended.md](README_extended.md).

## Installation
Canapy requires Python 3.11. If you don't have it, you can create a virtual environment with the right version using conda or uv:

### conda
```bash
conda create -n canapy python=3.11
conda activate canapy
```

### uv
```bash
pip install uv
uv venv canapy --python 3.11
source canapy/bin/activate   # macOS/Linux
canapy\Scripts\activate      # Windows
```

### venv (if Python 3.11 is already installed)
```bash
python3.11 -m venv canapy
source canapy/bin/activate   # macOS/Linux
canapy\Scripts\activate      # Windows
```

Once your environment is active, install Canapy:

### from PyPI
```bash
pip install canapy
```

### or from source
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

or

```text
song_dataset/
├── song1.wav
├── song1.csv
├── song2.wav
└── song2.csv
```

Aim for 30 min–1 hour of annotated data (10 min can already give good results on canary songs).

### 2. Launch the dashboard

```bash
canapy dash
```

The dashboard opens automatically at [localhost:9321](http://localhost:9321).

You can also pass paths directly as arguments:

```bash
canapy dash -a song_dataset/annotations -s song_dataset/audio -o output
```

---

## Using the dashboard

The dashboard is organized around a sidebar giving access to all pages. The typical workflow for training a new annotator is:

```
Load data → Preprocess → Train → Eval → (iterate) → Export → Annotate
```

### Home

The **Home** page gives an overview of Canapy and its pipelines, with a FAQ section and links to the GitHub repository and the Mnemosyne INRIA team.

### Load data

On the **Load data** page, specify where your data lives:

- **Source selection**: choose *combined folders* if audio and annotations are in the same directory, or *separate folders* otherwise.
- **Model directory**: if you want to annotate unlabeled data, point to a folder containing already-trained models (exported after using the training pipeline).
- **Output directory**: where models and annotations will be saved (defaults to `output/`).
- **Annotation format** and **audio extension**: set these to match your files.
- The **sampling rate** is auto-detected from your audio files. Enable *Downsample* if you want audio resampled to that rate at load time.

### Preprocess (Edit dataset)

Use this page to clean your dataset before training. It has two sections:

Two collapsible modules are always accessible at the top of the page, regardless of the active section:

- **Class spectrograms** — displays one representative mel spectrogram per class (the sample closest to the median duration). Useful for quickly spotting acoustically similar classes before merging.
- **Trim silences** *(Preprocess only)* — balances the proportion of silence in the dataset. Set the target silence ratio (e.g. 20%) and Canapy center-crops silence segments that exceed it. Trimmed files are saved to `output/audio_trimmed/` and `output/annots_trimmed/`.

**Class merge** — listen to each annotation class, compare them side by side, and rename or merge classes that are acoustically too similar. Type the new label in the text field and click *Apply*.

**Sample correction** — review individual samples per class. Select a class to listen to its samples one by one, view their spectrogram, and correct any mislabeled ones via the text input. Click *Save all* when done.

You can export corrected annotations at any time using the *Export Annotations* button at the top of the page.

### Train

**Hyperparameter search (optional):** Before training, you can run an automatic HP search to find better ESN parameters for your dataset. It uses TPE (sequential) or parallel random search. Key settings (on the **Settings** page): `opt_max_evals`, `opt_n_jobs`, `opt_max_percentage`. Optimized parameters are preserved across training iterations — no need to re-run the search each time.

Click *Start Training*. Three ESN-based models are trained:

| Model | Description |
|-------|-------------|
| `syn` | Trained on complete songs in order — uses sequential/syntactic context |
| `nsyn` | Trained on randomly shuffled, class-balanced samples — context-free |
| `ensemble` | Combines `syn` and `nsyn` by majority vote |

### Eval

After training, the **Eval** page shows:

- **Confusion matrix** — which classes are being confused with each other.
- **Per-class metrics table** — precision, recall, F1-score for each class (1.0 = perfect).
- **Class merge / sample correction** — same tools as Preprocessing, but focused on misclassified samples from the model's output.

If results are not satisfying, correct samples or merge classes, then retrain. **3–4 iterations** of train → eval is typically enough to converge. All corrections are preserved across iterations.

### Export

When you are satisfied with the model's performance, go to the **Export** page to save the trained models to the output folder. These exported models are then used in the **Annotate** pipeline. You can also export the current configuration (ESN and species parameters) from this page — or from the **Settings** page — to reuse it as a personal preset in future sessions.

### Annotate unlabeled data

Load your unlabeled audio and trained models via the **Load data** page, or pass them directly at launch:

```bash
canapy dash -d song_dataset/audio -c output/model -o output/annotations
```

Select which model(s) to use (*Syn-ESN*, *NSyn-ESN*, *Ensemble*), click *Start annotation*, then *Export annotation* when done. The export folder is named with a timestamp (`YYYY-MM-DD_HHhMMminSS`).

> If you trained with multiple iterations, you can pick a specific one: load `output/model/3` instead of `output/model`.

### Settings

The **Settings** page lets you configure the parameters used by Canapy:

- **Species parameters**: `fmin`, `fmax` (frequency range), `win_length`, `hop_length`, `n_fft` (spectrogram).
- **ESN parameters**: `sr` (spectral radius), `leak` (leak rate), `iss`, `isd`, `isd2` (input scalings), `ridge` (regularization).
- **HP search parameters**: `opt_max_evals`, `opt_n_jobs`, `opt_max_percentage`.

Click *Validate* to apply changes to the current session.

**Presets** — the Settings page provides pre-configured profiles for specific species (canary, Bengalese finch, zebra finch, mouse, infant marmoset). The canary preset is loaded by default. You can also load your own configuration file via *Load config*. If you have tuned Canapy for a new species, feel free to send us your configuration so we can add it to the preset library.

---

## Quick commands (no dashboard)

```bash
# Train once and export models
canapy train -d song_dataset/data -o output

# Annotate unlabeled audio with trained models
canapy annotate -d song_dataset/audio -c models_folder -o output
```

## Support

Contact Axel Arnaud or Xavier Hinaut at Inria Mnemosyne:
<axel.arnaud@inria.fr> — <xavier.hinaut@inria.fr>
