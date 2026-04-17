# Canapy

**Automatic audio annotation tools for animal vocalizations**

--------

Canapy trains automatic annotators for animal vocalizations using [Reservoir Computing](https://reservoirpy.readthedocs.io/) (Echo State Networks). It comes with an interactive dashboard to guide you through the full pipeline: dataset preparation, model training, evaluation, and annotation.

> For the full reference documentation, see [README_extended.md](README_extended.md).

## Installation

```bash
git clone -b canapy_2026 git@github.com:birds-canopy/canapy.git
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
└── song1.csv
└── song2.wav
└── song2.csv
```
Aim for 30 min–1 hour of annotated data (10 min can already give good results on canary songs).

### 2. Launch the dashboard

```bash
canapy dash -a song_dataset/annotations -s song_dataset/audio -o output
```

Or launch without arguments and use the **Load data** page in the dashboard (recommanded):

```bash
canapy dash
```

The dashboard automatically opens at [localhost:9321](http://localhost:9321).

---

## Using the dashboard

The home page gives access to three pipelines. The typical workflow for training a new annotator is:

```
Load data → Preprocess → Train → Eval → (iterate) → Export
```

### Load data
<img width="404" height="413" alt="image" src="https://github.com/user-attachments/assets/458edafc-3ea8-4e3f-a36e-9a616b13db86" />

On the **Load data** page, specify where your data lives:

- **Source selection**: choose *combined folders* if audio and annotations are in the same directory, or *separate folders* otherwise.
- **Model directory**: if you want to annotate unlabeled data, point to a folder containing already-trained models (exported after using the training pipeline).
- **Output directory**: where models and annotations will be saved (defaults to `output/`).
- **Annotation format** and **audio extension**: set these to match your files.
- The **sampling rate** is auto-detected from your audio files. Enable *Downsample* if you want audio resampled to that rate at load time.

### Pipeline 1: Preprocess (Edit dataset)
<img width="955" height="474" alt="image" src="https://github.com/user-attachments/assets/01e2926e-6722-482d-a4d7-8eac07dc8a9f" />

Use this page to clean your dataset before training. It has three sections:

**1. Class merge** — listen to each annotation class, compare them side by side, and rename/merge classes that are acoustically too similar. Type the new label in the text field and click *Apply*.

**2. Sample correction** — review individual samples per class. After clicking *Calculate stats*, you can see the distribution of duration, frequency centroid, and mean slope for each class. Select a class to listen to its samples one by one and correct any mislabeled ones. Click *Save all* when done.

**3. Trim silences** — balance the proportion of silence in your dataset. Use the **target silence ratio** slider (e.g. 0.2 = 20% silence) and Canapy will center-crop silence segments that exceed this ratio. Silence percentage below 50% are best. You should consider silences as an annotation label that Canapy will annotate, so rule of thumb : desired percentage of silence = 100/(number of annotation classes + 1) (except if you don't want Canapy to annotate silences). 
Trimmed files are saved to `output/audio_trimmed/` and `output/annots_trimmed/`.

You can then export corrected annotations and/or go back to the Home page and use these in training as corrected annotations are kept in memory.

### Pipeline 2: Train models

#### Preprocessing step
Same page as Pipeline 1. Apply any corrections you need, or skip directly to the next step.

#### Train step
<img width="961" height="234" alt="image" src="https://github.com/user-attachments/assets/5e02f7b1-8737-4b4a-a525-751119b75165" />

**Hyperparameter search (optional):** Before training, you can run an automatic HP search to find better ESN parameters for your dataset. It uses TPE (sequential) or parallel random search. Key settings (on the **Settings** page): `opt_max_evals`, `opt_n_jobs`, `opt_max_percentage`. Optimized parameters are preserved across training iterations — no need to re-run the search each time.

Click *Start Training*. Three ESN-based models are trained:

| Model | Description |
|-------|-------------|
| `syn` | Trained on complete songs in order — uses sequential/syntactic context |
| `nsyn` | Trained on randomly shuffled, class-balanced samples — context-free |
| `ensemble` | Combines `syn` and `nsyn` by majority vote |



#### Eval step

After training, the **Eval** page shows:

- **Confusion matrix** — which classes are being confused with each other.
- **Per-class metrics table** — precision, recall, F1-score for each class (1.0 = perfect).
- **Class merge / sample correction** — same tools as Preprocessing, but applied to misclassified samples from the model's output.

If the results are not satisfying, you can merge classes and correct samples, and click *Next step* to retrain. **3–4 iterations** of train → eval is typical to converge. Corrections made in Eval are preserved across iterations.

#### Export step

When you are satisfied with performance, click *Export* to save the trained models to the output folder.

### Pipeline 3: Annotate unlabeled data

Load your unlabeled audio and trained models (via **Load data** or in **CLI** with the `-c` argument at launch):

```bash
canapy dash -d song_dataset/audio -c output/model -o output/annotations
```

Select which model(s) to use (*Syn-ESN*, *NSyn-ESN*, *Ensemble*), click *Start annotation*, then *Export annotation* when done.

> If you trained with multiple iterations, you can pick a specific one: load `output/model/3` instead of `output/model`.

### Settings

The **Settings** page lets you configure the parameters used by Canapy:

- **Species parameters**: `fmin`, `fmax` (frequency range), `win_length`, `hop_length`, `n_fft` (spectrogram).
- **ESN parameters**: `sr` (spectral radius), `leak` (leak rate), `iss`, `isd`, `isd2` (input scalings), `ridge` (regularization).
- **HP search parameters**: `opt_max_evals`, `opt_n_jobs`, `opt_max_percentage`.

Click *Validate* to apply changes to the current session.

#### Presets

The **Presets** section in Setting provides pre-configured profiles for specific species (canary, bengalese finch, zebra finch, mouse, infant marmoset). 
By default, the parameters loaded in Canapy are those from the canary preset.

If you have used Canapy on a new species, You are welcomed to send us your configuration so we can add it to the preset folder. 

You can also load your own configuration by clicking on **Load config**.

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
