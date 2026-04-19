# EE6405 Sentiment Project

Group project for EE6405 NLP assignment — a multi-model, multi-dataset study of
English sentiment/emotion classification. Six approaches spanning classical ML,
recurrent networks, and transformers are compared across **IMDb**, **SST-2**, and
the **dair-ai Emotion** datasets, with a unified interactive dashboard to explore
all results side-by-side.

---

## ⭐ The Dashboard Lives in `comparison_app/`

**`comparison_app/` is the main deliverable of this project — it is the interactive
Streamlit dashboard that unifies every teammate's results into one place.** All
other folders (`src/`, `experiments/`, `gru_attention/`, `notebooks/`, `demo/`)
produce the artifacts (metrics JSONs, training logs, figures) that the dashboard
reads from `comparison_app/data/` and `comparison_app/assets/`.

### Launching the dashboard

```bash
cd comparison_app
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens at **http://localhost:8501**. It is self-contained: all
metrics and figures are pre-bundled, so **no training or GPU is required** to run
it.

### What's inside the dashboard

Seven tabs — one per model family plus a cross-model overview:

| Tab | Model | Owner |
|---|---|---|
| **Overall** | Side-by-side F1 / accuracy across every model × dataset, best-model-per-dataset breakdowns, efficiency frontier (F1 vs trainable params) | — |
| **SVM + DistilBERT** | TF-IDF + SVM baseline and full-FT DistilBERT | Wang Yuekai |
| **LoRA** | LoRA fine-tuning on RoBERTa-large / BERTweet / DistilBERT at rank 8 and 16 — rank comparison, per-epoch learning curves, parameter efficiency, bf16→fp16 NaN case study, per-class F1 | Shivaangii Jaiswal |
| **BERTweet Full-FT** | Full fine-tuning of `vinai/bertweet-base` with LIME explanations | Wai Yar Aung |
| **Bi-GRU + Attention** | From-scratch recurrent baseline with Bahdanau attention | Yu Juncheng |
| **BiLSTM + Attention** | From-scratch BiLSTM with attention | Joanna SJ |
| **TF-IDF + LogReg** | Classical baseline with SHAP explanations | Kevin Wong |

See [`comparison_app/README.md`](comparison_app/README.md) for the full dashboard
structure and notes on test-set sizes.

---

## Project Goal

Build a trustworthy English sentiment classification system with:

- A traditional baseline: TF-IDF + SVM
- A modern model: DistilBERT
- Cross-domain evaluation: IMDb <-> SST-2
- Robustness checks: simple text perturbations
- Explainability: token/word-level attribution
- Lightweight GUI demo for visualization
- **A unified interactive dashboard (`comparison_app/`)** combining every
  teammate's results — classical, recurrent, and transformer — into one place.

## Suggested Directory Structure

```text
ee6405-group/
├── README.md
├── requirements.txt
├── .gitignore
├── comparison_app/          ⭐ MAIN DASHBOARD — unified Streamlit UI
│   ├── app.py
│   ├── tabs/                # one module per model tab
│   ├── data/                # pre-computed metrics (CSV / JSON)
│   ├── assets/              # pre-computed figures per model
│   ├── requirements.txt
│   └── README.md
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── artifacts/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── training/
│   ├── robustness/
│   ├── explainability/
│   ├── visualization/
│   └── utils/
├── experiments/
├── gru_attention/           # Bi-GRU + Bahdanau attention (Yu Juncheng)
├── notebooks/
│   ├── WaiYarAung/          # BERTweet full-FT training scripts
│   └── joannasj-bilstm-attention.ipynb
├── results/
│   ├── figures/
│   ├── tables/
│   └── logs/
├── checkpoints/
├── demo/                    # per-teammate standalone Streamlit demos
├── docs/
├── report/
├── slides/
└── video/
```

## Quick Start

> **Prerequisites:** Python 3.10+ is required.

### Option A — Just run the dashboard (no training needed, recommended for reviewers)

```bash
cd comparison_app
pip install -r requirements.txt
streamlit run app.py
```

### Option B — One-command full setup

Creates venv, installs deps, runs experiments, launches demo:

```bash
bash setup.sh
```

### Option C — Manual step-by-step

Follow the steps below.

### 1) Create environment

```bash
python3 -m venv .venv
```

Activate the virtual environment:

- **macOS / Linux:** `source .venv/bin/activate`
- **Windows (cmd):** `.venv\Scripts\activate.bat`
- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`

Then install dependencies:

```bash
pip install -r requirements.txt
```

### 2) Prepare data

```bash
python3 -m src.main prepare-data
```

### 3) Run experiments

```bash
python3 -m experiments.run_baseline --dataset imdb
python3 -m experiments.run_baseline --dataset sst2
python3 -m experiments.run_distilbert --dataset imdb --epochs 3 --batch-size 16 --max-length 128
python3 -m experiments.run_distilbert --dataset sst2 --epochs 3 --batch-size 16 --max-length 128
python3 -m experiments.run_cross_domain
python3 -m experiments.run_robustness
```
Logistic Regression + SHAP
```bash
python3 -m experiments.run_Logistic_Regression --dataset imdb --max-features 10000
python3 -m experiments.run_Logistic_Regression --dataset sst2 --max-features 10000
```
Generated outputs:

- Processed datasets: `data/processed/*.csv`
- Statistics and labels: `data/artifacts/*.json`
- Model checkpoints: `checkpoints/baseline/*` and `checkpoints/distilbert/*`
- Metrics tables: `results/tables/*.csv` and `results/tables/*.json`
- Figures: `results/figures/*.png`

Logistic Regression + SHAP
- Processed datasets (Same)
- Statistics and labels (Same)
- Model checkpoints: `checkpoints/logistic_regression/*`
- Metrics tables (Same)
- Figures (Same)

### 4) Launch GUI demo

The main unified dashboard (recommended):

```bash
cd comparison_app && streamlit run app.py
```

Standalone per-teammate demos are also available:

```bash
streamlit run demo/gui_yuekai.py
```

The dashboard will open at `http://localhost:8501`.

Logistic Regression + SHAP
```bash
python3 -m streamlit run demo/gui_3.py
```



## Deliverables Checklist

- [ ] Reproducible code (`src/` + `experiments/`)
- [ ] Metrics table (Accuracy / Precision / Recall / F1)
- [ ] Core figures (distribution, confusion matrix, training curves, comparison)
- [ ] **Unified GUI dashboard → `comparison_app/`**
- [ ] Report PDF
- [ ] Presentation slides
- [ ] <=10 min video
- [ ] Group reflection + GAI declaration

## Submission Checklist

1. Run data + experiments once from scratch (new terminal session).
2. Confirm key files exist in `results/figures` and `results/tables`.
3. Open the unified dashboard (`comparison_app/app.py`) and verify every tab loads.
4. Open standalone GUIs and verify prediction/explain/robustness/benchmark tabs.
5. Finalize report, slides, and video from templates.
6. Push all code and docs to GitHub with a clean `git status`.

---



# YU JUNCHEG — GRU with Attention Model

This module implements a Bi-directional GRU with Bahdanau Attention for text classification (emotion, IMDB, SST-2). 
More detail you can find in `gru_attention/README.md`
To run:

**Navigate to the gru_attention directory:**
```bash
cd gru_attention
```

**Install dependencies** (if not already installed):
```bash
pip install -r ../requirements.txt
```

## No train the models:
** Prepare data ** 
The data is from the .\GROUP-PROJECT-6045\data\processed
If you not find or miss any '.csv' data file, you can try this way to rebuild them:
```bash
cd ..
python -m src.main prepare-data
cd gru_attention
```

**Train the models:** 
```bash
python train.py
```

**Evaluate the models and generate reports:**
```bash
python test.py
```

## Have already train the models/ Just want ti use my trained models:
**Launch the interactive demo:**
```bash
streamlit run gru_demo.py
```

**Clean up generated files && data** (optional):
```bash
python cleanup.py
```


# WaiYarAung — BERTweet Sentiment & Emotion Suite

Fine-tuning **BERTweet** (`vinai/bertweet-base`) on three datasets for text classification
with a full training pipeline, hyperparameter search, stratified cross-validation,
LIME explainability, and a combined Streamlit demo app.

## Contribution Structure

```
notebooks/
└── WaiYarAung/
    ├── WaiYarAung_train_imdb.py          <- Training pipeline: IMDB binary sentiment
    ├── WaiYarAung_train_sst2.py          <- Training pipeline: SST-2 binary sentiment
    ├── WaiYarAung_train_emotion.py       <- Training pipeline: 6-class emotion
    ├── WaiYarAung_gen_artifacts_imdb.py  <- Regenerate eval artifacts (IMDB)
    ├── WaiYarAung_gen_artifacts_sst2.py  <- Regenerate eval artifacts (SST-2)
    ├── WaiYarAung_gen_artifacts_emotion.py <- Regenerate eval artifacts (Emotion)
    └── WaiYarAung_requirements.txt       <- All Python dependencies
demo/
└── WaiYarAung_app.py                 <- Combined 3-model Streamlit app (port 8504)
checkpoints/
├── bertweet_imdb/     <- Tokenizer + config files (WaiYarAung_*)
├── bertweet_sst2/
└── bertweet_emotion/  <- Model weights excluded (>500 MB each, run train script to reproduce)
results/
├── figures/
│   ├── WaiYarAung_imdb_confusion_matrix.png
│   ├── WaiYarAung_sst2_confusion_matrix.png
│   └── WaiYarAung_emotion_confusion_matrix.png
└── tables/
    ├── WaiYarAung_*_classification_report.json
    ├── WaiYarAung_*_class_names.json
    └── WaiYarAung_*_lime_explanations.json
docs/
└── WaiYarAung_presentation.pdf       <- Project presentation (9 slides)
```

## Datasets

| Script / Notebook | Dataset | HF Identifier | Task | Classes |
|---|---|---|---|---|
| IMDB | Movie reviews | `imdb` | Binary sentiment | Negative, Positive |
| SST-2 | Stanford Sentiment Treebank | `glue/sst2` | Binary sentiment | Negative, Positive |
| Emotion | Twitter emotions | `dair-ai/emotion` | 6-class emotion | sadness, joy, love, anger, fear, surprise |

## Model & Training Pipeline

**BERTweet** — `vinai/bertweet-base` (RoBERTa pre-trained on 850M tweets)

| Parameter | IMDB | SST-2 | Emotion |
|---|---|---|---|
| Max seq length | 128 | 64 | 64 |
| Train samples | 600 (balanced) | 600 (balanced) | 1800 (300/class) |
| Epochs | 2 | 3 | 3 |
| Batch size | 16 | 16 | 16 |
| Best LR | 5e-5 | 5e-5 | 5e-5 |
| CV folds | 2 | 2 | 2 |

Every training script runs:
1. Balanced data loading (equal samples per class)
2. LR grid search over {2e-5, 5e-5} with Stratified K-Fold CV
3. AdamW + linear warmup scheduler (10% warmup, gradient clipping)
4. Best checkpoint saved via `save_pretrained()`
5. Confusion matrix, classification report, and LIME explanations exported

## Results

| Model | Val Accuracy | Macro F1 |
|---|---|---|
| IMDB Sentiment | **86.5%** | **0.865** |
| SST-2 Sentiment | **85.3%** | **0.853** |
| Emotion Classifier | **86.6%** | **0.842** |

## Running the Training Scripts

### 1. Install dependencies
```bash
pip install -r notebooks/WaiYarAung/WaiYarAung_requirements.txt
```

### 2. Train each model
```bash
python notebooks/WaiYarAung/WaiYarAung_train_imdb.py
python notebooks/WaiYarAung/WaiYarAung_train_sst2.py
python notebooks/WaiYarAung/WaiYarAung_train_emotion.py
```
> Training takes ~30-60 min per model on CPU, ~5 min on GPU.
> If interrupted, run `python WaiYarAung_gen_artifacts_<dataset>.py` to regenerate results.

## Running the Combined Demo App

The combined app loads all three models and offers:
- **Text Prediction**: classify any text with all 3 models + LIME word explanations
- **YouTube Comments**: fetch up to 80 comments from any YouTube URL and classify them (no API key needed)
- **Evaluation Results**: confusion matrices, classification reports, per-class F1 charts

```bash
streamlit run demo/WaiYarAung_app.py --server.port 8504
```

## Explainability — LIME

Each model includes LIME (Local Interpretable Model-agnostic Explanations):
- **Green bars** — words that support the predicted class
- **Red bars** — words that oppose it
- Pre-computed explanations saved in `results/tables/WaiYarAung_*_lime_explanations.json`

---

# Shivaangii Jaiswal — LoRA Fine-Tuning Study

Parameter-efficient fine-tuning with **LoRA adapters** on three backbone models
(**RoBERTa-large**, **BERTweet**, **DistilBERT**) at ranks **r=8** and **r=16**,
across all three datasets. Results — including rank comparison, per-epoch
learning curves, parameter efficiency analysis, a bf16→fp16 NaN case study, and
per-class F1 / precision / recall — are rendered inside the **LoRA** tab of the
unified `comparison_app/` dashboard.

**Artifacts:**
- Per-run bundles + train logs → `results/tables/lora/` (mirrored into
  `comparison_app/data/lora_*`)
- Efficiency frontier CSV → `comparison_app/data/lora_efficiency_frontier.csv`

**Run LoRA experiments:**
```bash
python3 -m experiments.run_lora --model roberta-large --rank 8  --dataset imdb
# Full matrix lives in experiments/run_efficiency_sweep.sh
bash experiments/run_efficiency_sweep.sh
```

---

# joannasj — BiLSTM + Attention

From-scratch BiLSTM with attention, implemented in
`notebooks/joannasj-bilstm-attention.ipynb`. Classification reports are
transcribed into `comparison_app/data/bilstm_classification_reports.json` and
rendered in the **BiLSTM** tab of the dashboard.

---

# Kevin Wong — TF-IDF + Logistic Regression + SHAP

Classical baseline with SHAP-based global and local explanations. See
`experiments/run_Logistic_Regression.py` above for training, and run the
standalone demo with:

```bash
streamlit run demo/gui_3.py
```

Metrics feed into the **TF-IDF + LogReg** tab of the dashboard.
