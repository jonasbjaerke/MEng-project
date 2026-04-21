# Repost Prediction on Bluesky

This repository contains the code used for a Master’s project on **repost prediction on Bluesky**.

The task is: given a post and a candidate user, predict whether that user will repost the post.

The repository is organized around a reproducible pipeline in `src/`:

- `src.collect` collects raw posts and user data from Bluesky
- `src.process` builds text-derived features from the collected data
- `src.dataset` constructs experiment datasets
- `src.model` trains and evaluates the models

This README focuses on **how to reproduce the results**.

---

## Repository structure

```text
.
├── src/
│   ├── collect/
│   ├── process/
│   ├── dataset/
│   ├── model/
│   └── config/
├── results/
├── feature_names.txt
├── pyproject.toml
└── uv.lock
```

Only the code under `src/` is needed for the main experimental pipeline.

---

## What is in this repository?

The repository supports four main stages:

1. **Data collection** from Bluesky
2. **Feature processing** for text and user/post information
3. **Dataset generation** for different feature setups and class ratios
4. **Model training and evaluation** with XGBoost and BERT-based experiments

The main result families are:

- **User-only** models
- **Message-only** models
- **Hybrid** models
- **BERT** models

Each family is evaluated using predefined experiment configs.

---

## Environment setup

The project uses `pyproject.toml` and includes a `uv.lock` file.

### Recommended setup

```bash
git clone https://github.com/jonasbjaerke/MEng-project.git
cd MEng-project
uv sync
```

Then run commands with `uv run`:

```bash
uv run python -m src.dataset.runner --config USER_1TO1
```

### Alternative setup

You can also create a Python virtual environment manually and install dependencies yourself.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

If you do not use `uv`, make sure all dependencies from `pyproject.toml` are installed.

---

## Reproducibility overview

There are two practical ways to reproduce the project.

### Option 1: Reproduce experiments from existing intermediate data
This is the best option if your goal is to regenerate the experiment outputs and compare them with the saved files in `results/`.

### Option 2: Reproduce the full pipeline from scratch
This includes recollecting data from Bluesky and rebuilding the datasets from the beginning.

> **Important:** exact end-to-end reproduction from live Bluesky collection is unlikely unless the same data snapshots are preserved. Posts, users, and engagement behavior can change over time.

For the most faithful reproduction of the reported model outputs, prefer using the original intermediate data.

---

## Expected data layout

The code expects a `data/` directory rooted at the repository.

A practical layout is:

```text
data/
├── raw/
│   ├── hashtags/
│   ├── posts/
│   │   ├── posts_new.json
│   │   └── postsFinal.json
│   └── users/
│       ├── users_new.json
│       ├── usersFinal.json
│       └── texts/
└── processed/
    └── datasets/
```

The important thing is that downstream steps expect the following files:

- `data/raw/posts/postsFinal.json`
- `data/raw/users/usersFinal.json`

The collection pipeline writes:

- `data/raw/posts/posts_new.json`
- `data/raw/users/users_new.json`

So after collection, you may need to rename or copy the files:

```bash
cp data/raw/posts/posts_new.json data/raw/posts/postsFinal.json
cp data/raw/users/users_new.json data/raw/users/usersFinal.json
```

---

## Full pipeline: how to reproduce from scratch

## 1. Collect raw data

Run:

```bash
uv run python -m src.collect.runner
```

The collection config is defined in `src/config/collect.py`.

The current collection configuration uses:

- hashtags:
  - `AI`
  - `Anime`
  - `BlackHistoryMonth`
  - `Booksky`
  - `Gaza`
  - `ICE`
  - `Pokemon`
  - `Superbowl`
  - `TheTraitors`
  - `Trump`
- date range:
  - start: `2026-01-15T00:00:00Z`
  - end: `2026-03-02T00:00:00Z`
- hashtag collection limits:
  - minimum 9000 posts
  - maximum 11000 posts

If you change any of those settings, you are no longer reproducing the same collection setup.

### Notes

- This step depends on Bluesky access and whatever credentials or API setup the collection code requires.
- If Bluesky content has changed since the original run, your collected data may differ.

---

## 2. Prepare the filenames expected downstream

After collection, make sure the downstream filenames exist:

```bash
cp data/raw/posts/posts_new.json data/raw/posts/postsFinal.json
cp data/raw/users/users_new.json data/raw/users/usersFinal.json
```

This matters because the dataset and processing configs are wired to the `*Final.json` filenames.

---

## 3. Run feature/text processing

Run:

```bash
uv run python -m src.process.runner
```

This stage reads the collected post and user JSON files and generates the processed text features needed by later steps.

Run this only after `postsFinal.json` and `usersFinal.json` are in place.

---

## 4. Build datasets

The repository defines **eight dataset configs**.

Run them with:

```bash
uv run python -m src.dataset.runner --config <CONFIG_NAME>
```

### Available dataset configs

| Config name | Description | Output file |
|---|---|---|
| `USER_1TO1` | User-only dataset, 1:1 class ratio | `dataset_user_1to1.csv` |
| `USER_1TO5` | User-only dataset, 1:5 class ratio | `dataset_user_1to5.csv` |
| `MESSAGE_1TO1` | Message-only dataset, 1:1 class ratio | `dataset_message_1to1.csv` |
| `MESSAGE_1TO5` | Message-only dataset, 1:5 class ratio | `dataset_message_1to5.csv` |
| `HYBRID_1TO1` | Hybrid dataset, 1:1 class ratio | `dataset_hybrid_1to1.csv` |
| `HYBRID_1TO5` | Hybrid dataset, 1:5 class ratio | `dataset_hybrid_1to5.csv` |
| `BERT_1TO1` | BERT dataset, 1:1 class ratio | `dataset_bert_1to1.csv` |
| `BERT_1TO5` | BERT dataset, 1:5 class ratio | `dataset_bert_1to5.csv` |

### Rebuild all datasets

```bash
uv run python -m src.dataset.runner --config USER_1TO1
uv run python -m src.dataset.runner --config USER_1TO5
uv run python -m src.dataset.runner --config MESSAGE_1TO1
uv run python -m src.dataset.runner --config MESSAGE_1TO5
uv run python -m src.dataset.runner --config HYBRID_1TO1
uv run python -m src.dataset.runner --config HYBRID_1TO5
uv run python -m src.dataset.runner --config BERT_1TO1
uv run python -m src.dataset.runner --config BERT_1TO5
```

### Dataset defaults

The dataset configs use:

- `postsFinal.json`
- `usersFinal.json`
- `dataset_seed = 42`

If you want to reproduce the same datasets, keep those defaults unchanged.

---

## 5. Run the experiments

The repository defines **eight experiment configs**.

Run them with:

```bash
uv run python -m src.model.runner --config <CONFIG_NAME> --save y
```

### Available experiment configs

| Config name | Model | Dataset family | Ratio | Evaluation mode |
|---|---|---|---|---|
| `XGB_USER_1TO1` | XGBoost | User | 1:1 | `all` |
| `XGB_USER_1TO5` | XGBoost | User | 1:5 | `all` |
| `XGB_MESSAGE_1TO1` | XGBoost | Message | 1:1 | `all` |
| `XGB_MESSAGE_1TO5` | XGBoost | Message | 1:5 | `all` |
| `XGB_HYBRID_1TO1` | XGBoost | Hybrid | 1:1 | `all` |
| `XGB_HYBRID_1TO5` | XGBoost | Hybrid | 1:5 | `all` |
| `BERT_1TO1` | BERT | Message/BERT | 1:1 | `mixed` |
| `BERT_1TO5` | BERT | Message/BERT | 1:5 | `mixed` |

### Run all experiments

```bash
uv run python -m src.model.runner --config XGB_USER_1TO1 --save y
uv run python -m src.model.runner --config XGB_USER_1TO5 --save y
uv run python -m src.model.runner --config XGB_MESSAGE_1TO1 --save y
uv run python -m src.model.runner --config XGB_MESSAGE_1TO5 --save y
uv run python -m src.model.runner --config XGB_HYBRID_1TO1 --save y
uv run python -m src.model.runner --config XGB_HYBRID_1TO5 --save y
uv run python -m src.model.runner --config BERT_1TO1 --save y
uv run python -m src.model.runner --config BERT_1TO5 --save y
```

---

## Evaluation modes

The code supports the following evaluation settings:

- `mixed` — a standard mixed split
- `id` — in-distribution evaluation
- `ood` — out-of-distribution evaluation
- `all` — run all of the above

In the provided experiment configs:

- all **XGBoost** experiments use `all`
- all **BERT** experiments use `mixed`

So if you are trying to match the saved outputs exactly, do not change those config definitions.

---

## Where results are saved

Saved results are written under `results/`.

The important output locations are:

```text
results/
├── xgb/
│   └── feature_analysis/
└── Bert/
```

In practice:

- XGBoost experiments save JSON result files under `results/xgb/`
- XGBoost feature importance/gain analysis is saved under `results/xgb/feature_analysis/`
- BERT experiment results are saved under `results/Bert/`

---

## Recommended reproduction workflow

If your goal is to reproduce the model results as closely as possible, use this order.

### A. Set up the environment

```bash
uv sync
```

### B. If needed, recollect the raw data

```bash
uv run python -m src.collect.runner
cp data/raw/posts/posts_new.json data/raw/posts/postsFinal.json
cp data/raw/users/users_new.json data/raw/users/usersFinal.json
```

### C. Process text features

```bash
uv run python -m src.process.runner
```

### D. Build the datasets

```bash
uv run python -m src.dataset.runner --config USER_1TO1
uv run python -m src.dataset.runner --config USER_1TO5
uv run python -m src.dataset.runner --config MESSAGE_1TO1
uv run python -m src.dataset.runner --config MESSAGE_1TO5
uv run python -m src.dataset.runner --config HYBRID_1TO1
uv run python -m src.dataset.runner --config HYBRID_1TO5
uv run python -m src.dataset.runner --config BERT_1TO1
uv run python -m src.dataset.runner --config BERT_1TO5
```

### E. Run the models

```bash
uv run python -m src.model.runner --config XGB_USER_1TO1 --save y
uv run python -m src.model.runner --config XGB_USER_1TO5 --save y
uv run python -m src.model.runner --config XGB_MESSAGE_1TO1 --save y
uv run python -m src.model.runner --config XGB_MESSAGE_1TO5 --save y
uv run python -m src.model.runner --config XGB_HYBRID_1TO1 --save y
uv run python -m src.model.runner --config XGB_HYBRID_1TO5 --save y
uv run python -m src.model.runner --config BERT_1TO1 --save y
uv run python -m src.model.runner --config BERT_1TO5 --save y
```

---

## Fast path: reproduce the saved experiment outputs

If you already have the original raw and processed data available locally, you do **not** need to rerun collection.

Instead:

```bash
uv sync
uv run python -m src.process.runner
uv run python -m src.dataset.runner --config USER_1TO1
uv run python -m src.dataset.runner --config USER_1TO5
uv run python -m src.dataset.runner --config MESSAGE_1TO1
uv run python -m src.dataset.runner --config MESSAGE_1TO5
uv run python -m src.dataset.runner --config HYBRID_1TO1
uv run python -m src.dataset.runner --config HYBRID_1TO5
uv run python -m src.dataset.runner --config BERT_1TO1
uv run python -m src.dataset.runner --config BERT_1TO5
uv run python -m src.model.runner --config XGB_USER_1TO1 --save y
uv run python -m src.model.runner --config XGB_USER_1TO5 --save y
uv run python -m src.model.runner --config XGB_MESSAGE_1TO1 --save y
uv run python -m src.model.runner --config XGB_MESSAGE_1TO5 --save y
uv run python -m src.model.runner --config XGB_HYBRID_1TO1 --save y
uv run python -m src.model.runner --config XGB_HYBRID_1TO5 --save y
uv run python -m src.model.runner --config BERT_1TO1 --save y
uv run python -m src.model.runner --config BERT_1TO5 --save y
```

---

## What to compare when checking reproduction

To verify that your run matches the project outputs, compare the following.

### Dataset-level checks

Check that:

- the expected CSV files are created under `data/processed/datasets/`
- the class ratio matches the config (`1:1` or `1:5`)
- the expected identifier and label columns are present
- feature columns align with `feature_names.txt`

### Model-level checks

Check that:

- result JSON files are written to the expected `results/` subdirectory
- XGBoost runs contain outputs for the `mixed`, `id`, and `ood` evaluations
- BERT runs contain the expected `mixed` evaluation output
- feature gain outputs are written for the XGBoost runs that enable them

### Environment checks

Check that:

- you are running from the repository root
- dependencies match the locked environment as closely as possible
- you did not change the config definitions

---

## Reproducibility caveats

### 1. Live Bluesky data changes
The collection pipeline depends on live platform data. If posts disappear, users change behavior, or timelines shift, recollected data will differ.

### 2. Configs are part of the experiment definition
The configs under `src/config/` are not just convenience wrappers. They define the actual datasets, ratios, paths, hashtags, and evaluation modes used by the experiments.

### 3. File naming matters
The pipeline expects the `postsFinal.json` and `usersFinal.json` filenames in later stages, even though collection writes `posts_new.json` and `users_new.json`.

### 4. Exact metric reproduction depends on exact data snapshots
If you need exact numerical agreement with earlier runs, reuse the original intermediate data rather than recollecting from the platform.

---

## Troubleshooting

### `ModuleNotFoundError` or import errors
Run commands from the repository root and ensure the environment is installed correctly.

### Missing `postsFinal.json` or `usersFinal.json`
Copy the collected files into the expected names:

```bash
cp data/raw/posts/posts_new.json data/raw/posts/postsFinal.json
cp data/raw/users/users_new.json data/raw/users/usersFinal.json
```

### Dataset config not recognized
Make sure you are using one of the valid dataset config names listed above.

### Experiment config not recognized
Make sure you are using one of the valid experiment config names listed above.

### Results differ from earlier saved outputs
The most common reasons are:

- different collected data
- changed config values
- different dependency versions
- mismatched dataset files
- running from a different class ratio or evaluation mode

---

## Feature reference

The file `feature_names.txt` contains the feature glossary used in the project.

Use it to interpret:

- dataset columns
- engineered feature groups
- model feature importance outputs

---

## Minimal reproduction commands

If you want the shortest complete pipeline reference, use this:

```bash
# Setup
uv sync

# Collect raw data
uv run python -m src.collect.runner
cp data/raw/posts/posts_new.json data/raw/posts/postsFinal.json
cp data/raw/users/users_new.json data/raw/users/usersFinal.json

# Process
uv run python -m src.process.runner

# Build datasets
uv run python -m src.dataset.runner --config USER_1TO1
uv run python -m src.dataset.runner --config USER_1TO5
uv run python -m src.dataset.runner --config MESSAGE_1TO1
uv run python -m src.dataset.runner --config MESSAGE_1TO5
uv run python -m src.dataset.runner --config HYBRID_1TO1
uv run python -m src.dataset.runner --config HYBRID_1TO5
uv run python -m src.dataset.runner --config BERT_1TO1
uv run python -m src.dataset.runner --config BERT_1TO5

# Run experiments
uv run python -m src.model.runner --config XGB_USER_1TO1 --save y
uv run python -m src.model.runner --config XGB_USER_1TO5 --save y
uv run python -m src.model.runner --config XGB_MESSAGE_1TO1 --save y
uv run python -m src.model.runner --config XGB_MESSAGE_1TO5 --save y
uv run python -m src.model.runner --config XGB_HYBRID_1TO1 --save y
uv run python -m src.model.runner --config XGB_HYBRID_1TO5 --save y
uv run python -m src.model.runner --config BERT_1TO1 --save y
uv run python -m src.model.runner --config BERT_1TO5 --save y
```

---

## Summary

To reproduce the results in this repository:

1. install the locked environment
2. collect or reuse the raw data
3. make sure the expected `postsFinal.json` and `usersFinal.json` files exist
4. run the processing stage
5. build the datasets using the named dataset configs
6. run the experiments using the named experiment configs
7. compare your outputs with the files already present in `results/`

If you keep the environment, data snapshot, and config definitions fixed, the pipeline is straightforward to rerun from the `src/` entry points.

