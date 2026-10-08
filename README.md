# Online News Popularity Prediction

**Jailer 3** · Yashvanth S and Gokul Krishna Balaji

This repository contains the training and inference code for our ML Challenge CS_CLUB entry. It uses only the CSV files provided by the organizers. No external datasets, pretrained models, private labels, or Kaggle prediction files are used.

## Approach

We predict `target_popularity` for each article with a blend of CatBoost and LightGBM classifiers. Row-wise feature construction adds a small set of count, keyword, topic, and sentiment summaries to the supplied numeric features; `id` and the target are never model inputs. Five-fold shuffled stratified cross-validation generates out-of-fold (OOF) training probabilities and fold-averaged test probabilities for each model. Multiple fixed split seeds reduce sensitivity to one partition. Class weights are computed using each training fold only. The final blend and its class decision rule were chosen from training-only OOF experiments; the test labels were never available.

We use **Macro F1** for validation, with per-class F1 and confusion matrices printed by the training script.

## Setup

Use Python 3.12 and install the dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

LightGBM needs an OpenMP runtime. On macOS, install `libomp` through Homebrew if LightGBM reports a missing `libomp.dylib`; on Debian/Ubuntu install `libgomp1`. The pinned Python packages match the training environment. CPU training is supported; allow time for all CV fits.

Place the organizer's files in `dataset/` without changing their row order or names:

```text
dataset/
  train.csv
  test.csv
  sample_submission.csv
```

These CSV files are ignored by Git and are **not** included in this repository.

## Train and generate the CSV

From this repository's root, run:

```bash
python train.py --data-dir dataset --output artifacts/probabilities.npz
python predict.py --data-dir dataset --archive artifacts/probabilities.npz --output submissions/submission.csv
```

`train.py` trains the selected classifiers on organizer training folds and writes OOF and test probability arrays. `predict.py` combines the saved arrays and validates that the output has exactly `id,target_popularity`, one valid `A`–`E` label per test ID, and the same ID order as both `test.csv` and `sample_submission.csv`. The generated CSV and probability archive remain local and are excluded from Git.

The source is organized as `ml_core.py` (features, fold-local weights, selected model fits), `train.py` (CV and diagnostics), and `predict.py` (ensemble and CSV validation). `solution.ipynb` is a notebook entry point that calls those same scripts. Fixed seeds, feature transformations, hyperparameters, and the selected decision rule are present in the code. A fresh run can differ slightly across CPU libraries and platforms; verify the output before using the competition's single submission opportunity.

For optional parallel CPU training on Modal, install `modal==1.6.1` in the environment, configure your own Modal profile locally (never commit credentials), and run `modal run modal_verify.py` from this directory. This uses the same `fit_one_fold` model definition as `train.py`, checkpoints each finished fold under `artifacts/modal_run/`, and runs the same CSV validation at the end. Resume an interrupted job with `modal run modal_verify.py --resume artifacts/modal_run`. Cloud usage incurs charges; the standard local commands above require no Modal account.

**Submission and repository registration:** The Team Leader alone should upload the checked CSV to Kaggle once, then provide the public repository URL through the organizers' Google Form. Running these scripts does neither automatically.
