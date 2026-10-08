# Online News Popularity Prediction

An end-to-end multiclass machine-learning solution for the **ML Challenge CS_CLUB** competition.

The task is deceptively difficult: predict one of five popularity classes (`A`-`E`) from 59 numeric article features, while being evaluated with **Macro F1**. The target is highly imbalanced, the signal is weak, and mistakes on rare classes matter just as much as mistakes on the majority class.

This repository turns that problem into a reproducible pipeline built around:

- leakage-safe feature engineering;
- repeated stratified cross-validation;
- class-aware CatBoost and LightGBM models;
- out-of-fold probability ensembling;
- Macro-F1-specific decision calibration;
- strict submission integrity checks.

## Why This Is Hard

The training labels are strongly imbalanced:

| Class | Training rows | Share |
|---|---:|---:|
| A | 1,715 | 4.6% |
| B | 3,945 | 10.6% |
| C | 7,230 | 19.5% |
| D | 13,507 | 36.4% |
| E | 10,747 | 28.9% |

An ordinary classifier can achieve reasonable accuracy by concentrating on `D` and `E`, but that performs poorly under Macro F1. Macro F1 computes an F1 score for every class and averages them equally, so the rare `A` class is just as important as the majority `D` class.

The strongest individual feature has only modest correlation with the target. There is no single shortcut feature, so performance comes from combining many weak signals and making better final classification decisions.

## Solution Overview

```text
Organizer CSVs
      |
      v
Schema and integrity checks
      |
      v
106 leakage-safe numeric features
      |
      v
3 complementary boosted-tree configurations
      |
      v
4 seeds x 5 stratified folds
      |
      v
Out-of-fold and test probabilities
      |
      v
OOF-selected probability blend
      |
      v
Nested Macro-F1 decision calibration
      |
      v
Validated submission.csv + report.json
```

## 1. Leakage-Safe Feature Engineering

`ml_core.py` converts the 59 organizer features into 106 model inputs. Every transformation is row-wise and stateless, which means a validation or test row cannot learn anything from another row.

The feature set includes:

- logarithmic compression for heavily skewed count features;
- link, image, video, and keyword density ratios;
- keyword minimum/average/maximum spreads;
- explicit handling of the documented `kw_min_min = -1` sentinel;
- empty-content and zero-count indicators;
- normalized latent-topic summaries;
- topic entropy, range, minimum, and maximum;
- positive-versus-negative sentiment balance;
- polarity spans and reference-share dispersion;
- channel-presence indicators.

The pipeline also handles real data defects rather than silently trusting every value:

- documented rate columns are clipped to their valid range;
- missing values remain available to models with native missing-value handling;
- `id` is excluded from training;
- the target column is explicitly removed from every feature frame;
- train and test feature columns are checked for exact parity.

## 2. Models Chosen for Diversity

The final ensemble uses three configurations:

| Variant | Model | Class strategy | Purpose |
|---|---|---|---|
| `catboost_bal` | CatBoost | Inverse-frequency balanced | Strong rare-class recall |
| `catboost_sqrt` | CatBoost | Square-root balanced | Better precision/recall compromise |
| `lgbm_bal` | LightGBM | Balanced | Different tree-growth behavior and errors |

The two CatBoost variants are intentionally not identical. Full inverse-frequency weighting pushes harder toward rare classes, while square-root weighting is less aggressive. LightGBM contributes a different boosting implementation and split strategy.

This diversity is important: the ensemble improves because its models make different mistakes, not because the same model is copied several times.

## 3. Repeated Out-of-Fold Validation

The pipeline uses:

- **5 stratified folds** per run;
- **4 independent split seeds**: `42`, `2026`, `7`, and `1337`;
- **3 model configurations**;
- **60 total fold fits**.

For every fold:

1. The model fits only the other four folds.
2. The held-out fold is used for early stopping and probability prediction.
3. The same fitted model predicts the competition test set.
4. Held-out predictions are written into an OOF matrix.
5. Test probabilities are averaged across folds and seeds.

This produces a prediction for every training row from a model whose trees did not fit that row. These OOF probabilities are then used for model comparison, blending, and final decision calibration.

## 4. Probability Blending

The models are combined at the probability level instead of voting on hard labels.

The notebook searches a compact simplex of blend weights and scores every candidate with OOF Macro F1. Test probabilities are not used to select the blend.

Probability blending preserves confidence information. For example, a model predicting `[0.35, 0.34, 0.20, ...]` is treated differently from one predicting `[0.90, 0.05, 0.03, ...]`, even when both have the same argmax label.

## 5. Macro-F1 Decision Calibration

The class with the largest raw probability is not necessarily the decision rule that maximizes Macro F1.

The notebook applies bounded per-class multipliers:

```python
prediction = argmax(probabilities * class_multipliers)
```

These multipliers are optimized using OOF predictions. A second nested stratified check measures whether calibration improves unseen OOF partitions. Calibration is accepted only when most held-out partitions improve.

This creates an important separation:

- models learn probabilities using multiclass log loss;
- the final decision rule is tuned for the competition's actual Macro F1 metric.

## 6. Optional Balanced-Test Assignment

`solution.ipynb` includes an optional Hungarian assignment step for competitions where the organizer explicitly guarantees equal hidden-test class counts.

Instead of independently taking the argmax for every row, it solves a global maximum-likelihood assignment under exact class quotas.

```python
TEST_IS_KNOWN_BALANCED = False
```

This remains disabled by default. It should only be enabled when class balance is known from the competition design, never inferred from leaderboard feedback.

## Cloud Execution with Modal

Sixty boosted-tree fits are independent, so they can be distributed across cloud workers. `modal_verify.py` fans fold jobs out to Modal CPU containers, checkpoints completed folds, combines the resulting probability matrices, and runs the same submission validator.

High-end GPUs such as H100s were considered during experimentation, but CPU workers were the better engineering choice for this modest tabular dataset. The committed reproducibility runner therefore uses parallel CPU execution rather than making an unsupported GPU-speed claim. Parallelism still turns a long serial experiment into many simultaneous fold jobs.

Modal is optional. The complete pipeline also runs locally, although all 60 fits require more time.

## Repository Structure

| File | Purpose |
|---|---|
| `solution.ipynb` | Complete competition workflow, explanation, training, blending, calibration, and submission generation |
| `ml_core.py` | Feature engineering, class weights, CatBoost, and LightGBM fitting |
| `train.py` | Repeated stratified CV and probability archive generation |
| `predict.py` | Probability blending, final decision rule, and strict CSV validation |
| `modal_verify.py` | Optional parallel Modal execution with fold checkpointing |
| `requirements.txt` | Pinned Python environment |
| `.gitignore` | Excludes organizer data, predictions, credentials, caches, and environments |

## Reproduce the Pipeline

### 1. Create the environment

Python 3.12 is recommended.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

LightGBM requires an OpenMP runtime:

```bash
# macOS
brew install libomp

# Debian or Ubuntu
sudo apt-get install libgomp1
```

### 2. Add the organizer files

Create `dataset/` and place the three competition files inside it:

```text
dataset/
  train.csv
  test.csv
  sample_submission.csv
```

Expected shapes:

```text
train.csv              37,144 x 61
test.csv                2,500 x 60
sample_submission.csv   2,500 x 2
```

The data is intentionally excluded from Git.

### 3. Run locally

```bash
python train.py \
  --data-dir dataset \
  --output artifacts/probabilities.npz

python predict.py \
  --data-dir dataset \
  --archive artifacts/probabilities.npz \
  --output submissions/submission.csv
```

Alternatively, run `solution.ipynb` from top to bottom for the fully explained workflow, including OOF blend selection and nested decision calibration.

### 4. Run folds in parallel on Modal

```bash
python -m pip install modal==1.6.1
modal token set
modal run modal_verify.py
```

Interrupted runs can resume from saved fold checkpoints:

```bash
modal run modal_verify.py --resume artifacts/modal_run
```

Cloud execution may incur charges.

## Output Validation

The submission is written only after checking all of the following:

- exactly two columns: `id,target_popularity`;
- exactly 2,500 predictions;
- IDs match `test.csv` and `sample_submission.csv` in the original order;
- every ID is unique;
- no prediction is missing;
- every label belongs to `{A, B, C, D, E}`;
- all probability matrices have the expected shape;
- probabilities are finite, non-negative, and normalized.

The scripts do not upload anything automatically.

## Reproducible Result

The selected historical ensemble achieved approximately **0.3305 selection OOF Macro F1**. A clean rerun of the simplified public pipeline produced approximately **0.3304**. Small differences can occur across operating systems and numerical-library versions.

This is the result supported by the code in this repository. OOF performance and a hidden Kaggle score are measurements on different samples and should not be presented as interchangeable.

## Engineering Principles

- Use only organizer-provided data.
- Never fit preprocessing on validation or test labels.
- Optimize the metric the competition actually evaluates.
- Prefer diverse models over duplicated models.
- Validate improvements across multiple split seeds.
- Treat probability prediction and final decision-making as separate stages.
- Keep datasets, predictions, credentials, and fitted artifacts out of Git.
- Make every reported result traceable to code and OOF evidence.

## Team

**Jailer 3**

Yashvanth S and Gokul Krishna Balaji
