"""Train the selected ensemble with stratified, out-of-fold prediction storage.

Run from this directory: python train.py --data-dir dataset
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from ml_core import CLASS_TO_INDEX, CLASSES, TARGET_COLUMN, build_features, fit_fold

SEEDS = (42, 2026, 7, 1337)
N_SPLITS = 5
VARIANTS = {
    "catboost_bal": ("catboost", "balanced"),
    "catboost_sqrt": ("catboost", "sqrt_balanced"),
    "lgbm_bal": ("lgbm", "balanced"),
}


def fit_one_fold(variant: str, seed: int, fold: int, data_dir: Path) -> dict:
    """The shared local/Modal training unit; fit only the selected training fold."""
    train, test, _, y = load_data(data_dir)
    X = build_features(train, kind="v2")
    X_test = build_features(test, kind="v2")
    if list(X.columns) != list(X_test.columns):
        raise ValueError("Train/test feature mismatch")
    model, weights = VARIANTS[variant]
    splits = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    fit_idx, valid_idx = list(splits.split(X, y))[fold]
    valid, heldout, best_iteration = fit_fold(
        model_name=model,
        X_train=X.iloc[fit_idx], y_train=y[fit_idx],
        X_valid=X.iloc[valid_idx], y_valid=y[valid_idx],
        X_test=X_test, seed=seed * 1000 + fold,
        weight_mode=weights, use_gpu=False,
    )
    valid = np.asarray(valid, dtype=np.float32)
    heldout = np.asarray(heldout, dtype=np.float32)
    if valid.shape != (len(valid_idx), len(CLASSES)) or heldout.shape != (len(test), len(CLASSES)):
        raise ValueError("Model produced an unexpected probability shape")
    if not np.isfinite(valid).all() or not np.isfinite(heldout).all():
        raise ValueError("Model produced nonfinite probabilities")
    return dict(variant=variant, seed=seed, fold=fold, valid_index=valid_idx,
                valid_probs=valid, test_probs=heldout, best_iteration=best_iteration)


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    if not {"id", TARGET_COLUMN}.issubset(train.columns) or TARGET_COLUMN in test:
        raise ValueError("Unexpected training/test target schema")
    if list(sample.columns) != ["id", TARGET_COLUMN]:
        raise ValueError("Unexpected sample-submission columns")
    if not train.id.is_unique or not test.id.is_unique or set(train.id) & set(test.id):
        raise ValueError("IDs must be unique and disjoint")
    if not sample.id.equals(test.id):
        raise ValueError("Sample-submission IDs must match test order")
    if train[TARGET_COLUMN].isna().any() or not train[TARGET_COLUMN].isin(CLASSES).all():
        raise ValueError("Training target contains missing or unknown labels")
    y = train[TARGET_COLUMN].map(CLASS_TO_INDEX).to_numpy(dtype=int)
    if min(np.bincount(y)) < N_SPLITS:
        raise ValueError("Each class needs at least five training examples")
    return train, test, sample, y


def train_all(data_dir: Path, output: Path) -> None:
    train, test, _, y = load_data(data_dir)
    X = build_features(train, kind="v2")
    X_test = build_features(test, kind="v2")
    if list(X.columns) != list(X_test.columns) or not all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes):
        raise ValueError("Train/test feature columns or numeric dtypes do not match")
    if not np.isfinite(X.to_numpy(dtype=float)[~X.isna().to_numpy()]).all():
        raise ValueError("Nonfinite training feature")

    arrays: dict[str, np.ndarray] = {}
    for variant in VARIANTS:
        for seed in SEEDS:
            oof = np.zeros((len(train), len(CLASSES)), dtype=np.float32)
            test_probs = np.zeros((len(test), len(CLASSES)), dtype=np.float32)
            seen = np.zeros(len(train), dtype=bool)
            fold_scores = []
            for fold in range(N_SPLITS):
                result = fit_one_fold(variant, seed, fold, data_dir)
                valid_idx = result["valid_index"]
                valid = result["valid_probs"]
                heldout = result["test_probs"]
                if valid.shape != (len(valid_idx), len(CLASSES)) or heldout.shape != (len(test), len(CLASSES)):
                    raise ValueError("Model produced an unexpected probability shape")
                if not np.isfinite(valid).all() or not np.isfinite(heldout).all():
                    raise ValueError("Model produced nonfinite probabilities")
                if seen[valid_idx].any():
                    raise ValueError("Overlapping validation folds")
                seen[valid_idx] = True
                oof[valid_idx] = valid
                test_probs += heldout / N_SPLITS
                score = f1_score(y[valid_idx], valid.argmax(axis=1), labels=range(len(CLASSES)), average="macro", zero_division=0)
                fold_scores.append(score)
                print(f"{variant} seed={seed} fold={fold} Macro F1={score:.5f}", flush=True)
            if not seen.all():
                raise ValueError("Incomplete out-of-fold coverage")
            arrays[f"{variant}__{seed}__oof"] = oof
            arrays[f"{variant}__{seed}__test"] = test_probs
            print(f"{variant} seed={seed} mean fold Macro F1={np.mean(fold_scores):.5f} (sd={np.std(fold_scores):.5f})")
            print("Per-class F1:", f1_score(y, oof.argmax(axis=1), labels=range(len(CLASSES)), average=None, zero_division=0))
            print("Confusion matrix:\n", confusion_matrix(y, oof.argmax(axis=1), labels=range(len(CLASSES))))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    print(f"Probability archive: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/probabilities.npz"))
    args = parser.parse_args()
    train_all(args.data_dir, args.output)
