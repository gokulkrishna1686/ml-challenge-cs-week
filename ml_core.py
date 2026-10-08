"""Organizer-only row-wise features and selected fold-local model fits."""

from __future__ import annotations

import numpy as np
import pandas as pd

CLASSES = ("A", "B", "C", "D", "E")
CLASS_TO_INDEX = {label: index for index, label in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)
TARGET_COLUMN = "target_popularity"
ID_COLUMN = "id"

# Documented token ratios. A single corrupt training row reaches 1042, far
# outside the documented [0, 1] domain, so the domain is enforced explicitly.
_RATE_COLUMNS = ("n_unique_tokens", "n_non_stop_words", "n_non_stop_unique_tokens")

_LOG_COLUMNS = (
    "n_tokens_content",
    "num_keywords",
    "num_hrefs",
    "num_self_hrefs",
    "num_imgs",
    "num_videos",
    "kw_max_min",
    "kw_avg_min",
    "kw_min_max",
    "kw_max_max",
    "kw_avg_max",
    "kw_min_avg",
    "kw_max_avg",
    "kw_avg_avg",
    "self_reference_min_shares",
    "self_reference_max_shares",
    "self_reference_avg_sharess",
)


_ZERO_COUNT_COLUMNS = ("n_tokens_content", "num_hrefs", "num_imgs", "num_self_hrefs")

def _safe_div(numerator: pd.Series, denominator: pd.Series, eps: float = 1.0) -> pd.Series:
    """Divide while keeping a non-zero denominator, preserving NaN as NaN."""
    return numerator / (denominator + eps)


def build_features(frame: pd.DataFrame, kind: str = "v2") -> pd.DataFrame:
    """Build numeric features using only information in each individual row."""
    if kind not in {"raw", "v2"}:
        raise ValueError(f"Unknown feature kind: {kind!r}")

    out = frame.drop(columns=[ID_COLUMN, TARGET_COLUMN], errors="ignore").copy()

    for column in _RATE_COLUMNS:
        if column in out:
            out[column] = out[column].clip(0.0, 1.0)

    if kind == "raw":
        return out

    # Skewed count-like columns are compressed; log1p leaves NaN as NaN so the
    # gradient-boosted models keep using their native missing-value handling.
    for column in _LOG_COLUMNS:
        if column in out:
            out[f"log1p__{column}"] = np.log1p(out[column].clip(lower=0.0))

    # `kw_min_min` uses -1 as a documented sentinel in 58% of rows, which is a
    # category rather than a magnitude, so flag it instead of transforming it.
    if "kw_min_min" in out:
        out["kw_min_min_is_sentinel"] = (out["kw_min_min"] < 0).astype(np.int8)

    # Empty articles are a real subpopulation (2.9% of train), not errors.
    for column in _ZERO_COUNT_COLUMNS:
        if column in out:
            out[f"{column}_is_zero"] = (out[column] == 0).astype(np.int8)

    # Structural density ratios.
    ratios = (
        ("num_hrefs", "n_tokens_content", "hrefs_per_token"),
        ("num_self_hrefs", "num_hrefs", "self_hrefs_per_href"),
        ("num_imgs", "n_tokens_content", "imgs_per_token"),
        ("num_videos", "n_tokens_content", "videos_per_token"),
        ("num_keywords", "n_tokens_content", "keywords_per_token"),
        ("n_unique_tokens", "n_non_stop_unique_tokens", "unique_vs_nonstop"),
    )
    for numerator, denominator, name in ratios:
        if numerator in out and denominator in out:
            out[name] = _safe_div(out[numerator], out[denominator])

    # Keyword-statistic spreads: the shape of the distribution matters more than
    # any single order statistic.
    spreads = (
        ("kw_max_avg", "kw_min_avg", "kw_spread_avg"),
        ("kw_max_max", "kw_min_max", "kw_spread_max"),
        ("kw_max_min", "kw_min_min", "kw_spread_min"),
        ("kw_avg_avg", "kw_min_avg", "kw_avg_minus_min_avg"),
        ("kw_avg_avg", "kw_avg_max", "kw_avg_minus_avg_max"),
    )
    for high, low, name in spreads:
        if high in out and low in out:
            out[name] = out[high] - out[low]
    if {"kw_max_avg", "kw_min_avg"}.issubset(out):
        out["kw_avg_to_max_ratio"] = _safe_div(out["kw_avg_avg"], out["kw_max_avg"], eps=1e-6)

    # Referenced-article share dispersion.
    if {"self_reference_max_shares", "self_reference_min_shares"}.issubset(out):
        out["self_ref_spread"] = (
            out["self_reference_max_shares"] - out["self_reference_min_shares"]
        )
        out["self_ref_avg_to_max"] = _safe_div(
            out["self_reference_avg_sharess"], out["self_reference_max_shares"], eps=1e-6
        )

    # Latent topic distributions: 984 training rows do not sum to 1, so expose
    # the normalised shape rather than assuming a simplex.
    lda_columns = [c for c in out if c.startswith("LDA_")]
    if lda_columns:
        topic = out[lda_columns]
        total = topic.sum(axis=1)
        out["lda_sum"] = total
        normalised = topic.div(total.replace(0.0, np.nan), axis=0)
        out["lda_max"] = topic.max(axis=1)
        out["lda_min"] = topic.min(axis=1)
        out["lda_spread"] = out["lda_max"] - out["lda_min"]
        probabilities = normalised.to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["lda_entropy"] = -np.nansum(
                np.where(probabilities > 0, probabilities * np.log(probabilities), 0.0), axis=1
            )

    # Sentiment balance and polarity spans.
    if {"global_rate_positive_words", "global_rate_negative_words"}.issubset(out):
        out["global_pos_minus_neg"] = (
            out["global_rate_positive_words"] - out["global_rate_negative_words"]
        )
    if {"rate_positive_words", "rate_negative_words"}.issubset(out):
        # A larger epsilon than the other ratios: `rate_negative_words` reaches
        # exactly 0, and a near-zero denominator would otherwise create ratios
        # in the millions and dominate the tree splits.
        out["pos_neg_rate_ratio"] = _safe_div(
            out["rate_positive_words"], out["rate_negative_words"], eps=1e-3
        ).clip(upper=1000.0)
    spans = (
        ("max_positive_polarity", "min_positive_polarity", "positive_polarity_span"),
        ("max_negative_polarity", "min_negative_polarity", "negative_polarity_span"),
    )
    for high, low, name in spans:
        if high in out and low in out:
            out[name] = out[high] - out[low]

    # Indicator groups: the topic channel and weekday are one-hot blocks.
    channel_columns = [c for c in out if c.startswith("data_channel_is_")]
    weekday_columns = [c for c in out if c.startswith("weekday_is_")]
    if channel_columns:
        out["channel_count"] = out[channel_columns].sum(axis=1)
        out["channel_is_none"] = (out["channel_count"] == 0).astype(np.int8)
    # `weekday_count` is deliberately not created: exactly one weekday indicator
    # is set in every row, so the sum is a constant 1.0 and carries no signal.

    return out


def class_weight_vector(y: np.ndarray, mode: str = "none") -> np.ndarray | None:
    """Compute class weights from this training fold only."""
    if mode == "none":
        return None
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    if (counts == 0).any():
        raise ValueError("A class is missing from this fold; cannot weight.")
    base = len(y) / (N_CLASSES * counts)
    if mode == "balanced":
        return base
    if mode == "sqrt_balanced":
        return np.sqrt(base)
    raise ValueError(f"Unknown weight mode: {mode!r}")


def sample_weights(y: np.ndarray, mode: str) -> np.ndarray | None:
    """Expand per-class weights into per-row weights where supported."""
    weights = class_weight_vector(y, mode)
    if weights is None:
        return None
    return weights[y]


def fit_fold(
    model_name: str, X_train: pd.DataFrame, y_train: np.ndarray,
    X_valid: pd.DataFrame, y_valid: np.ndarray, X_test: pd.DataFrame,
    seed: int, weight_mode: str = "none", use_gpu: bool = False,
    params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit only on the training fold; validation is used for early stopping."""
    params = dict(params or {})

    def merge(base: dict, locked: set[str]) -> dict:
        return {**base, **{k: v for k, v in params.items() if k not in locked}}


    if model_name == "catboost":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            **merge(
                dict(
                    loss_function="MultiClass",
                    # Use a smooth metric for early stopping, score Macro F1 outside.
                    eval_metric="MultiClass",
                    iterations=4000,
                    learning_rate=0.03,
                    depth=8,
                    l2_leaf_reg=6.0,
                    random_strength=0.5,
                    bagging_temperature=0.5,
                    border_count=128,
                    class_weights=class_weight_vector(y_train, weight_mode),
                    random_seed=seed,
                    task_type="GPU" if use_gpu else "CPU",
                    devices="0" if use_gpu else None,
                    od_type="Iter",
                    od_wait=250,
                    allow_writing_files=False,
                    verbose=False,
                ),
                locked={"class_weights", "random_seed", "loss_function", "task_type"},
            )
        )
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=False)
        best_iteration = int(model.get_best_iteration() or 0)
        valid_probs = np.asarray(model.predict_proba(X_valid), dtype=np.float64)
        test_probs = np.asarray(model.predict_proba(X_test), dtype=np.float64)
        return valid_probs, test_probs, best_iteration

    if model_name == "lgbm":
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            **merge(
                dict(
                    objective="multiclass",
                    num_class=N_CLASSES,
                    n_estimators=4000,
                    learning_rate=0.03,
                    num_leaves=63,
                    min_child_samples=40,
                    subsample=0.85,
                    subsample_freq=1,
                    colsample_bytree=0.8,
                    reg_lambda=2.0,
                    reg_alpha=0.5,
                    max_bin=255,
                    # Built-in balancing uses the current fit-fold labels.
                    class_weight="balanced" if weight_mode == "balanced" else None,
                    random_state=seed,
                    n_jobs=-1,
                    verbose=-1,
                ),
                locked={"class_weight", "random_state", "objective", "num_class"},
            )
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="multi_logloss",
            sample_weight=(None if weight_mode == "balanced" else sample_weights(y_train, weight_mode)),
            callbacks=[lgb.early_stopping(250, verbose=False), lgb.log_evaluation(0)],
        )
        return (
            model.predict_proba(X_valid),
            model.predict_proba(X_test),
            int(model.best_iteration_ or 0),
        )

    raise ValueError(f"Unknown model: {model_name!r}")
