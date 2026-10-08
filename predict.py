"""Create the validated-format submission from saved OOF and test probabilities."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ml_core import CLASSES, TARGET_COLUMN
from train import SEEDS, VARIANTS, load_data

# The fixed ensemble and final decision rule were chosen using training-only
# out-of-fold predictions; no hidden test labels are used here.
CLASS_MULTIPLIERS = np.array([1.0, 1.0, 1.0, 1.0, 0.85])


def create_submission(data_dir: Path, archives: list[Path], output: Path) -> pd.DataFrame:
    train, test, sample, y = load_data(data_dir)
    arrays: dict[str, np.ndarray] = {}
    for archive_path in archives:
        with np.load(archive_path, allow_pickle=False) as archive:
            for key in archive.files:
                if key in arrays:
                    raise ValueError(f"Duplicate probability key: {key}")
                arrays[key] = archive[key]

    blended_oof = np.zeros((len(train), len(CLASSES)), dtype=np.float64)
    blended_test = np.zeros((len(test), len(CLASSES)), dtype=np.float64)
    for variant in VARIANTS:
        for kind, target, nrows in (
            ("oof", blended_oof, len(train)),
            ("test", blended_test, len(test)),
        ):
            matrices = []
            for seed in SEEDS:
                key = f"{variant}__{seed}__{kind}"
                if key not in arrays:
                    raise ValueError(f"Missing probability matrix: {key}")
                matrix = arrays[key]
                if matrix.shape != (nrows, len(CLASSES)) or not np.isfinite(matrix).all():
                    raise ValueError(f"Invalid probability matrix: {key}")
                if (matrix < -1e-6).any() or not np.allclose(matrix.sum(axis=1), 1, atol=2e-4):
                    raise ValueError(f"Non-normalized probability matrix: {key}")
                matrices.append(matrix)
            target += np.mean(matrices, axis=0) / len(VARIANTS)

    print("OOF Macro F1:", f1_score(y, (blended_oof * CLASS_MULTIPLIERS).argmax(axis=1), labels=range(len(CLASSES)), average="macro", zero_division=0))
    indices = (blended_test * CLASS_MULTIPLIERS).argmax(axis=1)
    submission = pd.DataFrame({"id": test.id.to_numpy(), TARGET_COLUMN: np.asarray(CLASSES)[indices]})
    if list(submission.columns) != ["id", TARGET_COLUMN] or len(submission) != len(test):
        raise ValueError("Invalid submission schema")
    if not submission.id.is_unique or not submission.id.equals(test.id) or not submission.id.equals(sample.id):
        raise ValueError("Submission IDs differ from the competition files")
    if submission[TARGET_COLUMN].isna().any() or not submission[TARGET_COLUMN].isin(CLASSES).all():
        raise ValueError("Missing or invalid prediction labels")
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    print(f"Validated submission: {output}")
    return submission


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--archive", action="append", required=True, type=Path,
                        help="Probability archive; repeat to combine separate seed runs")
    parser.add_argument("--output", type=Path, default=Path("submissions/submission.csv"))
    args = parser.parse_args()
    create_submission(args.data_dir, args.archive, args.output)
