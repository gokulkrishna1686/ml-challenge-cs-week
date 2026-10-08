"""Optional Modal CPU fan-out using the same training unit as train.py.

Run from this repository root: modal run modal_verify.py
Interrupted jobs can resume with --resume artifacts/modal_run.
Data and generated predictions are mounted locally; none are committed to Git.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgomp1")
    .pip_install(
        "numpy==2.5.3", "pandas==3.0.6", "scikit-learn==1.9.1",
        "catboost==1.2.10", "lightgbm==4.7.0",
    )
    .add_local_dir(DATA, remote_path="/data", copy=True)
    .add_local_python_source("ml_core", "train")
)
app = modal.App("news-popularity-repo-verification")


@app.function(image=image, cpu=4, memory=8192, timeout=45 * 60, retries=1)
def remote_fold(variant: str, seed: int, fold: int) -> dict:
    from train import fit_one_fold

    return fit_one_fold(variant, seed, fold, Path("/data"))


@app.local_entrypoint()
def main(resume: str = "", batch_size: int = 8) -> None:
    import numpy as np
    from sklearn.metrics import confusion_matrix, f1_score

    from train import N_SPLITS, SEEDS, VARIANTS, load_data

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    train, test, _, y = load_data(DATA)
    run_dir = Path(resume) if resume else ROOT / "artifacts" / "modal_run"
    fold_dir = run_dir / "folds"
    fold_dir.mkdir(parents=True, exist_ok=True)
    specs = [(variant, seed, fold) for variant in VARIANTS for seed in SEEDS for fold in range(N_SPLITS)]

    def file_for(spec: tuple[str, int, int]) -> Path:
        return fold_dir / f"{spec[0]}__{spec[1]}__{spec[2]}.npz"

    pending = [spec for spec in specs if not file_for(spec).exists()]
    print(f"Saved: {len(specs) - len(pending)}; remaining: {len(pending)}", flush=True)
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        for result in remote_fold.starmap(batch, order_outputs=False):
            spec = result["variant"], result["seed"], result["fold"]
            np.savez_compressed(
                file_for(spec), valid_index=result["valid_index"],
                valid_probs=result["valid_probs"], test_probs=result["test_probs"],
                best_iteration=result["best_iteration"],
            )
            print(f"Saved {spec}", flush=True)

    archive: dict[str, np.ndarray] = {}
    diagnostics = {}
    for variant in VARIANTS:
        for seed in SEEDS:
            oof = np.zeros((len(train), 5), dtype=np.float32)
            heldout = np.zeros((len(test), 5), dtype=np.float32)
            seen = np.zeros(len(train), dtype=bool)
            fold_scores = []
            for fold in range(N_SPLITS):
                with np.load(file_for((variant, seed, fold)), allow_pickle=False) as saved:
                    index = saved["valid_index"]
                    if seen[index].any() or len(index) != len(saved["valid_probs"]):
                        raise ValueError("Overlapping or invalid validation fold")
                    seen[index] = True
                    oof[index] = saved["valid_probs"]
                    heldout += saved["test_probs"] / N_SPLITS
                    fold_scores.append(float(f1_score(
                        y[index], saved["valid_probs"].argmax(axis=1),
                        labels=range(5), average="macro", zero_division=0,
                    )))
            if not seen.all():
                raise ValueError("Incomplete OOF coverage")
            archive[f"{variant}__{seed}__oof"] = oof
            archive[f"{variant}__{seed}__test"] = heldout
            diagnostics[f"{variant}__{seed}"] = dict(
                fold_macro_f1=fold_scores,
                mean_fold_macro_f1=float(np.mean(fold_scores)),
                per_class_f1=f1_score(y, oof.argmax(axis=1), labels=range(5),
                                      average=None, zero_division=0).tolist(),
                confusion_matrix=confusion_matrix(y, oof.argmax(axis=1), labels=range(5)).tolist(),
            )
            print(f"{variant} seed={seed} fold-mean Macro F1={np.mean(fold_scores):.5f}")

    output = run_dir / "probabilities.npz"
    np.savez_compressed(output, **archive)
    (run_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(f"Validated fold archive: {output}")
    from predict import create_submission

    create_submission(DATA, [output], ROOT / "submissions" / "submission.csv")
