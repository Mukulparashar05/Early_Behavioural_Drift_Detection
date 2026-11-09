"""Evaluate a reproducible baseline on the NASA C-MAPSS RUL benchmark.

This is intentionally separate from the AI4I anomaly detector: C-MAPSS is a
run-to-failure time-series/RUL task, not a row-level failure classification
dataset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "archive"
OUTPUT = ROOT / "outputs" / "metrics" / "nasa_cmapps_results.json"
NAMES = ["unit", "cycle", "setting_1", "setting_2", "setting_3"] + [f"sensor_{i}" for i in range(1, 22)]


def load_matrix(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text().splitlines():
        values = line.split()
        if values:
            if len(values) != 26:
                raise ValueError(f"{path.name} contains a row with {len(values)} columns; expected 26")
            rows.append([float(value) for value in values])
    if not rows:
        raise ValueError(f"{path.name} is empty")
    return np.asarray(rows, dtype=float)


def load_rul(path: Path) -> np.ndarray:
    values = [float(line.strip()) for line in path.read_text().splitlines() if line.strip()]
    if not values:
        raise ValueError(f"{path.name} is empty")
    return np.asarray(values, dtype=float)


def ridge_fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, alpha: float = 10.0) -> np.ndarray:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x = (train_x - mean) / scale
    xt = (test_x - mean) / scale
    design = np.column_stack([np.ones(len(x)), x])
    test_design = np.column_stack([np.ones(len(xt)), xt])
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    # Solve the regularized least-squares system directly; normal equations are
    # needlessly ill-conditioned for the highly correlated C-MAPSS sensors.
    regularizer = np.sqrt(alpha) * np.eye(design.shape[1] - 1)
    augmented_x = np.vstack([design, np.column_stack([np.zeros(len(regularizer)), regularizer])])
    augmented_y = np.concatenate([train_y, np.zeros(len(regularizer))])
    weights = np.linalg.lstsq(augmented_x, augmented_y, rcond=1e-10)[0]
    predictions = np.maximum(0.0, np.einsum("ij,j->i", test_design, weights))
    if not np.isfinite(predictions).all():
        raise FloatingPointError("NASA RUL baseline produced non-finite predictions")
    return predictions


def nasa_score(errors: np.ndarray) -> float:
    score = 0.0
    for error in errors:
        score += np.exp(-error / 13.0) - 1.0 if error < 0 else np.exp(error / 10.0) - 1.0
    return float(score)


def evaluate_subset(subset: str, alpha: float = 10.0) -> Dict[str, float]:
    train = load_matrix(ARCHIVE / f"train_{subset}.txt")
    test = load_matrix(ARCHIVE / f"test_{subset}.txt")
    actual_rul = load_rul(ARCHIVE / f"RUL_{subset}.txt")
    units = np.unique(train[:, 0]).astype(int)
    if len(actual_rul) != len(np.unique(test[:, 0])):
        raise ValueError(f"{subset}: RUL count does not match test engine count")
    max_cycles = {unit: train[train[:, 0] == unit, 1].max() for unit in units}
    train_y = np.minimum(125.0, np.asarray([max_cycles[int(unit)] - cycle for unit, cycle in train[:, :2]]))
    # Exclude identifiers; retain operating settings, cycle, and sensor signals.
    train_x = train[:, 1:]
    final_rows = np.asarray([test[test[:, 0] == unit][-1] for unit in np.unique(test[:, 0])])
    predictions = ridge_fit_predict(train_x, train_y, final_rows[:, 1:], alpha)
    errors = predictions - actual_rul
    return {
        "subset": subset,
        "train_engines": int(len(units)),
        "test_engines": int(len(actual_rul)),
        "train_cycles": int(len(train)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "nasa_score": nasa_score(errors),
        "mean_signed_error": float(np.mean(errors)),
        "predictions": predictions.round(4).tolist(),
        "actual_rul": actual_rul.tolist(),
    }


def run(subsets: Tuple[str, ...] = ("FD001", "FD002", "FD003", "FD004")) -> Dict[str, object]:
    results = [evaluate_subset(subset) for subset in subsets]
    summary = {
        "task": "Remaining useful life regression",
        "dataset": "NASA C-MAPSS",
        "subsets": results,
        "limitations": [
            "The baseline is a regularized linear model, not a production prognostics model.",
            "RUL targets are used only for evaluation; training labels are derived from run-to-failure training trajectories.",
            "Results are benchmark estimates and do not establish aircraft-engine deployment performance.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=["FD001", "FD002", "FD003", "FD004"], action="append")
    args = parser.parse_args()
    result = run(tuple(args.subset) if args.subset else ("FD001", "FD002", "FD003", "FD004"))
    for item in result["subsets"]:
        print(f"{item['subset']}: MAE={item['mae']:.3f}, RMSE={item['rmse']:.3f}, NASA score={item['nasa_score']:.3f}")


if __name__ == "__main__":
    main()
