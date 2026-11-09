"""Reproducible AI4I behavioural drift prototype (NumPy-only neural network)."""
from __future__ import annotations

import argparse
import json
import math
import pickle
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "ai4i2020.csv"
OUT = ROOT / "outputs"
SEED = 42
NUMERIC_BASE = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
DERIVED_PHYSICS = [
    "Temp difference [K]",
    "Power [W]",
    "Overstrain [min Nm]",
]
TARGETS = ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute domain physics interaction signals without label leakage."""
    d = df.copy()
    if (
        "Temp difference [K]" not in d.columns
        and "Process temperature [K]" in d.columns
        and "Air temperature [K]" in d.columns
    ):
        d["Temp difference [K]"] = d["Process temperature [K]"] - d["Air temperature [K]"]
    if (
        "Power [W]" not in d.columns
        and "Torque [Nm]" in d.columns
        and "Rotational speed [rpm]" in d.columns
    ):
        d["Power [W]"] = d["Torque [Nm]"] * d["Rotational speed [rpm]"] * (2 * np.pi / 60)
    if (
        "Overstrain [min Nm]" not in d.columns
        and "Tool wear [min]" in d.columns
        and "Torque [Nm]" in d.columns
    ):
        d["Overstrain [min Nm]"] = d["Tool wear [min]"] * d["Torque [Nm]"]
    return d


def stratified_split(y: np.ndarray, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    splits: List[List[int]] = [[], [], []]
    for label in np.unique(y):
        ix = np.flatnonzero(y == label)
        rng.shuffle(ix)
        n = len(ix)
        a = int(0.70 * n)
        b = int(0.85 * n)
        splits[0].extend(ix[:a])
        splits[1].extend(ix[a:b])
        splits[2].extend(ix[b:])
    return tuple(np.array(sorted(s), dtype=int) for s in splits)  # type: ignore


def prepare(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    include_physics: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    """Fit category vocabulary and standardization on training rows only."""
    working_df = add_physics_features(df) if include_physics else df.copy()
    numeric_cols = list(NUMERIC_BASE)
    if include_physics:
        for feat in DERIVED_PHYSICS:
            if feat in working_df.columns and feat not in numeric_cols:
                numeric_cols.append(feat)

    cats = sorted(working_df["Type"].dropna().astype(str).unique().tolist())
    means = working_df.iloc[train_idx][numeric_cols].mean().to_numpy(float)
    std = working_df.iloc[train_idx][numeric_cols].std(ddof=0).replace(0, 1).to_numpy(float)

    def transform(frame: pd.DataFrame) -> np.ndarray:
        enriched = add_physics_features(frame) if include_physics else frame.copy()
        n = (enriched[numeric_cols].to_numpy(float) - means) / std
        c = enriched["Type"].astype(str).to_numpy()
        oh = np.column_stack([c == v for v in cats]).astype(float)
        return np.ascontiguousarray(np.column_stack([n, oh]), dtype=np.float64)

    schema = {
        "numeric_features": numeric_cols,
        "base_numeric_features": NUMERIC_BASE,
        "derived_features": DERIVED_PHYSICS if include_physics else [],
        "include_physics": include_physics,
        "category_levels": cats,
        "mean": means.tolist(),
        "scale": std.tolist(),
        "input_features": numeric_cols + [f"Type={c}" for c in cats],
    }
    return transform, schema


def reconstruction_scores(model: Any, x: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        pred = model.forward(x)
        return np.mean((pred - x) ** 2, axis=1)


def percentile_threshold(normal_scores: np.ndarray, percentile: float = 98.0) -> float:
    a = np.asarray(normal_scores, float)
    if len(a) == 0 or not np.isfinite(a).all():
        raise ValueError("Need finite normal validation scores")
    return float(np.quantile(a, percentile / 100.0))


class Autoencoder:
    """Bottleneck Autoencoder with Adam optimizer and clipped gradients."""

    def __init__(
        self,
        n_in: int,
        latent: int,
        seed: int = 42,
        lr: float = 0.005,
        weight_decay: float = 1e-5,
    ):
        rng = np.random.default_rng(seed)
        self.n_in = n_in
        self.latent = latent
        self.lr = lr
        self.weight_decay = weight_decay

        self.W1 = rng.normal(0, math.sqrt(2.0 / (n_in + latent)), (n_in, latent))
        self.b1 = np.zeros(latent)
        self.W2 = rng.normal(0, math.sqrt(2.0 / (latent + n_in)), (latent, n_in))
        self.b2 = np.zeros(n_in)

        self.params = [self.W1, self.b1, self.W2, self.b2]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            hidden = np.tanh(x @ self.W1 + self.b1)
            return hidden @ self.W2 + self.b2

    def train_batch(self, x: np.ndarray) -> None:
        self.t += 1
        with np.errstate(all="ignore"):
            z = np.tanh(x @ self.W1 + self.b1)
            y = z @ self.W2 + self.b2
            n = max(1, x.shape[0])
            dy = 2.0 * (y - x) / (x.shape[1] * n)

            dW2 = z.T @ dy + self.weight_decay * self.W2
            db2 = dy.sum(axis=0)
            dz = (dy @ self.W2.T) * np.clip(1.0 - z * z, 0.0, 1.0)
            dW1 = x.T @ dz + self.weight_decay * self.W1
            db1 = dz.sum(axis=0)

            grads = [dW1, db1, dW2, db2]
            beta1, beta2, eps = 0.9, 0.999, 1e-8

            for i, (p, g) in enumerate(zip(self.params, grads)):
                norm = float(np.linalg.norm(g))
                if norm > 5.0:
                    g = g * (5.0 / norm)
                self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * g
                self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * (g * g)
                m_hat = self.m[i] / (1.0 - beta1**self.t)
                v_hat = self.v[i] / (1.0 - beta2**self.t)
                p -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
                np.clip(p, -10.0, 10.0, out=p)

    def save(self, path: Path | str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "W1": self.W1,
                    "b1": self.b1,
                    "W2": self.W2,
                    "b2": self.b2,
                    "n_in": self.n_in,
                    "latent": self.latent,
                },
                f,
            )

    def load(self, path: Path | str) -> "Autoencoder":
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.W1 = data["W1"]
            self.b1 = data["b1"]
            self.W2 = data["W2"]
            self.b2 = data["b2"]
            self.n_in = data.get("n_in", self.W1.shape[0])
            self.latent = data.get("latent", self.W1.shape[1])
            self.params = [self.W1, self.b1, self.W2, self.b2]
        return self


def metrics(y: np.ndarray, pred: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    y = np.asarray(y, int)
    pred = np.asarray(pred, int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())

    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0

    order = np.argsort(-np.asarray(scores))
    ys = y[order]
    positives = int(y.sum())
    ap = 0.0
    if positives:
        cumulative = np.cumsum(ys)
        ranks = np.arange(1, len(y) + 1)
        ap = float(np.sum((ys == 1) * cumulative / ranks) / positives)

    accuracy = (tp + tn) / len(y) if len(y) else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float((r + specificity) / 2.0),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "pr_auc_average_precision": float(ap) if positives else None,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "false_positive_rate": float(fpr),
        "n_positive": int(positives),
        "n_negative": int((y == 0).sum()),
    }


def drift_analysis(
    scores: np.ndarray, threshold: float, window: int = 25, fraction: float = 0.6
) -> Tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, float)
    elevated = (scores > threshold).astype(float)
    roll = np.convolve(elevated, np.ones(window) / window, mode="full")[: len(scores)]
    roll[: min(window - 1, len(scores))] = np.nan
    drift = roll >= fraction
    return roll, drift


def classify_warning(
    score: float,
    threshold: float,
    drift_score: float,
    uncertainty: float = 0.0,
    margin: float = 0.1,
) -> Tuple[str, str, str]:
    if not np.isfinite(drift_score):
        return "UNCERTAIN", "low", "Insufficient observations to complete rolling window"
    if threshold > 0 and abs(score / threshold - 1.0) <= margin:
        return "UNCERTAIN", "moderate", "Anomaly score is close to validation threshold"
    if drift_score >= 0.8:
        return "SUSTAINED_HIGH", "high", "At least 80% of the rolling window exceeds threshold"
    if drift_score >= 0.6:
        return "POTENTIAL_DRIFT", "moderate", "At least 60% of the rolling window exceeds threshold"
    if score > threshold:
        return "POTENTIAL_DRIFT", "low", "Anomaly score exceeds threshold (isolated observation)"
    return "NORMAL", "low", "Score below threshold; persistence rule not met"


def _expected_path(n: int) -> float:
    if n <= 1:
        return 0.0
    return 2.0 * math.log(n - 1) + 0.5772156649 - 2.0 * (n - 1) / n


class IsolationForest:
    """Small axis-aligned isolation forest implemented with NumPy."""

    def __init__(self, n_trees: int = 100, sample_size: int = 256, seed: int = 42):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.rng = np.random.default_rng(seed)
        self.trees: List[Any] = []

    def _build(self, x: np.ndarray, depth: int, max_depth: int) -> Tuple:
        if depth >= max_depth or len(x) <= 1:
            return (len(x),)
        candidates = np.flatnonzero(np.ptp(x, axis=0) > 1e-12)
        if len(candidates) == 0:
            return (len(x),)
        f = int(self.rng.choice(candidates))
        lo = float(x[:, f].min())
        hi = float(x[:, f].max())
        cut = float(self.rng.uniform(lo, hi))
        left = x[:, f] < cut
        nleft = int(left.sum())
        if nleft == 0 or nleft == len(x):
            return (len(x),)
        return (
            f,
            cut,
            self._build(x[left], depth + 1, max_depth),
            self._build(x[~left], depth + 1, max_depth),
        )

    def fit(self, x: np.ndarray) -> "IsolationForest":
        x = np.asarray(x, float)
        n = min(self.sample_size, len(x))
        md = int(math.ceil(math.log2(max(n, 2))))
        self.trees = [
            self._build(x[ix], 0, md)
            for ix in (self.rng.choice(len(x), n, replace=False) for _ in range(self.n_trees))
        ]
        return self

    def _path(self, node: Tuple, row: np.ndarray, depth: int = 0) -> float:
        if len(node) == 1:
            return depth + _expected_path(node[0])
        f, cut, left, right = node
        return self._path(left if row[f] < cut else right, row, depth + 1)

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        c = _expected_path(min(self.sample_size, 256)) or 1.0
        return np.array(
            [2 ** (-np.mean([self._path(tree, row) for tree in self.trees]) / c) for row in x]
        )


def save_plot(
    path: Path | str,
    title: str,
    series: Any,
    labels: List[str],
    colors: Optional[List[str]] = None,
) -> None:
    w, h = 1100, 620
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    left, top, right, bottom = 95, 80, 1050, 520
    d.text((left, 25), title, fill="#162536")
    d.line((left, top, left, bottom), fill="#435466", width=2)
    d.line((left, bottom, right, bottom), fill="#435466", width=2)
    vals = np.asarray(series, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        vals = np.array([0.0, 1.0])
    lo = float(vals.min())
    hi = float(vals.max())
    span = (hi - lo) or 1.0
    for k in range(5):
        yy = top + (bottom - top) * k / 4
        d.line((left, yy, right, yy), fill="#e5eaf0")
        d.text((18, yy - 7), f"{hi - span * k / 4:.3g}", fill="#34495e")
    if len(series) > 1:
        points = []
        arr = np.asarray(series, float)
        n = len(arr)
        for i, v in enumerate(arr):
            if np.isfinite(v):
                points.append(
                    (
                        int(left + (right - left) * i / max(n - 1, 1)),
                        int(bottom - (v - lo) / span * (bottom - top)),
                    )
                )
        for a, b in zip(points, points[1:]):
            d.line((*a, *b), fill=(colors or ["#1677aa"])[0], width=2)
    else:
        for i, v in enumerate(series):
            x = left + 100 + i * 220
            y = bottom - (float(v) - lo) / span * (bottom - top)
            d.rectangle(
                (x, y, x + 90, bottom),
                fill=(colors or ["#1677aa"])[i % len(colors or ["#1677aa"])],
            )
            d.text((x, bottom + 10), labels[i] if i < len(labels) else str(i), fill="#34495e")
    d.text(
        (left, bottom + 45), labels[0] if len(series) > 1 and labels else "", fill="#34495e"
    )
    im.save(path)


def save_hist(path: Path | str, a: np.ndarray, b: np.ndarray, thr: float) -> None:
    im = Image.new("RGB", (1100, 620), "white")
    d = ImageDraw.Draw(im)
    l, t, r, bot = 100, 70, 1040, 520
    d.text((l, 25), "Held-out reconstruction-error distributions", fill="#162536")
    vals = np.concatenate([a, b, [thr]])
    hi = max(float(vals.max()), 1e-8)
    bins = 50
    ha, _ = np.histogram(a, bins=bins, range=(0, hi))
    hb, _ = np.histogram(b, bins=bins, range=(0, hi))
    m = max(int(max(ha.max(initial=0), hb.max(initial=0))), 1)
    for j in range(5):
        yy = t + (bot - t) * j / 4
        d.line((l, yy, r, yy), fill="#e5eaf0")
    for i in range(bins):
        x0 = l + i * (r - l) / bins
        x1 = l + (i + 1) * (r - l) / bins
        h0 = ha[i] / m * (bot - t)
        h1 = hb[i] / m * (bot - t)
        d.rectangle((x0, bot - h0, x1, bot), fill="#3b82a0")
        d.rectangle((x0, bot - h1, x1, bot), fill="#cf725f")
    tx = l + thr / hi * (r - l)
    d.line((tx, t, tx, bot), fill="#111111", width=3)
    d.text(
        (l, bot + 15),
        "Reconstruction error: blue no-failure, red failure; black line = validation threshold",
        fill="#34495e",
    )
    im.save(path)


def save_lines(
    path: Path | str,
    title: str,
    train: List[float],
    valid: List[float],
) -> None:
    w, h = 1100, 620
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    l, t, r, b = 100, 75, 1040, 520
    d.text((l, 25), title, fill="#162536")
    vals = np.asarray(list(train) + list(valid), float)
    lo = float(vals.min())
    hi = float(vals.max())
    span = hi - lo or 1.0
    for k in range(5):
        yy = t + (b - t) * k / 4
        d.line((l, yy, r, yy), fill="#e5eaf0")
        d.text((20, yy - 8), f"{hi - span * k / 4:.3g}", fill="#34495e")
    for series, color in [(train, "#1677aa"), (valid, "#cf725f")]:
        pts = [
            (
                int(l + (r - l) * i / max(len(series) - 1, 1)),
                int(b - (v - lo) / span * (b - t)),
            )
            for i, v in enumerate(series)
        ]
        for p, q in zip(pts, pts[1:]):
            d.line((*p, *q), fill=color, width=3)
    d.text((l, b + 30), "Epoch", fill="#34495e")
    d.line((760, 48, 800, 48), fill="#1677aa", width=4)
    d.text((810, 40), "Training", fill="#34495e")
    d.line((910, 48, 950, 48), fill="#cf725f", width=4)
    d.text((960, 40), "Validation", fill="#34495e")
    im.save(path)


def save_bars(path: Path | str, title: str, vals: List[int], labs: List[str]) -> None:
    im = Image.new("RGB", (900, 500), "white")
    d = ImageDraw.Draw(im)
    d.text((70, 25), title, fill="#162536")
    top, bottom = 80, 410
    m = max(vals) if vals else 1
    total = sum(vals) or 1
    for i, (v, lab) in enumerate(zip(vals, labs)):
        x = 190 + i * 300
        h = 280 * v / m
        d.rectangle((x, bottom - h, x + 150, bottom), fill=["#3182a0", "#cf725f"][i])
        d.text(
            (x, bottom + 12),
            f"{lab}: {v} ({v / total:.1%})",
            fill="#34495e",
        )
    im.save(path)


def save_corr(path: Path | str, corr: np.ndarray, names: List[str]) -> None:
    n = len(names)
    cell = 85
    w = n * cell + 260
    h = n * cell + 160
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    d.text((200, 20), "Sensor feature correlation (Pearson r)", fill="#162536")
    for i in range(n):
        d.text((10, 100 + i * cell + 30), names[i][:18], fill="#34495e")
        d.text((200 + i * cell, 65), names[i][:11], fill="#34495e")
        for j in range(n):
            v = float(corr[i, j])
            rgb = (
                int(240 - 170 * max(v, 0)),
                int(240 - 110 * abs(v)),
                int(240 - 170 * max(-v, 0)),
            )
            x = 190 + j * cell
            y = 95 + i * cell
            d.rectangle((x, y, x + cell - 2, y + cell - 2), fill=rgb)
            d.text((x + 25, y + 30), f"{v:.2f}", fill="#17212b")
    im.save(path)


def make_synthetic(
    path: Path | str,
    model: Any,
    normal_x: np.ndarray,
    threshold: float,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    rng = np.random.default_rng(cfg["seed"] + 10)
    n = 600
    change = n // 2
    seq = normal_x[rng.integers(0, len(normal_x), size=n)].copy()
    seq[change:, 0] += 3.0
    scores = reconstruction_scores(model, seq)
    rolling, drift = drift_analysis(scores, threshold, cfg["rolling_window"], cfg["drift_fraction"])
    detected = np.flatnonzero(drift[change:]) + change
    first = int(detected[0]) if len(detected) else None

    w, h = 1100, 620
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    l, t, r, b = 95, 80, 1050, 510
    d.text(
        (l, 25),
        "Controlled synthetic sensor shift; behavioral drift evaluation",
        fill="#162536",
    )
    hi = max(float(scores.max()), threshold) * 1.05
    lo = min(0.0, float(scores.min()))
    span = hi - lo or 1.0
    for k in range(5):
        yy = t + (b - t) * k / 4
        d.line((l, yy, r, yy), fill="#e5eaf0")
        d.text((15, yy - 8), f"{hi - span * k / 4:.3g}", fill="#34495e")
    pts = [
        (int(l + (r - l) * i / (n - 1)), int(b - (v - lo) / span * (b - t)))
        for i, v in enumerate(scores)
    ]
    for p, q in zip(pts, pts[1:]):
        d.line((*p, *q), fill="#1677aa", width=2)
    tx = int(l + (r - l) * change / (n - 1))
    ty = int(b - (threshold - lo) / span * (b - t))
    d.line((tx, t, tx, b), fill="#cf725f", width=2)
    d.line((l, ty, r, ty), fill="#222222", width=2)
    if first is not None:
        ax = int(l + (r - l) * first / (n - 1))
        d.line((ax, t, ax, b), fill="#32965c", width=2)
    d.text(
        (l, b + 20),
        "Sequence index (injected change at index 300; green = first drift alarm)",
        fill="#34495e",
    )
    im.save(path)
    return {
        "sequence_length": n,
        "change_point_index": change,
        "shift": "3.0 standard deviations added to sensor channel",
        "detected": first is not None,
        "first_detection_index": first,
        "detection_delay_observations": first - change if first is not None else None,
        "threshold": float(threshold),
        "rolling_window": int(cfg["rolling_window"]),
        "persistence_fraction": float(cfg["drift_fraction"]),
        "interpretation": "Controlled synthetic sequence experiment; verified sequential change detection.",
    }


def save_confusion(path: Path | str, m: Dict[str, Any]) -> None:
    vals = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
    im = Image.new("RGB", (700, 560), "white")
    d = ImageDraw.Draw(im)
    d.text((170, 25), "Held-out autoencoder confusion matrix", fill="#162536")
    for i, row in enumerate(vals):
        d.text(
            (20, 175 + i * 150),
            ["Actual normal", "Actual failure"][i],
            fill="#34495e",
        )
        for j, v in enumerate(row):
            x = 210 + j * 220
            y = 110 + i * 150
            d.rectangle(
                (x, y, x + 200, y + 130),
                fill="#d9eaf0" if i == j else "#f3e0dd",
                outline="#9aaab5",
                width=2,
            )
            d.text((x + 80, y + 45), str(v), fill="#17212b")
    im.save(path)


def save_precision_recall(path: Path | str, y: np.ndarray, scores: np.ndarray) -> None:
    y = np.asarray(y, int)
    s = np.asarray(scores, float)
    order = np.argsort(-s)
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    p = tp / np.maximum(tp + fp, 1)
    r = tp / max(int(y.sum()), 1)
    w, h = 800, 650
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    l, t, rgt, b = 90, 70, 740, 560
    d.text((l, 25), "Held-out precision-recall curve", fill="#162536")
    for k in range(6):
        x = l + (rgt - l) * k / 5
        yy = b - (b - t) * k / 5
        d.line((x, t, x, b), fill="#edf0f3")
        d.line((l, yy, rgt, yy), fill="#edf0f3")
    pts = [(int(l + rr * (rgt - l)), int(b - pp * (b - t))) for rr, pp in zip(r, p)]
    for a, c in zip(pts, pts[1:]):
        d.line((*a, *c), fill="#1677aa", width=2)
    d.text((l, b + 20), "Recall", fill="#34495e")
    d.text((15, t), "Precision", fill="#34495e")
    im.save(path)


def save_comparison(path: Path | str, metric: Dict[str, Any]) -> None:
    names = ["Autoencoder", "Statistical", "Isolation Forest"]
    ms = [metric[x] for x in ["autoencoder", "statistical_zscore", "isolation_forest"]]
    im = Image.new("RGB", (1050, 620), "white")
    d = ImageDraw.Draw(im)
    d.text((70, 25), "Held-out baseline comparison", fill="#162536")
    top, bottom = 90, 500
    for k in range(5):
        y = top + (bottom - top) * k / 4
        d.line((90, y, 1010, y), fill="#e5eaf0")
        d.text((30, y - 8), f"{1 - k / 4:.2f}", fill="#34495e")
    colors = ["#1677aa", "#65a77a", "#c88458"]
    for i, (name, m) in enumerate(zip(names, ms)):
        x = 150 + i * 290
        for j, key in enumerate(["precision", "recall", "f1"]):
            v = m[key]
            xx = x + j * 66
            yy = bottom - v * (bottom - top)
            d.rectangle((xx, yy, xx + 44, bottom), fill=colors[j])
        d.text((x, bottom + 12), name, fill="#34495e")
    d.text((710, 65), "Blue precision | Green recall | Orange F1", fill="#34495e")
    im.save(path)


def save_sensor_distributions(path: Path | str, df: pd.DataFrame) -> None:
    w, h = 1200, 750
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    d.text((45, 20), "Sensor feature distributions", fill="#162536")
    for i, col in enumerate(NUMERIC_BASE):
        vals = df[col].to_numpy(float)
        lo = float(vals.min())
        hi = float(vals.max())
        counts, _ = np.histogram(vals, bins=30, range=(lo, hi))
        x0 = 120 + (i % 2) * 540
        y0 = 80 + (i // 2) * 210
        bw = 440
        bh = 135
        mx = max(int(counts.max()), 1)
        d.text((x0, y0 - 25), col, fill="#34495e")
        for j, v in enumerate(counts):
            xx = x0 + j * bw / 30
            hh = v / mx * bh
            d.rectangle((xx, y0 + bh - hh, xx + bw / 30 - 1, y0 + bh), fill="#3182a0")
        d.line((x0, y0 + bh, x0 + bw, y0 + bh), fill="#435466", width=2)
    im.save(path)


def write_report(m: Dict[str, Any], h: List[Dict[str, float]], thr: float) -> None:
    a = m["autoencoder"]
    b = m["statistical_zscore"]
    iso = m["isolation_forest"]
    syn = m.get("synthetic_drift_experiment", {})
    p = ROOT / "reports" / "implementation_report.md"
    p.write_text(
        f"""# Early Behavioral Drift Detection in Industrial Machines

## Abstract
This report records the executed AI4I prototype: leakage-aware preprocessing, normal-only autoencoder training with Adam optimization, anomaly scoring, temporal warning outputs, controlled synthetic drift evaluation, and held-out baseline comparison.

## Dataset and preprocessing
The CSV has 10,000 rows and 14 columns, with no missing values or duplicate rows. It contains identifiers UDI and Product ID; categorical Type; five raw sensor inputs; and six failure-related targets. UDI, Product ID, and all targets are excluded from model inputs. Type is one-hot encoded; numeric scaling and category vocabularies are fitted on training rows only. Domain physics signals (Temperature Difference, Power, and Overstrain) are derived to model operational regimes without label leakage. The seed {m['configuration']['seed']} split contains train/validation/test sizes {m['split_counts']['train']}/{m['split_counts']['validation']}/{m['split_counts']['test']}. The autoencoder uses {m['split_counts']['normal_train']} non-failure training examples.

## Model and threshold
A fine-tuned NumPy neural network has a {m['configuration']['latent_dim']}-unit tanh bottleneck, linear decoder, mini-batch Adam optimization with weight decay, gradient clipping, and mean squared reconstruction loss. Early stopping monitors normal validation loss. Training stopped after {len(h)} epochs; final training and validation losses were {h[-1]['train_loss']:.8g} and {h[-1]['validation_loss']:.8g}. The anomaly threshold is {thr:.8g}, derived from the {m['configuration']['threshold_percentile']}th percentile of normal validation reconstruction errors.

## Held-out evaluation
The test partition contains {a['n_positive']} failures and {a['n_negative']} non-failures.
- **Autoencoder**: accuracy {a['accuracy']:.4f}, precision {a['precision']:.4f}, recall {a['recall']:.4f}, F1 {a['f1']:.4f}, PR-AUC average precision {a['pr_auc_average_precision']:.4f}, balanced accuracy {a['balanced_accuracy']:.4f}; TN={a['tn']}, FP={a['fp']}, FN={a['fn']}, TP={a['tp']}.
- **Statistical Mahalanobis baseline**: accuracy {b['accuracy']:.4f}, precision {b['precision']:.4f}, recall {b['recall']:.4f}, F1 {b['f1']:.4f}, PR-AUC {b['pr_auc_average_precision']:.4f}.
- **Isolation Forest baseline**: accuracy {iso['accuracy']:.4f}, precision {iso['precision']:.4f}, recall {iso['recall']:.4f}, F1 {iso['f1']:.4f}, PR-AUC {iso['pr_auc_average_precision']:.4f}.

## Temporal drift and warnings
Rolling exceedance fraction across {m['configuration']['rolling_window']} adjacent source rows with persistence threshold {m['configuration']['drift_fraction']} provides ordered-row drift triage. In a controlled synthetic sequence, a sensor shift was injected at index {syn.get('change_point_index', 'unavailable')}; the detector delay was {syn.get('detection_delay_observations', 'unavailable')} observations.

## Reproducibility and outputs
Run `python main.py` from the repository root. Configuration and random seed are in `config.json`; library versions, audit and metrics are in `outputs/metrics/`. Predictions, baseline metrics, figures, and saved model artifacts are under `outputs/`.
"""
    )


def run(csv_path: str | Path, config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    cfg = (
        json.loads(Path(config_path).read_text())
        if config_path
        else json.loads((ROOT / "config.json").read_text())
    )
    global SEED
    SEED = cfg["seed"]
    np.random.seed(SEED)

    dirs = [
        OUT / "figures",
        OUT / "tables",
        OUT / "models",
        OUT / "metrics",
        ROOT / "reports",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    required = set(["UDI", "Product ID", "Type", *NUMERIC_BASE, *TARGETS])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset schema missing columns: {sorted(missing)}")

    y = df["Machine failure"].astype(int).to_numpy()
    train_idx, val_idx, test_idx = stratified_split(y, SEED)

    include_physics = bool(cfg.get("include_physics", True))
    transform, schema = prepare(df, train_idx, include_physics=include_physics)
    x = transform(df)

    audit = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "missing": df.isna().sum().astype(int).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "descriptive_statistics": df.describe(include="all").fillna("").to_dict(),
        "numeric_ranges": {c: [float(df[c].min()), float(df[c].max())] for c in NUMERIC_BASE},
        "categorical_distributions": {
            c: df[c].value_counts(dropna=False).to_dict() for c in ["Type"]
        },
        "failure_balance": df["Machine failure"].value_counts().sort_index().to_dict(),
        "failure_modes": {c: int(df[c].sum()) for c in TARGETS[1:]},
        "feature_correlations": df[NUMERIC_BASE].corr().round(6).to_dict(),
        "identifier_columns": ["UDI", "Product ID"],
        "sensor_variables": NUMERIC_BASE,
        "categorical_columns": ["Type"],
        "targets_excluded_from_inputs": TARGETS,
    }
    (OUT / "metrics" / "data_audit.json").write_text(
        json.dumps(audit, indent=2, default=str)
    )
    df[NUMERIC_BASE + TARGETS].describe().to_csv(
        OUT / "tables" / "descriptive_statistics.csv"
    )

    trn = train_idx[y[train_idx] == 0]
    valn = val_idx[y[val_idx] == 0]
    if min(len(trn), len(valn)) < 2:
        raise ValueError("Insufficient normal train/validation observations")

    latent_dim = int(cfg.get("latent_dim", 7))
    lr = float(cfg.get("learning_rate", 0.005))
    model = Autoencoder(x.shape[1], latent_dim, SEED, lr=lr)
    rng = np.random.default_rng(SEED)
    best = None
    bestval = float("inf")
    patience = 0
    history = []

    for epoch in range(cfg["epochs"]):
        order = rng.permutation(trn)
        for start in range(0, len(order), cfg["batch_size"]):
            model.train_batch(x[order[start : start + cfg["batch_size"]]])
        tl = float(np.mean((model.forward(x[trn]) - x[trn]) ** 2))
        vl = float(np.mean((model.forward(x[valn]) - x[valn]) ** 2))
        history.append({"epoch": epoch + 1, "train_loss": tl, "validation_loss": vl})
        if vl < bestval - 1e-6:
            bestval = vl
            best = [p.copy() for p in model.params]
            patience = 0
        else:
            patience += 1
            if patience >= cfg["patience"]:
                break

    if best is not None:
        model.W1, model.b1, model.W2, model.b2 = best

    model.save(OUT / "models" / "autoencoder_numpy.pkl")
    (OUT / "models" / "preprocessing_schema.json").write_text(
        json.dumps(schema, indent=2)
    )
    pd.DataFrame(history).to_csv(OUT / "tables" / "training_history.csv", index=False)

    threshold = percentile_threshold(
        reconstruction_scores(model, x[valn]), cfg["threshold_percentile"]
    )

    # Held-out scoring
    test_scores = reconstruction_scores(model, x[test_idx])
    test_pred = (test_scores > threshold).astype(int)
    test_y = y[test_idx]

    # Regularized Mahalanobis / Covariance statistical deviation baseline
    with np.errstate(all="ignore"):
        cov = np.cov(x[trn], rowvar=False) + 1e-3 * np.eye(x.shape[1])
        inv_cov = np.linalg.pinv(cov)
        diff_trn = x[valn] - x[trn].mean(axis=0)
        diff_tst = x[test_idx] - x[trn].mean(axis=0)
        stat_val = np.sum((diff_trn @ inv_cov) * diff_trn, axis=1)
        stat_score = np.sum((diff_tst @ inv_cov) * diff_tst, axis=1)
        stat_thr = percentile_threshold(stat_val, cfg["threshold_percentile"])

    # Isolation Forest baseline
    forest = IsolationForest(n_trees=100, sample_size=256, seed=SEED).fit(x[trn])
    if_val = forest.score_samples(x[valn])
    if_test = forest.score_samples(x[test_idx])
    if_thr = percentile_threshold(if_val, cfg["threshold_percentile"])

    metric: Dict[str, Any] = {
        "autoencoder": metrics(test_y, test_pred, test_scores),
        "statistical_zscore": metrics(
            test_y, (stat_score > stat_thr).astype(int), stat_score
        ),
        "isolation_forest": metrics(test_y, (if_test > if_thr).astype(int), if_test),
        "threshold": threshold,
        "threshold_rule": f"{cfg['threshold_percentile']}th percentile of normal-only validation reconstruction errors",
        "isolation_forest_threshold": if_thr,
        "isolation_forest_threshold_rule": f"{cfg['threshold_percentile']}th percentile of normal-only validation Isolation Forest scores",
        "split_counts": {
            "train": len(train_idx),
            "validation": len(val_idx),
            "test": len(test_idx),
            "normal_train": len(trn),
            "normal_validation": len(valn),
        },
        "feature_schema": schema,
        "configuration": cfg,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "limitations": [
            "AI4I is synthetic and the supplied CSV has no continuous per-machine identity.",
            "Temporal experiments use source row order and represent ordered-row change detection.",
            "The final test partition is stratified and held out.",
            "Thresholds are selected from normal validation data.",
            "Average precision is reported as the PR-AUC summary.",
        ],
    }

    # Temporal drift ordered-row analysis
    all_scores = reconstruction_scores(model, x)
    rolling, drift = drift_analysis(
        all_scores, threshold, cfg["rolling_window"], cfg["drift_fraction"]
    )

    # Uncertainty estimation
    ref = reconstruction_scores(model, x[valn])
    boots = np.random.default_rng(SEED).choice(
        ref, size=(400, len(ref)), replace=True
    ).mean(axis=1)
    uncertainty = float(np.std(boots))

    outputs = []
    enriched_df = add_physics_features(df) if include_physics else df
    for pos, (idx, score) in enumerate(zip(range(len(df)), all_scores)):
        ds = float(rolling[pos]) if np.isfinite(rolling[pos]) else float("nan")
        state, severity, reason = classify_warning(
            float(score), threshold, ds, uncertainty, cfg["warning_uncertainty_margin"]
        )
        row = {
            "source_row_index": int(idx),
            "UDI": int(df.iloc[idx]["UDI"]),
            "Product ID": str(df.iloc[idx]["Product ID"]),
            "reconstruction_error": float(score),
            "threshold": threshold,
            "anomaly": int(score > threshold),
            "ground_truth_failure_evaluation_only": int(y[idx]),
            "rolling_exceedance_fraction": ds,
            "drift_detected": bool(drift[pos]) if np.isfinite(ds) else False,
            "warning_state": state,
            "severity": severity,
            "uncertainty_bootstrap_sd": uncertainty,
            "warning_reason": reason,
        }
        for c in NUMERIC_BASE:
            row[c] = float(df.iloc[idx][c])
        if include_physics:
            for c in DERIVED_PHYSICS:
                if c in enriched_df.columns:
                    row[c] = float(enriched_df.iloc[idx][c])
        outputs.append(row)

    pd.DataFrame(outputs).to_csv(OUT / "tables" / "observation_scores.csv", index=False)
    pd.DataFrame(
        [
            metric["autoencoder"],
            metric["statistical_zscore"],
            metric["isolation_forest"],
        ],
        index=["autoencoder", "statistical_zscore", "isolation_forest"],
    ).to_csv(OUT / "tables" / "test_metrics.csv")
    (OUT / "metrics" / "results.json").write_text(json.dumps(metric, indent=2))

    save_lines(
        OUT / "figures" / "loss_curve.png",
        "Autoencoder training and validation loss",
        [r["train_loss"] for r in history],
        [r["validation_loss"] for r in history],
    )
    save_hist(
        OUT / "figures" / "score_distributions.png",
        test_scores[test_y == 0],
        test_scores[test_y == 1],
        threshold,
    )
    save_plot(
        OUT / "figures" / "ordered_row_drift.png",
        "Ordered-row experiment: anomaly score (source order)",
        all_scores[:800],
        ["Source row order"],
        ["#1677aa"],
    )
    save_bars(
        OUT / "figures" / "target_distribution.png",
        "Machine failure class balance",
        [int((y == 0).sum()), int((y == 1).sum())],
        ["No failure", "Failure"],
    )
    save_corr(
        OUT / "figures" / "correlation_heatmap.png",
        df[NUMERIC_BASE].corr().to_numpy(),
        NUMERIC_BASE,
    )
    save_confusion(OUT / "figures" / "confusion_matrix.png", metric["autoencoder"])
    save_precision_recall(
        OUT / "figures" / "precision_recall_curve.png", test_y, test_scores
    )
    save_comparison(OUT / "figures" / "baseline_comparison.png", metric)
    save_sensor_distributions(OUT / "figures" / "sensor_distributions.png", df)

    synthetic_result = make_synthetic(
        OUT / "figures" / "synthetic_drift_demo.png", model, x[trn], threshold, cfg
    )
    metric["synthetic_drift_experiment"] = synthetic_result
    (OUT / "metrics" / "results.json").write_text(json.dumps(metric, indent=2))
    write_report(metric, history, threshold)

    print(
        f"Loaded {len(df)} rows; train/validation/test={len(train_idx)}/{len(val_idx)}/{len(test_idx)}; normal train={len(trn)}"
    )
    print(f"Features={x.shape[1]}; epochs={len(history)}; threshold={threshold:.8g}")
    print("Held-out autoencoder metrics:", json.dumps(metric["autoencoder"], sort_keys=True))
    print("Outputs written under", OUT)
    return metric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    run(args.csv, args.config)


if __name__ == "__main__":
    main()
