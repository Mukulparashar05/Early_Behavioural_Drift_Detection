from __future__ import annotations

import csv
import io
import json
import os
import pickle
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import (
    Autoencoder,
    add_physics_features,
    classify_warning,
    drift_analysis,
    reconstruction_scores,
)

DATASET = ROOT / "ai4i2020.csv"
RESULTS = ROOT / "outputs" / "metrics" / "results.json"
AUDIT = ROOT / "outputs" / "metrics" / "data_audit.json"
OBSERVATIONS = ROOT / "outputs" / "tables" / "observation_scores.csv"
NASA_RESULTS = ROOT / "outputs" / "metrics" / "nasa_cmapps_results.json"
MODEL = ROOT / "outputs" / "models" / "autoencoder_numpy.pkl"
SCHEMA = ROOT / "outputs" / "models" / "preprocessing_schema.json"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

lock = threading.Lock()
active_upload: Dict[str, Any] = {}
active_results: Optional[pd.DataFrame] = None
active_threshold: Optional[float] = None


class ThresholdUpdate(BaseModel):
    threshold: float = Field(gt=0, description="Reconstruction-error threshold")


def _json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise HTTPException(503, f"Required artifact is unavailable: {path.name}")
    return json.loads(path.read_text())


def _load_observations() -> pd.DataFrame:
    if not OBSERVATIONS.exists():
        raise HTTPException(503, "Prediction results are unavailable; run the training pipeline first")
    return pd.read_csv(OBSERVATIONS)


def _validate_frame(frame: pd.DataFrame, require_labels: bool = False) -> None:
    required = {
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    }
    if require_labels:
        required.add("Machine failure")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise HTTPException(422, {"message": "Dataset is missing required columns", "columns": missing})
    if frame.empty:
        raise HTTPException(422, "CSV contains no records")
    numeric = [c for c in required if c != "Type"]
    if frame[numeric].isna().any().any():
        bad = frame[numeric].columns[frame[numeric].isna().any()].tolist()
        raise HTTPException(422, {"message": "Numeric sensor columns contain missing values", "columns": bad})
    for column in numeric:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise HTTPException(422, f"Sensor column is not numeric: {column}")


def _dataset_info(frame: pd.DataFrame, provenance: str) -> Dict[str, Any]:
    timestamp_candidates = [
        c for c in frame.columns if c.lower() in {"timestamp", "time", "datetime", "date", "event_time"}
    ]
    machine_candidates = [
        c for c in frame.columns if c.lower() in {"machine_id", "machine", "unit", "engine_id", "asset_id"}
    ]
    timestamp_info: Dict[str, Any] = {
        "available": bool(timestamp_candidates),
        "column": timestamp_candidates[0] if timestamp_candidates else None,
        "valid": False,
        "invalid_count": 0,
        "ordered": False,
    }
    if timestamp_candidates:
        parsed = pd.to_datetime(frame[timestamp_candidates[0]], errors="coerce")
        timestamp_info.update({
            "valid": bool(parsed.notna().all()),
            "invalid_count": int(parsed.isna().sum()),
            "ordered": bool(parsed.dropna().is_monotonic_increasing),
        })
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "column_names": frame.columns.tolist(),
        "dtypes": {key: str(value) for key, value in frame.dtypes.items()},
        "missing": frame.isna().sum().astype(int).to_dict(),
        "duplicate_rows": int(frame.duplicated().sum()),
        "provenance": provenance,
        "has_labels": "Machine failure" in frame.columns,
        "timestamp": timestamp_info,
        "machine_id_column": machine_candidates[0] if machine_candidates else None,
        "machine_count": int(frame[machine_candidates[0]].nunique()) if machine_candidates else None,
    }


def _predict(frame: pd.DataFrame, threshold: Optional[float] = None) -> pd.DataFrame:
    _validate_frame(frame)
    if not MODEL.exists() or not SCHEMA.exists():
        raise HTTPException(503, "Autoencoder artifacts are unavailable; run python3 main.py first")
    schema = _json(SCHEMA)
    results = _json(RESULTS)

    model = Autoencoder(n_in=len(schema["input_features"]), latent=results["configuration"].get("latent_dim", 7))
    model.load(MODEL)

    enriched = add_physics_features(frame) if schema.get("include_physics", True) else frame.copy()
    numeric_cols = schema["numeric_features"]
    means = np.asarray(schema["mean"], dtype=float)
    scale = np.asarray(schema["scale"], dtype=float)
    categories = schema["category_levels"]

    category_values = enriched["Type"].astype(str).to_numpy()
    unknown_categories = sorted(set(category_values) - set(categories))
    if unknown_categories:
        raise HTTPException(
            422,
            {
                "message": "Dataset contains unsupported Type categories",
                "categories": unknown_categories,
                "supported": categories,
            },
        )

    values = (enriched[numeric_cols].to_numpy(float) - means) / scale
    encoded = np.column_stack([category_values == category for category in categories]).astype(float)
    x = np.ascontiguousarray(np.column_stack([values, encoded]), dtype=np.float64)

    scores = reconstruction_scores(model, x)

    global active_threshold
    if threshold is not None:
        effective_threshold = float(threshold)
    elif active_threshold is not None:
        effective_threshold = float(active_threshold)
    else:
        effective_threshold = float(results["threshold"])

    cfg = results.get("configuration", {})
    window = int(cfg.get("rolling_window", 25))
    fraction = float(cfg.get("drift_fraction", 0.6))
    margin = float(cfg.get("warning_uncertainty_margin", 0.1))

    rolling, drift = drift_analysis(scores, effective_threshold, window, fraction)

    outputs = pd.DataFrame({
        "source_row_index": np.arange(len(frame), dtype=int),
        "reconstruction_error": scores,
        "threshold": effective_threshold,
        "anomaly": (scores > effective_threshold).astype(int),
        "rolling_exceedance_fraction": rolling,
        "drift_detected": drift.astype(bool),
    })

    warning_states = []
    severities = []
    reasons = []
    for pos, sc in enumerate(scores):
        rf = float(rolling[pos]) if np.isfinite(rolling[pos]) else float("nan")
        st, sev, rsn = classify_warning(float(sc), effective_threshold, rf, margin=margin)
        warning_states.append(st)
        severities.append(sev)
        reasons.append(rsn)

    outputs["warning_state"] = warning_states
    outputs["severity"] = severities
    outputs["warning_reason"] = reasons

    for column in ("UDI", "Product ID", "Machine failure"):
        if column in frame.columns:
            outputs[column] = frame[column].values
    for column in frame.columns:
        if column.lower() in {
            "timestamp", "time", "datetime", "date", "event_time",
            "machine_id", "machine", "unit", "engine_id", "asset_id",
        }:
            outputs[column] = frame[column].values
    for column in schema["numeric_features"]:
        if column in enriched.columns:
            outputs[column] = enriched[column].values

    return outputs


def _psi(reference: np.ndarray, current: np.ndarray) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, 11)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_rate = np.maximum(ref_counts / max(len(reference), 1), 1e-6)
    cur_rate = np.maximum(cur_counts / max(len(current), 1), 1e-6)
    return float(np.sum((cur_rate - ref_rate) * np.log(cur_rate / ref_rate)))


def _analyze_uploaded(frame: pd.DataFrame) -> Dict[str, Any]:
    reference = pd.read_csv(DATASET)
    info = _dataset_info(frame, "uploaded analysis")
    timestamp_column = info["timestamp"]["column"]
    if timestamp_column and not info["timestamp"]["valid"]:
        raise HTTPException(
            422,
            {
                "message": "Timestamp column contains invalid values",
                "column": timestamp_column,
                "invalid_count": info["timestamp"]["invalid_count"],
            },
        )
    if timestamp_column:
        frame = (
            frame.assign(__parsed_timestamp=pd.to_datetime(frame[timestamp_column]))
            .sort_values("__parsed_timestamp")
            .drop(columns="__parsed_timestamp")
            .reset_index(drop=True)
        )
        info = _dataset_info(frame, "uploaded analysis")

    predictions = _predict(frame)
    results = _json(RESULTS)
    schema = _json(SCHEMA)
    ref_enriched = add_physics_features(reference) if schema.get("include_physics", True) else reference
    frame_enriched = add_physics_features(frame) if schema.get("include_physics", True) else frame

    drift_features = []
    for column in schema["numeric_features"]:
        if column in ref_enriched.columns and column in frame_enriched.columns:
            ref = ref_enriched[column].to_numpy(float)
            current = frame_enriched[column].to_numpy(float)
            psi = _psi(ref, current)
            mean_shift = abs(float(current.mean() - ref.mean())) / max(float(ref.std()), 1e-8)
            drift_features.append({
                "feature": column,
                "psi": psi,
                "standardized_mean_shift": mean_shift,
                "drifted": bool(psi >= 0.2 or mean_shift >= 0.5),
            })

    ref_type = reference["Type"].value_counts(normalize=True)
    cur_type = frame["Type"].value_counts(normalize=True)
    type_shift = float(
        sum(
            abs(float(cur_type.get(v, 0.0)) - float(ref_type.get(v, 0.0)))
            for v in set(ref_type.index) | set(cur_type.index)
        )
        / 2
    )

    reference_scores = _load_observations()["reconstruction_error"].to_numpy(float)
    anomaly_rate = float(predictions["anomaly"].mean())
    reference_rate = float((reference_scores > float(results["threshold"])).mean())
    score_shift = abs(
        float(predictions["reconstruction_error"].mean() - reference_scores.mean())
    ) / max(float(reference_scores.std()), 1e-8)

    signals = [item for item in drift_features if item["drifted"]]
    if type_shift >= 0.15:
        signals.append({
            "feature": "Type",
            "psi": None,
            "standardized_mean_shift": type_shift,
            "drifted": True,
        })
    score_drift = anomaly_rate > reference_rate + max(
        0.05, 3 * np.sqrt(max(reference_rate * (1 - reference_rate), 1e-8) / len(frame))
    )
    if score_drift:
        signals.append({
            "feature": "reconstruction_error",
            "psi": None,
            "standardized_mean_shift": score_shift,
            "drifted": True,
        })

    if not signals:
        drift_type = "NO_SIGNIFICANT_DRIFT"
        summary = "Uploaded records are within reference operational ranges and anomaly-rate tolerance."
    elif score_drift and len(signals) == 1:
        drift_type = "BEHAVIORAL_DRIFT"
        summary = "The autoencoder is flagging a statistically elevated fraction of anomaly scores."
    elif any(
        item["feature"] in ("Type", "Air temperature [K]", "Process temperature [K]", "Temp difference [K]")
        for item in signals
    ):
        drift_type = "OPERATING_CONDITION_DRIFT"
        summary = "Thermal operating conditions or categorical mix differ significantly from the reference baseline."
    else:
        drift_type = "SENSOR_DISTRIBUTION_DRIFT"
        summary = "One or more physical sensor distributions exhibit persistent statistical shift."

    checks = [
        {
            "priority": 1,
            "check": "Confirm sensor calibration, units, and timestamp alignment",
            "reason": "Calibration drift or synchronization error shifts several sensor channels simultaneously.",
        },
        {
            "priority": 2,
            "check": "Inspect flagged sensor channels against maintenance and replacement logs",
            "reason": "Physical mechanical degradation or recent tooling replacement alters baseline telemetry.",
        },
        {
            "priority": 3,
            "check": "Verify operating regime, machine Type mix, and production schedule",
            "reason": "An intentional change in machine workload or product recipe can cause benign drift.",
        },
        {
            "priority": 4,
            "check": "Inspect the highest-scoring anomaly records with a domain operator",
            "reason": "Scores triage anomalous risk; operators confirm physical machine health.",
        },
    ]

    sample_warning = (
        None
        if len(frame) >= 200
        else "Small upload sample: treat drift signals as initial triage evidence and verify with additional records."
    )
    score_series = predictions["reconstruction_error"].to_numpy(float)
    temporal = {"available": bool(timestamp_column), "ordered": bool(info["timestamp"].get("ordered", False))}
    if timestamp_column:
        parsed = pd.to_datetime(frame[timestamp_column])
        temporal.update({
            "start": parsed.min().isoformat(),
            "end": parsed.max().isoformat(),
            "duration_seconds": float((parsed.max() - parsed.min()).total_seconds()),
        })
        if len(score_series) >= 5:
            half = max(1, len(score_series) // 2)
            temporal["late_mean_score_shift"] = float(
                score_series[half:].mean() - score_series[:half].mean()
            )
            temporal["chronological_anomalies"] = int(predictions.iloc[half:]["anomaly"].sum())

    return {
        "dataset": info,
        "model": "autoencoder",
        "threshold": float(predictions["threshold"].iloc[0]),
        "drift_detected": bool(signals),
        "drift_type": drift_type,
        "summary": summary,
        "sample_size": len(frame),
        "sample_warning": sample_warning,
        "anomaly_count": int(predictions["anomaly"].sum()),
        "anomaly_rate": anomaly_rate,
        "reference_anomaly_rate": reference_rate,
        "signals": signals,
        "feature_comparison": drift_features,
        "type_distribution_shift": type_shift,
        "temporal_analysis": temporal,
        "investigation_checks": checks,
        "results": predictions.fillna("").to_dict("records"),
    }


app = FastAPI(title="Behavioral Drift Detection API", version="1.1.0")
origins = [
    item.strip()
    for item in os.getenv("FRONTEND_ORIGINS", "http://localhost:5173,http://localhost:3000,*").split(",")
    if item.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if "*" in origins else origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "artifacts_ready": MODEL.exists() and RESULTS.exists(),
        "dataset_available": DATASET.exists(),
        "active_threshold": active_threshold,
        "active_upload": active_results is not None,
    }


@app.get("/api/dataset/info")
def dataset_info() -> Dict[str, Any]:
    if not DATASET.exists():
        raise HTTPException(503, "Project dataset is unavailable")
    frame = pd.read_csv(DATASET)
    return _dataset_info(frame, "repository: ai4i2020.csv")


@app.post("/api/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(415, "Upload a CSV file")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "CSV exceeds the 10 MB upload limit")
    try:
        frame = pd.read_csv(io.BytesIO(data))
    except Exception as exc:
        raise HTTPException(422, "CSV could not be parsed") from exc
    _validate_frame(frame)
    return {
        "dataset": _dataset_info(frame, f"upload: {file.filename}"),
        "preview": frame.head(10).fillna("").to_dict("records"),
    }


@app.get("/api/dataset/active-analysis")
def get_active_analysis() -> Dict[str, Any]:
    global active_upload, active_results
    with lock:
        if active_upload:
            return active_upload
        if not DATASET.exists():
            raise HTTPException(503, "Project dataset is unavailable")
        frame = pd.read_csv(DATASET)
        analysis = _analyze_uploaded(frame)
        active_upload = analysis
        active_results = pd.DataFrame(analysis["results"])
        return analysis


@app.post("/api/dataset/analyze-reference")
def analyze_reference() -> Dict[str, Any]:
    global active_upload, active_results
    if not DATASET.exists():
        raise HTTPException(503, "Project dataset is unavailable")
    frame = pd.read_csv(DATASET)
    with lock:
        analysis = _analyze_uploaded(frame)
        active_upload = analysis
        active_results = pd.DataFrame(analysis["results"])
        return analysis


@app.post("/api/dataset/analyze")
async def analyze_dataset(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(415, "Upload a CSV file")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "CSV exceeds the 10 MB upload limit")
    try:
        frame = pd.read_csv(io.BytesIO(data))
    except Exception as exc:
        raise HTTPException(422, "CSV could not be parsed") from exc
    _validate_frame(frame)
    global active_upload, active_results
    with lock:
        analysis = _analyze_uploaded(frame)
        active_upload = analysis
        active_results = pd.DataFrame(analysis["results"])
        return analysis


@app.post("/api/detection/run")
def run_detection(threshold: Optional[float] = None) -> Dict[str, Any]:
    global active_results
    with lock:
        frame = pd.read_csv(DATASET)
        predictions = _predict(frame, threshold)
        active_results = predictions
    return {
        "model": "autoencoder",
        "count": len(predictions),
        "anomalies": int(predictions["anomaly"].sum()),
        "threshold": float(predictions["threshold"].iloc[0]),
        "results": predictions.head(1000).fillna("").to_dict("records"),
    }


@app.get("/api/detection/results")
def detection_results(
    limit: int = 100, offset: int = 0, status: Optional[str] = None
) -> Dict[str, Any]:
    frame = active_results.copy() if active_results is not None else _load_observations()
    if status == "anomaly":
        frame = frame[frame["anomaly"] == 1]
    elif status == "normal":
        frame = frame[frame["anomaly"] == 0]
    total = int(len(frame))
    page = frame.iloc[max(offset, 0) : max(offset, 0) + min(limit, 1000)]
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "active_upload": active_results is not None,
        "results": page.fillna("").to_dict("records"),
    }


@app.get("/api/models")
def models() -> Dict[str, Any]:
    results = _json(RESULTS)
    effective_thr = active_threshold if active_threshold is not None else results["threshold"]
    return {
        "active": "autoencoder",
        "threshold": float(effective_thr),
        "models": [
            {
                "id": "autoencoder",
                "name": "NumPy Bottleneck Autoencoder (Adam)",
                "status": "ready",
                "threshold": float(effective_thr),
                "metrics": results.get("autoencoder"),
            },
            {
                "id": "statistical_zscore",
                "name": "Statistical Mahalanobis Baseline",
                "status": "evaluated",
                "metrics": results.get("statistical_zscore"),
            },
            {
                "id": "isolation_forest",
                "name": "Isolation Forest",
                "status": "evaluated",
                "metrics": results.get("isolation_forest"),
            },
        ],
    }


@app.get("/api/evaluation")
def evaluation() -> Dict[str, Any]:
    results = _json(RESULTS)
    return {
        "dataset": "AI4I 2020 held-out test split",
        "split": results["split_counts"],
        "threshold": float(active_threshold if active_threshold is not None else results["threshold"]),
        "models": {
            key: results[key]
            for key in ("autoencoder", "statistical_zscore", "isolation_forest")
        },
    }


@app.get("/api/nasa/evaluation")
def nasa_evaluation() -> Dict[str, Any]:
    return _json(NASA_RESULTS)


@app.get("/api/drift/analysis")
def drift_analysis_endpoint() -> Dict[str, Any]:
    results = _json(RESULTS)
    frame = active_results.copy() if active_results is not None else _load_observations()
    cols = [
        "source_row_index",
        "reconstruction_error",
        "threshold",
        "rolling_exceedance_fraction",
        "drift_detected",
    ]
    available_cols = [c for c in cols if c in frame.columns]
    series = frame[available_cols].tail(1000).fillna("").to_dict("records")
    return {
        "ordered_rows": True,
        "timestamp_available": "timestamp" in frame.columns,
        "warning": "Source row order represents sequential change analysis.",
        "active_dataset": "uploaded" if active_results is not None else "ai4i2020.csv",
        "synthetic_experiment": results.get("synthetic_drift_experiment"),
        "series": series,
    }


@app.get("/api/alerts")
def alerts() -> Dict[str, Any]:
    frame = active_results.copy() if active_results is not None else _load_observations()
    flagged = frame[frame["anomaly"] == 1].sort_values("reconstruction_error", ascending=False).head(100)
    return {
        "count": int(len(flagged)),
        "active_dataset": "uploaded" if active_results is not None else "ai4i2020.csv",
        "alerts": flagged.fillna("").to_dict("records"),
    }


@app.patch("/api/settings/threshold")
def update_threshold(payload: ThresholdUpdate) -> Dict[str, Any]:
    global active_threshold, active_results, active_upload
    with lock:
        active_threshold = float(payload.threshold)
        if active_results is not None:
            active_results["threshold"] = active_threshold
            active_results["anomaly"] = (
                active_results["reconstruction_error"] > active_threshold
            ).astype(int)
            if active_upload:
                active_upload["threshold"] = active_threshold
                active_upload["anomaly_count"] = int(active_results["anomaly"].sum())
                active_upload["anomaly_rate"] = float(active_results["anomaly"].mean())
    return {
        "threshold": payload.threshold,
        "persisted": False,
        "active": True,
        "message": f"Active threshold set to {payload.threshold:.4f} for all subsequent detection requests.",
    }


@app.get("/api/export")
def export_results() -> StreamingResponse:
    frame = active_results.copy() if active_results is not None else _load_observations()
    stream = io.StringIO()
    frame.to_csv(stream, index=False)
    filename = "active_predictions.csv" if active_results is not None else "observation_scores.csv"
    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
