# Sentinel ML — Early Behavioral Drift Detection in Industrial Machines

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![NumPy Only](https://img.shields.io/badge/core_model-NumPy_Only-orange.svg)](https://numpy.org/)
[![Streamlit UI](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![Unit Tests](https://img.shields.io/badge/tests-6%2F6_passing-brightgreen.svg)](tests/test_pipeline.py)

An end-to-end, auditable machine learning system designed to detect **early behavioral drift and mechanical degradation** in industrial manufacturing equipment before catastrophic failure occurs.

Built around the **AI4I 2020 Predictive Maintenance Dataset**, the core modeling engine is written purely in **NumPy** (zero PyTorch, TensorFlow, or scikit-learn dependencies for inference), featuring domain physics feature engineering, an unsupervised Adam-trained autoencoder, adaptive thresholding, and rolling persistence alarms.

---

## ⚡ Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/amlpro.git
cd amlpro

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Streamlit Application (Primary UI)

```bash
streamlit run streamlit_app.py
```

The interactive dashboard opens at **`http://localhost:8501`**.

### 3. Run Pipeline Unit Tests

```bash
python3 -m unittest tests/test_pipeline.py -v
```

### 4. Retrain / Regenerate Model Artifacts

```bash
python3 main.py
```

---

## 🖥️ Streamlit Application Workflow

The user interface is structured into **3 core pages**:

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  1. Upload Data │  ──▶  │   2. Analysis   │  ──▶  │  3. Methodology │
└─────────────────┘       └─────────────────┘       └─────────────────┘
```

### Page 1: 📂 Ingest Machine Telemetry
- **1-Click Reference Ingestion**: Instantly load the complete AI4I 2020 baseline dataset (10,000 records).
- **Custom CSV Upload**: Ingest telemetry with drag-and-drop support.
- **Automated Validation**: Schema verification, missing-value check, duplicate detection, and memory audit.

### Page 2: 📊 Behavioral Drift & Anomaly Analysis
- **Drift Verdict Banner**: Instant status notification (`BEHAVIORAL DRIFT DETECTED` vs `NOMINAL REGIME`).
- **Live Key Performance Indicators**: Records scored, flagged anomaly count, active threshold, and peak reconstruction errors.
- **Interactive Threshold Slider**: Adjust sensitivity dynamically with real-time recalculation without retraining.
- **Reconstruction Error Chart**: Time-series visualization with dynamic threshold boundary line.
- **Rolling Exceedance Plot**: 25-observation sliding window tracking the $60\%$ drift trigger threshold.
- **Sensor Signal Summary**: Feature-level statistics (mean, std dev, range) across raw and physics-derived signals.
- **Prioritised Investigation Checklist**: Actionable mechanical triage priorities (Sensor Calibration $\rightarrow$ Component Wear $\rightarrow$ Operating Regime $\rightarrow$ Operator Review).
- **Interactive Observation Table**: Filter by anomalies/normal, sort by error or sensor values, and download results as `.csv`.

### Page 3: 📖 Auditable Detection Methodology
- Comprehensive 7-stage explanation of the ML pipeline from raw sensor data to operational alarms.
- LaTeX mathematical formulas for physics features, loss function, Adam optimizer, threshold selection, and rolling persistence.
- Complete confusion matrix, performance metrics table, model hyperparameters, and interactive Graphviz architecture diagram.

---

## 🧠 Methodology & Machine Learning Architecture

```
Raw Telemetry (5 Sensors + Type)
               │
               ▼
   [Physics Feature Engineering]
   • Temp Difference (ΔT = T_process - T_air)
   • Mechanical Power (P = τ · ω · 2π / 60)
   • Overstrain Proxy (OS = wear · τ)
               │
               ▼
   [Standardisation & Encoding]
   • Z-score scaling (fitted on train normal only)
   • One-hot encoding of Machine Type (H, L, M)
               │
               ▼
   [NumPy Autoencoder (Adam)]
   11 ──▶ 64 ──▶ 32 ──▶ [7] ──▶ 32 ──▶ 64 ──▶ 11
               │
               ▼
   [Reconstruction Error Score]
   MSE: L = (1/d) Σ (x_i - x̂_i)²
               │
               ▼
   [Adaptive Threshold τ]
   98th percentile of normal validation errors
               │
               ▼
   [Rolling Persistence Filter]
   Window = 25 obs · Exceedance Fraction ≥ 60%
        ┌──────┴──────┐
        ▼             ▼
   [DRIFT ALARM]   [NOMINAL]
```

### 1. Domain Physics Features
Without physics features, subtle failure modes like Overstrain (OSF) and Thermal Dissipation (HDF) go largely unnoticed. Three domain interaction signals are computed without label leakage:
- **Thermal Gradient**: $\Delta T = T_{\text{process}} - T_{\text{air}}$
- **Mechanical Power**: $P = \tau \cdot \omega \cdot \frac{2\pi}{60} \quad [\text{Watts}]$
- **Overstrain Proxy**: $OS = \text{Tool wear} \cdot \tau \quad [\text{min}\cdot\text{N}\cdot\text{m}]$

> **Impact**: PR-AUC improves from **0.310 to 0.474 (+52.9%)** and Overstrain Failure (OSF) recall rises from **17.3% to 40.8%**.

### 2. Unsupervised Autoencoder (NumPy)
- **Normal-Only Training**: Only observations with $\text{Machine failure} = 0$ are used during training, ensuring zero target leakage.
- **Architecture**: Symmetric encoder-decoder ($11 \rightarrow 64 \rightarrow 32 \rightarrow 7 \rightarrow 32 \rightarrow 64 \rightarrow 11$) with ReLU activations.
- **Adam Optimizer**: Momentum vectors ($m_t, v_t$), bias correction, weight decay ($\lambda = 10^{-5}$), and gradient norm clipping ($5.0$).

### 3. Adaptive Threshold & Rolling Persistence Rule
- **Threshold**: Set at the **98.0th percentile** of normal validation reconstruction errors ($\tau \approx 0.02115$).
- **Rolling Window Persistence**: Transient spikes are filtered using a 25-observation sliding window. An alarm is triggered only when at least **60%** of consecutive observations exceed $\tau$:
  $$R_t = \frac{1}{W}\sum_{k=t-W+1}^{t} \mathbf{1}[\mathcal{L}(x_k) > \tau] \ge 0.60$$

---

## 📊 Held-Out Test Performance

Evaluated on the held-out stratified test set (1,501 rows; 51 failures, 1,450 normal):

| Metric | Autoencoder (Ours) | Statistical Z-Score | Isolation Forest |
|:---|:---:|:---:|:---:|
| **PR-AUC (Avg Precision)** | **0.4743** | 0.4227 | 0.2457 |
| **Balanced Accuracy** | **0.7078** | 0.7074 | 0.5918 |
| **F1-Score** | **0.4583** | 0.4536 | 0.2532 |
| **Recall** | **0.4314** | 0.4314 | 0.1961 |
| **Precision** | **0.4889** | 0.4783 | 0.3571 |
| **False Positive Rate** | **1.59%** | 1.66% | 1.24% |

---

## 📁 Repository Structure

```
amlpro/
├── ai4i2020.csv                    # AI4I 2020 Predictive Maintenance Dataset
├── config.json                     # Pipeline hyperparameters & configuration
├── main.py                         # Complete reproducible ML training pipeline
├── streamlit_app.py                # 3-Page Streamlit application
├── requirements.txt                # Python project dependencies
├── nasa_cmapps.py                  # Standalone NASA C-MAPSS RUL evaluation
│
├── outputs/
│   ├── figures/                    # Generated loss curves, PR curves, heatmaps
│   ├── metrics/                    # results.json, data_audit.json
│   ├── models/                     # autoencoder_numpy.pkl, preprocessing_schema.json
│   └── tables/                     # Test metrics, scores, descriptive statistics
│
├── tests/
│   └── test_pipeline.py            # Unit test suite (6 passing tests)
│
├── backend/                        # Optional FastAPI service
│   ├── app.py                      # REST endpoints for scoring and upload
│   └── requirements.txt
│
└── frontend/                       # Optional React / Vite UI
    ├── package.json
    └── src/
```

---

## 🔬 Standalone NASA C-MAPSS Evaluation

In addition to AI4I drift detection, the repository contains a standalone **Remaining Useful Life (RUL)** baseline for turbofan engine degradation (`archive/train_FD001-4.txt`).

To execute the regularized Ridge regression baseline across all four subsets:

```bash
python3 nasa_cmapps.py
```

Results are stored in `outputs/metrics/nasa_cmapps_results.json`.

| Subset | Engines (Train / Test) | Operating Conditions | Fault Modes | MAE (cycles) | RMSE |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **FD001** | 100 / 100 | 1 (Sea Level) | 1 (HPC Degradation) | **17.79** | 22.55 |
| **FD002** | 260 / 259 | 6 Conditions | 1 (HPC Degradation) | **24.05** | 31.90 |
| **FD003** | 100 / 100 | 1 (Sea Level) | 2 (HPC + Fan) | **18.18** | 22.56 |
| **FD004** | 248 / 249 | 6 Conditions | 2 (HPC + Fan) | **29.23** | 36.14 |

---

## 📄 License & Integrity

- **Strict No Data Leakage**: All normalisation scalers and thresholds are strictly fitted on training splits with normal records only.
- **Deterministic**: Controlled by fixed random seeds (`seed: 42`).
- **Reproducible**: Every artifact, figure, and table can be regenerated with a single command (`python3 main.py`).
