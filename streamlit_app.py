"""
Sentinel ML — Industrial Behavioral Drift Detection
Streamlit UI  ·  3 pages: Upload → Analysis → Methodology

Run:  .venv/bin/streamlit run streamlit_app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from main import (
    Autoencoder,
    add_physics_features,
    drift_analysis,
    reconstruction_scores,
)

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent
DATASET = ROOT / "ai4i2020.csv"
RESULTS = ROOT / "outputs" / "metrics" / "results.json"
SCHEMA  = ROOT / "outputs" / "models" / "preprocessing_schema.json"
MODEL   = ROOT / "outputs" / "models" / "autoencoder_numpy.pkl"

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentinel ML | Industrial Drift Detection",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background:#0d1117; color:#c9d1d9; }
[data-testid="stSidebar"]          { background:#161b22; border-right:1px solid #21262d; }
[data-testid="stHeader"]           { background:transparent !important; }
footer, #MainMenu { visibility:hidden; }

h1,h2,h3,h4 { color:#e6edf3 !important; }
p, li        { color:#8b949e; }
label        { color:#8b949e !important; }

[data-testid="metric-container"] {
    background:#161b22; border:1px solid #21262d;
    border-radius:10px; padding:1rem 1.2rem .8rem;
}
[data-testid="stMetricLabel"] {
    font-size:.72rem !important; letter-spacing:.1em;
    color:#8b949e !important; text-transform:uppercase;
}
[data-testid="stMetricValue"] {
    font-size:1.6rem !important; color:#e6edf3 !important; font-weight:700;
}
[data-testid="stMetricDelta"] { font-size:.8rem !important; }

.drift-alarm {
    background:linear-gradient(135deg,#3d1f1f,#1a0808);
    border:1.5px solid #f85149; border-radius:10px;
    padding:1.1rem 1.5rem; margin-bottom:1.2rem;
}
.drift-alarm .badge  { color:#f85149 !important; font-size:1.1rem; font-weight:800; letter-spacing:.06em; }
.drift-alarm .detail { color:#cba6a5 !important; font-size:.88rem; margin-top:.3rem; }

.drift-ok {
    background:linear-gradient(135deg,#1a2e1a,#0c1c0c);
    border:1.5px solid #3fb950; border-radius:10px;
    padding:1.1rem 1.5rem; margin-bottom:1.2rem;
}
.drift-ok .badge  { color:#3fb950 !important; font-size:1.1rem; font-weight:800; letter-spacing:.06em; }
.drift-ok .detail { color:#9ec9a5 !important; font-size:.88rem; margin-top:.3rem; }

.sec-head {
    border-bottom:1px solid #21262d; padding-bottom:.4rem;
    margin:1.8rem 0 .9rem 0; color:#58a6ff !important;
    font-size:.82rem; letter-spacing:.14em; text-transform:uppercase; font-weight:700;
}

.mcard {
    background:#161b22; border-left:3px solid #58a6ff;
    border-radius:0 8px 8px 0; padding:.9rem 1.3rem; margin-bottom:.75rem;
}
.mcard h4 { color:#58a6ff !important; margin:0 0 .35rem 0; font-size:.92rem; }
.mcard p  { color:#8b949e; margin:0; font-size:.87rem; line-height:1.55; }

[data-testid="stFileUploader"] {
    border:2px dashed #30363d; border-radius:10px; padding:1rem; background:#161b22;
}
[data-testid="stFileUploader"]:hover { border-color:#58a6ff; }

[data-testid="stExpander"] {
    background:#161b22; border:1px solid #21262d; border-radius:8px; margin-bottom:.6rem;
}
[data-testid="stExpander"] summary { color:#c9d1d9 !important; }

[data-testid="baseButton-primary"] {
    background:#1f6feb !important; border:1px solid #388bfd !important;
    color:#fff !important; border-radius:8px !important; font-weight:700 !important;
}
[data-testid="baseButton-secondary"] {
    background:#21262d !important; border:1px solid #30363d !important;
    color:#c9d1d9 !important; border-radius:8px !important;
}

[data-testid="stSlider"] > div > div > div > div { background:#58a6ff !important; }
[data-testid="stDataFrame"] { border:1px solid #21262d; border-radius:8px; }

.empty-state {
    background:#161b22; border:1px dashed #30363d;
    border-radius:10px; padding:2.8rem; text-align:center; margin-top:1.2rem;
}

div[role="radiogroup"] label { color:#c9d1d9 !important; font-size:.88rem; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
_DEFAULTS = {
    "active_df":        None,
    "dataset_name":     None,
    "analysis_done":    False,
    "scores":           None,
    "drift_flags":      None,
    "rolling":          None,
    "active_threshold": None,
    "results_df":       None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ═══════════════════════════════════════════════════════════════════════════════
#  CACHED HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading model …")
def _load_artifacts():
    schema  = json.loads(SCHEMA.read_text())
    results = json.loads(RESULTS.read_text())
    model   = Autoencoder(
        n_in=len(schema["input_features"]),
        latent=results["configuration"].get("latent_dim", 7),
    )
    model.load(MODEL)
    return model, schema, results


def _score_dataframe(df: pd.DataFrame, schema: dict, model: Autoencoder) -> np.ndarray:
    enriched   = add_physics_features(df)
    num_cols   = schema["numeric_features"]
    means      = np.asarray(schema["mean"],  dtype=float)
    scale      = np.asarray(schema["scale"], dtype=float)
    categories = schema["category_levels"]

    cat_vals = (
        enriched["Type"].astype(str).to_numpy()
        if "Type" in enriched.columns
        else np.full(len(enriched), "L")
    )
    encoded = np.column_stack([cat_vals == c for c in categories]).astype(float)

    values = np.zeros((len(enriched), len(num_cols)), dtype=float)
    for i, col in enumerate(num_cols):
        if col in enriched.columns:
            values[:, i] = enriched[col].to_numpy(float)

    values = (values - means) / scale
    x = np.ascontiguousarray(np.column_stack([values, encoded]), dtype=np.float64)
    return reconstruction_scores(model, x)


def _run_analysis(threshold: float) -> None:
    model, schema, _ = _load_artifacts()
    df     = st.session_state.active_df
    scores = _score_dataframe(df, schema, model)
    rolling, drift = drift_analysis(scores, threshold, window=25, fraction=0.6)

    st.session_state.scores           = scores
    st.session_state.drift_flags      = drift
    st.session_state.rolling          = rolling
    st.session_state.active_threshold = threshold
    st.session_state.analysis_done    = True

    def _col(name):
        c = df.get(name)
        return c.to_numpy() if c is not None else np.zeros(len(df))

    st.session_state.results_df = pd.DataFrame({
        "Row #":           np.arange(len(df)),
        "Error Score":     np.round(scores, 5),
        "Threshold":       np.round(threshold, 5),
        "Anomaly":         (scores > threshold).astype(int),
        "Rolling Exceed":  np.round(rolling, 3),
        "Drift Flag":      drift,
        "Torque [Nm]":     _col("Torque [Nm]"),
        "Speed [rpm]":     _col("Rotational speed [rpm]"),
        "Tool Wear [min]": _col("Tool wear [min]"),
        "Air Temp [K]":    _col("Air temperature [K]"),
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<div style='font-size:1.4rem;font-weight:800;letter-spacing:.12em;"
        "color:#58a6ff;padding:.5rem 0 .05rem'>◈ SENTINEL/ML</div>"
        "<div style='font-size:.62rem;letter-spacing:.22em;color:#30363d;"
        "margin-bottom:1.3rem'>INDUSTRIAL DRIFT DETECTION</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    page = st.radio(
        "nav",
        ["📂  Upload Data", "📊  Analysis", "📖  Methodology"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown(
        "<div style='font-size:.62rem;letter-spacing:.18em;color:#30363d;"
        "margin-bottom:.5rem'>DATASET</div>",
        unsafe_allow_html=True,
    )
    if st.session_state.active_df is not None:
        _t = st.session_state.active_df
        st.success(f"✔ {st.session_state.dataset_name}")
        st.caption(f"{len(_t):,} rows · {len(_t.columns)} cols")
    else:
        st.info("No dataset loaded")

    st.divider()
    st.markdown(
        "<div style='font-size:.62rem;letter-spacing:.18em;color:#30363d;"
        "margin-bottom:.5rem'>MODEL</div>",
        unsafe_allow_html=True,
    )
    if MODEL.exists() and SCHEMA.exists() and RESULTS.exists():
        _r  = json.loads(RESULTS.read_text())
        _ae = _r.get("autoencoder", {})
        st.success("✔ Autoencoder ready")
        st.caption(
            f"F1 {_ae.get('f1', 0):.3f}  ·  "
            f"PR-AUC {_ae.get('pr_auc_average_precision', 0):.3f}  ·  "
            f"Recall {_ae.get('recall', 0):.3f}"
        )
    else:
        st.error("✘ Run `python main.py` first")

    st.divider()
    st.markdown(
        "<div style='font-size:.65rem;color:#21262d'>"
        "v2.1 · NumPy Autoencoder · AI4I 2020</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📂  Upload Data":
    st.markdown("## 📂 Ingest Machine Telemetry")
    st.markdown(
        "<p>Upload manufacturing telemetry as a CSV, or use the built-in "
        "AI4I 2020 reference dataset to explore the full drift detection pipeline.</p>",
        unsafe_allow_html=True,
    )

    ref_col, _ = st.columns([2, 5])
    with ref_col:
        if st.button("⚡  Use AI4I 2020 Reference Dataset", use_container_width=True):
            with st.spinner("Loading …"):
                _ref = pd.read_csv(DATASET)
                st.session_state.active_df     = _ref
                st.session_state.dataset_name  = "AI4I 2020 Reference"
                st.session_state.analysis_done = False
            st.success(f"Loaded {len(_ref):,} records from the AI4I 2020 reference dataset.")

    st.markdown('<div class="sec-head">▸ Custom CSV Upload</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drag & drop a CSV or click to browse",
        type=["csv"],
        help="Required columns: Type · Air temperature [K] · Process temperature [K] "
             "· Rotational speed [rpm] · Torque [Nm] · Tool wear [min]",
    )
    if uploaded is not None:
        try:
            with st.spinner("Parsing CSV …"):
                _udf = pd.read_csv(uploaded)
            st.session_state.active_df     = _udf
            st.session_state.dataset_name  = uploaded.name
            st.session_state.analysis_done = False
            st.success(f"✔  Ingested **{len(_udf):,}** records from `{uploaded.name}`")
        except Exception as exc:
            st.error(f"Could not parse CSV: {exc}")

    with st.expander("📋  Required Column Schema", expanded=False):
        st.dataframe(
            pd.DataFrame({
                "Column":  ["Type", "Air temperature [K]", "Process temperature [K]",
                            "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"],
                "Dtype":   ["string (L/M/H)", "float", "float", "int", "float", "int"],
                "Notes":   ["Machine quality tier", "Ambient air temperature (Kelvin)",
                            "In-process temperature (Kelvin)", "Spindle angular velocity",
                            "Applied drive torque", "Cumulative tool wear duration"],
            }),
            width="stretch", hide_index=True,
        )
        st.caption(
            "Physics features (Temp difference [K], Power [W], Overstrain [min·Nm]) "
            "are computed automatically — do not include them in your upload."
        )

    if st.session_state.active_df is not None:
        _df = st.session_state.active_df

        st.markdown('<div class="sec-head">▸ Dataset Summary</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records",  f"{len(_df):,}")
        c2.metric("Total Columns",  len(_df.columns))
        c3.metric("Duplicate Rows", int(_df.duplicated().sum()))
        c4.metric("Memory (KB)",    f"{_df.memory_usage(deep=True).sum() / 1024:.1f}")

        st.markdown('<div class="sec-head">▸ Data Preview (first 8 rows)</div>', unsafe_allow_html=True)
        st.dataframe(_df.head(8), width="stretch", hide_index=True)

        with st.expander("🔍  Column Type Audit", expanded=False):
            st.dataframe(
                pd.DataFrame({
                    "Column":   _df.columns.tolist(),
                    "Dtype":    [str(_df[c].dtype) for c in _df.columns],
                    "Non-null": [int(_df[c].notna().sum()) for c in _df.columns],
                    "Nulls":    [int(_df[c].isna().sum())  for c in _df.columns],
                    "Unique":   [int(_df[c].nunique())      for c in _df.columns],
                }),
                width="stretch", hide_index=True,
            )

        st.info("✅  Dataset loaded — go to **📊 Analysis** in the sidebar to run drift detection.")

    else:
        st.markdown("""
        <div class="empty-state">
            <div style="font-size:2.2rem">📡</div>
            <div style="color:#58a6ff;font-size:1.05rem;font-weight:700;margin:.5rem 0 .3rem">
                No dataset loaded yet</div>
            <div style="color:#8b949e;font-size:.88rem">
                Upload a CSV above or click "Use AI4I 2020 Reference Dataset"</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Analysis":
    st.markdown("## 📊 Behavioral Drift & Anomaly Analysis")

    if st.session_state.active_df is None:
        st.warning("⚠️  No dataset loaded. Go to **📂 Upload Data** first.")
        st.stop()
    if not MODEL.exists() or not SCHEMA.exists() or not RESULTS.exists():
        st.error("⚠️  Model artifacts missing. Run `python main.py` first.")
        st.stop()

    _model, _schema, _results_meta = _load_artifacts()
    _default_thr = float(_results_meta["threshold"])

    # threshold slider
    st.markdown('<div class="sec-head">▸ Anomaly Decision Threshold</div>', unsafe_allow_html=True)
    tc1, tc2, tc3 = st.columns([5, 1, 1])
    with tc1:
        threshold = st.slider(
            "thr",
            min_value=0.005, max_value=0.12,
            value=float(st.session_state.active_threshold or _default_thr),
            step=0.001, format="%.4f",
            label_visibility="collapsed",
            help="Reconstruction error cutoff. Lower = more sensitive. Higher = more conservative.",
        )
    with tc2:
        st.metric("Active Cutoff",   f"{threshold:.4f}")
    with tc3:
        st.metric("Baseline (98th)", f"{_default_thr:.4f}")

    # run / re-run scoring
    _needs_run = (
        not st.session_state.analysis_done
        or abs((st.session_state.active_threshold or 0) - threshold) > 1e-7
    )
    if _needs_run:
        with st.spinner("Scoring observations …"):
            _run_analysis(threshold)

    _scores  = st.session_state.scores
    _drift   = st.session_state.drift_flags
    _rolling = st.session_state.rolling
    _res_df  = st.session_state.results_df
    _df      = st.session_state.active_df

    _mask      = _scores > threshold
    _n_anom    = int(_mask.sum())
    _anom_rate = _n_anom / len(_df) * 100
    _is_drift  = bool(_drift.any())
    _n_dwin    = int(_drift.sum())

    # verdict banner
    if _is_drift:
        st.markdown(f"""
        <div class="drift-alarm">
            <div class="badge">🚨 BEHAVIORAL DRIFT DETECTED</div>
            <div class="detail">
                {_n_dwin} rolling window(s) exceeded 60% persistence threshold.
                Operational alarm triggered — prioritise sensor and component inspection.
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="drift-ok">
            <div class="badge">✅ NOMINAL REGIME</div>
            <div class="detail">
                Machine telemetry within historical tolerance.
                {_n_anom} isolated anomalies ({_anom_rate:.1f}%) — no persistent drift at this threshold.
            </div>
        </div>""", unsafe_allow_html=True)

    # KPI cards
    st.markdown('<div class="sec-head">▸ Key Performance Indicators</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Records Scored",    f"{len(_df):,}")
    k2.metric("Anomalies Flagged", f"{_n_anom:,}",   f"{_anom_rate:.1f}% of total")
    k3.metric("Drift Windows",     f"{_n_dwin}",
              delta="⚠ ACTIVE" if _is_drift else "✔ NONE",
              delta_color="inverse" if _is_drift else "off")
    k4.metric("Peak Error Score",  f"{_scores.max():.5f}")
    k5.metric("Mean Error Score",  f"{_scores.mean():.5f}")

    # reconstruction error chart
    st.markdown('<div class="sec-head">▸ Reconstruction Error Time Series</div>', unsafe_allow_html=True)
    _N = min(len(_scores), 700)
    st.line_chart(
        pd.DataFrame({
            "Reconstruction Error": _scores[:_N],
            "Decision Threshold":   np.full(_N, threshold),
        }, index=np.arange(_N)),
        color=["#58a6ff", "#f85149"],
    )
    st.caption(
        f"First {_N:,} of {len(_scores):,} observations.  "
        f"Blue = error score · Red = decision threshold ({threshold:.4f})."
    )

    with st.expander("📈  Rolling Exceedance Fraction (25-obs window)", expanded=False):
        st.line_chart(
            pd.DataFrame({
                "Exceedance Fraction":  _rolling[:_N],
                "Drift Trigger (0.60)": np.full(_N, 0.60),
            }, index=np.arange(_N)),
            color=["#e3b341", "#f85149"],
        )
        st.caption(
            "Yellow = fraction of the last 25 observations exceeding threshold.  "
            "When yellow ≥ red line (0.60), a drift alarm fires."
        )

    # sensor signal summary
    st.markdown('<div class="sec-head">▸ Sensor Signal Summary</div>', unsafe_allow_html=True)
    _enriched = add_physics_features(_df)
    _sig_rows = []
    for _lbl, _col in {
        "Air Temp [K]":           "Air temperature [K]",
        "Process Temp [K]":       "Process temperature [K]",
        "Rotational Speed [rpm]": "Rotational speed [rpm]",
        "Torque [Nm]":            "Torque [Nm]",
        "Tool Wear [min]":        "Tool wear [min]",
        "Temp Diff [K]":          "Temp difference [K]",
        "Power [W]":              "Power [W]",
        "Overstrain [min·Nm]":    "Overstrain [min Nm]",
    }.items():
        _s = _enriched.get(_col)
        if _s is None:
            continue
        _v = _s.dropna().to_numpy(float)
        if len(_v) == 0:
            continue
        _sig_rows.append({
            "Signal":  _lbl,
            "Mean":    round(float(_v.mean()), 3),
            "Std Dev": round(float(_v.std()),  3),
            "Min":     round(float(_v.min()),  3),
            "Max":     round(float(_v.max()),  3),
            "Range":   round(float(_v.max() - _v.min()), 3),
        })
    if _sig_rows:
        st.dataframe(pd.DataFrame(_sig_rows), width="stretch", hide_index=True)

    # investigation checklist
    st.markdown('<div class="sec-head">▸ Prioritised Investigation Checklist</div>', unsafe_allow_html=True)
    for _t, _b in [
        ("🔴 Priority 1 — Sensor Calibration",
         "Verify transducer calibration, temperature probe drift, and unit alignments. "
         "A single miscalibrated sensor shifts the entire reconstruction error distribution."),
        ("🟠 Priority 2 — Component Wear",
         "Inspect tooling wear logs and bearings on components with elevated overstrain or power draw. "
         "Cross-reference Tool Wear [min] and Torque [Nm] against the maintenance schedule."),
        ("🟡 Priority 3 — Operating Regime Change",
         "Confirm no recipe, feed rate, or rotational speed change was made intentionally. "
         "Deliberate process changes can trigger false alarms — compare operator logs."),
        ("🔵 Priority 4 — Operator Field Review",
         "Confirm the highest-scoring observations with a plant technician. "
         "Sort the table by Error Score descending and review the top 10 manually."),
    ]:
        st.markdown(f'<div class="mcard"><h4>{_t}</h4><p>{_b}</p></div>', unsafe_allow_html=True)

    # observation table
    st.markdown('<div class="sec-head">▸ Scored Observations</div>', unsafe_allow_html=True)
    _fc1, _fc2 = st.columns([3, 4])
    with _fc1:
        _filter = st.radio(
            "f", ["All", "Anomalies Only", "Normal Only"],
            horizontal=True, label_visibility="collapsed",
        )
    with _fc2:
        _sort = st.selectbox(
            "s",
            ["Error Score", "Row #", "Rolling Exceed", "Torque [Nm]", "Speed [rpm]"],
            label_visibility="collapsed",
        )

    _disp = _res_df.copy()
    if _filter == "Anomalies Only": _disp = _disp[_disp["Anomaly"] == 1]
    elif _filter == "Normal Only":  _disp = _disp[_disp["Anomaly"] == 0]
    _disp = _disp.sort_values(_sort, ascending=(_sort == "Row #"))

    st.caption(f"Showing {min(len(_disp), 300):,} of {len(_disp):,} rows (max 300 rendered).")
    st.dataframe(_disp.head(300), width="stretch", hide_index=True)
    st.download_button(
        "⬇  Export Scored Telemetry (.csv)",
        data=_disp.to_csv(index=False).encode("utf-8"),
        file_name="sentinel_scored_telemetry.csv",
        mime="text/csv",
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — METHODOLOGY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📖  Methodology":
    st.markdown("## 📖 Auditable Detection Methodology")
    st.markdown(
        "<p>Full pipeline documentation — from raw sensor data to operational drift alarms — "
        "so that plant engineers, auditors, and ML practitioners can verify "
        "and reproduce every decision.</p>",
        unsafe_allow_html=True,
    )

    with st.expander("Stage 1 — Data Ingestion & Schema Validation", expanded=True):
        st.markdown("""
**What happens:**  
Raw manufacturing telemetry is ingested as a CSV. The pipeline validates that
five mandatory sensor channels are present and assigns a schema version for reproducibility.

| Column | Unit | Description |
|--------|------|-------------|
| `Type` | L / M / H | Machine quality tier |
| `Air temperature` | K | Ambient air temperature |
| `Process temperature` | K | In-process operating temperature |
| `Rotational speed` | rpm | Spindle angular velocity |
| `Torque` | N·m | Applied drive torque |
| `Tool wear` | min | Cumulative tool wear duration |

Kelvin is used (not Celsius) to avoid negative-number artefacts in physics calculations.
        """)

    with st.expander("Stage 2 — Domain Physics Feature Engineering", expanded=True):
        st.markdown(r"""
Three interaction features are derived from raw sensor channels without label leakage:

$$\Delta T = T_{\text{process}} - T_{\text{air}} \quad \text{[thermal gradient, K]}$$

$$P = \tau \cdot \omega \cdot \frac{2\pi}{60} \quad \text{[mechanical power, W]}$$

$$OS = \text{wear} \cdot \tau \quad \text{[overstrain proxy, min·N·m]}$$

**Impact on detection performance:**

| Metric | Without physics | With physics | Improvement |
|--------|----------------|--------------|-------------|
| PR-AUC | 0.310 | **0.474** | +52.9 % |
| Recall | 0.334 | **0.431** | +29.4 % |
| OSF recall | 17.3 % | **40.8 %** | +136 % |

The thermal gradient $\Delta T$ is the strongest single predictor of thermal failure modes.
        """)

    with st.expander("Stage 3 — Preprocessing & Normalisation", expanded=False):
        st.markdown(r"""
Numeric features are **z-score standardised** using statistics computed on the training split only
(no data leakage):

$$\hat{x}_i = \frac{x_i - \mu_i}{\sigma_i}$$

The categorical column `Type` is one-hot encoded into three binary columns (H, L, M).
The preprocessing schema (mean, std, category levels) is saved to
`outputs/models/preprocessing_schema.json` so new datasets are processed identically.

**Data split:** 70 % training · 15 % validation · 15 % test — stratified by failure label.

**Normal-only training:** Only `Machine failure = 0` rows enter the autoencoder training loop.
This prevents label leakage — the model learns what *normal* looks like, not failure patterns.
        """)

    with st.expander("Stage 4 — Autoencoder Architecture (NumPy · Adam)", expanded=True):
        col_arch, col_params = st.columns([3, 2])
        with col_arch:
            st.markdown(r"""
**Symmetric encoder-decoder:**
```
Input  (11)
  → Dense 64  → ReLU
  → Dense 32  → ReLU
  → Latent  7   ← bottleneck
  → Dense 32  → ReLU
  → Dense 64  → ReLU
  → Output (11)
```
**Loss — Mean Squared Error:**
$$\mathcal{L} = \frac{1}{d}\sum_{i=1}^{d}(x_i - \hat{x}_i)^2$$

**Why unsupervised?**  
Failure labels are expensive and inconsistently assigned in real plants.
An autoencoder trained on normal data generalises to any unlabelled production stream.
            """)
        with col_params:
            st.markdown(r"""
**Adam optimiser:**
$$m_t = \beta_1 m_{t-1} + (1{-}\beta_1)g_t$$
$$v_t = \beta_2 v_{t-1} + (1{-}\beta_2)g_t^2$$
$$\theta \mathrel{-}= \frac{\alpha}{\sqrt{\hat{v}_t}+\varepsilon}\hat{m}_t + \lambda\theta$$
            """)
            st.dataframe(
                pd.DataFrame({
                    "Hyperparameter": [
                        "Latent dim", "Learning rate", "β₁", "β₂",
                        "Weight decay", "Batch size", "Max epochs",
                        "Early-stop patience", "Gradient clip",
                    ],
                    "Value": ["7", "0.005", "0.9", "0.999",
                              "1×10⁻⁵", "128", "150", "15", "5.0"],
                }),
                width="stretch", hide_index=True,
            )

    with st.expander("Stage 5 — Adaptive Anomaly Threshold", expanded=True):
        st.markdown(r"""
After training, reconstruction errors are computed for every validation-split normal observation.
The threshold is set at the 98th percentile:

$$\tau = \text{percentile}_{98}\!\left(\{\mathcal{L}(x_i) : x_i \in \mathcal{D}_{\text{val, normal}}\}\right)$$

At the 98th percentile, ~2 % of normal observations trigger false positives —
acceptable for a monitoring alarm system.

The **threshold slider** on the Analysis page adjusts sensitivity in real time without retraining:
- Lower $\tau$ → higher recall (catches more anomalies, more false positives)
- Higher $\tau$ → higher precision (fewer, surer anomalies)
        """)

    with st.expander("Stage 6 — Rolling Persistence Drift Alarm", expanded=True):
        col_eq, col_tbl = st.columns([3, 2])
        with col_eq:
            st.markdown(r"""
A single anomalous point can be transient noise.
The drift alarm fires only when anomalies *persist* across a sliding window:

$$R_t = \frac{1}{W}\sum_{k=t-W+1}^{t}\mathbf{1}\!\left[\mathcal{L}(x_k) > \tau\right]$$

$$\text{DRIFT}_t = \begin{cases}1 & R_t \geq f \\ 0 & \text{otherwise}\end{cases}$$

True mechanical degradation is persistent — a bearing wearing out does not briefly
return to normal. Requiring 60 % of a 25-obs window eliminates transient spikes
from vibration, material variation, or sensor glitches.
            """)
        with col_tbl:
            st.dataframe(
                pd.DataFrame({
                    "Parameter":  ["Window W",   "Fraction f", "Alarm condition"],
                    "Value":      ["25 obs",      "0.60",       "R_t ≥ 0.60"],
                    "Rationale":  [
                        "~25 consecutive machine cycles",
                        "60% = sustained degradation",
                        "Yellow line ≥ red line on chart",
                    ],
                }),
                width="stretch", hide_index=True,
            )

    with st.expander("Stage 7 — Test-Set Performance", expanded=True):
        if RESULTS.exists():
            _r  = json.loads(RESULTS.read_text())
            _ae = _r.get("autoencoder", {})
            _p1, _p2 = st.columns(2)
            with _p1:
                st.markdown("**Classification metrics (held-out test split)**")
                st.dataframe(
                    pd.DataFrame({
                        "Metric": ["F1 Score", "Recall", "Precision",
                                   "PR-AUC", "Balanced Accuracy", "False Positive Rate"],
                        "Value":  [
                            f"{_ae.get('f1', 0):.4f}",
                            f"{_ae.get('recall', 0):.4f}",
                            f"{_ae.get('precision', 0):.4f}",
                            f"{_ae.get('pr_auc_average_precision', 0):.4f}",
                            f"{_ae.get('balanced_accuracy', 0):.4f}",
                            f"{_ae.get('false_positive_rate', 0):.4f}",
                        ],
                    }),
                    width="stretch", hide_index=True,
                )
            with _p2:
                st.markdown("**Confusion matrix (test split)**")
                _tn, _fp = _ae.get("tn", 0), _ae.get("fp", 0)
                _fn, _tp = _ae.get("fn", 0), _ae.get("tp", 0)
                st.dataframe(
                    pd.DataFrame({
                        "":               ["Pred Normal", "Pred Anomaly"],
                        "Actual Normal":  [f"TN = {_tn}", f"FP = {_fp}"],
                        "Actual Anomaly": [f"FN = {_fn}", f"TP = {_tp}"],
                    }),
                    width="stretch", hide_index=True,
                )
                st.markdown("**Training configuration**")
                _cfg = _r.get("configuration", {})
                st.dataframe(
                    pd.DataFrame({
                        "Parameter": list(_cfg.keys()),
                        "Value":     [str(v) for v in _cfg.values()],
                    }),
                    width="stretch", hide_index=True,
                )
        else:
            st.warning("Run `python main.py` to populate performance metrics.")

    # drift resolution guide
    st.markdown('<div class="sec-head">▸ How to Resolve a Drift Alarm</div>', unsafe_allow_html=True)
    for _step, _body in [
        ("1. Isolate the signal source",
         "Compare reconstruction errors per sensor channel. The sensor contributing most "
         "to the error increase indicates the degraded subsystem."),
        ("2. Check the maintenance schedule",
         "Cross-reference the alarm timestamp with the maintenance log. If Tool Wear [min] "
         "is near the manufacturer's rated limit, schedule a tool change."),
        ("3. Inspect for intentional process change",
         "Confirm with operations that no recipe, speed, or material change occurred. "
         "Intentional changes may require model re-calibration."),
        ("4. Validate sensor hardware",
         "For persistent thermal drift (high ΔT anomalies), inspect thermocouple "
         "placements and calibration certificates."),
        ("5. Re-baseline if the process changed permanently",
         "If the operating regime has permanently shifted, retrain on a new normal-operation "
         "window using `python main.py --csv new_baseline.csv`."),
    ]:
        st.markdown(
            f'<div class="mcard"><h4>{_step}</h4><p>{_body}</p></div>',
            unsafe_allow_html=True,
        )

    # pipeline flowchart
    st.markdown('<div class="sec-head">▸ Pipeline Architecture</div>', unsafe_allow_html=True)
    st.graphviz_chart("""
digraph sentinel {
    rankdir=LR
    bgcolor="#0d1117"
    node [fontname="Helvetica" fontsize=11 fontcolor="#e6edf3"
          style=filled fillcolor="#161b22" color="#21262d" margin="0.15,0.1"]
    edge [color="#58a6ff" fontcolor="#8b949e" fontsize=9]

    A [label="Raw CSV\\n(5 sensor channels)" shape=cylinder]
    B [label="Physics Engineering\\n(ΔT · P · OS)"    shape=box]
    C [label="Z-score Norm\\n+ One-hot Type"          shape=box]
    D [label="Autoencoder\\n11→64→32→7→32→64→11\\nAdam · normal-only only"
                                                       shape=box fillcolor="#1f3050" color="#388bfd"]
    E [label="Reconstruction\\nError Score"            shape=diamond fillcolor="#2d2000" color="#e3b341"]
    F [label="Threshold τ\\n(98th percentile)"         shape=box]
    G [label="Rolling Window\\nW=25 · f=0.60"          shape=box]
    H [label="DRIFT ALARM"  shape=oval fillcolor="#2d0a0a" color="#f85149"]
    I [label="NOMINAL"      shape=oval fillcolor="#0d2d0d" color="#3fb950"]

    A -> B -> C -> D -> E -> F -> G
    G -> H [label=" ≥ 60%"]
    G -> I [label=" < 60%"]
}
""")

    st.markdown("""
---
<p style="color:#30363d;font-size:.77rem;text-align:center;">
Sentinel ML v2.1 · NumPy Autoencoder · AI4I 2020 Benchmark ·
All physics features derived without label leakage
</p>""", unsafe_allow_html=True)
