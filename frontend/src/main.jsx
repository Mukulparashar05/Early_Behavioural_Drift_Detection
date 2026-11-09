import React, { useEffect, useState, useMemo } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function apiGet(path) {
  const response = await fetch(`${API}${path}`);
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail?.message || errorData.detail || `Request failed (${response.status})`);
  }
  return response.json();
}

async function apiPost(path, body = null) {
  const options = { method: "POST" };
  if (body) {
    options.headers = { "Content-Type": "application/json" };
    options.body = JSON.stringify(body);
  }
  const response = await fetch(`${API}${path}`, options);
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail?.message || errorData.detail || `Request failed (${response.status})`);
  }
  return response.json();
}

async function apiPatch(path, body) {
  const response = await fetch(`${API}${path}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail?.message || errorData.detail || `Request failed (${response.status})`);
  }
  return response.json();
}

function App() {
  const [page, setPage] = useState("Upload");
  const [analysis, setAnalysis] = useState(null);
  const [activeThreshold, setActiveThreshold] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [notification, setNotification] = useState("");

  const loadActiveAnalysis = async () => {
    try {
      setError("");
      const res = await apiGet("/api/dataset/active-analysis");
      setAnalysis(res);
      if (res.threshold) {
        setActiveThreshold(Number(res.threshold));
      }
      return res;
    } catch (err) {
      setError(err.message);
      return null;
    }
  };

  useEffect(() => {
    loadActiveAnalysis();
  }, []);

  const nav = ["Upload", "Analysis", "Methodology"];

  const handleThresholdChange = async (newThr) => {
    setActiveThreshold(newThr);
    try {
      await apiPatch("/api/settings/threshold", { threshold: newThr });
      await loadActiveAnalysis();
      setNotification(`Active decision threshold adjusted to ${newThr.toFixed(4)}.`);
      setTimeout(() => setNotification(""), 4000);
    } catch (e) {
      setError(e.message);
    }
  };

  return (
    <div className="app">
      <aside>
        <div className="brand">
          <span>◈</span> SENTINEL<span className="muted">/ ML</span>
        </div>
        <p className="eyebrow">INDUSTRIAL ANALYTICS</p>
        {nav.map((item) => (
          <button
            className={page === item ? "nav active" : "nav"}
            onClick={() => setPage(item)}
            key={item}
          >
            {item}
          </button>
        ))}
        <div className="side-note">
          <b>Pipeline status</b>
          <span className="dot" />
          Autoencoder Ready
          <br />
          <small>AI4I 2020 · Adam Optimizer</small>
        </div>
      </aside>

      <main>
        <header>
          <div>
            <p className="eyebrow">BEHAVIORAL DRIFT DETECTION SYSTEM</p>
            <h1>{page}</h1>
          </div>
          <a className="export" href={`${API}/api/export`} download>
            Export Predictions (.csv) ↗
          </a>
        </header>

        {notification && <div className="success">{notification}</div>}
        {error && <div className="error">{error}</div>}

        {page === "Upload" && (
          <UploadPage
            onAnalysisComplete={(result) => {
              setAnalysis(result);
              if (result.threshold) setActiveThreshold(Number(result.threshold));
              setNotification(`Dataset analyzed successfully! Transferred ${result.sample_size} records to Analysis.`);
              setTimeout(() => setNotification(""), 4000);
              setPage("Analysis");
            }}
            onError={(msg) => setError(msg)}
            onUseReference={async () => {
              setLoading(true);
              try {
                const res = await apiPost("/api/dataset/analyze-reference");
                setAnalysis(res);
                if (res.threshold) setActiveThreshold(Number(res.threshold));
                setNotification("Loaded reference dataset (AI4I 2020) into Analysis.");
                setTimeout(() => setNotification(""), 4000);
                setPage("Analysis");
              } catch (e) {
                setError(e.message);
              } finally {
                setLoading(false);
              }
            }}
            loading={loading}
          />
        )}

        {page === "Analysis" && (
          <AnalysisPage
            analysis={analysis}
            activeThreshold={activeThreshold}
            onThresholdChange={handleThresholdChange}
            onRefresh={loadActiveAnalysis}
            onNavigateUpload={() => setPage("Upload")}
          />
        )}

        {page === "Methodology" && <MethodologyPage />}
      </main>
    </div>
  );
}

function UploadPage({ onAnalysisComplete, onError, onUseReference, loading }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (!selected) return;
    setFile(selected);

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const text = event.target.result;
        const lines = text.split("\n").filter((l) => l.trim().length > 0);
        if (lines.length > 0) {
          const headers = lines[0].split(",").map((h) => h.trim().replace(/^"|"$/g, ""));
          const sampleRows = lines.slice(1, 6).map((line) => {
            const vals = line.split(",").map((v) => v.trim().replace(/^"|"$/g, ""));
            const obj = {};
            headers.forEach((h, i) => {
              obj[h] = vals[i] || "";
            });
            return obj;
          });
          setPreview({
            name: selected.name,
            sizeKb: (selected.size / 1024).toFixed(1),
            totalLines: lines.length - 1,
            headers,
            sampleRows,
          });
        }
      } catch (err) {
        console.error("Preview parse error", err);
      }
    };
    reader.readAsText(selected.slice(0, 32768));
  };

  const submitUpload = async () => {
    if (!file) {
      onError("Please select a CSV file first.");
      return;
    }
    setAnalyzing(true);
    onError("");
    const form = new FormData();
    form.append("file", file);

    try {
      const response = await fetch(`${API}/api/dataset/analyze`, {
        method: "POST",
        body: form,
      });
      const result = await response.json();
      if (response.ok) {
        onAnalysisComplete(result);
      } else {
        onError(result.detail?.message || result.detail || "Upload analysis failed.");
      }
    } catch (err) {
      onError(`Upload error: ${err.message}`);
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <>
      <section className="hero">
        <div>
          <p className="eyebrow">STEP 1: TELEMETRY INGESTION</p>
          <h2>Upload Machine Telemetry Data</h2>
          <p className="sub">
            Upload new operational CSV files or test with the reference AI4I 2020 dataset. Telemetry is validated,
            transformed with domain physics, and evaluated by the Adam-trained Autoencoder.
          </p>
        </div>
        <div>
          <button className="run" onClick={onUseReference} disabled={loading || analyzing}>
            {loading ? "LOADING BASELINE..." : "USE REFERENCE DATASET (AI4I 2020)"}
          </button>
        </div>
      </section>

      <div className="panel">
        <p className="eyebrow">FILE UPLOADER</p>
        <h3>Choose Telemetry CSV</h3>
        <label className="upload">
          <strong>Select or Drag & Drop a .csv File</strong>
          <input type="file" accept=".csv,text/csv" onChange={handleFileChange} disabled={analyzing} />
        </label>

        {preview && (
          <div style={{ marginTop: "20px" }}>
            <div className="cards">
              <Card label="SELECTED FILE" value={preview.name} note={`${preview.sizeKb} KB`} />
              <Card label="RECORDS FOUND" value={preview.totalLines.toLocaleString()} note="Parsed lines" />
              <Card label="COLUMNS" value={preview.headers.length} note="Detected headers" />
              <Card label="VALIDATION" value="READY" note="Schema check ready" />
            </div>

            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", margin: "20px 0 10px" }}>
              <h4>Sample Preview (First 5 Rows)</h4>
              <button className="run" style={{ marginTop: 0 }} onClick={submitUpload} disabled={analyzing}>
                {analyzing ? "ANALYZING DRIFT..." : "ANALYZE TELEMETRY FOR DRIFT →"}
              </button>
            </div>

            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    {preview.headers.slice(0, 8).map((h) => (
                      <th key={h}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {preview.sampleRows.map((row, i) => (
                    <tr key={i}>
                      {preview.headers.slice(0, 8).map((h) => (
                        <td key={h}>{row[h]}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>

      <div className="panel">
        <p className="eyebrow">REQUIRED SCHEMA SPECIFICATION</p>
        <h3>Expected Sensor Signals</h3>
        <p className="sub">
          The telemetry file should supply raw manufacturing signals: <code>Type</code> (L/M/H),{" "}
          <code>Air temperature [K]</code>, <code>Process temperature [K]</code>,{" "}
          <code>Rotational speed [rpm]</code>, <code>Torque [Nm]</code>, and <code>Tool wear [min]</code>. Optional
          identifiers like <code>Product ID</code> or timestamps are preserved and used for chronology.
        </p>
      </div>
    </>
  );
}

function AnalysisPage({
  analysis,
  activeThreshold,
  onThresholdChange,
  onRefresh,
  onNavigateUpload,
}) {
  if (!analysis) {
    return (
      <div className="panel" style={{ textAlign: "center", padding: "40px" }}>
        <h3>No active dataset loaded</h3>
        <p className="sub">Upload a telemetry file or run the reference baseline to inspect drift analysis.</p>
        <button className="run" onClick={onNavigateUpload}>
          Go to Upload Page →
        </button>
      </div>
    );
  }

  const rows = analysis.results || [];
  const isDrift = analysis.drift_detected;
  const signals = analysis.signals || [];
  const checks = analysis.investigation_checks || [];
  const maxScore = Math.max(...rows.map((r) => Number(r.reconstruction_error) || 0), 0.05);

  return (
    <>
      <div className="panel">
        <div className="panel-head">
          <div>
            <p className="eyebrow">DRIFT VERDICT & CLASSIFICATION</p>
            <h2>{isDrift ? analysis.drift_type?.replaceAll("_", " ") : "NO SIGNIFICANT DRIFT"}</h2>
          </div>
          <span className={isDrift ? "badge danger" : "badge good"}>
            {isDrift ? "INVESTIGATION RECOMMENDED" : "NOMINAL REGIME"}
          </span>
        </div>
        <p className="sub" style={{ fontSize: "14px", marginTop: "4px" }}>
          {analysis.summary}
        </p>
        {analysis.sample_warning && <div className="notice">{analysis.sample_warning}</div>}

        <div className="cards">
          <Card
            label="TELEMETRY ROWS"
            value={analysis.sample_size?.toLocaleString() || rows.length}
            note={analysis.dataset?.provenance || "Active dataset context"}
          />
          <Card
            label="FLAGGED ANOMALIES"
            value={analysis.anomaly_count ?? 0}
            note={`${((analysis.anomaly_rate || 0) * 100).toFixed(1)}% of uploaded records`}
          />
          <Card
            label="HISTORICAL BASELINE"
            value={`${((analysis.reference_anomaly_rate || 0) * 100).toFixed(1)}%`}
            note="Expected nominal anomaly rate"
          />
          <Card
            label="ACTIVE THRESHOLD"
            value={activeThreshold !== null ? activeThreshold.toFixed(4) : "—"}
            note="Reconstruction error cutoff"
          />
        </div>

        <div style={{ marginTop: "16px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          {activeThreshold !== null && (
            <div className="threshold-tuner">
              <label>Adjust Decision Threshold:</label>
              <input
                type="range"
                min="0.005"
                max="0.08"
                step="0.002"
                value={activeThreshold}
                onChange={(e) => onThresholdChange(parseFloat(e.target.value))}
              />
              <span>{activeThreshold.toFixed(4)}</span>
            </div>
          )}
          <button className="run" style={{ margin: 0 }} onClick={onRefresh}>
            ↻ RE-EVALUATE TELEMETRY
          </button>
        </div>
      </div>

      <div className="panel">
        <div className="panel-head">
          <div>
            <p className="eyebrow">SEQUENTIAL TELEMETRY TRACK</p>
            <h3>Reconstruction Error & Sequential Drift Persistence</h3>
          </div>
          <span className="legend">
            <i></i>Error Score
            <i className="amber"></i>Threshold Cutoff
            <i className="red"></i>Anomaly Spike
          </span>
        </div>
        <div className="spark">
          {rows.slice(-80).map((r, i) => {
            const score = Number(r.reconstruction_error) || 0;
            const pct = Math.max(4, Math.min(100, (score / maxScore) * 100));
            return (
              <span
                key={i}
                title={`Row #${r.source_row_index}: error=${score.toFixed(4)}, rolling=${(Number(r.rolling_exceedance_fraction || 0) * 100).toFixed(1)}%`}
                style={{ height: `${pct}%` }}
                className={r.anomaly ? "bar alert" : "bar"}
              />
            );
          })}
        </div>
      </div>

      <div className="panel">
        <p className="eyebrow">ROOT CAUSE SIGNALS</p>
        <h3>Flagged Sensor Channels & Shift Diagnostics</h3>
        {signals.length > 0 ? (
          <div>
            {signals.map((sig) => (
              <div className="signal" key={sig.feature}>
                <div>
                  <b style={{ color: "#ecf5f7" }}>{sig.feature}</b>
                  <div className="sub" style={{ fontSize: "11px", marginTop: "2px" }}>
                    {sig.feature === "Temp difference [K]"
                      ? "Thermodynamic heat dissipation discrepancy"
                      : sig.feature === "Power [W]"
                        ? "Rotational mechanical load aberration"
                        : sig.feature === "Overstrain [min Nm]"
                          ? "Excessive tooling stress / wear combination"
                          : "Distribution shift relative to training population"}
                  </div>
                </div>
                <span>
                  PSI: <b>{sig.psi != null ? sig.psi.toFixed(3) : "—"}</b> · Shift:{" "}
                  <b>{sig.standardized_mean_shift?.toFixed(2)}σ</b>
                </span>
              </div>
            ))}
          </div>
        ) : (
          <p className="sub">All sensor channels remain within configured statistical tolerance.</p>
        )}

        <h4 style={{ marginTop: "24px" }}>How to Solve: Prioritized Investigation Checks</h4>
        <ol className="checks">
          {checks.map((chk) => (
            <li key={chk.priority}>
              <b>
                Priority {chk.priority}: {chk.check}
              </b>
              <span>{chk.reason}</span>
            </li>
          ))}
        </ol>
        <p className="disclaimer">
          Prioritized checks are statistical triage hypotheses designed for industrial reliability engineers. Confirm
          all mechanical maintenance with domain operators.
        </p>
      </div>

      <Table rows={rows} />
    </>
  );
}

function Table({ rows }) {
  const [filter, setFilter] = useState("all");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(0);
  const pageSize = 15;

  const filteredRows = useMemo(() => {
    return rows.filter((r) => {
      if (filter === "anomalies" && r.anomaly !== 1) return false;
      if (filter === "normal" && r.anomaly !== 0) return false;
      if (search.trim()) {
        const query = search.trim().toLowerCase();
        const rowMatch = String(r.source_row_index).includes(query);
        const prodMatch = String(r["Product ID"] || "").toLowerCase().includes(query);
        const udiMatch = String(r.UDI || "").includes(query);
        return rowMatch || prodMatch || udiMatch;
      }
      return true;
    });
  }, [rows, filter, search]);

  const totalPages = Math.max(1, Math.ceil(filteredRows.length / pageSize));
  const pageRows = filteredRows.slice(page * pageSize, (page + 1) * pageSize);

  return (
    <div className="panel">
      <div className="panel-head">
        <div>
          <p className="eyebrow">OBSERVATIONS & SCORING RECORDS</p>
          <h3>Telemetry Observation Table</h3>
        </div>
      </div>

      <div className="controls-bar">
        <div className="btn-group">
          <button
            className={`filter-btn ${filter === "all" ? "active" : ""}`}
            onClick={() => {
              setFilter("all");
              setPage(0);
            }}
          >
            All Records ({rows.length})
          </button>
          <button
            className={`filter-btn ${filter === "anomalies" ? "active" : ""}`}
            onClick={() => {
              setFilter("anomalies");
              setPage(0);
            }}
          >
            Anomalies ({rows.filter((r) => r.anomaly === 1).length})
          </button>
          <button
            className={`filter-btn ${filter === "normal" ? "active" : ""}`}
            onClick={() => {
              setFilter("normal");
              setPage(0);
            }}
          >
            Normal ({rows.filter((r) => r.anomaly === 0).length})
          </button>
        </div>

        <input
          type="text"
          className="search-input"
          placeholder="Search by row #, product, or UDI..."
          value={search}
          onChange={(e) => {
            setSearch(e.target.value);
            setPage(0);
          }}
        />
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>ROW</th>
              <th>IDENTIFIER</th>
              <th>SCORE</th>
              <th>THRESHOLD</th>
              <th>STATUS</th>
              <th>WARNING STATE</th>
              <th>TORQUE</th>
              <th>SPEED [rpm]</th>
              <th>TOOL WEAR</th>
              <th>TEMP DIFF [K]</th>
            </tr>
          </thead>
          <tbody>
            {pageRows.map((r) => (
              <tr key={r.source_row_index}>
                <td>#{r.source_row_index}</td>
                <td>{r["Product ID"] || `UDI-${r.UDI}` || "—"}</td>
                <td>{Number(r.reconstruction_error).toFixed(4)}</td>
                <td>{Number(r.threshold).toFixed(4)}</td>
                <td>
                  <span className={r.anomaly ? "status danger" : "status"}>
                    {r.anomaly ? "ANOMALY" : "NORMAL"}
                  </span>
                </td>
                <td>
                  <span
                    className={
                      r.warning_state === "SUSTAINED_HIGH"
                        ? "status danger"
                        : r.warning_state === "POTENTIAL_DRIFT"
                          ? "status warning"
                          : "status"
                    }
                  >
                    {r.warning_state || "NORMAL"}
                  </span>
                </td>
                <td>{r["Torque [Nm]"] != null ? Number(r["Torque [Nm]"]).toFixed(1) : "—"}</td>
                <td>{r["Rotational speed [rpm]"] != null ? r["Rotational speed [rpm]"] : "—"}</td>
                <td>{r["Tool wear [min]"] != null ? r["Tool wear [min]"] : "—"}</td>
                <td>
                  {r["Temp difference [K]"] != null
                    ? Number(r["Temp difference [K]"]).toFixed(2)
                    : "—"}
                </td>
              </tr>
            ))}
            {pageRows.length === 0 && (
              <tr>
                <td colSpan="10" style={{ textAlign: "center", padding: "20px" }}>
                  No matching records found.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="pagination">
        <span>
          Showing {filteredRows.length ? page * pageSize + 1 : 0} to{" "}
          {Math.min((page + 1) * pageSize, filteredRows.length)} of {filteredRows.length} entries
        </span>
        <div style={{ display: "flex", gap: "8px" }}>
          <button
            className="page-btn"
            disabled={page === 0}
            onClick={() => setPage((p) => Math.max(0, p - 1))}
          >
            ← Previous
          </button>
          <span style={{ padding: "6px 10px" }}>
            Page {page + 1} of {totalPages}
          </span>
          <button
            className="page-btn"
            disabled={page >= totalPages - 1}
            onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
          >
            Next →
          </button>
        </div>
      </div>
    </div>
  );
}

function MethodologyPage() {
  return (
    <div className="panel prose">
      <p className="eyebrow">AUDITABLE SYSTEM ARCHITECTURE</p>
      <h2>Methodology: How We Find & Solve Machine Drift</h2>

      <h3>1. How We Detect Behavioral Drift</h3>
      <p>
        Manufacturing machines often exhibit subtle mechanical and thermal drift long before catastrophic failure. To
        detect drift accurately without label leakage, Sentinel ML uses a five-stage architecture:
      </p>

      <ul>
        <li>
          <b>Unsupervised Normal-Only Baseline:</b> The model trains strictly on normal machine records (where no
          failure occurred). It never receives failure targets as features, ensuring zero label leakage and high
          fidelity to the nominal operational manifold.
        </li>
        <li>
          <b>Domain Physics Integration:</b> Pure sensor standardization misses coupled physical failure laws. We
          derive key thermodynamics and mechanical equations:
          <ul>
            <li>
              <b>Thermal Gradient:</b> ΔT = T_process - T_air (Governs Heat Dissipation Failures).
            </li>
            <li>
              <b>Mechanical Power:</b> P = Torque × Speed (Governs Power Failures if P &lt; 3500 W or P &gt; 9000 W).
            </li>
            <li>
              <b>Overstrain Index:</b> OS = Tool_wear × Torque (Governs Overstrain Failures).
            </li>
          </ul>
        </li>
        <li>
          <b>Bottleneck Neural Autoencoder (Adam):</b> A compact neural architecture compresses observations into a
          latent bottleneck with tanh activations. Nominal telemetry reconstructs with minimal error; anomalous drift
          yields large reconstruction errors.
        </li>
        <li>
          <b>Empirical Validation Cutoff:</b> The anomaly decision boundary is derived from the normal validation
          percentile error distribution.
        </li>
        <li>
          <b>Rolling Persistence Rule:</b> Rather than alarming on a single outlier, a 25-observation rolling window
          calculates the exceedance fraction. If \(\ge 60\%\) of consecutive observations exceed threshold, a{" "}
          <code>POTENTIAL_DRIFT</code> alarm triggers; at \(\ge 80\%\), it escalates to <code>SUSTAINED_HIGH</code>.
        </li>
      </ul>

      <hr style={{ borderColor: "#273844", margin: "28px 0" }} />

      <h3>2. How We Solve and Triage Detected Drift</h3>
      <p>
        Once behavioral drift is flagged, the system does not simply emit an alarm—it provides structured triage evidence:
      </p>

      <ul>
        <li>
          <b>Drift Classification:</b>
          <ul>
            <li>
              <b>SENSOR_DISTRIBUTION_DRIFT:</b> Specific individual sensors (e.g. speed or torque) have drifted via
              high Population Stability Index (PSI \(\ge 0.2\)) or mean shift (\(\ge 0.5\sigma\)).
            </li>
            <li>
              <b>OPERATING_CONDITION_DRIFT:</b> Thermal ambient differences or product type mix have altered the
              baseline.
            </li>
            <li>
              <b>BEHAVIORAL_DRIFT:</b> Sensor distributions look individually nominal, but the autoencoder detects
              coordinated off-manifold degradation.
            </li>
          </ul>
        </li>
        <li>
          <b>Actionable Prioritized Checks:</b>
          <ol>
            <li>
              <b>Confirm Sensor Calibration & Units:</b> Ensure instrumentation has not drifted or desynchronized.
            </li>
            <li>
              <b>Inspect Flagged Channels & Maintenance Logs:</b> Check tooling replacement history and mechanical wear
              on components with high overstrain or power draw.
            </li>
            <li>
              <b>Verify Production Operating Regime:</b> Confirm if recipe, feed rate, or speed requirements changed
              intentionally.
            </li>
            <li>
              <b>Operator Visual Confirmation:</b> Inspect high-scoring records with domain engineers before shutting
              down machinery.
            </li>
          </ol>
        </li>
        <li>
          <b>Operational Threshold Fine-Tuning:</b> Operators can dynamically adjust the decision cutoff in the UI to
          tune alarm sensitivity according to operational risk appetite without retraining the model.
        </li>
      </ul>
    </div>
  );
}

function Card({ label, value, note }) {
  return (
    <div className="card">
      <p className="eyebrow">{label}</p>
      <strong>{value}</strong>
      <small>{note}</small>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
