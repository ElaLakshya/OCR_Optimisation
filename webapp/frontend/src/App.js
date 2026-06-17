import React, { useState, useRef, useCallback } from "react";

const API = "http://localhost:8000";

const STAGES = [
  "Extracting digital text",
  "Analysing document structure",
  "Running OCR on image regions",
  "Rendering output",
  "Done",
];

function App() {
  const [file, setFile]           = useState(null);
  const [dragging, setDragging]   = useState(false);
  const [jobId, setJobId]         = useState(null);
  const [status, setStatus]       = useState(null); // queued|running|done|error
  const [stage, setStage]         = useState("");
  const [progress, setProgress]   = useState(0);
  const [result, setResult]       = useState(null); // {html, stats, has_pdf}
  const [error, setError]         = useState(null);
  const [view, setView]           = useState("preview"); // preview|html
  
  // AI Summary States
  const [summary, setSummary]           = useState(null);
  const [isSummarizing, setIsSummarizing] = useState(false);
  const [summaryError, setSummaryError]   = useState(null);

  // AI Classification States
  const [classification, setClassification] = useState(null);
  const [isClassifying, setIsClassifying]   = useState(false);
  const [classifyError, setClassifyError]   = useState(null);

  const pollRef                   = useRef(null);
  const fileInputRef              = useRef(null);

  // ── File selection ──────────────────────────────────────────────────────
  const handleFile = (f) => {
    if (!f) return;
    const ext = f.name.split(".").pop().toLowerCase();
    if (!["pdf", "jpg", "jpeg", "png"].includes(ext)) {
      setError("Only PDF, JPG, and PNG files are supported.");
      return;
    }
    setFile(f);
    setError(null);
    setResult(null);
    setJobId(null);
    setStatus(null);
    setProgress(0);
    setStage("");
    
    // Reset AI states
    setSummary(null);
    setIsSummarizing(false);
    setSummaryError(null);
    setClassification(null);
    setIsClassifying(false);
    setClassifyError(null);
  };

  const onFileChange = (e) => handleFile(e.target.files[0]);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  }, []);

  const onDragOver = (e) => { e.preventDefault(); setDragging(true); };
  const onDragLeave = () => setDragging(false);

  // ── Polling ─────────────────────────────────────────────────────────────
  const startPolling = (id) => {
    pollRef.current = setInterval(async () => {
      try {
        const res  = await fetch(`${API}/status/${id}`);
        const data = await res.json();
        setStage(data.stage);
        setProgress(data.progress);
        setStatus(data.status);

        if (data.status === "done") {
          clearInterval(pollRef.current);
          const rRes  = await fetch(`${API}/result/${id}`);
          const rData = await rRes.json();
          setResult(rData);
        } else if (data.status === "error") {
          clearInterval(pollRef.current);
          setError(data.error || "An error occurred during processing.");
        }
      } catch (e) {
        clearInterval(pollRef.current);
        setError("Lost connection to server.");
      }
    }, 1000);
  };

  // ── Upload ───────────────────────────────────────────────────────────────
  const handleUpload = async () => {
    if (!file) return;
    setError(null);
    setResult(null);
    setProgress(0);
    setStage("Uploading...");
    setStatus("queued");

    const form = new FormData();
    form.append("file", file);

    try {
      const res  = await fetch(`${API}/upload`, { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Upload failed");
      setJobId(data.job_id);
      setStatus("running");
      startPolling(data.job_id);
    } catch (e) {
      setError(e.message);
      setStatus(null);
    }
  };

  // ── AI Summarization ─────────────────────────────────────────────────────
  const handleSummarize = async () => {
    if (!jobId) return;
    setIsSummarizing(true);
    setSummaryError(null);
    try {
      const res = await fetch(`${API}/summarize/${jobId}`, { method: "POST" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Summarization failed");
      setSummary(data.summary);
    } catch (e) {
      setSummaryError(e.message);
    } finally {
      setIsSummarizing(false);
    }
  };

  // ── AI Classification ────────────────────────────────────────────────────
  const handleClassify = async () => {
    if (!jobId) return;
    setIsClassifying(true);
    setClassifyError(null);
    try {
      const res = await fetch(`${API}/classify/${jobId}`, { method: "POST" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Classification failed");
      setClassification(data.classification);
    } catch (e) {
      setClassifyError(e.message);
    } finally {
      setIsClassifying(false);
    }
  };

  // ── Reset ────────────────────────────────────────────────────────────────
  const handleReset = () => {
    clearInterval(pollRef.current);
    setFile(null);
    setJobId(null);
    setStatus(null);
    setStage("");
    setProgress(0);
    setResult(null);
    setError(null);
    setView("preview");
    setSummary(null);
    setIsSummarizing(false);
    setSummaryError(null);
    setClassification(null);
    setIsClassifying(false);
    setClassifyError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  // ── Render ───────────────────────────────────────────────────────────────
  return (
    <div style={styles.root}>
      <header style={styles.header}>
        <h1 style={styles.title}>OCR Pipeline</h1>
        <p style={styles.subtitle}>
          Upload a PDF or image — digital text is extracted instantly,
          image regions are processed with Surya OCR.
        </p>
      </header>

      <main style={styles.main}>
        {/* ── Upload panel ── */}
        {!status && (
          <div
            style={{ ...styles.dropzone, ...(dragging ? styles.dropzoneDrag : {}) }}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.jpg,.jpeg,.png"
              style={{ display: "none" }}
              onChange={onFileChange}
            />
            {file ? (
              <div style={styles.fileInfo}>
                <span style={styles.fileIcon}>📄</span>
                <span style={styles.fileName}>{file.name}</span>
                <span style={styles.fileSize}>
                  {(file.size / 1024).toFixed(1)} KB
                </span>
              </div>
            ) : (
              <div style={styles.dropHint}>
                <span style={styles.dropIcon}>⬆️</span>
                <p style={styles.dropText}>
                  Drag & drop or <u>click to browse</u>
                </p>
                <p style={styles.dropSub}>PDF, JPG, PNG supported</p>
              </div>
            )}
          </div>
        )}

        {error && <div style={styles.errorBox}>{error}</div>}

        {/* ── Upload button ── */}
        {file && !status && (
          <button style={styles.uploadBtn} onClick={handleUpload}>
            Process Document
          </button>
        )}

        {/* ── Progress ── */}
        {status && status !== "done" && status !== "error" && (
          <div style={styles.progressCard}>
            <div style={styles.progressHeader}>
              <span style={styles.progressStage}>{stage}</span>
              <span style={styles.progressPct}>{progress}%</span>
            </div>
            <div style={styles.progressTrack}>
              <div
                style={{ ...styles.progressBar, width: `${progress}%` }}
              />
            </div>
            <div style={styles.stageList}>
              {STAGES.map((s) => (
                <div key={s} style={styles.stageItem}>
                  <span
                    style={{
                      ...styles.stageDot,
                      background: stage === s
                        ? "#4f46e5"
                        : progress === 100 ? "#22c55e" : "#d1d5db",
                    }}
                  />
                  <span style={{
                    ...styles.stageLabel,
                    color: stage === s ? "#4f46e5" : "#6b7280",
                    fontWeight: stage === s ? 600 : 400,
                  }}>
                    {s}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── Result ── */}
        {status === "done" && result && (
          <div style={styles.resultCard}>
            {/* Stats bar */}
            <div style={styles.statsBar}>
              <StatPill label="Pages"     value={result.stats?.n_pages ?? "—"} />
              <StatPill label="Images"    value={result.stats?.n_image_regions ?? "—"} />
              <StatPill label="Cache hits" value={result.stats?.cache_hits ?? "—"} />
              <StatPill label="Total time" value={result.stats?.total ? `${result.stats.total}s` : "—"} />
            </div>

            {/* AI Control Panel */}
            <div style={styles.summarySection}>
              
              <div style={styles.buttonRow}>
                {!summary && !isSummarizing && (
                  <button style={styles.aiBtn} onClick={handleSummarize}>
                    ✨ Generate Case Summary
                  </button>
                )}
                {!classification && !isClassifying && (
                  <button style={{ ...styles.aiBtn, background: "#059669" }} onClick={handleClassify}>
                    🏷️ Classify FIR
                  </button>
                )}
              </div>
              
              {/* Classification Results */}
              {(isClassifying || classification || classifyError) && (
                <div style={{ marginBottom: "16px" }}>
                  {isClassifying && (
                    <div style={{ ...styles.loadingPulse, color: "#059669" }}>
                      <span className="spinner">⏳</span> CPU is classifying...
                    </div>
                  )}
                  {classifyError && <div style={styles.errorBox}>{classifyError}</div>}
                  {classification && (
                    <div style={{
                      ...styles.classificationBadge,
                      background: classification.includes("Valid") && !classification.includes("Invalid") ? "#dcfce7" : "#fee2e2",
                      color: classification.includes("Valid") && !classification.includes("Invalid") ? "#166534" : "#991b1b",
                      border: `1px solid ${classification.includes("Valid") && !classification.includes("Invalid") ? "#bbf7d0" : "#fecaca"}`
                    }}>
                      {classification.includes("Valid") && !classification.includes("Invalid") ? "✅" : "❌"} {classification}
                    </div>
                  )}
                </div>
              )}

              {/* Summary Results */}
              {(isSummarizing || summary || summaryError) && (
                <div>
                  {isSummarizing && (
                    <div style={styles.loadingPulse}>
                      <span className="spinner">⏳</span> CPU is analyzing document... (This may take ~30s)
                    </div>
                  )}
                  {summaryError && <div style={styles.errorBox}>{summaryError}</div>}
                  {summary && (
                    <div style={styles.summaryBox}>
                      <h3 style={styles.summaryTitle}>📄 AI Case Summary</h3>
                      <pre style={styles.summaryText}>{summary}</pre>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* View toggle */}
            <div style={styles.viewToggle}>
              <button
                style={{ ...styles.toggleBtn, ...(view === "preview" ? styles.toggleActive : {}) }}
                onClick={() => setView("preview")}
              >
                Preview Document
              </button>
              <button
                style={{ ...styles.toggleBtn, ...(view === "html" ? styles.toggleActive : {}) }}
                onClick={() => setView("html")}
              >
                HTML Source
              </button>
            </div>

            {/* Output display */}
            {view === "preview" ? (
              <iframe
                srcDoc={result.html}
                style={styles.previewFrame}
                title="OCR Output Preview"
                sandbox="allow-same-origin"
              />
            ) : (
              <pre style={styles.htmlSource}>
                {result.html}
              </pre>
            )}

            {/* Download buttons */}
            <div style={styles.downloadRow}>
              <a
                href={`${API}/download/${jobId}/html`}
                download
                style={styles.downloadBtn}
              >
                ⬇ Download HTML
              </a>
              {result.has_pdf && (
                <a
                  href={`${API}/download/${jobId}/pdf`}
                  download
                  style={{ ...styles.downloadBtn, background: "#dc2626" }}
                >
                  ⬇ Download PDF
                </a>
              )}
              <button style={styles.resetBtn} onClick={handleReset}>
                Process another file
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

function StatPill({ label, value }) {
  return (
    <div style={styles.statPill}>
      <span style={styles.statValue}>{value}</span>
      <span style={styles.statLabel}>{label}</span>
    </div>
  );
}

// ── Styles ───────────────────────────────────────────────────────────────────
const styles = {
  root: { minHeight: "100vh", background: "#f9fafb", fontFamily: "'Inter', 'Segoe UI', Arial, sans-serif", color: "#111827" },
  header: { background: "#fff", borderBottom: "1px solid #e5e7eb", padding: "24px 40px" },
  title: { margin: 0, fontSize: 24, fontWeight: 700, color: "#111827" },
  subtitle: { margin: "6px 0 0", fontSize: 14, color: "#6b7280" },
  main: { maxWidth: 860, margin: "40px auto", padding: "0 24px" },
  dropzone: { border: "2px dashed #d1d5db", borderRadius: 12, padding: "48px 32px", textAlign: "center", cursor: "pointer", background: "#fff", transition: "border-color 0.2s, background 0.2s" },
  dropzoneDrag: { borderColor: "#4f46e5", background: "#eef2ff" },
  dropHint: { display: "flex", flexDirection: "column", alignItems: "center", gap: 8 },
  dropIcon: { fontSize: 40 },
  dropText: { margin: 0, fontSize: 16, color: "#374151" },
  dropSub:  { margin: 0, fontSize: 13, color: "#9ca3af" },
  fileInfo: { display: "flex", alignItems: "center", gap: 12, justifyContent: "center" },
  fileIcon: { fontSize: 32 },
  fileName: { fontSize: 16, fontWeight: 600, color: "#111827" },
  fileSize: { fontSize: 13, color: "#6b7280" },
  errorBox: { marginTop: 16, padding: "12px 16px", background: "#fef2f2", border: "1px solid #fecaca", borderRadius: 8, color: "#dc2626", fontSize: 14 },
  uploadBtn: { marginTop: 20, display: "block", width: "100%", padding: "14px", background: "#4f46e5", color: "#fff", border: "none", borderRadius: 8, fontSize: 16, fontWeight: 600, cursor: "pointer" },
  progressCard: { background: "#fff", border: "1px solid #e5e7eb", borderRadius: 12, padding: "28px 32px" },
  progressHeader: { display: "flex", justifyContent: "space-between", marginBottom: 12 },
  progressStage: { fontSize: 15, fontWeight: 600, color: "#111827" },
  progressPct:   { fontSize: 15, fontWeight: 600, color: "#4f46e5" },
  progressTrack: { height: 8, background: "#e5e7eb", borderRadius: 4, overflow: "hidden", marginBottom: 24 },
  progressBar: { height: "100%", background: "#4f46e5", borderRadius: 4, transition: "width 0.4s ease" },
  stageList:  { display: "flex", flexDirection: "column", gap: 10 },
  stageItem:  { display: "flex", alignItems: "center", gap: 10 },
  stageDot:   { width: 10, height: 10, borderRadius: "50%", flexShrink: 0 },
  stageLabel: { fontSize: 14 },
  resultCard: { background: "#fff", border: "1px solid #e5e7eb", borderRadius: 12, overflow: "hidden" },
  statsBar: { display: "flex", borderBottom: "1px solid #e5e7eb", padding: "16px 24px", flexWrap: "wrap", gap: 16 },
  statPill: { display: "flex", flexDirection: "column", alignItems: "center", minWidth: 80 },
  statValue: { fontSize: 22, fontWeight: 700, color: "#4f46e5" },
  statLabel: { fontSize: 12, color: "#6b7280", marginTop: 2 },
  
  // AI Summary Styles
  summarySection: { padding: "24px", background: "#f5f3ff", borderBottom: "1px solid #e5e7eb" },
  buttonRow: { display: "flex", gap: "12px", marginBottom: "16px", flexWrap: "wrap" },
  aiBtn: { padding: "12px 24px", background: "#7c3aed", color: "#fff", border: "none", borderRadius: 8, fontSize: 15, fontWeight: 600, cursor: "pointer", display: "flex", alignItems: "center", gap: 8 },
  loadingPulse: { color: "#6d28d9", fontWeight: 500, fontSize: 14, display: "flex", alignItems: "center", gap: 8, marginBottom: 12 },
  summaryBox: { background: "#fff", border: "1px solid #ddd6fe", borderRadius: 8, padding: "16px", boxShadow: "0 1px 2px rgba(0,0,0,0.05)" },
  summaryTitle: { margin: "0 0 12px 0", color: "#6d28d9", fontSize: 16 },
  summaryText: { margin: 0, whiteSpace: "pre-wrap", fontFamily: "inherit", fontSize: 14, color: "#374151", lineHeight: 1.6 },
  classificationBadge: { display: "inline-block", padding: "8px 16px", borderRadius: "9999px", fontWeight: "600", fontSize: "14px" },

  viewToggle: { display: "flex", borderBottom: "1px solid #e5e7eb", padding: "0 24px" },
  toggleBtn: { padding: "12px 20px", border: "none", borderBottom: "2px solid transparent", background: "none", cursor: "pointer", fontSize: 14, color: "#6b7280", fontWeight: 500 },
  toggleActive: { color: "#4f46e5", borderBottomColor: "#4f46e5" },
  previewFrame: { width: "100%", height: 600, border: "none", display: "block" },
  htmlSource: { padding: 24, margin: 0, background: "#f9fafb", fontSize: 12, lineHeight: 1.6, overflowX: "auto", maxHeight: 600, overflowY: "auto", color: "#374151" },
  downloadRow: { display: "flex", gap: 12, padding: "16px 24px", borderTop: "1px solid #e5e7eb", flexWrap: "wrap", alignItems: "center" },
  downloadBtn: { padding: "10px 20px", background: "#4f46e5", color: "#fff", borderRadius: 8, textDecoration: "none", fontSize: 14, fontWeight: 600 },
  resetBtn: { padding: "10px 20px", background: "none", border: "1px solid #d1d5db", borderRadius: 8, fontSize: 14, cursor: "pointer", color: "#374151", marginLeft: "auto" },
};

export default App;