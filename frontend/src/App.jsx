import { useCallback, useMemo, useRef, useState } from "react";

// ── Palette & typography via inline CSS variables ─────────────────────
const CSS = `
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Space+Grotesk:wght@300;500;700&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:        #0a0c0f;
    --panel:     #111318;
    --border:    #1e2229;
    --accent:    #00ff88;
    --accent2:   #00cfff;
    --warn:      #ff6b35;
    --text:      #e8eaf0;
    --muted:     #5a6070;
    --mono:      'IBM Plex Mono', monospace;
    --sans:      'Space Grotesk', sans-serif;
  }

  body { background: var(--bg); color: var(--text); font-family: var(--sans); }

  .app {
    min-height: 100vh;
    display: grid;
    grid-template-rows: auto 1fr;
  }

  /* ── Header ── */
  .header {
    border-bottom: 1px solid var(--border);
    padding: 20px 32px;
    display: flex;
    align-items: center;
    gap: 16px;
    background: var(--panel);
  }
  .header-badge {
    background: var(--accent);
    color: #000;
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 600;
    padding: 3px 8px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }
  .header h1 {
    font-size: 20px;
    font-weight: 700;
    letter-spacing: -0.02em;
  }
  .header h1 span { color: var(--accent); }
  .header-sub {
    margin-left: auto;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
  }

  /* ── Main layout ── */
  .main {
    display: grid;
    grid-template-columns: 340px 1fr;
    gap: 0;
    height: calc(100vh - 65px);
  }

  /* ── Sidebar ── */
  .sidebar {
    background: var(--panel);
    border-right: 1px solid var(--border);
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 0;
  }

  .section {
    border-bottom: 1px solid var(--border);
    padding: 20px 24px;
  }
  .section-title {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.12em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 16px;
  }

  /* Upload zone */
  .upload-zone {
    border: 1px dashed var(--border);
    padding: 28px 16px;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
    position: relative;
  }
  .upload-zone:hover, .upload-zone.drag-over {
    border-color: var(--accent);
    background: rgba(0,255,136,0.04);
  }
  .upload-zone input {
    position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%;
  }
  .upload-icon { font-size: 28px; margin-bottom: 8px; }
  .upload-label { font-size: 13px; color: var(--muted); font-family: var(--mono); }
  .upload-label strong { color: var(--accent); display: block; margin-bottom: 4px; font-size: 12px; }

  /* Method selector */
  .method-tabs {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 4px;
  }
  .method-tab {
    padding: 10px 12px;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--muted);
    font-family: var(--mono);
    font-size: 11px;
    cursor: pointer;
    transition: all 0.15s;
    text-align: center;
  }
  .method-tab.active {
    border-color: var(--accent);
    color: var(--accent);
    background: rgba(0,255,136,0.06);
  }
  .method-desc {
    font-size: 11px;
    color: var(--muted);
    font-family: var(--mono);
    margin-top: 8px;
    line-height: 1.5;
  }

  /* Sliders */
  .param-row { margin-bottom: 16px; }
  .param-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 6px;
  }
  .param-name {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--text);
  }
  .param-value {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--accent);
  }
  input[type="range"] {
    width: 100%;
    -webkit-appearance: none;
    height: 2px;
    background: var(--border);
    outline: none;
    cursor: pointer;
  }
  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 12px; height: 12px;
    border-radius: 50%;
    background: var(--accent);
    cursor: pointer;
  }

  /* Run button */
  .run-btn {
    width: 100%;
    padding: 14px;
    background: var(--accent);
    color: #000;
    border: none;
    font-family: var(--mono);
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    cursor: pointer;
    transition: opacity 0.15s;
    margin-top: 4px;
  }
  .run-btn:hover { opacity: 0.85; }
  .run-btn:disabled { opacity: 0.35; cursor: not-allowed; }
  .run-btn.loading { background: var(--accent2); }

  /* Stats */
  .stats-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }
  .stat-card {
    background: var(--bg);
    border: 1px solid var(--border);
    padding: 12px;
  }
  .stat-label {
    font-family: var(--mono);
    font-size: 9px;
    color: var(--muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 4px;
  }
  .stat-value {
    font-family: var(--mono);
    font-size: 20px;
    font-weight: 600;
    color: var(--accent);
  }
  .stat-value.time { color: var(--accent2); }
  .stat-unit { font-size: 11px; color: var(--muted); margin-left: 2px; }

  /* Error */
  .error-box {
    border: 1px solid var(--warn);
    background: rgba(255,107,53,0.08);
    padding: 10px 14px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--warn);
    margin-top: 12px;
  }

  /* ── Canvas area ── */
  .canvas-area {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0;
    overflow: hidden;
  }
  .canvas-area.three {
    grid-template-columns: 1fr 1fr 1fr;
  }

  .image-panel {
    position: relative;
    display: flex;
    flex-direction: column;
    border-right: 1px solid var(--border);
    overflow: hidden;
  }
  .image-panel:last-child { border-right: none; }

  .panel-label {
    position: absolute;
    top: 12px;
    left: 12px;
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.1em;
    color: var(--muted);
    text-transform: uppercase;
    background: rgba(10,12,15,0.85);
    padding: 4px 8px;
    z-index: 2;
    border: 1px solid var(--border);
  }

  .image-container {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    padding: 16px;
  }
  .image-container img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    image-rendering: pixelated;
  }

  .placeholder {
    text-align: center;
    color: var(--muted);
  }
  .placeholder-icon { font-size: 48px; margin-bottom: 12px; opacity: 0.3; }
  .placeholder-text { font-family: var(--mono); font-size: 12px; line-height: 1.6; }

  /* Spinner */
  .spinner {
    width: 32px; height: 32px;
    border: 2px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin: 0 auto 16px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Corner table */
  .corners-list {
    max-height: 180px;
    overflow-y: auto;
    border: 1px solid var(--border);
    background: var(--bg);
  }
  .corner-row {
    display: grid;
    grid-template-columns: 60px 60px 1fr;
    gap: 8px;
    padding: 6px 12px;
    border-bottom: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    align-items: center;
  }
  .corner-row:first-child {
    color: var(--text);
    background: var(--panel);
    position: sticky;
    top: 0;
    font-size: 9px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .corner-x { color: var(--accent); }
  .corner-y { color: var(--accent2); }
  .corner-r {
    font-size: 9px;
    background: var(--border);
    padding: 2px 6px;
    text-align: right;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); }

  /* Tabs */
  .tabs {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 8px;
  }
  .tab {
    padding: 10px 10px;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--muted);
    font-family: var(--mono);
    font-size: 11px;
    cursor: pointer;
    transition: all 0.15s;
    text-align: center;
  }
  .tab.active {
    border-color: var(--accent2);
    color: var(--accent2);
    background: rgba(0,207,255,0.06);
  }
`;

const METHODS = {
  harris: {
    label: "Harris",
    formula: "R = det(M) − k·trace(M)²",
    desc: "Classic Harris operator. k ≈ 0.04–0.06 balances corner vs. edge sensitivity.",
  },
  lambda_minus: {
    label: "λ− (Min Eigenvalue)",
    formula: "λ− = (a+c)/2 − √(b²+(a−c)²)/2",
    desc: "Minimum eigenvalue of the structure tensor M. Corner if λ− >> 0.",
  },
};

const MATCH_METHODS = {
  ssd: {
    label: "SSD",
    desc: "Sum of Squared Differences (lower is better).",
  },
  ncc: {
    label: "NCC",
    desc: "Normalized Cross-Correlation (uses 1 − dot(desc1, desc2)).",
  },
};

export default function App() {
  const [mode, setMode] = useState("corners"); // corners | sift | match

  // ── Corners params ────────────────────────────────────────────────
  const [method, setMethod]       = useState("harris");
  const [k, setK]                 = useState(0.04);
  const [threshold, setThreshold] = useState(0.01);
  const [windowSize, setWindowSize] = useState(3);
  const [nmsSize, setNmsSize]     = useState(5);
  const [sigma, setSigma]         = useState(1.0);

  // ── SIFT params ───────────────────────────────────────────────────
  const [siftSigma0, setSiftSigma0] = useState(1.6);
  const [siftOctaves, setSiftOctaves] = useState(3);
  const [siftScales, setSiftScales] = useState(4);
  const [siftContrast, setSiftContrast] = useState(0.03);
  const [siftMaxKp, setSiftMaxKp] = useState(100);

  // ── Match params ──────────────────────────────────────────────────
  const [matchMethod, setMatchMethod] = useState("ssd");
  const [ratioThreshold, setRatioThreshold] = useState(0.8);
  const [matchMaxKp, setMatchMaxKp] = useState(75);

  // ── Inputs ────────────────────────────────────────────────────────
  const [imageA, setImageA] = useState(null); // base64 data-uri
  const [imageB, setImageB] = useState(null); // base64 data-uri (match mode)
  const [draggingA, setDraggingA] = useState(false);
  const [draggingB, setDraggingB] = useState(false);
  const fileARef = useRef();
  const fileBRef = useRef();

  // ── Outputs ───────────────────────────────────────────────────────
  const [cornerResultImage, setCornerResultImage] = useState(null);
  const [cornerPoints, setCornerPoints] = useState([]);
  const [cornerStats, setCornerStats] = useState(null);

  const [siftResultImage, setSiftResultImage] = useState(null);
  const [siftKeypoints, setSiftKeypoints] = useState([]);
  const [siftStats, setSiftStats] = useState(null);

  const [matchResultImage, setMatchResultImage] = useState(null);
  const [matchStats, setMatchStats] = useState(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const resetOutputs = useCallback(() => {
    setCornerResultImage(null);
    setCornerPoints([]);
    setCornerStats(null);
    setSiftResultImage(null);
    setSiftKeypoints([]);
    setSiftStats(null);
    setMatchResultImage(null);
    setMatchStats(null);
    setError(null);
  }, []);

  const loadFileInto = useCallback((file, setter) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      setter(e.target.result);
      resetOutputs();
    };
    reader.readAsDataURL(file);
  }, [resetOutputs]);

  const handleDropA = useCallback((e) => {
    e.preventDefault();
    setDraggingA(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith("image/")) loadFileInto(file, setImageA);
  }, [loadFileInto]);

  const handleDropB = useCallback((e) => {
    e.preventDefault();
    setDraggingB(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith("image/")) loadFileInto(file, setImageB);
  }, [loadFileInto]);

  const runCorners = useCallback(async () => {
    if (!imageA) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/detect-corners/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: imageA,
          method,
          k,
          threshold_ratio: threshold,
          window_size: windowSize,
          nms_size: nmsSize,
          sigma,
        }),
      });
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
      setCornerResultImage(data.result_image);
      setCornerPoints(data.corners || []);
      setCornerStats({
        num_corners: data.num_corners,
        computation_time_ms: data.computation_time_ms,
        width: data.image_size?.width,
        height: data.image_size?.height,
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [imageA, method, k, threshold, windowSize, nmsSize, sigma]);

  const runSift = useCallback(async () => {
    if (!imageA) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/sift/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: imageA,
          sigma0: siftSigma0,
          num_octaves: siftOctaves,
          num_scales: siftScales,
          contrast_threshold: siftContrast,
          max_keypoints: siftMaxKp,
        }),
      });
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
      setSiftResultImage(data.result_image);
      setSiftKeypoints(data.keypoints || []);
      setSiftStats({
        num_keypoints: data.num_keypoints,
        computation_time_ms: data.computation_time_ms,
        width: data.image_size?.width,
        height: data.image_size?.height,
        kp_density: data.image_size
          ? ((data.num_keypoints / (data.image_size.width * data.image_size.height)) * 1000).toFixed(2)
          : null,
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [imageA, siftSigma0, siftOctaves, siftScales, siftContrast, siftMaxKp]);

  const runMatching = useCallback(async () => {
    if (!imageA || !imageB) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/match-features/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image1: imageA,
          image2: imageB,
          method: matchMethod,
          ratio_threshold: ratioThreshold,
          sigma0: siftSigma0,
          num_octaves: siftOctaves,
          num_scales: siftScales,
          contrast_threshold: siftContrast,
          max_keypoints: matchMaxKp,
        }),
      });
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
      setMatchResultImage(data.result_image);
      setMatchStats({
        num_matches: data.num_matches,
        computation_time_ms: data.computation_time_ms,
        sift_time_ms: data.sift_time_ms,
        match_time_ms: data.match_time_ms,
        kp1: data.keypoints1?.length ?? 0,
        kp2: data.keypoints2?.length ?? 0,
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [
    imageA,
    imageB,
    matchMethod,
    ratioThreshold,
    siftSigma0,
    siftOctaves,
    siftScales,
    siftContrast,
    matchMaxKp,
  ]);

  const primaryAction = useMemo(() => {
    if (mode === "sift") return { label: loading ? "Processing…" : "▶  Run SIFT", onClick: runSift, disabled: !imageA || loading };
    if (mode === "match") return { label: loading ? "Matching…" : "▶  Run Matching", onClick: runMatching, disabled: !imageA || !imageB || loading };
    return { label: loading ? "Processing…" : "▶  Run Detection", onClick: runCorners, disabled: !imageA || loading };
  }, [mode, loading, imageA, imageB, runCorners, runSift, runMatching]);

  return (
    <>
      <style>{CSS}</style>
      <div className="app">
        {/* ── Header ── */}
        <header className="header">
          <span className="header-badge">CV Lab</span>
          <h1>Feature <span>Toolkit</span></h1>
          <span className="header-sub">
            Corners · SIFT · Matching (SSD / NCC)
          </span>
        </header>

        <div className="main">
          {/* ── Sidebar ── */}
          <aside className="sidebar">
            <div className="section">
              <div className="section-title">Mode</div>
              <div className="tabs">
                <button className={`tab ${mode === "corners" ? "active" : ""}`} onClick={() => { setMode("corners"); resetOutputs(); }}>
                  Corners
                </button>
                <button className={`tab ${mode === "sift" ? "active" : ""}`} onClick={() => { setMode("sift"); resetOutputs(); }}>
                  SIFT
                </button>
                <button className={`tab ${mode === "match" ? "active" : ""}`} onClick={() => { setMode("match"); resetOutputs(); }}>
                  Match
                </button>
              </div>
            </div>

            {/* Upload */}
            <div className="section">
              <div className="section-title">{mode === "match" ? "Images" : "Input Image"}</div>
              <div
                className={`upload-zone ${draggingA ? "drag-over" : ""}`}
                onDragOver={(e) => { e.preventDefault(); setDraggingA(true); }}
                onDragLeave={() => setDraggingA(false)}
                onDrop={handleDropA}
                onClick={() => fileARef.current?.click()}
              >
                <input
                  ref={fileARef}
                  type="file"
                  accept="image/*"
                  onChange={(e) => loadFileInto(e.target.files?.[0], setImageA)}
                  style={{ display: "none" }}
                />
                <div className="upload-icon">⬆</div>
                <div className="upload-label">
                  <strong>{imageA ? "✓ Image A loaded" : "Drop or click (Image A)"}</strong>
                  PNG / JPG accepted
                </div>
              </div>

              {mode === "match" && (
                <div style={{ marginTop: 12 }}>
                  <div
                    className={`upload-zone ${draggingB ? "drag-over" : ""}`}
                    onDragOver={(e) => { e.preventDefault(); setDraggingB(true); }}
                    onDragLeave={() => setDraggingB(false)}
                    onDrop={handleDropB}
                    onClick={() => fileBRef.current?.click()}
                  >
                    <input
                      ref={fileBRef}
                      type="file"
                      accept="image/*"
                      onChange={(e) => loadFileInto(e.target.files?.[0], setImageB)}
                      style={{ display: "none" }}
                    />
                    <div className="upload-icon">⬆</div>
                    <div className="upload-label">
                      <strong>{imageB ? "✓ Image B loaded" : "Drop or click (Image B)"}</strong>
                      PNG / JPG accepted
                    </div>
                  </div>
                </div>
              )}
            </div>

            {mode === "corners" && (
              <div className="section">
                <div className="section-title">Corner Operator</div>
                <div className="method-tabs">
                  {Object.entries(METHODS).map(([id, m]) => (
                    <button
                      key={id}
                      className={`method-tab ${method === id ? "active" : ""}`}
                      onClick={() => setMethod(id)}
                    >
                      {m.label}
                    </button>
                  ))}
                </div>
                <div className="method-desc">
                  <code style={{ color: "var(--accent2)", display: "block", marginBottom: 4, fontSize: 10 }}>
                    {METHODS[method].formula}
                  </code>
                  {METHODS[method].desc}
                </div>
              </div>
            )}

            {mode === "match" && (
              <div className="section">
                <div className="section-title">Matching Metric</div>
                <div className="method-tabs">
                  {Object.entries(MATCH_METHODS).map(([id, m]) => (
                    <button
                      key={id}
                      className={`method-tab ${matchMethod === id ? "active" : ""}`}
                      onClick={() => setMatchMethod(id)}
                    >
                      {m.label}
                    </button>
                  ))}
                </div>
                <div className="method-desc">{MATCH_METHODS[matchMethod].desc}</div>
              </div>
            )}

            {/* Parameters */}
            <div className="section">
              <div className="section-title">Parameters</div>

              {mode === "corners" && (
                <>
                  {method === "harris" && (
                    <div className="param-row">
                      <div className="param-header">
                        <span className="param-name">k (Harris constant)</span>
                        <span className="param-value">{k.toFixed(3)}</span>
                      </div>
                      <input type="range" min="0.01" max="0.15" step="0.005"
                        value={k} onChange={(e) => setK(parseFloat(e.target.value))} />
                    </div>
                  )}

                  <div className="param-row">
                    <div className="param-header">
                      <span className="param-name">Threshold ratio</span>
                      <span className="param-value">{threshold.toFixed(3)}</span>
                    </div>
                    <input type="range" min="0.001" max="0.1" step="0.001"
                      value={threshold} onChange={(e) => setThreshold(parseFloat(e.target.value))} />
                  </div>

                  <div className="param-row">
                    <div className="param-header">
                      <span className="param-name">Window size (M matrix)</span>
                      <span className="param-value">{windowSize}×{windowSize}</span>
                    </div>
                    <input type="range" min="3" max="9" step="2"
                      value={windowSize} onChange={(e) => setWindowSize(parseInt(e.target.value))} />
                  </div>

                  <div className="param-row">
                    <div className="param-header">
                      <span className="param-name">NMS window</span>
                      <span className="param-value">{nmsSize}×{nmsSize}</span>
                    </div>
                    <input type="range" min="3" max="15" step="2"
                      value={nmsSize} onChange={(e) => setNmsSize(parseInt(e.target.value))} />
                  </div>

                  <div className="param-row">
                    <div className="param-header">
                      <span className="param-name">Gaussian σ</span>
                      <span className="param-value">{sigma.toFixed(1)}</span>
                    </div>
                    <input type="range" min="0.5" max="3.0" step="0.1"
                      value={sigma} onChange={(e) => setSigma(parseFloat(e.target.value))} />
                  </div>
                </>
              )}

              {(mode === "sift" || mode === "match") && (
                <>
                  <div className="param-row">
                    <div className="param-header">
                      <span className="param-name">σ0 (initial)</span>
                      <span className="param-value">{siftSigma0.toFixed(1)}</span>
                    </div>
                    <input type="range" min="0.8" max="3.0" step="0.1"
                      value={siftSigma0} onChange={(e) => setSiftSigma0(parseFloat(e.target.value))} />
                  </div>

                  <div className="param-row">
                    <div className="param-header">
                      <span className="param-name">Octaves</span>
                      <span className="param-value">{siftOctaves}</span>
                    </div>
                    <input type="range" min="1" max="5" step="1"
                      value={siftOctaves} onChange={(e) => setSiftOctaves(parseInt(e.target.value))} />
                  </div>

                  <div className="param-row">
                    <div className="param-header">
                      <span className="param-name">Scales / octave</span>
                      <span className="param-value">{siftScales}</span>
                    </div>
                    <input type="range" min="3" max="6" step="1"
                      value={siftScales} onChange={(e) => setSiftScales(parseInt(e.target.value))} />
                  </div>

                  <div className="param-row">
                    <div className="param-header">
                      <span className="param-name">Contrast threshold</span>
                      <span className="param-value">{siftContrast.toFixed(3)}</span>
                    </div>
                    <input type="range" min="0.005" max="0.08" step="0.005"
                      value={siftContrast} onChange={(e) => setSiftContrast(parseFloat(e.target.value))} />
                  </div>
                </>
              )}

              {mode === "sift" && (
                <div className="param-row">
                  <div className="param-header">
                    <span className="param-name">Max keypoints</span>
                    <span className="param-value">{siftMaxKp}</span>
                  </div>
                  <input type="range" min="50" max="300" step="25"
                    value={siftMaxKp} onChange={(e) => setSiftMaxKp(parseInt(e.target.value))} />
                  <div className="method-desc" style={{marginTop:6}}>
                    ↑ more detail · ↓ faster (~{siftMaxKp <= 100 ? "5-15s" : siftMaxKp <= 200 ? "15-40s" : "40-90s"})
                  </div>
                </div>
              )}

              {mode === "match" && (
                <>
                  <div className="param-row">
                    <div className="param-header">
                      <span className="param-name">Max keypoints / image</span>
                      <span className="param-value">{matchMaxKp}</span>
                    </div>
                    <input type="range" min="50" max="200" step="25"
                      value={matchMaxKp} onChange={(e) => setMatchMaxKp(parseInt(e.target.value))} />
                    <div className="method-desc" style={{marginTop:6}}>
                      ↑ more matches · ↓ faster (~{matchMaxKp <= 75 ? "10-30s" : matchMaxKp <= 125 ? "30-60s" : "60-120s"})
                    </div>
                  </div>
                  <div className="param-row">
                    <div className="param-header">
                      <span className="param-name">Lowe ratio</span>
                      <span className="param-value">{ratioThreshold.toFixed(2)}</span>
                    </div>
                    <input type="range" min="0.4" max="0.95" step="0.01"
                      value={ratioThreshold} onChange={(e) => setRatioThreshold(parseFloat(e.target.value))} />
                  </div>
                </>
              )}

              <button
                className={`run-btn ${loading ? "loading" : ""}`}
                onClick={primaryAction.onClick}
                disabled={primaryAction.disabled}
              >
                {primaryAction.label}
              </button>

              {error && <div className="error-box">⚠ {error}</div>}
            </div>

            {/* Stats */}
            {mode === "corners" && cornerStats && (
              <div className="section">
                <div className="section-title">Results</div>
                <div className="stats-grid">
                  <div className="stat-card">
                    <div className="stat-label">Corners found</div>
                    <div className="stat-value">{cornerStats.num_corners}</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Compute time</div>
                    <div className="stat-value time">
                      {cornerStats.computation_time_ms < 1000
                        ? cornerStats.computation_time_ms.toFixed(0)
                        : (cornerStats.computation_time_ms / 1000).toFixed(2)}
                      <span className="stat-unit">
                        {cornerStats.computation_time_ms < 1000 ? "ms" : "s"}
                      </span>
                    </div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Image width</div>
                    <div className="stat-value" style={{ fontSize: 16 }}>{cornerStats.width}<span className="stat-unit">px</span></div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Image height</div>
                    <div className="stat-value" style={{ fontSize: 16 }}>{cornerStats.height}<span className="stat-unit">px</span></div>
                  </div>
                </div>
              </div>
            )}

            {mode === "sift" && siftStats && (
              <div className="section">
                <div className="section-title">Results — SIFT Report</div>
                <div className="stats-grid">
                  <div className="stat-card">
                    <div className="stat-label">Keypoints found</div>
                    <div className="stat-value">{siftStats.num_keypoints}</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Computation time</div>
                    <div className="stat-value time">
                      {siftStats.computation_time_ms < 1000
                        ? siftStats.computation_time_ms.toFixed(0)
                        : (siftStats.computation_time_ms / 1000).toFixed(2)}
                      <span className="stat-unit">
                        {siftStats.computation_time_ms < 1000 ? "ms" : "s"}
                      </span>
                    </div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Image size</div>
                    <div className="stat-value" style={{ fontSize: 14 }}>{siftStats.width}×{siftStats.height}<span className="stat-unit">px</span></div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Density (kp/kpx)</div>
                    <div className="stat-value" style={{ fontSize: 16 }}>{siftStats.kp_density}<span className="stat-unit"></span></div>
                  </div>
                </div>
              </div>
            )}

            {mode === "match" && matchStats && (
              <div className="section">
                <div className="section-title">Results — Match Report</div>
                <div className="stats-grid">
                  <div className="stat-card">
                    <div className="stat-label">Matches found</div>
                    <div className="stat-value">{matchStats.num_matches}</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Total time</div>
                    <div className="stat-value time">
                      {matchStats.computation_time_ms < 1000
                        ? matchStats.computation_time_ms.toFixed(0)
                        : (matchStats.computation_time_ms / 1000).toFixed(2)}
                      <span className="stat-unit">
                        {matchStats.computation_time_ms < 1000 ? "ms" : "s"}
                      </span>
                    </div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">SIFT time</div>
                    <div className="stat-value" style={{ fontSize: 15 }}>
                      {(matchStats.sift_time_ms / 1000).toFixed(2)}<span className="stat-unit">s</span>
                    </div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Match time</div>
                    <div className="stat-value" style={{ fontSize: 15 }}>
                      {matchStats.match_time_ms < 1000
                        ? matchStats.match_time_ms.toFixed(0)
                        : (matchStats.match_time_ms / 1000).toFixed(2)}
                      <span className="stat-unit">{matchStats.match_time_ms < 1000 ? "ms" : "s"}</span>
                    </div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">KP img A / B</div>
                    <div className="stat-value" style={{ fontSize: 14 }}>{matchStats.kp1} / {matchStats.kp2}</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Match quality</div>
                    <div className="stat-value" style={{ fontSize: 14 }}>
                      {matchStats.kp1 + matchStats.kp2 > 0
                        ? ((matchStats.num_matches / Math.min(matchStats.kp1, matchStats.kp2)) * 100).toFixed(0)
                        : "—"}
                      <span className="stat-unit">%</span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Corner table */}
            {mode === "corners" && cornerPoints.length > 0 && (
              <div className="section" style={{ flexGrow: 1 }}>
                <div className="section-title">Corner Points (top 200)</div>
                <div className="corners-list">
                  <div className="corner-row">
                    <span>X</span><span>Y</span><span>Response</span>
                  </div>
                  {cornerPoints.map(([x, y, r], i) => (
                    <div className="corner-row" key={i}>
                      <span className="corner-x">{x}</span>
                      <span className="corner-y">{y}</span>
                      <span className="corner-r">{r.toExponential(2)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </aside>

          {/* ── Image panels ── */}
          <div className={`canvas-area ${mode === "match" ? "three" : ""}`}>
            {/* Input */}
            <div className="image-panel">
              <span className="panel-label">{mode === "match" ? "Image A" : "Input"}</span>
              <div className="image-container">
                {imageA ? (
                  <img src={imageA} alt="Input" />
                ) : (
                  <div className="placeholder">
                    <div className="placeholder-icon">□</div>
                    <div className="placeholder-text">
                      Upload an image<br />to begin
                    </div>
                  </div>
                )}
              </div>
            </div>

            {mode === "match" && (
              <div className="image-panel">
                <span className="panel-label">Image B</span>
                <div className="image-container">
                  {imageB ? (
                    <img src={imageB} alt="Image B" />
                  ) : (
                    <div className="placeholder">
                      <div className="placeholder-icon">□</div>
                      <div className="placeholder-text">
                        Upload image B<br />to match
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Result */}
            <div className="image-panel">
              <span className="panel-label">
                {mode === "corners"
                  ? (method === "harris" ? "Harris Result" : "λ− Result")
                  : mode === "sift"
                    ? "SIFT Keypoints"
                    : "Matches"}
              </span>
              <div className="image-container">
                {loading ? (
                  <div className="placeholder">
                    <div className="spinner" />
                    <div className="placeholder-text">Running pipeline…</div>
                  </div>
                ) : (mode === "corners" && cornerResultImage) ? (
                  <img src={cornerResultImage} alt="Corner result" />
                ) : (mode === "sift" && siftResultImage) ? (
                  <img src={siftResultImage} alt="SIFT result" />
                ) : (mode === "match" && matchResultImage) ? (
                  <img src={matchResultImage} alt="Match result" />
                ) : (
                  <div className="placeholder">
                    <div className="placeholder-icon" style={{ opacity: 0.15 }}>◉</div>
                    <div className="placeholder-text">
                      {mode === "corners" ? "Corner overlay" : mode === "sift" ? "Keypoints overlay" : "Match visualization"}<br />
                      will appear here
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}