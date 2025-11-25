import { useEffect, useState } from "react";

const API_BASE = "http://127.0.0.1:8000";

function App() {
  // ===========================
  //  THEME
  // ===========================
  const [theme, setTheme] = useState(() => {
    if (typeof window === "undefined") return "dark";
    return localStorage.getItem("papersense-theme") || "dark";
  });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("papersense-theme", theme);
  }, [theme]);

  function toggleTheme() {
    setTheme((t) => (t === "dark" ? "light" : "dark"));
  }

  function themeLabel() {
    return theme === "dark" ? "Dark" : "Light";
  }

  // ===========================
  //  STATE
  // ===========================
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [history, setHistory] = useState([]);
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploadFile, setUploadFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [sortBy, setSortBy] = useState("score");
  const [sortOrder, setSortOrder] = useState("desc");

  // ===========================
  //  INITIAL LOAD
  // ===========================
  useEffect(() => {
    fetchFiles();
    fetchHistory();
  }, []);

  async function fetchFiles() {
    try {
      const res = await fetch(`${API_BASE}/files`);
      if (!res.ok) return;
      const data = await res.json();
      setFiles(data);
    } catch (e) {
      console.error("Error fetching files", e);
    }
  }

  async function fetchHistory() {
    try {
      const res = await fetch(`${API_BASE}/history`);
      if (!res.ok) return;
      const data = await res.json();
      if (Array.isArray(data.history)) {
        setHistory(data.history);
      }
    } catch (e) {
      console.error("Error fetching history", e);
    }
  }

  // ===========================
  //  SEARCH
  // ===========================
  async function handleSearch(e) {
    e?.preventDefault();
    setError("");

    if (!query.trim()) {
      setError("Please type something to search.");
      return;
    }

    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          sort_by: sortBy,
          sort_order: sortOrder,
        }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || "Search failed");
      }

      const data = await res.json();
      setResults(data.results || []);
      setHistory(data.history || []);
    } catch (err) {
      console.error(err);
      setError(err.message || "Search failed");
    } finally {
      setLoading(false);
    }
  }

  // ===========================
  //  UPLOAD PDF
  // ===========================
  async function handleUpload() {
    if (!uploadFile) {
      setError("Please choose a PDF file.");
      return;
    }

    setError("");
    setUploading(true);

    try {
      const form = new FormData();
      form.append("file", uploadFile);

      const res = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || "Upload failed");
      }

      await fetchFiles();

      if (query.trim()) {
        await handleSearch();
      }
    } catch (err) {
      console.error(err);
      setError(err.message || "Upload failed");
    } finally {
      setUploading(false);
      setUploadFile(null);
    }
  }

  // ===========================
  //  HELPERS
  // ===========================
  function handleHistoryClick(q) {
    setQuery(q);
    handleSearch();
  }

  function openPdf(url) {
    const full = `${API_BASE}${url}`;
    window.open(full, "_blank", "noopener,noreferrer");
  }

  function thumbUrl(filename) {
    return `${API_BASE}/thumbnail/${encodeURIComponent(filename)}`;
  }

  function scoreLabelColor(category) {
    if (category === "high") return "score-badge score-high";
    if (category === "medium") return "score-badge score-medium";
    return "score-badge score-low";
  }

  // ===========================
  //  JSX
  // ===========================
  return (
    <div className="app-shell">
      {/* Background orbits */}
      <div className="app-orbit"></div>
      <div className="app-orbit app-orbit-2"></div>

      <main className="app-main">
        {/* HEADER */}
        <header className="app-header glass">
          <div className="app-header-content">
            <div className="app-title-block">
              <h1 className="app-title">PaperSense</h1>
              <p className="app-subtitle">
                AI-powered semantic search across your local PDF knowledge base.
              </p>
            </div>

            <div className="app-header-right">
              <div className="theme-toggle">
                <span className="theme-label">{themeLabel()} mode</span>
                <button
                  type="button"
                  className={`theme-switch ${
                    theme === "dark" ? "is-dark" : "is-light"
                  }`}
                  onClick={toggleTheme}
                >
                  <span className="theme-knob">
                    {theme === "dark" ? "🌙" : "☀️"}
                  </span>
                </button>
              </div>
            </div>
          </div>
        </header>

        {/* TOP GRID */}
        <section className="top-grid">
          {/* LEFT: Search + Upload */}
          <div className="glass search-panel">
            <form onSubmit={handleSearch} className="search-form">
              <div className="search-input-row">
                <div className="search-input-wrap">
                  <span className="search-icon">🔍</span>
                  <input
                    type="text"
                    className="search-input"
                    placeholder="Ask anything from your PDFs…"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                  />
                </div>

                <button
                  type="submit"
                  className="pill-button primary"
                  disabled={loading}
                >
                  {loading ? "Searching…" : "Search"}
                </button>
              </div>

              <div className="search-options">
                <div className="search-sort-group">
                  <span className="label">Sort by</span>
                  <select
                    className="pill-select"
                    value={sortBy}
                    onChange={(e) => setSortBy(e.target.value)}
                  >
                    <option value="score">Score</option>
                    <option value="filename">Filename</option>
                    <option value="occurrences">Occurrences</option>
                  </select>

                  <select
                    className="pill-select"
                    value={sortOrder}
                    onChange={(e) => setSortOrder(e.target.value)}
                  >
                    <option value="desc">Desc</option>
                    <option value="asc">Asc</option>
                  </select>
                </div>
              </div>
            </form>

            {error && <div className="error-banner">{error}</div>}
            
            <div className="upload-section">
              <div className="upload-header">
                <span className="upload-title">Upload PDF</span>
                <span className="upload-subtitle">
                  New files are indexed automatically for semantic search.
                </span>
              </div>
              
              <div className="upload-box">
                <label className="pill-button subtle file-label">
                  <input
                    type="file"
                    accept="application/pdf"
                    onChange={(e) =>
                      setUploadFile(e.target.files?.[0] || null)
                    }
                  />
                  <span className="file-label-text">
                      {uploadFile ? uploadFile.name : "Choose file"}
                  </span>
                </label>
                <button
                  type="button"
                  className="pill-button success"
                  onClick={handleUpload}
                  disabled={uploading || !uploadFile}
                >
                  {uploading ? "Uploading…" : "Upload"}
                </button>
              </div>
            </div>
          </div>

          {/* RIGHT: History + Files */}
          <div className="right-column">
            {/* Recent Searches */}
            <div className="glass side-card">
              <div className="side-card-header">
                <span className="side-card-title">Recent searches</span>
              </div>
              <div className="side-card-body scrollable">
                {history.length === 0 ? (
                  <div className="empty-text">No searches yet.</div>
                ) : (
                  <ul className="chip-list">
                    {history.map((h, idx) => (
                      <li key={idx}>
                        <button
                          type="button"
                          className="chip"
                          onClick={() => handleHistoryClick(h)}
                        >
                          {h}
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>

            {/* Indexed PDFs */}
            <div className="glass side-card">
              <div className="side-card-header">
                <span className="side-card-title">Indexed PDFs</span>
                <span className="side-card-meta">{files.length} files</span>
              </div>
              <div className="side-card-body scrollable">
                {files.length === 0 ? (
                  <div className="empty-text">No PDFs found in /pdfs.</div>
                ) : (
                  <ul className="file-list">
                    {files.map((f) => (
                      <li key={f.filename} className="file-row">
                        <div className="file-icon">📄</div>
                        <div className="file-info">
                          <div className="file-name">{f.filename}</div>
                          <div className="file-meta">
                            {f.pages} pages · {f.size_kb} KB
                          </div>
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          </div>
        </section>

        {/* RESULTS */}
        <section className="results-section">
          <div className="results-header">
            <h2>Results</h2>
            <span className="results-meta">
              {results.length > 0
                ? `${results.length} matches`
                : "No results yet."}
            </span>
          </div>

          {results.length === 0 ? (
            <div className="glass empty-state">
              <p>
                Try searching for something like{" "}
                <span className="hint">"compensation structure"</span> or{" "}
                <span className="hint">"employee relations"</span>.
              </p>
            </div>
          ) : (
            <div className="results-list">
              {results.map((r, idx) => (
                <article key={idx} className="glass result-card">
                  {/* Thumbnail column */}
                  <div className="thumb-col">
                    <div className="thumb-wrapper">
                      <div className="thumb-inner">
                        <img
                          src={thumbUrl(r.filename)}
                          alt={r.filename}
                          className="thumb-img"
                          onError={(e) => {
                            e.target.style.display = "none";
                          }}
                        />
                      </div>
                    </div>
                    <div className="thumb-footer">
                      <span className="thumb-filename">{r.filename}</span>
                      <span className="thumb-page">Page {r.page}</span>
                    </div>
                  </div>

                  {/* Main result column */}
                  <div className="result-main">
                    <div className="result-header-line">
                      <div className="result-title-block">
                        <div className="result-title">{r.filename}</div>
                        <div className="result-submeta">
                          {r.metadata.pages} pages · {r.metadata.size_kb} KB
                        </div>
                      </div>
                      <div className="result-score-block">
                        <span className={scoreLabelColor(r.score_category)}>
                          {r.score_category.toUpperCase()}
                        </span>
                        <span className="score-detail">
                          score {r.score.toFixed(3)}
                        </span>
                        <span className="score-detail">
                          {r.occurrences} occurrence
                          {r.occurrences === 1 ? "" : "s"}
                        </span>
                      </div>
                    </div>

                    <div
                      className="result-snippet"
                      dangerouslySetInnerHTML={{ __html: r.snippet }}
                    />

                    <div className="result-actions">
                      <button
                        type="button"
                        className="pill-button ghost"
                        onClick={() => openPdf(r.url)}
                      >
                        Open PDF
                      </button>
                    </div>
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;