import React, { useState } from "react";
import axios from "axios";

type Match = {
  chunk_id: number;
  text: string;
  html: string;
  score: number;
};

export default function App() {
  const [url, setUrl] = useState("");
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<Match[]>([]);
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setResults([]);
    setLoading(true);
    try {
      const res = await axios.post("/search", { url, query });
      setResults(res.data);
    } catch (err: any) {
      setError(err?.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  }

  // Helpers: extract first N <p> texts from html string or fallback to plain text
  function getFirstParagraphs(html: string, fallbackText?: string, n = 2) {
    try {
      const parser = new DOMParser();
      const doc = parser.parseFromString(html || "", "text/html");
      const ps = Array.from(doc.querySelectorAll("p")).slice(0, n);
      const texts = ps
        .map((p) => p.textContent?.trim())
        .filter(Boolean) as string[];
      if (texts.length) return texts.join("\n\n");
    } catch (e) {
      // ignore parsing errors
    }
    // fallback: strip tags from provided fallbackText or html and take first n lines
    const source = (fallbackText || html || "").replace(/<[^>]+>/g, "").trim();
    if (!source) return "";
    const parts = source
      .split(/\n+/)
      .map((s) => s.trim())
      .filter(Boolean);
    return parts.slice(0, n).join("\n\n");
  }

  function truncate(str: string, n = 300) {
    if (!str) return "";
    return str.length > n ? str.slice(0, n) + "..." : str;
  }

  const [showFullHtml, setShowFullHtml] = useState<Record<number, boolean>>({});
  function toggleFull(idx: number) {
    setShowFullHtml((prev) => ({ ...prev, [idx]: !prev[idx] }));
  }

  return (
    <div className="container">
      <h1>Website HTML Content Search</h1>
      <form onSubmit={onSubmit} className="form">
        <label>Website URL</label>
        <input
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="https://example.com"
        />
        <label>Search Query</label>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search terms"
        />
        <button type="submit" disabled={loading}>
          {" "}
          {loading ? "Searching..." : "Search"}
        </button>
      </form>

      {error && <div className="error">{error}</div>}

      <div className="results">
        {results.length > 0 && (
          <div className="card single-card">
            <h2>Top {results.length} matches</h2>
            <ol className="match-list">
              {results.map((r, idx) => {
                const previewText = getFirstParagraphs(r.html || "", r.text, 2);
                const htmlContent = r.html || "";
                const isFull = !!showFullHtml[idx];
                const htmlPreview = isFull
                  ? htmlContent
                  : truncate(htmlContent, 300);
                return (
                  <li key={idx} className="match-item">
                    <div className="score-box">
                      Match: {(Number(r.score) * 100).toFixed(2)}%
                    </div>
                    <div className="meta">
                      {idx + 1}. Chunk #{r.chunk_id}
                    </div>
                    <div className="text">{previewText || <i>No text</i>}</div>
                    <details>
                      <summary>Raw HTML</summary>
                      <div className="html">
                        {htmlPreview || <i>No html</i>}
                      </div>
                      {htmlContent && htmlContent.length > 300 && (
                        <div>
                          <button
                            className="toggle-btn"
                            onClick={() => toggleFull(idx)}
                            aria-expanded={isFull}
                          >
                            {isFull ? "Show less" : "Show full HTML"}
                          </button>
                        </div>
                      )}
                    </details>
                  </li>
                );
              })}
            </ol>
          </div>
        )}
      </div>
    </div>
  );
}
