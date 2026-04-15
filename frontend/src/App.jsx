import { useState, useCallback, useRef, useEffect, useMemo } from "react";

// ==============================================================================
// CONFIGURATION
// ==============================================================================

const API_BASE_URL = import.meta.env.VITE_API_URL || "https://ai-product-review.onrender.com";
const ANALYZE_ENDPOINT = `${API_BASE_URL}/api/v1/analyze-raw`;

// ==============================================================================
// 🔥 FINAL API FUNCTION - All Edge Cases Handled
// ==============================================================================

async function analyzeReviews(rawText) {
  let response;

  // Handle network failures
  try {
    response = await fetch(ANALYZE_ENDPOINT, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(import.meta.env.VITE_API_KEY && { "X-API-Key": import.meta.env.VITE_API_KEY }),
      },
      body: JSON.stringify({ raw_text: rawText }),
    });
  } catch (networkError) {
    throw new Error("🌐 Network error. Please check your connection and try again.");
  }

  // Safe JSON parsing - handle empty/corrupted responses
  const text = await response.text();
  let data = null;

  if (text && text.trim()) {
    try {
      data = JSON.parse(text);
    } catch {
      throw new Error("⚠️ Invalid server response. Please try again.");
    }
  }

  if (!response.ok) {
    if (response.status === 400) {
      if (data?.detail?.error === "OUT_OF_SCOPE") {
        throw new Error("❌ Not a product review. Please provide actual product reviews.");
      }
      throw new Error(data?.detail?.message || "Invalid input format.");
    }

    if (response.status === 429) {
      throw new Error("⚠️ Rate limit exceeded. Please wait a moment.");
    }

    if (response.status >= 500) {
      throw new Error("🛠️ Server error. Please try again later.");
    }

    throw new Error(data?.detail?.message || "Request failed.");
  }

  return data;
}

// ==============================================================================
// UTILITY FUNCTIONS
// ==============================================================================

function toSafeNumber(value, fallback = 0) {
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
}

function toSafeList(value) {
  if (!Array.isArray(value)) return [];
  return value
    .map((item) => {
      if (typeof item === "string") return item.trim();
      if (item && typeof item === "object") {
        return item?.text || item?.content || "";
      }
      return "";
    })
    .filter(Boolean);
}

function formatPercent(value) {
  return `${toSafeNumber(value).toFixed(2)}%`;
}

function formatScore(value) {
  const safe = toSafeNumber(value);
  return safe > 0 ? `${safe.toFixed(2)} / 5` : "N/A";
}

// ==============================================================================
// SENTIMENT CHART
// ==============================================================================

function SentimentChart({ positive = 0, neutral = 0, negative = 0 }) {
  const maxPercent = Math.max(positive, neutral, negative, 1);

  return (
    <div className="w-full p-4">
      <div className="flex h-52 items-end justify-around gap-3">
        <div className="flex flex-1 flex-col items-center gap-2">
          <div className="relative flex h-full w-full flex-col justify-end">
            <div
              className="w-full rounded-t-lg bg-gradient-to-t from-emerald-600 to-emerald-400 transition-all duration-700 ease-out"
              style={{
                height: `${(positive / maxPercent) * 100}%`,
                minHeight: positive > 0 ? "15%" : "0%",
              }}
            />
          </div>
          <span className="text-sm font-semibold text-emerald-300">
            {positive.toFixed(1)}%
          </span>
        </div>

        <div className="flex flex-1 flex-col items-center gap-2">
          <div className="relative flex h-full w-full flex-col justify-end">
            <div
              className="w-full rounded-t-lg bg-gradient-to-t from-amber-600 to-amber-400 transition-all duration-700 ease-out"
              style={{
                height: `${(neutral / maxPercent) * 100}%`,
                minHeight: neutral > 0 ? "15%" : "0%",
              }}
            />
          </div>
          <span className="text-sm font-semibold text-amber-300">
            {neutral.toFixed(1)}%
          </span>
        </div>

        <div className="flex flex-1 flex-col items-center gap-2">
          <div className="relative flex h-full w-full flex-col justify-end">
            <div
              className="w-full rounded-t-lg bg-gradient-to-t from-rose-600 to-rose-400 transition-all duration-700 ease-out"
              style={{
                height: `${(negative / maxPercent) * 100}%`,
                minHeight: negative > 0 ? "15%" : "0%",
              }}
            />
          </div>
          <span className="text-sm font-semibold text-rose-300">
            {negative.toFixed(1)}%
          </span>
        </div>
      </div>

      <div className="mt-4 flex justify-around border-t border-white/10 pt-4">
        <div className="flex items-center gap-2">
          <div className="h-3 w-3 rounded-full bg-emerald-400" />
          <span className="text-xs text-slate-400">Positive</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="h-3 w-3 rounded-full bg-amber-400" />
          <span className="text-xs text-slate-400">Neutral</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="h-3 w-3 rounded-full bg-rose-400" />
          <span className="text-xs text-slate-400">Negative</span>
        </div>
      </div>
    </div>
  );
}

// ==============================================================================
// TOAST NOTIFICATION
// ==============================================================================

function Toast({ message, type = "info", onClose }) {
  useEffect(() => {
    const timer = setTimeout(onClose, 4000);
    return () => clearTimeout(timer);
  }, [onClose]);

  const styles = {
    success: "bg-emerald-600 border-emerald-500",
    error: "bg-rose-600 border-rose-500",
    warning: "bg-amber-600 border-amber-500",
    info: "bg-cyan-600 border-cyan-500",
  };

  const icons = {
    success: "✓",
    error: "✕",
    warning: "⚠",
    info: "ℹ",
  };

  return (
    <div
      className={`fixed bottom-6 right-6 z-50 animate-slide-up rounded-xl border-2 ${styles[type]} px-6 py-4 shadow-2xl`}
    >
      <div className="flex items-center gap-4">
        <span className="text-xl">{icons[type]}</span>
        <span className="text-sm font-medium text-white">{message}</span>
        <button
          onClick={onClose}
          className="ml-2 text-white/70 transition-colors hover:text-white"
        >
          <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
    </div>
  );
}

// ==============================================================================
// LOADING SKELETON
// ==============================================================================

function LoadingSkeleton() {
  return (
    <section className="animate-pulse rounded-[2rem] border border-white/10 bg-[rgba(7,23,19,0.72)] p-6 sm:p-8">
      <div className="mb-8 flex items-center gap-4">
        <div className="h-10 w-48 rounded-lg bg-white/5" />
        <div className="h-4 w-32 rounded bg-white/5" />
      </div>

      <div className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        {[...Array(6)].map((_, i) => (
          <div key={i} className="h-20 rounded-2xl bg-white/5" />
        ))}
      </div>

      <div className="grid gap-6 lg:grid-cols-[1fr_0.95fr]">
        <div className="space-y-6">
          <div className="h-32 rounded-[1.5rem] bg-white/5" />
          <div className="grid gap-6 sm:grid-cols-2">
            <div className="h-40 rounded-[1.5rem] bg-white/5" />
            <div className="h-40 rounded-[1.5rem] bg-white/5" />
          </div>
        </div>
        <div className="h-64 rounded-2xl bg-white/5" />
      </div>
    </section>
  );
}

// ==============================================================================
// MAIN APP COMPONENT
// ==============================================================================

export default function App() {
  const [inputText, setInputText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [toast, setToast] = useState(null);
  const [copySuccess, setCopySuccess] = useState(false);

  const isMountedRef = useRef(true);

  // ==============================================================================
  // EFFECTS
  // ==============================================================================

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    if (result && !loading) {
      document.getElementById("results")?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [result, loading]);

  // ==============================================================================
  // HELPERS
  // ==============================================================================

  const showToast = useCallback((message, type = "info") => {
    if (isMountedRef.current) {
      setToast({ message, type });
    }
  }, []);

  const reviews = useMemo(() => {
    return inputText
      .split(/\n+/)
      .map((line) => line.trim())
      .filter((line) => line.length > 0);
  }, [inputText]);

  const hasInput = inputText.trim().length >= 10;

  // ==============================================================================
  // MAIN HANDLER - CLEAN VERSION
  // ==============================================================================

  const handleAnalyze = async (e) => {
    e?.preventDefault();

    if (!hasInput) {
      setError("Please enter at least 10 characters.");
      return;
    }

    if (loading) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await analyzeReviews(inputText.trim());

      if (!isMountedRef.current) return;

      if (data?.out_of_scope) {
        setError("❌ Not a product review. Please provide actual product reviews.");
      } else {
        setResult(data);
        showToast("Analysis complete!", "success");
      }
    } catch (err) {
      if (!isMountedRef.current) return;
      setError(err.message || "❌ Something went wrong. Please try again.");
      showToast("Analysis failed", "error");
    } finally {
      if (isMountedRef.current) {
        setLoading(false);
      }
    }
  };

  const handleReset = () => {
    setInputText("");
    setResult(null);
    setError("");
  };

  const handleCopy = async () => {
    if (!result) return;
    try {
      await navigator.clipboard.writeText(JSON.stringify(result, null, 2));
      setCopySuccess(true);
      showToast("Copied to clipboard!", "success");
      setTimeout(() => setCopySuccess(false), 2000);
    } catch {
      showToast("Failed to copy", "error");
    }
  };

  const loadExample = () => {
    setInputText(
      "Battery life is excellent and lasts all day. Camera quality is amazing with great night mode. The display is bright but could be sharper. Sound quality is average. Fast charging is a big plus. Some apps run slow occasionally."
    );
  };

  // ==============================================================================
  // RENDER HELPERS - SAFE VERSIONS
  // ==============================================================================

  const renderList = (items, emptyMsg) => {
    if (!items?.length) {
      return <p className="mt-3 text-sm italic text-slate-500">{emptyMsg}</p>;
    }
    return (
      <ul className="mt-3 space-y-2">
        {items.map((item, i) => (
          <li key={i} className="rounded-xl bg-black/20 px-4 py-3 text-sm text-slate-100">
            {typeof item === "string"
              ? item
              : item?.text || item?.content || ""}
          </li>
        ))}
      </ul>
    );
  };

  // Safe neutral points extraction (handles both naming conventions)
  const neutralPoints = useMemo(() => {
    if (!result) return [];
    return toSafeList(result.neutral_points || result.neutralPoints || []);
  }, [result]);

  // ==============================================================================
  // RENDER
  // ==============================================================================

  return (
    <main className="relative min-h-screen overflow-hidden bg-gradient-to-br from-slate-950 via-slate-900 to-emerald-950 px-4 py-10 text-slate-100 sm:px-6 lg:px-8">
      {/* Background Effects */}
      <div aria-hidden="true" className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute left-[12%] top-[-4rem] h-72 w-72 rounded-full bg-emerald-500/10 blur-3xl" />
        <div className="absolute right-[-5rem] top-28 h-80 w-80 rounded-full bg-amber-500/8 blur-3xl" />
        <div className="absolute bottom-12 left-1/2 h-64 w-64 -translate-x-1/2 rounded-full bg-teal-500/8 blur-3xl" />
      </div>

      <div className="relative mx-auto flex max-w-6xl flex-col gap-8">
        {/* Header Section */}
        <section className="overflow-hidden rounded-[2rem] border border-white/10 bg-[rgba(7,23,19,0.78)] shadow-2xl">
          <div className="grid gap-6 lg:grid-cols-[minmax(0,1.1fr)_320px] lg:items-center">
            <div className="p-6 sm:p-8 lg:p-10">
              <div className="mb-4 flex items-center gap-3">
                <div className="inline-flex rounded-full border border-emerald-400/30 bg-emerald-400/10 px-3 py-1 text-xs font-semibold uppercase tracking-widest text-emerald-200">
                  React + FastAPI
                </div>
              </div>

              <h1 className="text-3xl font-bold tracking-tight sm:text-4xl lg:text-5xl">
                AI Product Review{" "}
                <span className="bg-gradient-to-r from-emerald-400 to-teal-300 bg-clip-text text-transparent">
                  Analyzer
                </span>
              </h1>
              <p className="mt-4 max-w-2xl text-sm text-slate-300/80 sm:text-base">
                Analyze product reviews to extract sentiment, pros, cons, and key insights powered by AI.
              </p>

              {/* Form */}
              <form className="mt-8 space-y-5" onSubmit={handleAnalyze}>
                <label className="block">
                  <div className="mb-2 flex items-center justify-between">
                    <span className="text-sm font-medium text-slate-200">Enter Reviews or Text</span>
                    <span className="rounded-full border border-white/10 px-3 py-1 text-xs text-slate-300">
                      {reviews.length} review{reviews.length !== 1 ? "s" : ""}
                    </span>
                  </div>

                  <textarea
                    value={inputText}
                    onChange={(e) => {
                      setInputText(e.target.value);
                      if (error) setError("");
                    }}
                    placeholder={`Battery life is excellent and setup was simple.\nThe camera is average for the price.\nPerformance feels slow sometimes.\n\nSeparate reviews with newlines!`}
                    className="min-h-48 w-full rounded-3xl border border-white/10 bg-slate-950/55 px-4 py-3 text-base text-slate-100 outline-none transition-all placeholder:text-slate-600 focus:border-emerald-400/60 focus:ring-2 focus:ring-emerald-400/20 sm:min-h-64 sm:px-5 sm:py-4 resize-y"
                    disabled={loading}
                    rows={6}
                  />

                  <div className="mt-2 flex items-center justify-between text-xs text-slate-500">
                    <span>Separate reviews with newlines</span>
                    <span>{inputText.length}/10,000</span>
                  </div>
                </label>

                {/* Error Display */}
                {error && (
                  <div className="rounded-xl border border-rose-400/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200 whitespace-pre-line">
                    {error}
                  </div>
                )}

                {/* Action Buttons */}
                <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                  <div className="flex flex-col gap-1 text-xs sm:text-sm">
                    {loading && (
                      <div className="flex items-center gap-2 text-emerald-300">
                        <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                        </svg>
                        Analyzing reviews...
                      </div>
                    )}
                  </div>

                  <div className="flex items-center gap-3">
                    {result && (
                      <button
                        type="button"
                        onClick={handleReset}
                        className="rounded-full border border-white/20 bg-white/5 px-4 py-2 text-sm font-medium text-slate-300 transition-all hover:bg-white/10"
                      >
                        New Analysis
                      </button>
                    )}

                    <button
                      type="submit"
                      disabled={!hasInput || loading}
                      className="inline-flex items-center justify-center gap-2 rounded-full bg-gradient-to-r from-emerald-500 via-teal-400 to-cyan-400 px-6 py-3 text-sm font-semibold text-slate-950 shadow-lg shadow-emerald-500/30 transition-all hover:-translate-y-0.5 hover:scale-105 hover:shadow-xl disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:translate-y-0 disabled:hover:scale-100"
                    >
                      {loading ? (
                        <>
                          <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                          </svg>
                          Analyzing...
                        </>
                      ) : (
                        <>
                          <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.357-1.743-1-2.366l-.548-.547z" />
                          </svg>
                          Analyze Reviews
                        </>
                      )}
                    </button>
                  </div>
                </div>
              </form>
            </div>

            {/* Example Sidebar */}
            <div className="px-6 pb-6 sm:px-8 lg:px-0 lg:pr-10">
              <div className="rounded-[1.75rem] border border-white/10 bg-gradient-to-b from-white/5 to-transparent p-5">
                <p className="text-xs font-semibold uppercase tracking-widest text-amber-200/75">
                  Try Example
                </p>
                <div
                  onClick={loadExample}
                  className="mt-4 cursor-pointer rounded-2xl border border-emerald-400/20 bg-black/20 px-4 py-4 text-sm leading-relaxed text-slate-200 transition-all hover:bg-black/30 hover:border-emerald-400/30"
                >
                  "Battery life is excellent. Camera is amazing but screen could be sharper..."
                </div>
                <p className="mt-3 text-xs text-slate-500">Click to load example</p>
              </div>
            </div>
          </div>
        </section>

        {/* Loading State */}
        {loading && <LoadingSkeleton />}

        {/* Empty State */}
        {!result && !loading && (
          <div className="flex flex-col items-center justify-center rounded-[2rem] border border-white/10 bg-[rgba(7,23,19,0.72)] py-16 text-center">
            <div className="text-5xl sm:text-6xl">🔍</div>
            <p className="mt-6 text-base text-slate-200 sm:text-lg">Ready to analyze your reviews</p>
            <p className="mt-2 text-sm text-slate-500">Paste product reviews above and click Analyze</p>
          </div>
        )}

        {/* Results Section */}
        {result && !loading && (
          <section id="results" className="rounded-[2rem] border border-white/10 bg-[rgba(7,23,19,0.72)] p-6 shadow-2xl sm:p-8">
            {/* Results Header */}
            <div className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h2 className="text-xl font-semibold text-white sm:text-2xl">Analysis Results</h2>
                <p className="mt-1 text-sm text-slate-300">AI-powered analysis complete</p>
              </div>

              <div className="flex items-center gap-3">
                <button
                  onClick={handleAnalyze}
                  className="rounded-full border border-emerald-400/30 bg-emerald-400/10 px-4 py-2 text-sm font-medium text-emerald-300 transition-all hover:bg-emerald-400/20"
                >
                  🔄 Re-analyze
                </button>
                <button
                  onClick={handleCopy}
                  className={`inline-flex items-center gap-2 rounded-full border px-4 py-2 text-sm font-medium transition-all ${
                    copySuccess
                      ? "border-emerald-400/30 bg-emerald-400/10 text-emerald-300"
                      : "border-white/20 bg-white/5 text-slate-300 hover:bg-white/10"
                  }`}
                >
                  {copySuccess ? (
                    <>
                      <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                      Copied!
                    </>
                  ) : (
                    <>
                      <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                      </svg>
                      Copy JSON
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Stats Grid */}
            <div className="mb-6 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
              <div className="rounded-xl border border-white/10 bg-black/20 px-3 py-3 sm:px-4">
                <span className="block text-xs text-slate-400">Reviews</span>
                <span className="text-lg font-semibold text-slate-100 sm:text-xl">
                  {result.sentiment?.total || reviews.length}
                </span>
              </div>
              <div className="rounded-xl border border-white/10 bg-black/20 px-3 py-3 sm:px-4">
                <span className="block text-xs text-slate-400">Score</span>
                <span className="text-lg font-semibold text-amber-200 sm:text-xl">
                  {formatScore(result.score)}
                </span>
              </div>
              <div className="rounded-xl border border-white/10 bg-black/20 px-3 py-3 sm:px-4">
                <span className="block text-xs text-slate-400">Confidence</span>
                <span className="text-lg font-semibold text-cyan-200 sm:text-xl">
                  {formatPercent(result.confidence)}
                </span>
              </div>
              <div className="rounded-xl border border-emerald-400/20 bg-emerald-400/10 px-3 py-3 sm:px-4">
                <span className="block text-xs text-emerald-300">Positive</span>
                <span className="text-lg font-semibold text-emerald-200 sm:text-xl">
                  {formatPercent(result.sentiment?.positive)}
                </span>
              </div>
              <div className="rounded-xl border border-amber-400/20 bg-amber-400/10 px-3 py-3 sm:px-4">
                <span className="block text-xs text-amber-300">Neutral</span>
                <span className="text-lg font-semibold text-amber-200 sm:text-xl">
                  {formatPercent(result.sentiment?.neutral)}
                </span>
              </div>
              <div className="rounded-xl border border-rose-400/20 bg-rose-400/10 px-3 py-3 sm:px-4">
                <span className="block text-xs text-rose-300">Negative</span>
                <span className="text-lg font-semibold text-rose-200 sm:text-xl">
                  {formatPercent(result.sentiment?.negative)}
                </span>
              </div>
            </div>

            {/* Content Grid */}
            <div className="grid gap-6 lg:grid-cols-[1fr_0.95fr]">
              <div className="space-y-6">
                {/* Summary */}
                <article className="rounded-xl border border-white/10 bg-slate-950/35 p-5">
                  <p className="text-xs font-semibold uppercase tracking-widest text-emerald-200/80">
                    Summary
                  </p>
                  <p className="mt-3 text-sm leading-relaxed text-slate-100 sm:text-base">
                    {result.summary || "No summary available."}
                  </p>
                </article>

                {/* Pros & Cons */}
                <div className="grid gap-4 sm:grid-cols-2">
                  <article className="rounded-xl border border-emerald-400/20 bg-emerald-500/10 p-5">
                    <p className="text-xs font-semibold uppercase tracking-widest text-emerald-100">
                      👍 Pros ({result.pros?.length || 0})
                    </p>
                    {renderList(result.pros, "No pros identified.")}
                  </article>

                  <article className="rounded-xl border border-rose-400/20 bg-rose-500/10 p-5">
                    <p className="text-xs font-semibold uppercase tracking-widest text-rose-100">
                      👎 Cons ({result.cons?.length || 0})
                    </p>
                    {renderList(result.cons, "No cons identified.")}
                  </article>
                </div>

                {/* Neutral Points - SAFE VERSION */}
                {neutralPoints.length > 0 && (
                  <article className="rounded-xl border border-amber-400/20 bg-amber-500/10 p-5">
                    <p className="text-xs font-semibold uppercase tracking-widest text-amber-100">
                      ➖ Neutral Points ({neutralPoints.length})
                    </p>
                    {renderList(neutralPoints, "No neutral points.")}
                  </article>
                )}
              </div>

              {/* Sentiment Chart */}
              <div className="mt-6 lg:mt-0">
                <p className="mb-4 text-xs font-semibold uppercase tracking-widest text-amber-100/80">
                  Sentiment Distribution
                </p>
                <div className="overflow-hidden rounded-2xl border border-white/10 bg-slate-950/20">
                  <SentimentChart
                    positive={result.sentiment?.positive || 0}
                    neutral={result.sentiment?.neutral || 0}
                    negative={result.sentiment?.negative || 0}
                  />
                </div>
              </div>
            </div>
          </section>
        )}

        {/* Footer */}
        <footer className="mt-10 text-center text-xs text-slate-500 sm:text-sm">
          Built with React + FastAPI • AI Product Review Analyzer
        </footer>
      </div>

      {/* Toast */}
      {toast && <Toast message={toast.message} type={toast.type} onClose={() => setToast(null)} />}

      {/* Animation Styles */}
      <style>{`
        @keyframes slide-up {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-slide-up { animation: slide-up 0.3s ease-out; }
      `}</style>
    </main>
  );
}
