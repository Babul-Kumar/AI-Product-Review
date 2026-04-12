import { useRef, useState, useEffect, useCallback } from "react";
import SentimentChart from "./components/SentimentChart";

// ==============================================================================
// CONFIGURATION
// ==============================================================================

// 🔥 LOG LEVEL CONTROL - Toggle for production
const LOG_LEVEL = {
  DEBUG: true,    // Set to false in production
  INFO: true,
  WARN: true,
  ERROR: true,
};

// Logging utility with levels
const log = {
  debug: (...args) => LOG_LEVEL.DEBUG && console.debug("[DEBUG]", ...args),
  info: (...args) => LOG_LEVEL.INFO && console.info("[INFO]", ...args),
  warn: (...args) => LOG_LEVEL.WARN && console.warn("[WARN]", ...args),
  error: (...args) => LOG_LEVEL.ERROR && console.error("[ERROR]", ...args),
};

// Environment Variables
const API_BASE_URL = import.meta.env.VITE_API_URL || "https://ai-product-review.onrender.com";
const ANALYZE_URL = `${API_BASE_URL}/analyze`;
const PING_URL = `${API_BASE_URL}/ping`;

// ✅ API Key from environment (required for protected endpoints)
const API_KEY = import.meta.env.VITE_API_KEY || "";

// Constants
const MIN_REVIEW_INPUT_LENGTH = 10;
const LOW_QUALITY_MESSAGE = "Please enter meaningful product reviews to get accurate insights.";
const COOLDOWN_MS = 3000;
const MAX_RETRIES = 2;
const RETRY_DELAY_BASE = 1500;
const REQUEST_TIMEOUT_MS = 10000;
const HEALTH_CHECK_INTERVAL = 30000;

// Non-meaningful patterns to filter out
const NON_MEANINGFUL_PATTERNS = [
  /^no summary was returned\.?$/i,
  /^no major recurring strengths were mentioned\.?$/i,
  /^no major recurring complaints were mentioned\.?$/i,
  /^no neutral feedback found\.?$/i,
  /^no valid review content found\.?$/i,
  /^please provide at least one non-empty review\.?$/i,
  /^input does not appear to be product reviews\.?$/i,
];

// Empty result template
const EMPTY_RESULT = {
  summary: "",
  pros: [],
  cons: [],
  neutral_points: [],
  sentiment: {
    positive: 0,
    neutral: 0,
    negative: 0,
    total: 0,
  },
  score: 0,
  confidence: 0,
};

// ==============================================================================
// UTILITY FUNCTIONS
// ==============================================================================

function toSafeNumber(value) {
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue : 0;
}

function toSafeList(value) {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.map((item) => String(item).trim()).filter(Boolean);
}

function normalizeResult(data) {
  const sentiment = data?.sentiment && typeof data.sentiment === "object" 
    ? data.sentiment 
    : EMPTY_RESULT.sentiment;

  return {
    summary: typeof data?.summary === "string" && data.summary.trim()
      ? data.summary.trim()
      : "No summary was returned.",
    pros: toSafeList(data?.pros),
    cons: toSafeList(data?.cons),
    neutral_points: toSafeList(data?.neutral_points),
    sentiment: {
      positive: toSafeNumber(sentiment.positive),
      neutral: toSafeNumber(sentiment.neutral),
      negative: toSafeNumber(sentiment.negative),
      total: toSafeNumber(sentiment.total),
    },
    score: toSafeNumber(data?.score),
    confidence: toSafeNumber(data?.confidence),
    // ✅ Include explained data if available
    explained_pros: Array.isArray(data?.explained_pros) ? data.explained_pros : [],
    explained_cons: Array.isArray(data?.explained_cons) ? data.explained_cons : [],
    feature_scores: Array.isArray(data?.feature_scores) ? data.feature_scores : [],
  };
}

function formatPercent(value) {
  return `${toSafeNumber(value).toFixed(2)}%`;
}

function formatScore(value) {
  const safeValue = toSafeNumber(value);
  return safeValue > 0 ? `${safeValue.toFixed(2)} / 5` : "Not available";
}

function isMeaningfulItem(item) {
  const text = String(item ?? "").trim();
  return text.length > 0 && !NON_MEANINGFUL_PATTERNS.some((pattern) => pattern.test(text));
}

function hasMeaningfulInsights(data) {
  return data.pros.some(isMeaningfulItem) || data.cons.some(isMeaningfulItem);
}

// ==============================================================================
// FETCH UTILITIES
// ==============================================================================

// Fetch with timeout using AbortController
function fetchWithTimeout(url, options, timeout = REQUEST_TIMEOUT_MS) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  return fetch(url, {
    ...options,
    signal: controller.signal,
  }).finally(() => clearTimeout(timeoutId));
}

// Combined retry + timeout with exponential backoff
async function fetchWithRetry(url, options, retries = MAX_RETRIES, retryCount = 0, requestId = "unknown") {
  try {
    const response = await fetchWithTimeout(url, options);

    if (response.ok) {
      log.debug(`✅ [${requestId}] Success on attempt ${retryCount + 1}`);
      return { success: true, response, retriesUsed: retryCount };
    }

    // Retry on server errors and rate limits
    const shouldRetry = response.status >= 500 || 
                        response.status === 429 || 
                        response.status === 503 || 
                        response.status === 504;

    if (shouldRetry && retryCount < retries) {
      const delay = RETRY_DELAY_BASE * Math.pow(2, retryCount);
      log.debug(`🔄 [${requestId}] Retry ${retryCount + 1}/${retries} after ${delay}ms (status: ${response.status})`);
      await new Promise((resolve) => setTimeout(resolve, delay));
      return fetchWithRetry(url, options, retries, retryCount + 1, requestId);
    }

    return { success: false, response, retriesUsed: retryCount };
  } catch (error) {
    const isAbort = error.name === "AbortError" || error.message?.includes("aborted");
    const isTimeout = isAbort || error.message?.includes("timeout");

    log.debug(`❌ [${requestId}] Attempt ${retryCount + 1} failed: ${error.message}${isTimeout ? " (TIMEOUT)" : ""}`);

    if (retryCount < retries && !isAbort) {
      const delay = RETRY_DELAY_BASE * Math.pow(2, retryCount);
      log.debug(`🔄 [${requestId}] Retrying after ${delay}ms...`);
      await new Promise((resolve) => setTimeout(resolve, delay));
      return fetchWithRetry(url, options, retries, retryCount + 1, requestId);
    }

    // Create timeout error with flag
    const finalError = new Error(isTimeout 
      ? `Request timed out after ${(retryCount + 1) * REQUEST_TIMEOUT_MS}ms` 
      : error.message);
    finalError.isTimeout = isTimeout;
    finalError.isAbort = isAbort;
    throw finalError;
  }
}

// ==============================================================================
// MAIN COMPONENT
// ==============================================================================

function App() {
  // State
  const [inputText, setInputText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [lowQualityMessage, setLowQualityMessage] = useState("");
  const [serverStatus, setServerStatus] = useState("checking");
  const [lastClickTime, setLastClickTime] = useState(0);
  const [copySuccess, setCopySuccess] = useState(false);
  const [retryInfo, setRetryInfo] = useState(null);
  const [requestId, setRequestId] = useState(null);

  // Refs
  const abortControllerRef = useRef(null);
  const inFlightRef = useRef(false);
  const lastRequestKeyRef = useRef("");
  const lastResultRef = useRef(null);
  const isMountedRef = useRef(true);

  // ==============================================================================
  // EFFECTS
  // ==============================================================================

  // Cleanup on unmount
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        log.debug("[Cleanup] Aborted pending request on unmount");
      }
    };
  }, []);

  // Backend health check
  useEffect(() => {
    let intervalId;

    const checkServerHealth = async () => {
      try {
        const response = await fetch(PING_URL, { method: "GET" });
        if (isMountedRef.current) {
          setServerStatus(response.ok ? "online" : "offline");
          log.debug(`[Health] Server ${response.ok ? "online" : "offline"}`);
        }
      } catch (err) {
        if (isMountedRef.current) {
          setServerStatus("offline");
          log.warn("[Health] Server unreachable");
        }
      }
    };

    checkServerHealth();
    intervalId = setInterval(checkServerHealth, HEALTH_CHECK_INTERVAL);

    return () => clearInterval(intervalId);
  }, []);

  // ==============================================================================
  // HELPERS
  // ==============================================================================

  const getCleanedReviews = useCallback((text) => {
    const rawReviews = text
      .split(/\n|\.\s+/)
      .map((line) => line.trim())
      .filter((line) => line.length > 0);

    return [...new Set(rawReviews)];
  }, []);

  const reviews = getCleanedReviews(inputText);
  const safeResult = result ?? EMPTY_RESULT;
  const trimmedInput = inputText.trim();
  const isInputTooShort = trimmedInput.length > 0 && trimmedInput.length < MIN_REVIEW_INPUT_LENGTH;
  const hasMeaningfulResult = result ? hasMeaningfulInsights(result) : false;
  const isInCooldown = Date.now() - lastClickTime < COOLDOWN_MS;

  const isAnalyzeDisabled = 
    loading || 
    reviews.length === 0 || 
    trimmedInput.length < MIN_REVIEW_INPUT_LENGTH || 
    isInCooldown || 
    serverStatus === "offline";

  // ==============================================================================
  // ACTIONS
  // ==============================================================================

  const copyToClipboard = () => {
    const textToCopy = JSON.stringify({
      summary: safeResult.summary,
      pros: safeResult.pros,
      cons: safeResult.cons,
      neutral_points: safeResult.neutral_points,
      sentiment: safeResult.sentiment,
      score: safeResult.score,
      confidence: safeResult.confidence,
      feature_scores: safeResult.feature_scores,
    }, null, 2);

    navigator.clipboard.writeText(textToCopy).then(() => {
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    }).catch((err) => {
      log.error("[Copy] Failed to copy to clipboard:", err);
    });
  };

  const analyzeReviews = async () => {
    // Prevent duplicate requests
    if (inFlightRef.current) {
      log.warn("[Request] Blocked - request already in flight");
      return;
    }

    // Abort any pending request
    if (abortControllerRef.current) {
      log.debug(`🔴 [${requestId}] Aborting previous request`);
      abortControllerRef.current.abort();
    }

    // Generate unique request ID
    const newRequestId = `REQ-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
    setRequestId(newRequestId);

    const cleanedReviews = getCleanedReviews(inputText);

    // Validations
    if (cleanedReviews.length === 0) {
      setError("Please enter at least one review before analyzing.");
      setLowQualityMessage("");
      setResult(null);
      setRetryInfo(null);
      return;
    }

    if (trimmedInput.length < MIN_REVIEW_INPUT_LENGTH) {
      setError("");
      setLowQualityMessage(LOW_QUALITY_MESSAGE);
      setResult(null);
      setRetryInfo(null);
      return;
    }

    // Client-side rate limiting
    const now = Date.now();
    if (now - lastClickTime < COOLDOWN_MS) {
      setError("Too many requests. Please wait a few seconds.");
      setLowQualityMessage("");
      setResult(null);
      setRetryInfo(null);
      return;
    }
    setLastClickTime(now);

    // Check for cached result
    const requestKey = cleanedReviews.join("|");
    if (requestKey === lastRequestKeyRef.current && lastResultRef.current) {
      log.debug(`📦 [${newRequestId}] Returning cached result`);
      setError("");
      if (hasMeaningfulInsights(lastResultRef.current)) {
        setLowQualityMessage("");
        setResult(lastResultRef.current);
      } else {
        setResult(null);
        setLowQualityMessage(LOW_QUALITY_MESSAGE);
      }
      setRetryInfo(null);
      return;
    }

    // Create new abort controller
    const controller = new AbortController();
    abortControllerRef.current = controller;

    // Update state
    setLoading(true);
    setError("");
    setLowQualityMessage("");
    setResult(null);
    setRetryInfo({ retries: 0, maxRetries: MAX_RETRIES, status: "connecting", requestId: newRequestId });
    inFlightRef.current = true;

    log.info(`🚀 [${newRequestId}] Starting with ${cleanedReviews.length} reviews`);

    // Build headers
    const headers = {
      "Content-Type": "application/json",
    };
    
    // ✅ Add API key header if configured
    if (API_KEY) {
      headers["X-API-Key"] = API_KEY;
    }

    const fetchOptions = {
      method: "POST",
      headers,
      body: JSON.stringify({ reviews: cleanedReviews }),
      signal: controller.signal,
    };

    try {
      log.debug(`📡 [${newRequestId}] Sending to ${ANALYZE_URL}`);
      
      const { success, response, retriesUsed } = await fetchWithRetry(
        ANALYZE_URL,
        fetchOptions,
        MAX_RETRIES,
        0,
        newRequestId
      );

      // Update retry info
      setRetryInfo((prev) => prev ? {
        ...prev,
        retries: retriesUsed,
        status: retriesUsed > 0 ? "retried" : "success",
      } : null);

      // Check for abort
      if (controller.signal.aborted) {
        log.debug(`⚠️ [${newRequestId}] Request aborted`);
        return;
      }

      // Parse response
      const responseText = await response.text();
      let data = null;

      if (responseText) {
        try {
          data = JSON.parse(responseText);
        } catch {
          log.error(`❌ [${newRequestId}] Invalid JSON`);
          throw new Error("The server returned invalid JSON.");
        }
      }

      // Handle HTTP errors
      if (!response.ok) {
        const message = typeof data?.detail === "string" 
          ? data.detail 
          : `Server error (${response.status})`;
        log.error(`❌ [${newRequestId}] ${message}`);
        throw new Error(message);
      }

      // Validate response
      if (!data || typeof data !== "object") {
        log.error(`❌ [${newRequestId}] Empty response`);
        throw new Error("The server returned an empty response.");
      }

      // Normalize and cache result
      const normalizedData = normalizeResult(data);
      lastRequestKeyRef.current = requestKey;
      lastResultRef.current = normalizedData;

      log.info(`✅ [${newRequestId}] Done - ${normalizedData.pros.length} pros, ${normalizedData.cons.length} cons`);

      if (hasMeaningfulInsights(normalizedData)) {
        setLowQualityMessage("");
        setResult(normalizedData);
      } else {
        setResult(null);
        setLowQualityMessage(LOW_QUALITY_MESSAGE);
      }
      setRetryInfo(null);

    } catch (requestError) {
      // Handle abort gracefully
      if (controller.signal.aborted || requestError.isAbort) {
        log.debug(`⚠️ [${newRequestId}] Request aborted by user`);
        return;
      }

      log.error(`❌ [${newRequestId}] Failed: ${requestError.message}`);

      setLowQualityMessage("");

      // Smart error messages
      let errorMessage;
      const errorText = requestError.message || String(requestError);

      if (requestError.isTimeout || errorText.includes("timed out")) {
        errorMessage = "⏱️ Request timed out. The server is taking too long. Please try again.";
      } else if (
        errorText.includes("Failed to fetch") ||
        errorText.includes("NetworkError") ||
        errorText.includes("Network request failed") ||
        errorText.includes("fetch failed")
      ) {
        errorMessage = "🔌 Network error. The server might be waking up. Please try again in 10-15 seconds.";
      } else if (errorText.includes("429")) {
        errorMessage = "⚠️ Too many requests. API rate limit reached. Please wait a moment.";
      } else if (errorText.includes("401") || errorText.includes("Invalid API key")) {
        errorMessage = "🔑 Invalid or missing API key. Please check your configuration.";
      } else if (errorText.includes("500") || errorText.includes("502") || errorText.includes("503") || errorText.includes("504")) {
        errorMessage = "🛠️ Server error. Please try again in a few moments.";
      } else {
        errorMessage = errorText || "Something went wrong while connecting to the API.";
      }

      setError(errorMessage);
      setRetryInfo(null);
    } finally {
      log.info(`🏁 [${newRequestId}] Complete`);
      
      // Only update state if component is still mounted
      if (isMountedRef.current) {
        setLoading(false);
        inFlightRef.current = false;
        abortControllerRef.current = null;
      }
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    await analyzeReviews();
  };

  // ==============================================================================
  // RENDER HELPERS
  // ==============================================================================

  const renderList = (items, emptyMessage) => {
    const values = items.length > 0 ? items : [emptyMessage];
    return (
      <ul className="mt-4 space-y-3 text-sm leading-6 text-slate-100">
        {values.map((item, index) => (
          <li 
            key={`item-${index}-${typeof item === 'string' ? item.substring(0, 20) : index}`}
            className="rounded-2xl bg-black/10 px-4 py-3"
          >
            {item}
          </li>
        ))}
      </ul>
    );
  };

  const renderSkeleton = () => (
    <section className="rounded-[2rem] border border-white/10 bg-[rgba(7,23,19,0.72)] p-6 shadow-[0_20px_90px_rgba(4,12,10,0.32)] backdrop-blur-xl sm:p-8">
      <div className="mb-8 animate-pulse">
        <div className="h-8 w-48 rounded-lg bg-white/5" />
        <div className="mt-2 h-4 w-64 rounded bg-white/5" />
      </div>

      <div className="mb-6 grid gap-4 sm:grid-cols-3 lg:grid-cols-6">
        {[...Array(6)].map((_, i) => (
          <div key={i} className="h-20 rounded-2xl bg-white/5" />
        ))}
      </div>

      <div className="grid gap-6 lg:grid-cols-[1fr_0.95fr]">
        <div className="space-y-6">
          <div className="h-32 rounded-[1.5rem] bg-white/5" />
          <div className="grid gap-6 md:grid-cols-2">
            <div className="h-40 rounded-[1.5rem] bg-white/5" />
            <div className="h-40 rounded-[1.5rem] bg-white/5" />
          </div>
          <div className="h-32 rounded-[1.5rem] bg-white/5" />
        </div>
        <div className="h-64 rounded-2xl bg-white/5" />
      </div>
    </section>
  );

  // ==============================================================================
  // RENDER
  // ==============================================================================

  return (
    <main className="relative min-h-screen overflow-hidden px-4 py-10 text-slate-100 sm:px-6 lg:px-8">
      {/* Background effects */}
      <div aria-hidden="true" className="pointer-events-none absolute inset-0">
        <div className="absolute left-[12%] top-[-4rem] h-72 w-72 rounded-full bg-emerald-300/10 blur-3xl" />
        <div className="absolute right-[-5rem] top-28 h-80 w-80 rounded-full bg-amber-200/8 blur-3xl" />
        <div className="absolute bottom-12 left-1/2 h-64 w-64 -translate-x-1/2 rounded-full bg-teal-300/8 blur-3xl" />
      </div>

      <div className="relative mx-auto flex max-w-6xl flex-col gap-8">
        {/* Header Section */}
        <section className="overflow-hidden rounded-[2rem] border border-white/10 bg-[rgba(7,23,19,0.78)] shadow-[0_25px_120px_rgba(4,12,10,0.45)] backdrop-blur-xl">
          <div className="grid gap-6 lg:grid-cols-[minmax(0,1.1fr)_320px] lg:items-center">
            <div className="p-6 sm:p-8 lg:p-10">
              {/* Badges */}
              <div className="mb-6 flex flex-wrap items-center gap-3">
                <div className="inline-flex rounded-full border border-emerald-400/30 bg-emerald-400/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.22em] text-emerald-200">
                  React + FastAPI
                </div>

                {/* Server Status Indicator */}
                <div className={`flex items-center gap-2 rounded-full px-3 py-1 text-xs font-medium ${
                  serverStatus === "online"
                    ? "border border-emerald-400/30 bg-emerald-400/10 text-emerald-300"
                    : serverStatus === "offline"
                    ? "border border-rose-400/30 bg-rose-400/10 text-rose-300"
                    : "border border-slate-400/30 bg-slate-400/10 text-slate-400"
                }`}>
                  <span className={`h-2 w-2 rounded-full ${
                    serverStatus === "online" ? "bg-emerald-400 animate-pulse" :
                    serverStatus === "offline" ? "bg-rose-400" : "animate-pulse bg-slate-400"
                  }`} />
                  {serverStatus === "online" ? "Server Online" :
                   serverStatus === "offline" ? "Server Offline" : "Checking..."}
                </div>
              </div>

              {/* Title & Description */}
              <h1 className="text-4xl font-bold tracking-tight text-white sm:text-5xl">
                AI Product Review Analyzer
              </h1>
              <p className="mt-3 max-w-2xl text-sm text-slate-300/80 sm:text-base">
                Analyze product reviews to extract sentiment, pros, cons, and key insights instantly.
              </p>

              {/* Form */}
              <form className="mt-8 space-y-5" onSubmit={handleSubmit}>
                <label className="block">
                  <div className="mb-2 flex flex-wrap items-center justify-between gap-3 text-sm font-medium text-slate-200">
                    <span>Enter Reviews</span>
                    <div className="flex items-center gap-3">
                      <span className="rounded-full border border-white/10 px-3 py-1 text-xs text-slate-300">
                        {reviews.length} review{reviews.length === 1 ? "" : "s"}
                      </span>
                      {reviews.length > 0 && (
                        <span className="rounded-full border border-emerald-400/30 bg-emerald-400/10 px-3 py-1 text-xs text-emerald-300">
                          {reviews.length} unique
                        </span>
                      )}
                    </div>
                  </div>

                  <textarea
                    value={inputText}
                    onChange={(event) => {
                      setInputText(event.target.value);
                      if (error) setError("");
                      if (lowQualityMessage) setLowQualityMessage("");
                    }}
                    placeholder={`Battery life is excellent and setup was simple.\nThe camera is average for the price.\nPerformance feels slow sometimes.\n\nYou can also use periods to separate reviews!`}
                    className="min-h-64 w-full rounded-3xl border border-white/10 bg-slate-950/55 px-5 py-4 text-base text-slate-100 outline-none transition focus:border-emerald-300/60 focus:ring-2 focus:ring-emerald-300/20"
                    disabled={loading}
                  />
                  <p className="mt-3 text-xs text-slate-400">
                    Separate reviews with newlines or periods. Duplicates are automatically removed.
                  </p>
                </label>

                {/* Submit Button & Status */}
                <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                  <div className="flex flex-col gap-1">
                    <div className="text-xs text-slate-500">
                      API: {API_BASE_URL.replace(/^https?:\/\//, "")}
                    </div>
                    
                    {isInCooldown && !error && (
                      <div className="text-xs text-amber-300 animate-pulse">
                        ⏳ Cooldown active... please wait
                      </div>
                    )}
                    
                    {retryInfo?.status === "retried" && !error && (
                      <div className="text-xs text-emerald-400 animate-pulse">
                        ✅ Recovered after {retryInfo.retries} auto-retry{retryInfo.retries !== 1 ? "s" : ""}
                      </div>
                    )}
                    
                    {requestId && loading && (
                      <div className="text-xs text-cyan-400 font-mono">
                        ID: {requestId}
                      </div>
                    )}
                  </div>

                  <button
                    type="submit"
                    disabled={isAnalyzeDisabled}
                    className="inline-flex items-center justify-center rounded-full bg-gradient-to-r from-emerald-400 via-teal-300 to-amber-300 px-6 py-3 text-sm font-semibold text-slate-950 shadow-[0_14px_40px_rgba(52,211,153,0.28)] transition duration-300 hover:-translate-y-0.5 hover:scale-105 disabled:cursor-not-allowed disabled:opacity-70"
                  >
                    {loading ? (
                      <span className="flex items-center gap-2">
                        <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                        </svg>
                        {retryInfo?.status === "retried" ? "Recovered ✓" :
                         retryInfo ? `Retrying (${retryInfo.retries}/${retryInfo.maxRetries})...` : "Analyzing..."}
                      </span>
                    ) : (
                      "Analyze Reviews"
                    )}
                  </button>
                </div>
              </form>

              {/* Messages */}
              {isInputTooShort && !error && !lowQualityMessage && (
                <div className="mt-5 rounded-2xl border border-amber-300/20 bg-amber-300/10 px-4 py-3 text-sm text-amber-100">
                  {LOW_QUALITY_MESSAGE}
                </div>
              )}

              {error && (
                <div className="mt-5 rounded-2xl border border-rose-300/20 bg-rose-400/10 px-4 py-3 text-sm text-rose-100">
                  {error}
                </div>
              )}

              {lowQualityMessage && (
                <div className="mt-5 rounded-2xl border border-amber-300/20 bg-amber-300/10 px-4 py-3 text-sm text-amber-100">
                  {lowQualityMessage}
                </div>
              )}
            </div>

            {/* Example Section */}
            <div className="px-6 pb-6 sm:px-8 lg:px-0 lg:pr-10">
              <div className="rounded-[1.75rem] border border-white/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.08),rgba(255,255,255,0.03))] p-5 shadow-[0_18px_50px_rgba(4,12,10,0.22)]">
                <p className="text-xs font-semibold uppercase tracking-[0.22em] text-amber-100/75">
                  Try Example
                </p>
                <div 
                  className="mt-4 rounded-2xl border border-emerald-300/12 bg-black/15 px-4 py-4 text-sm leading-7 text-slate-100 cursor-pointer hover:bg-black/25 transition"
                  onClick={() => setInputText("The camera is great but battery drains fast. Performance is excellent for the price. Display is bright but could be sharper. Sound quality is average.")}
                >
                  "The camera is great but battery drains fast. Performance is excellent for the price."
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Loading Skeleton */}
        {loading && renderSkeleton()}

        {/* Results Section */}
        {result && hasMeaningfulResult && !isInputTooShort && !lowQualityMessage && !loading && (
          <section className="rounded-[2rem] border border-white/10 bg-[rgba(7,23,19,0.72)] p-6 shadow-[0_20px_90px_rgba(4,12,10,0.32)] backdrop-blur-xl sm:p-8">
            {/* Results Header */}
            <div className="mb-8 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h2 className="text-2xl font-semibold text-white">Analysis Result</h2>
                <p className="mt-1 text-sm text-slate-300">
                  Response data from the deployed FastAPI backend.
                </p>
              </div>

              <button
                onClick={copyToClipboard}
                className={`inline-flex items-center gap-2 rounded-full border px-4 py-2 text-sm font-medium transition ${
                  copySuccess
                    ? "border-emerald-400/30 bg-emerald-400/10 text-emerald-300"
                    : "border-white/10 bg-white/5 text-slate-300 hover:border-white/20 hover:bg-white/10"
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
                    Copy Results
                  </>
                )}
              </button>
            </div>

            {/* Stats Grid */}
            <div className="mb-6 flex flex-wrap gap-3">
              <div className="rounded-2xl border border-white/10 bg-black/10 px-4 py-3 text-sm">
                <span className="block text-slate-400">Total Reviews</span>
                <span className="font-semibold text-slate-100">
                  {safeResult.sentiment.total || reviews.length}
                </span>
              </div>
              <div className="rounded-2xl border border-white/10 bg-black/10 px-4 py-3 text-sm">
                <span className="block text-slate-400">Score</span>
                <span className="font-semibold text-amber-100">{formatScore(safeResult.score)}</span>
              </div>
              <div className="rounded-2xl border border-white/10 bg-black/10 px-4 py-3 text-sm">
                <span className="block text-slate-400">Confidence</span>
                <span className="font-semibold text-cyan-200">{formatPercent(safeResult.confidence)}</span>
              </div>
              <div className="rounded-2xl border border-white/10 bg-black/10 px-4 py-3 text-sm">
                <span className="block text-slate-400">Positive</span>
                <span className="font-semibold text-emerald-300">{formatPercent(safeResult.sentiment.positive)}</span>
              </div>
              <div className="rounded-2xl border border-white/10 bg-black/10 px-4 py-3 text-sm">
                <span className="block text-slate-400">Neutral</span>
                <span className="font-semibold text-amber-200">{formatPercent(safeResult.sentiment.neutral)}</span>
              </div>
              <div className="rounded-2xl border border-white/10 bg-black/10 px-4 py-3 text-sm">
                <span className="block text-slate-400">Negative</span>
                <span className="font-semibold text-rose-300">{formatPercent(safeResult.sentiment.negative)}</span>
              </div>
            </div>

            {/* Content Grid */}
            <div className="grid gap-6 lg:grid-cols-[1fr_0.95fr]">
              <div className="space-y-6">
                {/* Summary */}
                <article className="rounded-[1.5rem] border border-white/10 bg-slate-950/35 p-5">
                  <p className="text-sm uppercase tracking-[0.18em] text-emerald-200/80">Summary</p>
                  <p className="mt-3 text-base leading-7 text-slate-100">{safeResult.summary}</p>
                </article>

                {/* Pros & Cons Grid */}
                <div className="grid gap-6 md:grid-cols-2">
                  <article className="rounded-[1.5rem] border border-emerald-300/15 bg-emerald-300/8 p-5">
                    <p className="text-sm uppercase tracking-[0.18em] text-emerald-100">Pros</p>
                    {renderList(safeResult.pros, "No pros returned.")}
                  </article>

                  <article className="rounded-[1.5rem] border border-rose-300/15 bg-rose-300/8 p-5">
                    <p className="text-sm uppercase tracking-[0.18em] text-rose-100">Cons</p>
                    {renderList(safeResult.cons, "No cons returned.")}
                  </article>
                </div>

                {/* Neutral Points */}
                <article className="rounded-[1.5rem] border border-amber-300/15 bg-amber-300/8 p-5">
                  <p className="text-sm uppercase tracking-[0.18em] text-amber-100">Neutral Points</p>
                  {renderList(safeResult.neutral_points, "No neutral feedback found.")}
                </article>
              </div>

              {/* Chart */}
              <div>
                <p className="mb-4 text-sm uppercase tracking-[0.18em] text-amber-100/80">Sentiment Chart</p>
                <SentimentChart
                  positive={safeResult.sentiment.positive}
                  neutral={safeResult.sentiment.neutral}
                  negative={safeResult.sentiment.negative}
                  score={safeResult.score}
                  confidence={safeResult.confidence}
                />
              </div>
            </div>
          </section>
        )}

        {/* Footer */}
        <footer className="mt-10 text-center text-sm text-slate-400/80">
          Developed by Babul Kumar • Ultra-resilient with auto-retry + timeout
        </footer>
      </div>
    </main>
  );
}

export default App;
