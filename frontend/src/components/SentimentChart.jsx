import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import Chart from "chart.js/auto";

// ==============================================================================
// CENTER TEXT PLUGIN (Registered once globally)
// ==============================================================================

const centerTextPlugin = {
  id: "sentimentCenterText",
  beforeDraw(chart, args, pluginOptions) {
    const { ctx, chartArea } = chart;
    const text = pluginOptions?.text || "Sentiment";

    if (!chartArea) {
      return;
    }

    const centerX = (chartArea.left + chartArea.right) / 2;
    const centerY = (chartArea.top + chartArea.bottom) / 2;
    const chartWidth = chartArea.right - chartArea.left;
    const maxTextWidth = chartWidth * 0.54;
    let fontSize = Math.max(12, Math.min(chartWidth / 10, 22));

    ctx.save();
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = "#f8fafc";
    ctx.font = `600 ${fontSize}px Space Grotesk`;

    const measuredTextWidth = ctx.measureText(text).width;
    if (measuredTextWidth > maxTextWidth) {
      fontSize = Math.max(11, Math.floor(fontSize * (maxTextWidth / measuredTextWidth)));
      ctx.font = `600 ${fontSize}px Space Grotesk`;
    }

    ctx.fillText(text, centerX, centerY);
    ctx.restore();
  },
};

// ✅ FIX: Register plugin once with proper guard
const registerPlugin = () => {
  if (!globalThis.__sentimentCenterTextPluginRegistered) {
    Chart.register(centerTextPlugin);
    globalThis.__sentimentCenterTextPluginRegistered = true;
  }
};
registerPlugin();

// ==============================================================================
// CONSTANTS
// ==============================================================================

const FLOAT_TOLERANCE = 0.001; // ✅ FIX: For floating point comparison
const ANIMATION_DURATION = 800;
const TRANSITION_DURATION = 320;

// ==============================================================================
// UTILITY FUNCTIONS
// ==============================================================================

// ✅ FIX: Safe number comparison for floating point
const areNumbersEqual = (a, b, tolerance = FLOAT_TOLERANCE) => {
  const numA = Number(a) || 0;
  const numB = Number(b) || 0;
  return Math.abs(numA - numB) < tolerance;
};

// ✅ FIX: Safe array comparison
const hasDataChanged = (oldData, newData) => {
  if (!Array.isArray(oldData) || !Array.isArray(newData)) return true;
  if (oldData.length !== newData.length) return true;
  return oldData.some((value, index) => !areNumbersEqual(value, newData[index]));
};

// ==============================================================================
// COMPONENT
// ==============================================================================

function SentimentChart({ positive, neutral, negative, score, confidence }) {
  const canvasRef = useRef(null);
  const chartRef = useRef(null);
  const lastLoggedRef = useRef(null);
  const previousTotalRef = useRef(0);
  const isMountedRef = useRef(true);

  // ✅ FIX: Proper cleanup on unmount
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      if (chartRef.current) {
        chartRef.current.destroy();
        chartRef.current = null;
      }
    };
  }, []);

  // ✅ FIX: Memoize with correct dependency order
  const chartValues = useMemo(() => {
    return [positive, neutral, negative].map((value) =>
      typeof value === "number" && Number.isFinite(value) ? Math.max(0, value) : 0
    );
  }, [positive, neutral, negative]);

  const [safePositive, safeNeutral, safeNegative] = chartValues;

  const total = useMemo(
    () => chartValues.reduce((sum, value) => sum + value, 0),
    [chartValues]
  );

  const [isChartVisible, setIsChartVisible] = useState(false);

  // ✅ FIX: Stable center text with proper emoji handling
  const centerText = useMemo(() => {
    if (total === 0) return "No Data";
    if (typeof score === "number" && Number.isFinite(score)) {
      return `${score.toFixed(1)}`;
    }
    return "Sentiment";
  }, [score, total]);

  // ✅ FIX: Proper ARIA label
  const ariaLabel = useMemo(
    () =>
      `Sentiment analysis: ${safePositive.toFixed(1)}% positive, ${safeNeutral.toFixed(1)}% neutral, ${safeNegative.toFixed(1)}% negative`,
    [safePositive, safeNeutral, safeNegative]
  );

  // Handle visibility transitions
  useEffect(() => {
    if (total === 0) {
      setIsChartVisible(false);
      previousTotalRef.current = 0;
      return;
    }

    // Animate in when data first appears
    if (previousTotalRef.current === 0) {
      setIsChartVisible(false);
      const frameId = requestAnimationFrame(() => {
        if (isMountedRef.current) {
          setIsChartVisible(true);
        }
      });
      previousTotalRef.current = total;
      return () => cancelAnimationFrame(frameId);
    }

    setIsChartVisible(true);
    previousTotalRef.current = total;
  }, [total]);

  // Chart creation and updates
  useEffect(() => {
    // Destroy chart when no data
    if (total === 0) {
      if (chartRef.current) {
        chartRef.current.destroy();
        chartRef.current = null;
      }
      return;
    }

    // Create new chart if needed
    if (!chartRef.current && canvasRef.current) {
      chartRef.current = new Chart(canvasRef.current, {
        type: "pie",
        data: {
          labels: ["Positive", "Neutral", "Negative"],
          datasets: [
            {
              label: "Sentiment",
              data: [safePositive, safeNeutral, safeNegative],
              backgroundColor: [
                "rgba(34, 197, 94, 0.85)",
                "rgba(234, 179, 8, 0.85)",
                "rgba(244, 63, 94, 0.85)",
              ],
              hoverBackgroundColor: [
                "rgba(74, 222, 128, 0.98)",
                "rgba(250, 204, 21, 0.98)",
                "rgba(251, 113, 133, 0.98)",
              ],
              borderColor: ["#0f172a", "#0f172a", "#0f172a"],
              borderWidth: 2,
              spacing: 2,
              hoverBorderColor: "rgba(248, 250, 252, 0.9)",
              hoverBorderWidth: 3,
              hoverOffset: 14,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          cutout: "58%",
          onHover(event, elements, chart) {
            const canvas = chart?.canvas || event?.native?.target;
            if (canvas) {
              canvas.style.cursor = elements.length > 0 ? "pointer" : "default";
            }
          },
          animation: {
            animateRotate: true,
            duration: ANIMATION_DURATION,
          },
          plugins: {
            centerText: {
              text: centerText,
            },
            legend: {
              position: "bottom",
              labels: {
                color: "#e2e8f0",
                padding: 18,
                font: {
                  family: "Space Grotesk",
                  size: 13,
                },
                generateLabels(chart) {
                  const labels = chart.data.labels || [];
                  const dataset = chart.data.datasets[0];

                  return labels.map((label, index) => ({
                    text: `${label}: ${(dataset.data[index] || 0).toFixed(1)}%`,
                    fillStyle: dataset.backgroundColor[index],
                    strokeStyle: dataset.borderColor[index],
                    lineWidth: dataset.borderWidth,
                    hidden: !chart.getDataVisibility(index),
                    index,
                  }));
                },
              },
            },
            tooltip: {
              callbacks: {
                label(context) {
                  const value = context.parsed ?? 0;
                  return ` ${context.label}: ${value.toFixed(1)}%`;
                },
              },
            },
          },
        },
      });
      return;
    }

    // Update existing chart
    if (chartRef.current) {
      const nextData = [safePositive, safeNeutral, safeNegative];
      const dataset = chartRef.current.data.datasets[0];

      // ✅ FIX: Use safe comparison
      const dataNeedsUpdate = hasDataChanged(dataset.data, nextData);
      const centerTextChanged = chartRef.current.options.plugins?.centerText?.text !== centerText;

      if (dataNeedsUpdate) {
        dataset.data = nextData;
      }

      if (centerTextChanged && chartRef.current.options.plugins?.centerText) {
        chartRef.current.options.plugins.centerText.text = centerText;
      }

      if (dataNeedsUpdate || centerTextChanged) {
        chartRef.current.update();
      }
    }
  }, [centerText, safeNegative, safeNeutral, safePositive, total]);

  // ✅ FIX: Debounced resize handler
  useEffect(() => {
    let resizeTimeout;

    const handleResize = () => {
      clearTimeout(resizeTimeout);
      resizeTimeout = setTimeout(() => {
        if (chartRef.current && isMountedRef.current) {
          chartRef.current.resize();
        }
      }, 100);
    };

    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
      clearTimeout(resizeTimeout);
    };
  }, []);

  // Debug logging (dev only)
  useEffect(() => {
    if (
      import.meta.env.DEV &&
      typeof confidence === "number" &&
      Number.isFinite(confidence) &&
      lastLoggedRef.current !== confidence
    ) {
      console.debug("[Chart] Confidence:", confidence.toFixed(2));
      lastLoggedRef.current = confidence;
    }
  }, [confidence]);

  // Empty state
  if (total === 0) {
    return (
      <div className="flex h-72 items-center justify-center rounded-3xl border border-white/10 bg-slate-900/40 px-6 text-center text-sm text-slate-300">
        No sentiment data available yet.
      </div>
    );
  }

  // Chart container
  return (
    <div
      className="h-72 rounded-3xl border border-white/10 bg-slate-900/40 p-4 shadow-[0_18px_60px_rgba(15,23,42,0.25)]"
      style={{
        opacity: isChartVisible ? 1 : 0,
        transition: `opacity ${TRANSITION_DURATION}ms ease`,
      }}
    >
      <canvas
        ref={canvasRef}
        aria-label={ariaLabel}
        role="img"
        tabIndex={0}
        className="rounded-2xl focus:outline-none focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-emerald-400"
      />
    </div>
  );
}

export default SentimentChart;
