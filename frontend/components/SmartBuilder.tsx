"use client"

import { useState } from "react"
import TickerSearch from "@/components/TickerSearch"
import { apiClient } from "@/lib/apiClient"
import type { SearchResult, OptimizeGoal, OptimizeResult } from "@/types/api"

// ---------------------------------------------------------------------------
// Suggested tickers — well-known S&P 500 / ETF constituents guaranteed to be
// in the preloaded DB. Mix of sectors + ETFs for natural diversification.
// ---------------------------------------------------------------------------
const SUGGESTIONS: { ticker: string; label: string }[] = [
  { ticker: "AAPL",  label: "AAPL"  },
  { ticker: "MSFT",  label: "MSFT"  },
  { ticker: "GOOGL", label: "GOOGL" },
  { ticker: "AMZN",  label: "AMZN"  },
  { ticker: "NVDA",  label: "NVDA"  },
  { ticker: "META",  label: "META"  },
  { ticker: "BRK-B", label: "BRK-B" },
  { ticker: "JPM",   label: "JPM"   },
  { ticker: "JNJ",   label: "JNJ"   },
  { ticker: "V",     label: "V"     },
  { ticker: "SPY",   label: "SPY"   },
  { ticker: "QQQ",   label: "QQQ"   },
  { ticker: "GLD",   label: "GLD"   },
  { ticker: "XOM",   label: "XOM"   },
  { ticker: "UNH",   label: "UNH"   },
]

const GOAL_OPTIONS: { value: OptimizeGoal; label: string; desc: string }[] = [
  { value: "sharpe",   label: "MAX SHARPE",     desc: "Best return per unit of risk" },
  { value: "min_vol",  label: "MIN VOLATILITY", desc: "Lowest portfolio volatility"  },
  { value: "max_cagr", label: "MAX RETURN",     desc: "Highest expected annual return" },
]

const LOOKBACK_OPTIONS = [
  { years: 3,  label: "3Y" },
  { years: 5,  label: "5Y" },
  { years: 10, label: "10Y" },
]

const MIN_TICKERS = 5

interface Props {
  onLoad?: (weights: Record<string, number>) => void
}

export default function SmartBuilder({ onLoad }: Props) {
  const [candidates,    setCandidates]    = useState<SearchResult[]>([])
  const [goal,          setGoal]          = useState<OptimizeGoal>("sharpe")
  const [lookback,      setLookback]      = useState(5)
  const [isOptimizing,  setIsOptimizing]  = useState(false)
  const [result,        setResult]        = useState<OptimizeResult | null>(null)
  const [error,         setError]         = useState<string | null>(null)

  function addTicker(r: SearchResult) {
    if (candidates.find((c) => c.ticker === r.ticker)) return
    setCandidates((prev) => [...prev, r])
    setResult(null)
  }

  function addSuggestion(ticker: string) {
    if (candidates.find((c) => c.ticker === ticker)) return
    const stub: SearchResult = { ticker, name: "", sector: null, tier: "preloaded" }
    setCandidates((prev) => [...prev, stub])
    setResult(null)
  }

  function removeTicker(ticker: string) {
    setCandidates((prev) => prev.filter((c) => c.ticker !== ticker))
    setResult(null)
  }

  async function handleOptimize() {
    if (candidates.length < MIN_TICKERS || isOptimizing) return
    setIsOptimizing(true)
    setError(null)
    setResult(null)
    try {
      const data = await apiClient.optimize({
        tickers:        candidates.map((c) => c.ticker),
        goal,
        lookback_years: lookback,
      })
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Optimization failed")
    } finally {
      setIsOptimizing(false)
    }
  }

  const canRun = candidates.length >= MIN_TICKERS && !isOptimizing

  return (
    <div className="space-y-5">

      {/* ── Suggested tickers ───────────────────────────────────────────── */}
      <div>
        <p className="text-xs mb-2" style={{ color: "var(--text-muted)" }}>
          Quick add — click to add to your candidate pool
        </p>
        <div className="flex flex-wrap gap-2">
          {SUGGESTIONS.map(({ ticker }) => {
            const added = !!candidates.find((c) => c.ticker === ticker)
            return (
              <button
                key={ticker}
                onClick={() => !added && addSuggestion(ticker)}
                disabled={added}
                className="px-2.5 py-1 rounded text-xs tracking-widest font-mono transition-all"
                style={
                  added
                    ? { background: "rgba(0,229,160,0.08)", border: "1px solid rgba(0,229,160,0.25)", color: "var(--accent)", opacity: 0.45, cursor: "default" }
                    : { background: "transparent", border: "1px solid var(--border-bright)", color: "var(--text-muted)", cursor: "pointer" }
                }
                onMouseEnter={(e) => {
                  if (!added) {
                    (e.currentTarget as HTMLElement).style.borderColor = "rgba(0,229,160,0.35)"
                    ;(e.currentTarget as HTMLElement).style.color = "var(--accent)"
                  }
                }}
                onMouseLeave={(e) => {
                  if (!added) {
                    (e.currentTarget as HTMLElement).style.borderColor = "var(--border-bright)"
                    ;(e.currentTarget as HTMLElement).style.color = "var(--text-muted)"
                  }
                }}
              >
                {ticker}
              </button>
            )
          })}
        </div>
      </div>

      {/* ── Search ──────────────────────────────────────────────────────── */}
      <TickerSearch
        onSelect={addTicker}
        disabledTickers={candidates.map((c) => c.ticker)}
      />

      {/* ── Candidate pool ──────────────────────────────────────────────── */}
      {candidates.length > 0 && (
        <div className="space-y-1.5">
          <p className="text-xs" style={{ color: "var(--text-muted)" }}>
            Candidate pool — {candidates.length} ticker{candidates.length !== 1 ? "s" : ""}
            {candidates.length < MIN_TICKERS && (
              <span style={{ color: "#fbbf24" }}>
                {" "}· need {MIN_TICKERS - candidates.length} more to run
              </span>
            )}
          </p>
          <div className="flex flex-wrap gap-2">
            {candidates.map((c) => (
              <span
                key={c.ticker}
                className="flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-mono tracking-widest"
                style={{ background: "var(--bg-card-2)", border: "1px solid var(--border-bright)", color: "var(--accent)" }}
              >
                {c.ticker}
                <button
                  onClick={() => removeTicker(c.ticker)}
                  className="opacity-40 hover:opacity-100 transition-opacity leading-none"
                  style={{ color: "#f87171" }}
                >
                  ✕
                </button>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* ── Goal + lookback controls ─────────────────────────────────────── */}
      <div className="flex flex-wrap gap-4">

        {/* Goal selector */}
        <div className="space-y-1.5">
          <p className="text-xs" style={{ color: "var(--text-muted)" }}>Optimize for</p>
          <div className="flex rounded overflow-hidden" style={{ border: "1px solid var(--border-bright)" }}>
            {GOAL_OPTIONS.map((opt) => (
              <button
                key={opt.value}
                onClick={() => { setGoal(opt.value); setResult(null) }}
                className="px-3 py-1.5 text-xs tracking-widest transition-colors"
                title={opt.desc}
                style={
                  goal === opt.value
                    ? { background: "rgba(0,229,160,0.12)", color: "var(--accent)" }
                    : { background: "transparent", color: "var(--text-muted)" }
                }
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>

        {/* Lookback selector */}
        <div className="space-y-1.5">
          <p className="text-xs" style={{ color: "var(--text-muted)" }}>Lookback period</p>
          <div className="flex gap-2">
            {LOOKBACK_OPTIONS.map((opt) => (
              <button
                key={opt.years}
                onClick={() => { setLookback(opt.years); setResult(null) }}
                className="px-3 py-1.5 rounded text-xs tracking-widest font-mono transition-colors"
                style={
                  lookback === opt.years
                    ? { background: "rgba(0,229,160,0.12)", color: "var(--accent)", border: "1px solid rgba(0,229,160,0.3)" }
                    : { background: "transparent", color: "var(--text-muted)", border: "1px solid var(--border)" }
                }
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>

      </div>

      {/* ── Run button ──────────────────────────────────────────────────── */}
      <button
        onClick={handleOptimize}
        disabled={!canRun}
        className="w-full py-2.5 rounded text-xs tracking-widest uppercase transition-all"
        style={
          canRun
            ? { background: "rgba(0,229,160,0.12)", border: "1px solid rgba(0,229,160,0.4)", color: "var(--accent)", cursor: "pointer" }
            : { background: "transparent", border: "1px solid var(--border)", color: "var(--text-dim)", cursor: "not-allowed", opacity: 0.5 }
        }
      >
        {isOptimizing ? "OPTIMIZING..." : `RUN OPTIMIZER · ${candidates.length}/${MIN_TICKERS} TICKERS`}
      </button>

      {/* ── Error ───────────────────────────────────────────────────────── */}
      {error && (
        <div className="rounded px-3 py-2 text-xs leading-relaxed"
          style={{ background: "rgba(248,113,113,0.07)", border: "1px solid rgba(248,113,113,0.2)", color: "#f87171" }}>
          {error}
        </div>
      )}

      {/* ── Results ─────────────────────────────────────────────────────── */}
      {result && <OptimizeResults result={result} onLoad={onLoad} />}

    </div>
  )
}

// ---------------------------------------------------------------------------
// Results view
// ---------------------------------------------------------------------------

function OptimizeResults({ result, onLoad }: { result: OptimizeResult; onLoad?: (weights: Record<string, number>) => void }) {
  const sorted = Object.entries(result.weights).sort(([, a], [, b]) => b - a)
  const { metrics } = result

  const goalLabel = {
    sharpe:   "MAX SHARPE",
    min_vol:  "MIN VOLATILITY",
    max_cagr: "MAX RETURN",
  }[result.goal]

  return (
    <div className="space-y-4 pt-2" style={{ borderTop: "1px solid var(--border)" }}>

      {/* Header */}
      <div className="flex items-center justify-between text-xs tracking-widest"
        style={{ color: "var(--text-muted)" }}>
        <span>SUGGESTED ALLOCATION</span>
        <span className="px-2 py-0.5 rounded font-mono"
          style={{ background: "rgba(0,229,160,0.07)", border: "1px solid rgba(0,229,160,0.2)", color: "var(--accent)" }}>
          {goalLabel}
        </span>
      </div>

      {/* Weights table */}
      <div className="space-y-2">
        {sorted.map(([ticker, weight]) => (
          <div key={ticker} className="flex items-center gap-3">
            <span className="font-mono text-xs tracking-widest w-14 shrink-0"
              style={{ color: "var(--accent)" }}>
              {ticker}
            </span>
            {/* Weight bar */}
            <div className="flex-1 h-1.5 rounded-full overflow-hidden"
              style={{ background: "var(--border-bright)" }}>
              <div className="h-full rounded-full transition-all duration-500"
                style={{ width: `${(weight * 100).toFixed(1)}%`, background: "var(--accent)", opacity: 0.7 }} />
            </div>
            <span className="font-mono text-xs tabular-nums w-12 text-right"
              style={{ color: "var(--text-primary)" }}>
              {(weight * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>

      {/* Metrics row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2"
        style={{ borderTop: "1px solid var(--border)" }}>
        <ResultMetric label="SHARPE"      value={metrics.sharpe.toFixed(2)}        color={metrics.sharpe >= 1 ? "var(--accent)" : metrics.sharpe >= 0.5 ? "#fbbf24" : "#f87171"} />
        <ResultMetric label="CAGR"        value={pct(metrics.cagr)}                color={metrics.cagr >= 0 ? "var(--accent)" : "#f87171"} />
        <ResultMetric label="MAX DRAWDOWN" value={pct(metrics.max_drawdown)}       color="#f87171" />
        <ResultMetric label="VOLATILITY"  value={pct(metrics.volatility)}          color="var(--text-muted)" />
      </div>

      {onLoad && (
        <button
          onClick={() => onLoad(result.weights)}
          className="w-full py-2.5 rounded text-xs tracking-widest uppercase transition-all"
          style={{ background: "rgba(0,229,160,0.12)", border: "1px solid rgba(0,229,160,0.4)", color: "var(--accent)", cursor: "pointer" }}
        >
          LOAD INTO PORTFOLIO BUILDER ↑
        </button>
      )}

      <p className="text-xs leading-relaxed" style={{ color: "var(--text-dim)" }}>
        Weights are optimized over the selected lookback window. Loading will replace
        any existing tickers in the Portfolio Builder.
      </p>
    </div>
  )
}

function ResultMetric({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="rounded p-3" style={{ background: "var(--bg-card-2)", border: "1px solid var(--border)" }}>
      <p className="text-xs tracking-widest mb-1.5" style={{ color: "var(--text-muted)" }}>{label}</p>
      <p className="text-xl font-mono" style={{ color }}>{value}</p>
    </div>
  )
}

function pct(v: number): string {
  return (v >= 0 ? "+" : "") + (v * 100).toFixed(1) + "%"
}
