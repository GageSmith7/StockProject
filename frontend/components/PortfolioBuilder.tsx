"use client"

import { useState, useEffect } from "react"
import { apiClient } from "@/lib/apiClient"
import type { SearchResult, PortfolioResponse } from "@/types/api"

// Two input modes:
//   "percent" — user types 60 / 40, must sum to 100
//   "dollar"  — user types $5468 / $3992, weights are auto-calculated
//               e.g. $5468 + $3992 = $9460 total
//               AAPL weight = 5468 / 9460 = 57.8%, MSFT = 42.2%
// The API always receives weights (0.0–1.0). The mode is purely a UI concern.

type InputMode = "percent" | "dollar"

interface Props {
  holdings:      SearchResult[]
  startDate:     string
  onRemove:      (ticker: string) => void
  onResult:      (data: PortfolioResponse, portfolioValue: number, submittedHoldings: { ticker: string; weight: number }[]) => void
  // presetWeights: percent integers (e.g. 25 = 25%) loaded from the Smart Builder optimizer.
  // When set, switches to percent mode and pre-fills the weight inputs.
  presetWeights?: Record<string, number>
}

export default function PortfolioBuilder({
  holdings,
  startDate,
  onRemove,
  onResult,
  presetWeights,
}: Props) {
  const [mode,       setMode]       = useState<InputMode>("percent")
  const [values,     setValues]     = useState<Record<string, number>>({})
  const [loading,    setLoading]    = useState(false)
  const [error,      setError]      = useState<string | null>(null)
  const [onDemand,   setOnDemand]   = useState<string[]>([])
  // Draft value for the editable PORTFOLIO TOTAL field in dollar mode.
  // null = not editing (display the live computed sum).
  const [totalDraft, setTotalDraft] = useState<string | null>(null)

  // Sync values when holdings or presetWeights change.
  // If presetWeights are provided (loaded from Smart Builder), switch to
  // percent mode and pre-fill each ticker's value from the preset.
  // Otherwise fall back to the previous value or 0 for new tickers.
  useEffect(() => {
    if (presetWeights) setMode("percent")
    setValues((prev) => {
      const next: Record<string, number> = {}
      for (const h of holdings) {
        next[h.ticker] = presetWeights?.[h.ticker] ?? prev[h.ticker] ?? 0
      }
      return next
    })
  }, [holdings, presetWeights])

  // ── Derived weights (always 0.0–1.0) ──────────────────────────
  // In percent mode: divide each value by 100
  // In dollar mode:  divide each amount by the total portfolio value
  // Either way, the API sees the same shape.
  const totalDollar  = Object.values(values).reduce((s, v) => s + v, 0)
  const totalPercent = Object.values(values).reduce((s, v) => s + v, 0)

  function getWeight(ticker: string): number {
    if (mode === "percent") {
      return (values[ticker] ?? 0) / 100
    } else {
      return totalDollar > 0 ? (values[ticker] ?? 0) / totalDollar : 0
    }
  }

  // Validation differs by mode:
  //   percent → must sum to exactly 100
  //   dollar  → every holding must have an amount > 0
  const isValid =
    holdings.length > 0 &&
    (mode === "percent"
      ? totalPercent === 100
      : holdings.every((h) => (values[h.ticker] ?? 0) > 0))

  function handleValueChange(ticker: string, raw: string) {
    const parsed = parseFloat(raw)
    const clamped =
      isNaN(parsed) ? 0 :
      mode === "percent" ? Math.min(100, Math.max(0, Math.round(parsed))) :
      Math.max(0, parsed)
    setValues((prev) => ({ ...prev, [ticker]: clamped }))
  }

  // When switching percent → dollar: carry over the percentages applied to a
  // $10,000 base (e.g. 60% → $6,000). Going dollar → percent: reset to 0.
  function handleModeSwitch(next: InputMode) {
    setTotalDraft(null)
    if (next === "dollar" && mode === "percent") {
      const base = 10000
      setValues((prev) =>
        Object.fromEntries(
          Object.keys(prev).map((k) => [k, Math.round(((prev[k] ?? 0) / 100) * base * 100) / 100])
        )
      )
    } else if (next === "percent") {
      setValues((prev) => Object.fromEntries(Object.keys(prev).map((k) => [k, 0])))
    }
    setMode(next)
  }

  // Scale all ticker amounts proportionally to a new total.
  // Called when the user edits and commits the PORTFOLIO TOTAL field.
  function commitTotalChange() {
    if (totalDraft === null) return
    const newTotal = parseFloat(totalDraft)
    setTotalDraft(null)
    if (isNaN(newTotal) || newTotal <= 0 || totalDollar === 0) return
    const ratio = newTotal / totalDollar
    setValues((prev) =>
      Object.fromEntries(
        Object.entries(prev).map(([k, v]) => [k, Math.round(v * ratio * 100) / 100])
      )
    )
  }

  async function handleSubmit() {
    if (!isValid) return
    setLoading(true)
    setError(null)
    setOnDemand([])

    try {
      const data = await apiClient.portfolio({
        holdings:   holdings.map((h) => ({
          ticker: h.ticker,
          weight: getWeight(h.ticker),  // already 0.0–1.0 regardless of mode
        })),
        start_date: startDate,
      })
      setOnDemand(data.on_demand_tickers ?? [])
      const portfolioValue      = mode === "dollar" ? totalDollar : 10000
      const submittedHoldings   = holdings.map((h) => ({
        ticker: h.ticker,
        weight: getWeight(h.ticker),
      }))
      onResult(data, portfolioValue, submittedHoldings)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed")
    } finally {
      setLoading(false)
    }
  }

  // ── Progress bar (percent mode only) ──────────────────────────
  const barPct   = Math.min(totalPercent, 100)
  const barColor =
    totalPercent === 100 ? "#00e5a0" :
    totalPercent >   100 ? "#f87171" :
                           "#fbbf24"

  return (
    <div className="space-y-4">

      {/* Mode toggle */}
      <div
        className="flex rounded overflow-hidden text-xs tracking-widest"
        style={{ border: "1px solid var(--border-bright)" }}
      >
        {(["percent", "dollar"] as InputMode[]).map((m) => (
          <button
            key={m}
            onClick={() => handleModeSwitch(m)}
            className="flex-1 py-1.5 transition-colors"
            style={
              mode === m
                ? { background: "rgba(0,229,160,0.12)", color: "var(--accent)" }
                : { background: "transparent",           color: "var(--text-muted)" }
            }
          >
            {m === "percent" ? "% WEIGHT" : "$ AMOUNT"}
          </button>
        ))}
      </div>

      {/* On-demand fetch notice */}
      {onDemand.length > 0 && (
        <div className="rounded px-3 py-2 text-xs tracking-wider leading-relaxed"
          style={{ background: "rgba(251,191,36,0.07)", border: "1px solid rgba(251,191,36,0.2)", color: "#fbbf24" }}>
          FETCHING {onDemand.join(", ")} FOR THE FIRST TIME — THIS MAY TAKE UP TO 30 SECONDS.
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="rounded px-3 py-2 text-xs tracking-wider"
          style={{ background: "rgba(248,113,113,0.07)", border: "1px solid rgba(248,113,113,0.2)", color: "#f87171" }}>
          {error.toUpperCase()}
        </div>
      )}

      {/* Holdings list */}
      {holdings.length === 0 ? (
        <p className="text-xs tracking-wider py-2" style={{ color: "var(--text-dim)" }}>
          SEARCH AND ADD TICKERS ABOVE
        </p>
      ) : (
        <div className="space-y-2">
          {holdings.map((h) => {
            const autoPercent =
              mode === "dollar" && totalDollar > 0
                ? ((values[h.ticker] ?? 0) / totalDollar * 100).toFixed(1)
                : null

            return (
              <div key={h.ticker}
                className="flex items-center gap-2 px-3 py-2 rounded"
                style={{ background: "var(--bg-card-2)", border: "1px solid var(--border)" }}>

                {/* Ticker */}
                <span className="text-sm tracking-widest w-14 shrink-0 font-mono"
                  style={{ color: "var(--accent)" }}>
                  {h.ticker}
                </span>

                {/* Dollar sign prefix */}
                {mode === "dollar" && (
                  <span className="text-xs shrink-0" style={{ color: "var(--text-muted)" }}>$</span>
                )}

                {/* Value input */}
                <input
                  type="number"
                  min={0}
                  max={mode === "percent" ? 100 : undefined}
                  step={mode === "dollar" ? 1 : 1}
                  value={values[h.ticker] ?? 0}
                  onChange={(e) => handleValueChange(h.ticker, e.target.value)}
                  className="bg-transparent outline-none text-sm text-right tracking-wider font-mono"
                  style={{
                    width:        mode === "dollar" ? "80px" : "56px",
                    color:        "var(--text-primary)",
                    caretColor:   "var(--accent)",
                    border:       "1px solid var(--border-bright)",
                    borderRadius: "4px",
                    padding:      "2px 6px",
                  }}
                />

                {/* % suffix (percent mode) */}
                {mode === "percent" && (
                  <span className="text-xs shrink-0" style={{ color: "var(--text-muted)" }}>%</span>
                )}

                {/* Auto-calculated weight (dollar mode) */}
                {autoPercent !== null && (
                  <span className="text-xs shrink-0 tabular-nums font-mono"
                    style={{ color: "var(--accent-dim)" }}>
                    {autoPercent}%
                  </span>
                )}

                <div className="flex-1" />

                {/* Remove */}
                <button onClick={() => onRemove(h.ticker)}
                  className="text-xs opacity-40 hover:opacity-100 transition-opacity"
                  style={{ color: "#f87171" }}>
                  ✕
                </button>
              </div>
            )
          })}
        </div>
      )}

      {/* Summary row */}
      {holdings.length > 0 && (
        <div className="space-y-1.5">
          {mode === "percent" ? (
            <>
              <div className="flex justify-between text-xs tracking-widest">
                <span style={{ color: "var(--text-muted)" }}>TOTAL</span>
                <span className="font-mono" style={{ color: barColor }}>
                  {totalPercent}%{totalPercent === 100 ? " ✓" : ""}
                </span>
              </div>
              <div className="h-1 rounded-full overflow-hidden"
                style={{ background: "var(--border-bright)" }}>
                <div className="h-full rounded-full transition-all duration-200"
                  style={{ width: `${barPct}%`, background: barColor }} />
              </div>
              {totalPercent !== 100 && (
                <p className="text-xs tracking-wider" style={{ color: "var(--text-dim)" }}>
                  {totalPercent < 100
                    ? `${100 - totalPercent}% REMAINING`
                    : `${totalPercent - 100}% OVER — REDUCE WEIGHTS`}
                </p>
              )}
            </>
          ) : (
            // Dollar mode: show total portfolio value + allocation bar per ticker
            <>
              <div className="flex justify-between items-center text-xs tracking-widest">
                <span style={{ color: "var(--text-muted)" }}>PORTFOLIO TOTAL</span>
                <div className="flex items-center gap-1">
                  <span className="text-xs shrink-0" style={{ color: "var(--text-muted)" }}>$</span>
                  <input
                    type="number"
                    min={0}
                    step={100}
                    value={totalDraft ?? totalDollar.toFixed(2)}
                    onFocus={() => setTotalDraft(totalDollar.toFixed(2))}
                    onChange={(e) => setTotalDraft(e.target.value)}
                    onBlur={commitTotalChange}
                    onKeyDown={(e) => { if (e.key === "Enter") commitTotalChange() }}
                    className="bg-transparent outline-none text-right font-mono"
                    style={{
                      width:        "96px",
                      color:        totalDollar > 0 ? "var(--accent)" : "var(--text-dim)",
                      caretColor:   "var(--accent)",
                      border:       "1px solid var(--border-bright)",
                      borderRadius: "4px",
                      padding:      "2px 6px",
                    }}
                    title="Edit to rescale all allocations proportionally"
                  />
                </div>
              </div>

              {/* Stacked allocation bar — one colour segment per ticker */}
              {totalDollar > 0 && (
                <div className="h-1 rounded-full overflow-hidden flex"
                  style={{ background: "var(--border-bright)" }}>
                  {holdings.map((h, i) => {
                    const pct = totalDollar > 0 ? (values[h.ticker] ?? 0) / totalDollar * 100 : 0
                    const segColors = ["#00e5a0", "#00b37d", "#fbbf24", "#60a5fa", "#a78bfa", "#f87171"]
                    return (
                      <div key={h.ticker}
                        className="h-full transition-all duration-200"
                        style={{ width: `${pct}%`, background: segColors[i % segColors.length] }} />
                    )
                  })}
                </div>
              )}

              {!isValid && totalDollar > 0 && (
                <p className="text-xs tracking-wider" style={{ color: "var(--text-dim)" }}>
                  ALL POSITIONS MUST BE GREATER THAN $0
                </p>
              )}
            </>
          )}
        </div>
      )}

      {/* Submit */}
      <button
        onClick={handleSubmit}
        disabled={!isValid || loading}
        className="w-full py-2.5 rounded text-xs tracking-widest uppercase transition-all"
        style={
          isValid && !loading
            ? { background: "rgba(0,229,160,0.12)", border: "1px solid rgba(0,229,160,0.4)", color: "var(--accent)", cursor: "pointer" }
            : { background: "transparent", border: "1px solid var(--border)", color: "var(--text-dim)", cursor: "not-allowed", opacity: 0.5 }
        }
      >
        {loading ? "LOADING..." : "RUN ANALYSIS"}
      </button>
    </div>
  )
}
