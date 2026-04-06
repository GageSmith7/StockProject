"use client"

import { useState, useEffect, useRef, useCallback } from "react"
import TickerSearch      from "@/components/TickerSearch"
import PortfolioBuilder  from "@/components/PortfolioBuilder"
import HistoricalChart   from "@/components/HistoricalChart"
import ProjectionChart   from "@/components/ProjectionChart"
import PredictionsTable  from "@/components/PredictionsTable"
import SmartBuilder      from "@/components/SmartBuilder"
import { apiClient }     from "@/lib/apiClient"
import type { SearchResult, PortfolioResponse, SimulateResponse, PredictResponse, ModelHealthResponse, Holding, DataPoint } from "@/types/api"

const DEFAULT_START = "2020-01-01"

const RANGE_MAP: Record<string, string> = {
  "1M":  offsetDate(-1,   "months"),
  "3M":  offsetDate(-3,   "months"),
  "6M":  offsetDate(-6,   "months"),
  "1Y":  offsetDate(-1,   "years"),
  "3Y":  offsetDate(-3,   "years"),
  "5Y":  offsetDate(-5,   "years"),
  "10Y": offsetDate(-10,  "years"),
  "MAX": "2010-01-01",
}

export default function Home() {
  const [holdings,       setHoldings]       = useState<SearchResult[]>([])
  const [portfolioData,  setPortfolioData]  = useState<PortfolioResponse | null>(null)
  const [portfolioValue, setPortfolioValue] = useState<number>(10000)
  const [defaultSpyData, setDefaultSpyData] = useState<DataPoint[] | null>(null)
  const [startDate,      setStartDate]      = useState(DEFAULT_START)
  const [activeRange,    setActiveRange]    = useState("5Y")
  const [isRefetching,   setIsRefetching]   = useState(false)
  const [simulationData, setSimulationData] = useState<SimulateResponse | null>(null)
  const [simError,       setSimError]       = useState<string | null>(null)
  const [isSimulating,   setIsSimulating]   = useState(false)
  const [simYears,        setSimYears]        = useState(10)
  const [simCount,        setSimCount]        = useState(1000)
  const [simContribution, setSimContribution] = useState(0)
  const [simUseRegime,    setSimUseRegime]    = useState(false)
  const [predictionData,    setPredictionData]    = useState<PredictResponse | null>(null)
  const [isPredicting,      setIsPredicting]      = useState(false)
  const [modelHealth,       setModelHealth]       = useState<ModelHealthResponse | null>(null)
  const [modelHealthError,  setModelHealthError]  = useState(false)
  const [presetWeights,     setPresetWeights]     = useState<Record<string, number> | undefined>(undefined)

  const builderRef = useRef<HTMLDivElement>(null)

  // lastSubmit stores the holdings+weights from the most recent Run Analysis.
  // It's a ref (not state) because changing it should NOT trigger a re-render —
  // we only read it inside the auto-rerun effect.
  const lastSubmit = useRef<{ holdings: Holding[]; portfolioValue: number } | null>(null)

  // ── Fetch predictions + model health once on mount ───────────────────────
  useEffect(() => {
    setIsPredicting(true)
    apiClient
      .predict({})
      .then((data) => setPredictionData(data))
      .catch(() => {})
      .finally(() => setIsPredicting(false))

    apiClient
      .modelHealth()
      .then((data) => setModelHealth(data))
      .catch(() => setModelHealthError(true))
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // ── Single effect handles two cases based on whether analysis has been run ──
  //
  // Case A — no portfolio yet (lastSubmit is null):
  //   Fetch SPY on its own so the chart is never empty.
  //   Fires on mount AND whenever the range changes before first analysis.
  //
  // Case B — portfolio has been run (lastSubmit is set):
  //   Replay the same holdings with the new startDate.
  //   Dims the chart while the request is in flight.
  //
  // useRef (not useState) for lastSubmit so updating it never re-triggers
  // this effect — we only want startDate changes to drive it.
  useEffect(() => {
    setIsRefetching(true)

    if (!lastSubmit.current) {
      // Case A: default SPY chart
      apiClient
        .portfolio({ holdings: [{ ticker: "SPY", weight: 1.0 }], start_date: startDate })
        .then((data) => setDefaultSpyData(data.portfolio_returns))
        .catch(() => {})
        .finally(() => setIsRefetching(false))
    } else {
      // Case B: rerun user's portfolio
      const { holdings: h, portfolioValue: pv } = lastSubmit.current
      apiClient
        .portfolio({ holdings: h, start_date: startDate })
        .then((data) => {
          setPortfolioData(data)
          setPortfolioValue(pv)
        })
        .catch(() => {})
        .finally(() => setIsRefetching(false))
    }
  }, [startDate]) // eslint-disable-line react-hooks/exhaustive-deps

  function handleTickerSelect(result: SearchResult) {
    if (holdings.find((h) => h.ticker === result.ticker)) return
    setHoldings((prev) => [...prev, result])
  }

  function handleRemove(ticker: string) {
    setHoldings((prev) => prev.filter((h) => h.ticker !== ticker))
  }

  function handlePortfolioResult(
    data: PortfolioResponse,
    value: number,
    submittedHoldings: Holding[]
  ) {
    setPortfolioData(data)
    setPortfolioValue(value)
    // Store for auto-rerun — update the ref directly, no re-render needed
    lastSubmit.current = { holdings: submittedHoldings, portfolioValue: value }
    // Auto-run simulation with current param values
    triggerSimulation(submittedHoldings, value, simYears, simCount, simContribution, simUseRegime)
  }

  // Called from SmartBuilder when user clicks "Load into Portfolio Builder".
  // Converts decimal weights (0.25) → percent integers (25), fixes rounding
  // so they always sum to exactly 100, then scrolls up to the builder.
  const handleLoadToBuilder = useCallback((decimalWeights: Record<string, number>) => {
    const tickers = Object.keys(decimalWeights)

    // Build SearchResult stubs for each ticker
    const newHoldings: SearchResult[] = tickers.map((t) => ({
      ticker: t, name: "", sector: null, tier: "preloaded" as const,
    }))

    // Convert to percent integers and fix rounding drift
    const rawPcts  = tickers.map((t) => Math.round(decimalWeights[t] * 100))
    const diff     = 100 - rawPcts.reduce((a, b) => a + b, 0)
    rawPcts[0]    += diff   // absorb remainder into the largest-weight ticker

    setHoldings(newHoldings)
    setPresetWeights(Object.fromEntries(tickers.map((t, i) => [t, rawPcts[i]])))
    builderRef.current?.scrollIntoView({ behavior: "smooth", block: "start" })
  }, [])

  function handleRangeClick(range: string) {
    setActiveRange(range)
    setStartDate(RANGE_MAP[range])
  }

  function triggerSimulation(holdings: Holding[], value: number, years: number, sims: number, contribution: number = 0, useRegime: boolean = false) {
    setIsSimulating(true)
    setSimError(null)
    apiClient
      .simulate({ holdings, years, n_simulations: sims, initial_value: value, monthly_contribution: contribution, use_regime: useRegime })
      .then((data) => setSimulationData(data))
      .catch((err) => setSimError(err instanceof Error ? err.message : "Simulation failed"))
      .finally(() => setIsSimulating(false))
  }

  return (
    <div className="space-y-6">

      {/* ── Top row: Builder + Charts ─────────────────────────────── */}
      <div className="flex gap-6 items-start">

        {/* Left: Portfolio Builder */}
        <aside ref={builderRef} className="w-80 shrink-0 space-y-4">
          <Section stepLabel="STEP 3–4" title="Portfolio Builder">
            <div className="space-y-3">
              <TickerSearch
                onSelect={handleTickerSelect}
                disabledTickers={holdings.map((h) => h.ticker)}
              />
              <PortfolioBuilder
                holdings={holdings}
                startDate={startDate}
                onRemove={handleRemove}
                onResult={handlePortfolioResult}
                presetWeights={presetWeights}
              />
            </div>
          </Section>
        </aside>

        {/* Right: Charts */}
        <div className="flex-1 space-y-4">
          <Section stepLabel="STEP 5" title="Historical Performance">
            <HistoricalChart
              portfolioData={portfolioData}
              defaultSpyData={defaultSpyData}
              initialValue={portfolioValue}
              isRefetching={isRefetching}
            />
            <div className="flex flex-wrap gap-2 mt-3">
              {["1M", "3M", "6M", "1Y", "3Y", "5Y", "10Y", "MAX"].map((r) => (
                <RangePill
                  key={r}
                  label={r}
                  active={r === activeRange}
                  onClick={() => handleRangeClick(r)}
                />
              ))}
            </div>
          </Section>

          <Section stepLabel="STEP 6" title="Metrics">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <MetricCard label="CAGR"
                value={portfolioData ? pct(portfolioData.metrics.cagr) : "--"}
                colorType="signed"
                rawValue={portfolioData?.metrics.cagr ?? null} />
              <MetricCard label="SHARPE RATIO"
                value={portfolioData ? portfolioData.metrics.sharpe.toFixed(2) : "--"}
                colorType="sharpe"
                rawValue={portfolioData?.metrics.sharpe ?? null} />
              <MetricCard label="MAX DRAWDOWN"
                value={portfolioData ? pct(portfolioData.metrics.max_drawdown) : "--"}
                colorType="always-red"
                rawValue={portfolioData?.metrics.max_drawdown ?? null} />
              <MetricCard label="TOTAL RETURN"
                value={portfolioData ? pct(portfolioData.metrics.total_return) : "--"}
                colorType="signed"
                rawValue={portfolioData?.metrics.total_return ?? null} />
            </div>
          </Section>
        </div>
      </div>

      {/* ── Projection Section ────────────────────────────────────── */}
      <Section stepLabel="STEP 7" title="Monte Carlo Projection">
        <div className="flex flex-wrap gap-4 items-end mb-5">
          <NumberInput label="YEARS"           value={simYears}        onChange={setSimYears}        min={1}   max={30}    />
          <NumberInput label="SIMULATIONS"    value={simCount}        onChange={setSimCount}        min={100} max={10000} step={100} />
          <NumberInput label="MONTHLY CONTRIB ($)" value={simContribution} onChange={setSimContribution} min={0}   max={100000} step={100} />
          {/* Regime toggle */}
          <div className="flex flex-col gap-1">
            <label className="text-xs tracking-widest" style={{ color: "var(--text-muted)" }}>REGIME FILTER</label>
            <button
              onClick={() => setSimUseRegime(v => !v)}
              className="px-3 py-1.5 rounded text-xs tracking-widest uppercase transition-all"
              style={simUseRegime
                ? { background: "rgba(0,229,160,0.12)", border: "1px solid rgba(0,229,160,0.4)", color: "var(--accent)", cursor: "pointer" }
                : { background: "transparent", border: "1px solid var(--border-bright)", color: "var(--text-dim)", cursor: "pointer" }}
            >
              {simUseRegime ? "ON" : "OFF"}
            </button>
          </div>
          {/* Regime badge — shown after simulation runs with regime enabled */}
          {simulationData?.current_regime && (
            <div className="flex flex-col gap-1 self-end pb-0.5">
              <span className="text-xs tracking-widest px-2 py-1 rounded"
                style={{
                  background: simulationData.current_regime === "expansion"
                    ? "rgba(0,229,160,0.08)" : "rgba(248,113,113,0.08)",
                  border: simulationData.current_regime === "expansion"
                    ? "1px solid rgba(0,229,160,0.3)" : "1px solid rgba(248,113,113,0.3)",
                  color: simulationData.current_regime === "expansion"
                    ? "var(--accent)" : "#f87171",
                }}>
                {simulationData.current_regime === "expansion" ? "▲ EXPANSION" : "▼ CONTRACTION"}
                {simulationData.regime_fallback && " (FALLBACK)"}
              </span>
            </div>
          )}
          <button
            disabled={!portfolioData || isSimulating}
            onClick={() => {
              if (!lastSubmit.current) return
              triggerSimulation(lastSubmit.current.holdings, lastSubmit.current.portfolioValue, simYears, simCount, simContribution, simUseRegime)
            }}
            className="self-end px-5 py-1.5 rounded text-xs tracking-widest uppercase transition-all"
            style={
              portfolioData && !isSimulating
                ? { background: "rgba(0,229,160,0.1)", border: "1px solid rgba(0,229,160,0.35)", color: "var(--accent)", cursor: "pointer" }
                : { background: "transparent", border: "1px solid var(--border)", color: "var(--text-dim)", cursor: "not-allowed", opacity: 0.5 }
            }
          >
            {isSimulating ? "SIMULATING..." : "RUN PROJECTION"}
          </button>
        </div>
        {simError && (
          <div className="rounded px-3 py-2 text-xs tracking-wider mb-3"
            style={{ background: "rgba(248,113,113,0.07)", border: "1px solid rgba(248,113,113,0.2)", color: "#f87171" }}>
            {simError.toUpperCase()}
          </div>
        )}
        <ProjectionChart data={simulationData} isSimulating={isSimulating} />
        <p className="text-xs mt-4 pt-4 tracking-wide leading-relaxed"
          style={{ color: "var(--text-muted)", borderTop: "1px solid var(--border)" }}>
          DISCLAIMER — Projections are based on the historical return distribution of your
          portfolio. Past performance does not predict future results. This is not financial advice.
        </p>
      </Section>

      {/* ── ML Predictions ───────────────────────────────────────── */}
      <Section stepLabel="STEP 8" title="ML Predictions · 21-Day Forecast">
        <PredictionsTable data={predictionData} isLoading={isPredicting} modelHealth={modelHealth} modelHealthError={modelHealthError} />
      </Section>

      {/* ── Smart Portfolio Builder ───────────────────────────────── */}
      <Section stepLabel="STEP 9" title="Smart Portfolio Builder">
        <SmartBuilder onLoad={handleLoadToBuilder} />
      </Section>

    </div>
  )
}

/* ── Helpers ────────────────────────────────────────────────────── */

// Returns a date string offset from today by the given amount and unit
function offsetDate(value: number, unit: "days" | "months" | "years" = "years"): string {
  const d = new Date()
  if (unit === "years")  d.setFullYear(d.getFullYear() + value)
  if (unit === "months") d.setMonth(d.getMonth() + value)
  if (unit === "days")   d.setDate(d.getDate() + value)
  return d.toISOString().slice(0, 10)
}

// Format a decimal as a percentage string: 0.142 → "+14.2%"
function pct(v: number): string {
  return (v >= 0 ? "+" : "") + (v * 100).toFixed(1) + "%"
}

/* ── Scaffold helpers ───────────────────────────────────────────── */

function Section({ stepLabel, title, children }: {
  stepLabel: string; title: string; children: React.ReactNode
}) {
  return (
    <div className="rounded-lg p-5"
      style={{ background: "var(--bg-card)", border: "1px solid var(--border)" }}>
      <div className="flex items-center gap-2 mb-4">
        <span className="text-xs tracking-widest px-2 py-0.5 rounded"
          style={{ color: "var(--accent-dim)", background: "rgba(0,229,160,0.07)", border: "1px solid rgba(0,229,160,0.15)" }}>
          {stepLabel}
        </span>
        <h2 className="text-xs tracking-widest uppercase" style={{ color: "var(--text-primary)" }}>
          {title}
        </h2>
      </div>
      {children}
    </div>
  )
}

function ChartPlaceholder({ height, label }: { height: string; label: string }) {
  return (
    <div className={`${height} rounded flex items-center justify-center`}
      style={{ border: "1px dashed var(--border-bright)", background: "rgba(0,0,0,0.2)" }}>
      <span className="text-xs tracking-widest" style={{ color: "var(--text-dim)" }}>{label}</span>
    </div>
  )
}

// ---------------------------------------------------------------------------
// MetricCard info lookup
// Each entry describes what the metric measures, why it matters, and how
// it's calculated — shown in the popup when the user clicks the i button.
// ---------------------------------------------------------------------------
const METRIC_INFO: Record<string, { measures: string; importance: string; formula: string }> = {
  "CAGR": {
    measures: "Compound Annual Growth Rate — your portfolio's smoothed yearly return if it grew at a constant rate.",
    importance: "Lets you compare portfolios across different time periods on equal footing.",
    formula: "(End Value ÷ Start Value)^(1 ÷ Years) − 1",
  },
  "SHARPE RATIO": {
    measures: "Return earned per unit of risk (volatility) taken.",
    importance: "A high return means nothing if you took enormous risk to get it. Sharpe normalizes for that. < 0.5 is poor, 0.5–1.0 is acceptable, ≥ 1.0 is good.",
    formula: "(Portfolio Return − Risk-Free Rate) ÷ Std Dev of Returns",
  },
  "MAX DRAWDOWN": {
    measures: "The largest peak-to-trough decline in your portfolio's value.",
    importance: "Worst-case loss you'd have experienced if you bought at the peak and sold at the trough.",
    formula: "(Trough Value − Peak Value) ÷ Peak Value",
  },
  "TOTAL RETURN": {
    measures: "Overall gain or loss over the entire period, not annualized.",
    importance: "The simplest measure of how much your investment grew or shrunk.",
    formula: "(End Value − Start Value) ÷ Start Value",
  },
}

// colorType controls how the value is colored:
//   "signed"     — green if positive, red if negative (CAGR, Total Return)
//   "sharpe"     — green ≥1.0, yellow ≥0.5, red <0.5
//   "always-red" — always red regardless of value (Max Drawdown)
type ColorType = "signed" | "sharpe" | "always-red"

function MetricCard({ label, value, colorType, rawValue }: {
  label:     string
  value:     string
  colorType: ColorType
  rawValue:  number | null   // raw number for color logic; null when no data yet
}) {
  const [open, setOpen] = useState(false)
  const cardRef = useRef<HTMLDivElement>(null)

  // Close popup on outside click — only active while popup is open
  useEffect(() => {
    if (!open) return
    function handleOutside(e: MouseEvent) {
      if (cardRef.current && !cardRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener("mousedown", handleOutside)
    return () => document.removeEventListener("mousedown", handleOutside)
  }, [open])

  const isPopulated = value !== "--"

  // Determine display color from colorType + rawValue
  let color = "var(--text-dim)"
  if (isPopulated && rawValue !== null) {
    if (colorType === "always-red") {
      color = "#f87171"
    } else if (colorType === "sharpe") {
      color = rawValue >= 1.0 ? "var(--accent)" : rawValue >= 0.5 ? "#fbbf24" : "#f87171"
    } else {
      // signed
      color = rawValue >= 0 ? "var(--accent)" : "#f87171"
    }
  }

  const info = METRIC_INFO[label]

  return (
    <div ref={cardRef} className="relative rounded p-3"
      style={{ background: "var(--bg-card-2)", border: "1px solid var(--border)" }}>

      {/* Label row with i button */}
      <div className="flex items-center justify-between mb-2">
        <p className="text-xs tracking-widest" style={{ color: "var(--text-muted)" }}>{label}</p>
        {info && (
          <button
            onClick={() => setOpen((o) => !o)}
            className="w-4 h-4 rounded-full flex items-center justify-center text-xs shrink-0 transition-colors"
            style={{
              border:     "1px solid var(--border-bright)",
              color:      open ? "var(--accent)" : "var(--text-dim)",
              background: open ? "rgba(0,229,160,0.08)" : "transparent",
              fontStyle:  "italic",
              lineHeight: 1,
            }}
          >
            i
          </button>
        )}
      </div>

      {/* Value */}
      <p className="text-2xl font-mono" style={{ color }}>{value}</p>

      {/* Info popup — positioned below the card, above other content via z-50 */}
      {open && info && (
        <div
          className="absolute z-50 left-0 top-full mt-1 rounded p-3 space-y-2 text-xs leading-relaxed"
          style={{
            width:      "220px",
            background: "var(--bg-card)",
            border:     "1px solid var(--border-bright)",
            boxShadow:  "0 4px 24px rgba(0,0,0,0.6)",
          }}
        >
          <p style={{ color: "var(--text-primary)" }}>
            <span style={{ color: "var(--accent)" }}>WHAT · </span>{info.measures}
          </p>
          <p style={{ color: "var(--text-primary)" }}>
            <span style={{ color: "var(--accent)" }}>WHY · </span>{info.importance}
          </p>
          <p style={{ color: "var(--text-muted)" }}>
            <span style={{ color: "var(--accent-dim)" }}>CALC · </span>{info.formula}
          </p>
        </div>
      )}
    </div>
  )
}

function RangePill({ label, active, onClick }: {
  label: string; active: boolean; onClick?: () => void
}) {
  return (
    <button onClick={onClick}
      className="text-xs tracking-widest px-3 py-1 rounded transition-colors"
      style={active
        ? { background: "rgba(0,229,160,0.12)", color: "var(--accent)", border: "1px solid rgba(0,229,160,0.3)" }
        : { background: "transparent", color: "var(--text-muted)", border: "1px solid var(--border)" }}>
      {label}
    </button>
  )
}

function NumberInput({ label, value, onChange, min, max, step = 1 }: {
  label:    string
  value:    number
  onChange: (v: number) => void
  min?:     number
  max?:     number
  step?:    number
}) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-xs tracking-widest" style={{ color: "var(--text-muted)" }}>{label}</label>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => {
          const v = parseInt(e.target.value)
          if (!isNaN(v)) onChange(v)
        }}
        className="px-3 py-1.5 rounded text-sm w-28 outline-none font-mono"
        style={{ background: "var(--bg-card-2)", border: "1px solid var(--border-bright)", color: "var(--text-primary)" }}
      />
    </div>
  )
}
