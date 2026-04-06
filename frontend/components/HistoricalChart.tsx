"use client"

import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts"
import type { PortfolioResponse, DataPoint } from "@/types/api"

// Extended point — includes trendline values alongside actual values
interface ChartPoint {
  date:           string
  portfolio:      number
  spy:            number | null
  portfolioTrend: number
  spyTrend:       number | null
}

interface Props {
  portfolioData:  PortfolioResponse | null
  defaultSpyData?: DataPoint[] | null  // SPY-only data shown before any portfolio is built
  initialValue?:  number               // actual $ to scale Y-axis — defaults to $10k
  isRefetching?:  boolean              // dims the chart while auto-rerun is in flight
}

// ---------------------------------------------------------------------------
// Linear regression
//
// Takes an array of Y values (equally spaced in time — index = X).
// Returns a new array of the same length containing the fitted straight line.
//
// Math:
//   slope     = Σ((x - x̄)(y - ȳ)) / Σ((x - x̄)²)
//   intercept = ȳ - slope * x̄
//   fitted[i] = slope * i + intercept
//
// Why no library: it's 10 lines of arithmetic. Adding a library for this
// would be overkill and make the logic harder to follow.
// ---------------------------------------------------------------------------
function linearRegression(values: number[]): number[] {
  const n    = values.length
  const xMean = (n - 1) / 2
  const yMean = values.reduce((a, b) => a + b, 0) / n

  let num = 0
  let den = 0
  for (let i = 0; i < n; i++) {
    num += (i - xMean) * (values[i] - yMean)
    den += (i - xMean) ** 2
  }

  const slope     = den !== 0 ? num / den : 0
  const intercept = yMean - slope * xMean

  return values.map((_, i) => slope * i + intercept)
}

function mergeChartData(data: PortfolioResponse, scale: number): ChartPoint[] {
  const spyMap = new Map(
    data.benchmark_returns.map((p) => [p.date, p.value])
  )

  // Raw merged points (scaled)
  const points = data.portfolio_returns.map((p) => ({
    date:      p.date,
    portfolio: p.value * scale,
    spy:       (spyMap.get(p.date) ?? null) !== null
                 ? (spyMap.get(p.date) as number) * scale
                 : null,
  }))

  // Compute trendlines — regression needs a clean array with no nulls.
  // For SPY, filter out any null entries before regressing, then re-insert.
  const portfolioTrends = linearRegression(points.map((p) => p.portfolio))

  // SPY trendline: only computed over indices where spy is not null
  const spyIndices = points
    .map((p, i) => (p.spy !== null ? i : -1))
    .filter((i) => i !== -1)
  const spyValues   = spyIndices.map((i) => points[i].spy as number)
  const spyFitted   = linearRegression(spyValues)
  // Re-map fitted values back to their original indices
  const spyTrendMap = new Map(spyIndices.map((origIdx, fittedIdx) => [origIdx, spyFitted[fittedIdx]]))

  return points.map((p, i) => ({
    ...p,
    portfolioTrend: portfolioTrends[i],
    spyTrend:       spyTrendMap.get(i) ?? null,
  }))
}

function formatDate(iso: string): string {
  const d = new Date(iso + "T00:00:00")
  return d.toLocaleDateString("en-US", { month: "short", year: "2-digit" })
    .toUpperCase()
    .replace(" ", " '")
}

export default function HistoricalChart({
  portfolioData,
  defaultSpyData,
  initialValue = 10000,
  isRefetching  = false,
}: Props) {
  // ── Case: no portfolio yet, but we have default SPY data ────────
  // Show SPY on its own with a trendline so the chart is never empty.
  if (!portfolioData && defaultSpyData && defaultSpyData.length > 0) {
    const spyPoints  = defaultSpyData.map((p) => ({ date: p.date, spy: p.value }))
    const spyTrends  = linearRegression(spyPoints.map((p) => p.spy))
    const spyChartData = spyPoints.map((p, i) => ({ ...p, spyTrend: spyTrends[i] }))
    const xInterval  = Math.max(1, Math.floor(spyChartData.length / 8))
    const vals       = spyChartData.flatMap((p) => [p.spy, p.spyTrend])
    const minV       = Math.min(...vals)
    const maxV       = Math.max(...vals)
    const pad        = (maxV - minV) * 0.05

    return (
      <div style={{ width: "100%", height: 320, opacity: isRefetching ? 0.4 : 1, transition: "opacity 0.2s" }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={spyChartData} margin={{ top: 8, right: 16, bottom: 0, left: 8 }}>
            <CartesianGrid strokeDasharray="4 4" stroke="rgba(255,255,255,0.04)" vertical={false} />
            <XAxis dataKey="date"
              tick={{ fill: "var(--text-muted)" as string, fontSize: 10, fontFamily: "var(--font-space-mono)" }}
              tickLine={false} axisLine={false} interval={xInterval} tickFormatter={formatDate} />
            <YAxis domain={[Math.floor(minV - pad), Math.ceil(maxV + pad)]}
              tick={{ fill: "var(--text-muted)" as string, fontSize: 10, fontFamily: "var(--font-space-mono)" }}
              tickLine={false} axisLine={false} width={56}
              tickFormatter={(v: number) => "$" + (v >= 1000 ? (v / 1000).toFixed(1) + "k" : v.toFixed(0))} />
            <Tooltip content={<CustomTooltip />} />
            <Line type="monotone" dataKey="spyTrend" stroke="#6b7280"
              strokeWidth={1} strokeDasharray="6 4" dot={false} name="SPY TREND" opacity={0.5} />
            <Line type="monotone" dataKey="spy" stroke="#6b7280"
              strokeWidth={1.5} dot={false} name="S&P 500" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    )
  }

  // ── Case: no portfolio, no SPY data yet (initial load in flight) ─
  if (!portfolioData) {
    return (
      <div className="flex items-center justify-center rounded"
        style={{ height: 320, border: "1px dashed var(--border-bright)", background: "rgba(0,0,0,0.2)" }}
      >
        <p className="text-xs tracking-widest" style={{ color: "var(--text-dim)" }}>
          {isRefetching ? "LOADING S&P 500 DATA..." : "RUN ANALYSIS TO VIEW PERFORMANCE"}
        </p>
      </div>
    )
  }

  const scale     = initialValue / 10000
  const chartData = mergeChartData(portfolioData, scale)

  const xInterval = Math.max(1, Math.floor(chartData.length / 8))

  const allValues = chartData.flatMap((p) =>
    [p.portfolio, p.spy, p.portfolioTrend, p.spyTrend]
      .filter((v): v is number => v !== null)
  )
  const minVal  = Math.min(...allValues)
  const maxVal  = Math.max(...allValues)
  const padding = (maxVal - minVal) * 0.05
  const yDomain = [Math.floor(minVal - padding), Math.ceil(maxVal + padding)]

  return (
    <div style={{ width: "100%", height: 320, opacity: isRefetching ? 0.4 : 1, transition: "opacity 0.2s" }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 8, right: 16, bottom: 0, left: 8 }}>

          <CartesianGrid
            strokeDasharray="4 4"
            stroke="rgba(255,255,255,0.04)"
            vertical={false}
          />

          <XAxis
            dataKey="date"
            tick={{ fill: "var(--text-muted)" as string, fontSize: 10, fontFamily: "var(--font-space-mono)" }}
            tickLine={false}
            axisLine={false}
            interval={xInterval}
            tickFormatter={formatDate}
          />

          <YAxis
            domain={yDomain}
            tick={{ fill: "var(--text-muted)" as string, fontSize: 10, fontFamily: "var(--font-space-mono)" }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v: number) =>
              "$" + (v >= 1000 ? (v / 1000).toFixed(1) + "k" : v.toFixed(0))
            }
            width={56}
          />

          <Tooltip content={<CustomTooltip />} />

          {/* ── Trendlines (dashed) — rendered UNDER actual lines ── */}
          {/* Rendered first so actual lines draw on top */}

          <Line
            type="monotone"
            dataKey="spyTrend"
            stroke="#6b7280"
            strokeWidth={1}
            strokeDasharray="6 4"
            dot={false}
            name="SPY TREND"
            connectNulls
            opacity={0.5}
          />

          <Line
            type="monotone"
            dataKey="portfolioTrend"
            stroke="#00e5a0"
            strokeWidth={1}
            strokeDasharray="6 4"
            dot={false}
            name="PORTFOLIO TREND"
            opacity={0.5}
          />

          {/* ── Actual lines — rendered on top ── */}

          <Line
            type="monotone"
            dataKey="spy"
            stroke="#6b7280"
            strokeWidth={1.5}
            dot={false}
            name="SPY"
            connectNulls
          />

          <Line
            type="monotone"
            dataKey="portfolio"
            stroke="#00e5a0"
            strokeWidth={2}
            dot={false}
            name="PORTFOLIO"
          />

        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

// Trendline entries are filtered out of the tooltip — they'd just repeat
// the same color as the actual line and add noise.
function CustomTooltip({
  active,
  payload,
  label,
}: {
  active?:  boolean
  payload?: { name: string; value: number; color: string }[]
  label?:   string
}) {
  if (!active || !payload?.length) return null

  const visible = payload.filter(
    (p) => !p.name.includes("TREND") && p.value != null
  )
  if (!visible.length) return null

  return (
    <div
      className="rounded px-3 py-2 text-xs space-y-1.5"
      style={{ background: "var(--bg-card)", border: "1px solid var(--border-bright)" }}
    >
      <p className="tracking-widest mb-2" style={{ color: "var(--text-muted)" }}>
        {label ? formatDate(label) : ""}
      </p>
      {visible.map((p) => (
        <div key={p.name} className="flex items-center gap-3">
          <span className="w-2 h-2 rounded-full shrink-0" style={{ background: p.color }} />
          <span className="tracking-wider w-20" style={{ color: "var(--text-muted)" }}>
            {p.name}
          </span>
          <span className="tabular-nums font-mono" style={{ color: p.color }}>
            ${p.value.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
        </div>
      ))}
    </div>
  )
}
