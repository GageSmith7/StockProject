"use client"

import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts"
import type { SimulateResponse } from "@/types/api"

// ---------------------------------------------------------------------------
// Linear regression — identical to HistoricalChart's copy.
// Kept local so each component is self-contained (no shared utility needed
// for 10 lines of math at MVP scale).
// ---------------------------------------------------------------------------
function linearRegression(values: number[]): number[] {
  const n     = values.length
  const xMean = (n - 1) / 2
  const yMean = values.reduce((a, b) => a + b, 0) / n
  let num = 0, den = 0
  for (let i = 0; i < n; i++) {
    num += (i - xMean) * (values[i] - yMean)
    den += (i - xMean) ** 2
  }
  const slope     = den !== 0 ? num / den : 0
  const intercept = yMean - slope * xMean
  return values.map((_, i) => slope * i + intercept)
}

// ---------------------------------------------------------------------------
// ConePoint — one data point for the chart.
//
// The stacked-area trick:
//   recharts Area with stackId stacks values visually.
//   • p10   — invisible base; pushes the band up to start at p10
//   • band  — (p90 - p10); the visible mint-green fill sits on top of p10
//   Together they render the shaded p10→p90 cone.
//
// p10 and band are the original values in the tooltip payload (recharts
// reports raw values, not stacked totals), so p90 = p10 + band in the tooltip.
// ---------------------------------------------------------------------------
interface ConePoint {
  label:    string
  p10:      number
  band:     number   // p90 - p10
  p50:      number
  p50Trend: number
}

function buildConeData(data: SimulateResponse): ConePoint[] {
  const p50Trends = linearRegression(data.percentiles.p50)
  return data.time_axis.map((ym, i) => ({
    label:    formatLabel(ym),
    p10:      data.percentiles.p10[i],
    band:     data.percentiles.p90[i] - data.percentiles.p10[i],
    p50:      data.percentiles.p50[i],
    p50Trend: p50Trends[i],
  }))
}

// time_axis format is "YYYY-MM" → "JAN '26"
function formatLabel(ym: string): string {
  const [year, month] = ym.split("-")
  const d = new Date(parseInt(year), parseInt(month) - 1, 1)
  return d.toLocaleDateString("en-US", { month: "short", year: "2-digit" })
    .toUpperCase()
    .replace(" ", " '")
}

interface Props {
  data:         SimulateResponse | null
  isSimulating: boolean
}

export default function ProjectionChart({ data, isSimulating }: Props) {
  // ── No data yet ──────────────────────────────────────────────────────────
  if (!data) {
    return (
      <div
        className="flex items-center justify-center rounded"
        style={{ height: 288, border: "1px dashed var(--border-bright)", background: "rgba(0,0,0,0.2)" }}
      >
        <p className="text-xs tracking-widest" style={{ color: "var(--text-dim)" }}>
          {isSimulating ? "RUNNING SIMULATION..." : "RUN ANALYSIS TO GENERATE PROJECTION"}
        </p>
      </div>
    )
  }

  const coneData  = buildConeData(data)
  const xInterval = Math.max(1, Math.floor(coneData.length / 8))

  // Y-axis domain from actual p10/p90 extremes
  const allVals = [
    ...data.percentiles.p10,
    ...data.percentiles.p90,
    ...data.percentiles.p50,
  ]
  const minVal  = Math.min(...allVals)
  const maxVal  = Math.max(...allVals)
  const pad     = (maxVal - minVal) * 0.05
  const yDomain = [Math.floor(minVal - pad), Math.ceil(maxVal + pad)]

  return (
    <div style={{ width: "100%", height: 288, opacity: isSimulating ? 0.4 : 1, transition: "opacity 0.2s" }}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={coneData} margin={{ top: 8, right: 16, bottom: 0, left: 8 }}>

          {/* Vertical gradient: red (worst / bottom) → amber → green (best / top) */}
          <defs>
            <linearGradient id="coneGradient" x1="0" y1="1" x2="0" y2="0">
              <stop offset="0%"   stopColor="#f87171" stopOpacity={0.18} />
              <stop offset="50%"  stopColor="#fbbf24" stopOpacity={0.07} />
              <stop offset="100%" stopColor="#00e5a0" stopOpacity={0.18} />
            </linearGradient>
          </defs>

          <CartesianGrid
            strokeDasharray="4 4"
            stroke="rgba(255,255,255,0.04)"
            vertical={false}
          />

          <XAxis
            dataKey="label"
            tick={{ fill: "var(--text-muted)" as string, fontSize: 10, fontFamily: "var(--font-space-mono)" }}
            tickLine={false}
            axisLine={false}
            interval={xInterval}
          />

          <YAxis
            domain={yDomain}
            tick={{ fill: "var(--text-muted)" as string, fontSize: 10, fontFamily: "var(--font-space-mono)" }}
            tickLine={false}
            axisLine={false}
            width={56}
            tickFormatter={(v: number) =>
              "$" + (v >= 1000 ? (v / 1000).toFixed(1) + "k" : v.toFixed(0))
            }
          />

          <Tooltip content={<ConeTooltip />} />

          {/* Invisible p10 base — pushes band up to start at p10 visually */}
          <Area
            type="monotone"
            dataKey="p10"
            stackId="cone"
            fill="transparent"
            stroke="none"
            legendType="none"
          />

          {/* Visible cone fill — stacked on p10, reaches p90 */}
          <Area
            type="monotone"
            dataKey="band"
            stackId="cone"
            fill="url(#coneGradient)"
            stroke="none"
            legendType="none"
          />

          {/* P50 trendline — dashed, rendered before solid line */}
          <Line
            type="monotone"
            dataKey="p50Trend"
            stroke="#00e5a0"
            strokeWidth={1}
            strokeDasharray="6 4"
            dot={false}
            name="P50 TREND"
            opacity={0.5}
          />

          {/* P50 median — solid, on top */}
          <Line
            type="monotone"
            dataKey="p50"
            stroke="#00e5a0"
            strokeWidth={2}
            dot={false}
            name="P50 MEDIAN"
          />

        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Tooltip — reconstructs p90 from p10 + band (raw, pre-stack values).
// Hides trendline entry. Shows P90 → P50 → P10 top-to-bottom.
// ---------------------------------------------------------------------------
function ConeTooltip({
  active,
  payload,
  label,
}: {
  active?:  boolean
  payload?: { name: string; dataKey: string; value: number }[]
  label?:   string
}) {
  if (!active || !payload?.length) return null

  const p10Val  = payload.find((p) => p.dataKey === "p10")?.value ?? null
  const bandVal = payload.find((p) => p.dataKey === "band")?.value ?? null
  const p50Val  = payload.find((p) => p.dataKey === "p50")?.value ?? null
  const p90Val  = p10Val !== null && bandVal !== null ? p10Val + bandVal : null

  if (p50Val === null) return null

  return (
    <div
      className="rounded px-3 py-2 text-xs space-y-1.5"
      style={{ background: "var(--bg-card)", border: "1px solid var(--border-bright)" }}
    >
      <p className="tracking-widest mb-2" style={{ color: "var(--text-muted)" }}>
        {label ?? ""}
      </p>
      {p90Val !== null && (
        <TooltipRow label="P90" value={p90Val} color="rgba(0,229,160,0.5)" />
      )}
      <TooltipRow label="P50" value={p50Val} color="#00e5a0" />
      {p10Val !== null && (
        <TooltipRow label="P10" value={p10Val} color="#f87171" />
      )}
    </div>
  )
}

function TooltipRow({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex items-center gap-3">
      <span className="w-2 h-2 rounded-full shrink-0" style={{ background: color }} />
      <span className="tracking-wider w-8" style={{ color: "var(--text-muted)" }}>{label}</span>
      <span className="tabular-nums font-mono" style={{ color }}>
        ${value.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
      </span>
    </div>
  )
}
