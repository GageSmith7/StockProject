"use client"

import { useState } from "react"
import type { ModelHealthResponse, PredictResponse } from "@/types/api"

interface Props {
  data:             PredictResponse | null
  isLoading:        boolean
  modelHealth:      ModelHealthResponse | null
  modelHealthError: boolean
}

export default function PredictionsTable({ data, isLoading, modelHealth, modelHealthError }: Props) {
  // Must be declared before any early returns — hooks cannot be conditional
  const [visibleCount, setVisibleCount] = useState(20)

  // ── Health bar — always rendered regardless of prediction state ─────────
  const healthBar = (
    <div className="flex items-center justify-between text-xs tracking-widest mb-3"
      style={{ color: "var(--text-muted)" }}>
      <span>MODEL STATUS</span>
      {modelHealth ? (
        <ModelHealthBadge health={modelHealth} />
      ) : (
        <span className="px-2 py-0.5 rounded text-xs tracking-wider"
          style={{
            border: "1px solid var(--border)",
            color: modelHealthError ? "#f87171" : "var(--text-dim)",
          }}>
          {modelHealthError ? "HEALTH CHECK FAILED" : "CONNECTING..."}
        </span>
      )}
    </div>
  )

  // ── Loading ──────────────────────────────────────────────────────────────
  if (isLoading) {
    return <>{healthBar}<Placeholder label="LOADING PREDICTIONS..." /></>
  }

  // ── No data yet (fetch failed silently) ──────────────────────────────────
  if (!data) {
    return <>{healthBar}<Placeholder label="PREDICTIONS UNAVAILABLE" /></>
  }

  // ── Model not trained ────────────────────────────────────────────────────
  if (!data.model_available) {
    return <>{healthBar}<Placeholder label="MODEL NOT AVAILABLE — RUN TRAINING FIRST" /></>
  }

  const sorted   = [...data.predictions].sort((a, b) => a.rank - b.rank)
  const rows     = sorted.slice(0, visibleCount)
  const hasMore  = visibleCount < sorted.length

  return (
    <div className="space-y-3">

      {/* Health status + as-of date + model version */}
      <div className="flex items-center justify-between text-xs tracking-widest"
        style={{ color: "var(--text-muted)" }}>
        <span>AS OF <span className="font-mono">{data.as_of_date}</span></span>
        <div className="flex items-center gap-3">
          {modelHealth && <ModelHealthBadge health={modelHealth} />}
          <span className="font-mono" style={{ color: "var(--text-dim)" }}>{data.model_version}</span>
        </div>
      </div>

      {/* Scrollable table */}
      <div className="overflow-y-auto" style={{ maxHeight: "280px" }}>
        <table className="w-full text-xs">
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border-bright)" }}>
              <Th align="left"  label="RANK"         />
              <Th align="left"  label="TICKER"       />
              <Th align="right" label="21D FORECAST" />
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const positive = row.predicted_return_21d >= 0
              const color    = positive ? "var(--accent)" : "#f87171"
              const pct      = (positive ? "+" : "") +
                (row.predicted_return_21d * 100).toFixed(2) + "%"

              return (
                <tr
                  key={row.ticker}
                  style={{ borderBottom: "1px solid var(--border)" }}
                >
                  <td className="py-2 pr-4 tabular-nums w-10 font-mono"
                    style={{ color: "var(--text-dim)" }}>
                    {row.rank}
                  </td>
                  <td className="py-2 tracking-widest font-mono"
                    style={{ color: "var(--text-primary)" }}>
                    {row.ticker}
                  </td>
                  <td className="py-2 text-right tabular-nums tracking-wider font-mono"
                    style={{ color }}>
                    {pct}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* Load more */}
      {hasMore && (
        <button
          onClick={() => setVisibleCount((c) => c + 20)}
          className="w-full py-2 rounded text-xs tracking-widest uppercase transition-colors"
          style={{
            background: "transparent",
            border:     "1px solid var(--border-bright)",
            color:      "var(--text-muted)",
          }}
          onMouseEnter={(e) => {
            (e.currentTarget as HTMLElement).style.borderColor = "rgba(0,229,160,0.3)"
            ;(e.currentTarget as HTMLElement).style.color = "var(--accent)"
          }}
          onMouseLeave={(e) => {
            (e.currentTarget as HTMLElement).style.borderColor = "var(--border-bright)"
            ;(e.currentTarget as HTMLElement).style.color = "var(--text-muted)"
          }}
        >
          LOAD MORE · SHOWING {rows.length} OF {sorted.length}
        </button>
      )}

      {/* Disclaimer */}
      <p className="text-xs tracking-wide leading-relaxed pt-3"
        style={{ color: "var(--text-dim)", borderTop: "1px solid var(--border)" }}>
        {data.disclaimer}
      </p>
    </div>
  )
}

function Th({ label, align }: { label: string; align: "left" | "right" }) {
  return (
    <th
      className={`py-2 text-xs tracking-widest font-normal ${align === "right" ? "text-right" : "text-left"}`}
      style={{ color: "var(--text-muted)" }}
    >
      {label}
    </th>
  )
}

function ModelHealthBadge({ health }: { health: ModelHealthResponse }) {
  const cfg = {
    HEALTHY:         { label: "HEALTHY",         color: "var(--accent)",  bg: "rgba(0,229,160,0.07)",  border: "rgba(0,229,160,0.2)"  },
    DEGRADED:        { label: "DEGRADED",        color: "#fbbf24",        bg: "rgba(251,191,36,0.07)", border: "rgba(251,191,36,0.2)" },
    CIRCUIT_BREAKER: { label: "MODEL DEGRADED",  color: "#f87171",        bg: "rgba(248,113,113,0.07)",border: "rgba(248,113,113,0.2)"},
    NO_DATA:         { label: "AWAITING DATA",   color: "var(--text-dim)",bg: "transparent",           border: "var(--border)"        },
  }[health.gate_status] ?? { label: health.gate_status, color: "var(--text-dim)", bg: "transparent", border: "var(--border)" }

  const ic = health.rolling_ic !== null
    ? `IC: ${health.rolling_ic.toFixed(3)}`
    : health.training_ic !== null
      ? `IC: ${health.training_ic.toFixed(3)} (train)`
      : null

  return (
    <span
      className="flex items-center gap-1.5 px-2 py-0.5 rounded text-xs tracking-wider"
      style={{ background: cfg.bg, border: `1px solid ${cfg.border}`, color: cfg.color }}
    >
      <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: cfg.color }} />
      {cfg.label}{ic ? <span className="font-mono"> · {ic}</span> : ""}
    </span>
  )
}

function Placeholder({ label }: { label: string }) {
  return (
    <div className="flex items-center justify-center rounded"
      style={{ height: 120, border: "1px dashed var(--border-bright)", background: "rgba(0,0,0,0.2)" }}>
      <p className="text-xs tracking-widest" style={{ color: "var(--text-dim)" }}>{label}</p>
    </div>
  )
}
