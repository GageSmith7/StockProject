"use client"

import { useState, useEffect, useRef } from "react"
import { apiClient } from "@/lib/apiClient"
import type { SearchResult } from "@/types/api"

// ----------------------------------------------------------------------------
// Props
// ----------------------------------------------------------------------------
// onSelect is a callback the parent gives us.
// When the user picks a ticker, we call onSelect(ticker) and the parent
// decides what to do with it (add it to the holdings list).
// TickerSearch doesn't own the holdings list — it just reports selections up.
// ----------------------------------------------------------------------------

interface Props {
  onSelect: (result: SearchResult) => void
  disabledTickers?: string[] // tickers already in the portfolio — greyed out
}

export default function TickerSearch({ onSelect, disabledTickers = [] }: Props) {
  const [query,     setQuery]     = useState("")       // what the user typed
  const [results,   setResults]   = useState<SearchResult[]>([])
  const [isOpen,    setIsOpen]    = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error,     setError]     = useState<string | null>(null)

  // We attach this ref to the wrapper <div> so we can detect clicks outside
  // and close the dropdown.
  const wrapperRef = useRef<HTMLDivElement>(null)

  // --------------------------------------------------------------------------
  // Debounce — wait 300ms after the user stops typing before calling the API.
  // Without this, every keystroke fires a request. With it, only the final
  // character in a pause triggers the fetch.
  // --------------------------------------------------------------------------
  useEffect(() => {
    if (query.trim().length === 0) {
      setResults([])
      setIsOpen(false)
      return
    }

    setIsLoading(true)
    setError(null)

    // setTimeout returns a timer ID we can cancel
    const timer = setTimeout(async () => {
      try {
        const data = await apiClient.search(query.trim(), 8)
        setResults(data.results)
        setIsOpen(true)
      } catch {
        setError("Search unavailable")
        setResults([])
      } finally {
        setIsLoading(false)
      }
    }, 300)

    // Cleanup: if the user types again before 300ms is up, cancel the old timer.
    // This is what makes it a true debounce — only the last timer fires.
    return () => clearTimeout(timer)
  }, [query]) // re-run whenever query changes

  // --------------------------------------------------------------------------
  // Close dropdown on outside click.
  // useEffect sets up a listener on the document. When a click happens, we
  // check if it was inside wrapperRef — if not, close the dropdown.
  // --------------------------------------------------------------------------
  useEffect(() => {
    function handleOutsideClick(e: MouseEvent) {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target as Node)) {
        setIsOpen(false)
      }
    }
    document.addEventListener("mousedown", handleOutsideClick)
    return () => document.removeEventListener("mousedown", handleOutsideClick)
  }, [])

  function handleSelect(result: SearchResult) {
    onSelect(result)      // tell the parent
    setQuery("")          // clear the input
    setIsOpen(false)      // close the dropdown
  }

  return (
    <div ref={wrapperRef} className="relative">
      {/* Input */}
      <div
        className="flex items-center gap-2 px-3 py-2 rounded"
        style={{
          background:   "var(--bg-card-2)",
          border:       "1px solid var(--border-bright)",
        }}
      >
        {/* Search icon */}
        <svg
          className="w-3.5 h-3.5 shrink-0"
          style={{ color: "var(--text-muted)" }}
          fill="none" stroke="currentColor" viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M21 21l-4.35-4.35M17 11A6 6 0 1 1 5 11a6 6 0 0 1 12 0z" />
        </svg>

        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value.toUpperCase())}
          placeholder="SEARCH TICKER OR COMPANY..."
          className="flex-1 bg-transparent outline-none text-sm tracking-widest placeholder:text-xs"
          style={{
            color:             "var(--text-primary)",
            caretColor:        "var(--accent)",
          }}
          // Also open the dropdown if there are cached results and user re-focuses
          onFocus={() => { if (results.length > 0) setIsOpen(true) }}
        />

        {/* Loading spinner */}
        {isLoading && (
          <div
            className="w-3.5 h-3.5 rounded-full border border-t-transparent animate-spin shrink-0"
            style={{ borderColor: "var(--accent)", borderTopColor: "transparent" }}
          />
        )}
      </div>

      {/* Error */}
      {error && (
        <p className="text-xs mt-1 tracking-wider" style={{ color: "#f87171" }}>
          {error}
        </p>
      )}

      {/* Dropdown */}
      {isOpen && results.length > 0 && (
        <ul
          className="absolute z-50 w-full mt-1 rounded overflow-hidden"
          style={{
            background: "var(--bg-card-2)",
            border:     "1px solid var(--border-bright)",
          }}
        >
          {results.map((r) => {
            const disabled = disabledTickers.includes(r.ticker)
            return (
              <li key={r.ticker}>
                <button
                  disabled={disabled}
                  onClick={() => handleSelect(r)}
                  className="w-full text-left px-3 py-2.5 flex items-center justify-between gap-3 transition-colors"
                  style={
                    disabled
                      ? { opacity: 0.35, cursor: "not-allowed" }
                      : { cursor: "pointer" }
                  }
                  onMouseEnter={(e) => {
                    if (!disabled)
                      (e.currentTarget as HTMLElement).style.background = "rgba(0,229,160,0.06)"
                  }}
                  onMouseLeave={(e) => {
                    (e.currentTarget as HTMLElement).style.background = "transparent"
                  }}
                >
                  {/* Left: ticker + name */}
                  <span className="flex items-center gap-3 min-w-0">
                    <span
                      className="text-sm tracking-widest shrink-0 font-mono"
                      style={{ color: disabled ? "var(--text-muted)" : "var(--accent)" }}
                    >
                      {r.ticker}
                    </span>
                    <span
                      className="text-xs truncate"
                      style={{ color: "var(--text-muted)" }}
                    >
                      {r.name}
                    </span>
                  </span>

                  {/* Right: sector + tier badge */}
                  <span className="flex items-center gap-2 shrink-0">
                    {r.sector && (
                      <span
                        className="text-xs hidden sm:block"
                        style={{ color: "var(--text-dim)" }}
                      >
                        {r.sector.toUpperCase()}
                      </span>
                    )}
                    {r.tier === "on_demand" && (
                      <span
                        className="text-xs px-1.5 py-0.5 rounded tracking-wider"
                        style={{
                          color:      "#fbbf24",
                          background: "rgba(251,191,36,0.08)",
                          border:     "1px solid rgba(251,191,36,0.2)",
                        }}
                        title="Data fetched live from yFinance on first use. Not in ML training set — no predictions available."
                      >
                        LIVE · NO ML
                      </span>
                    )}
                  </span>
                </button>
              </li>
            )
          })}
        </ul>
      )}

      {/* No results */}
      {isOpen && !isLoading && results.length === 0 && query.length > 0 && (
        <div
          className="absolute z-50 w-full mt-1 rounded px-3 py-3 text-xs tracking-wider"
          style={{
            background: "var(--bg-card-2)",
            border:     "1px solid var(--border-bright)",
            color:      "var(--text-muted)",
          }}
        >
          NO RESULTS FOR &quot;{query}&quot;
        </div>
      )}
    </div>
  )
}
