// ---------------------------------------------------------------------------
// Shared
// ---------------------------------------------------------------------------

export interface Holding {
  ticker: string
  weight: number // 0.0 – 1.0
}

// ---------------------------------------------------------------------------
// GET /api/search
// ---------------------------------------------------------------------------

export interface SearchResult {
  ticker: string
  name:   string
  sector: string | null
  tier:   "preloaded" | "on_demand"
}

export interface SearchResponse {
  results: SearchResult[]
}

// ---------------------------------------------------------------------------
// POST /api/portfolio
// ---------------------------------------------------------------------------

export interface PortfolioRequest {
  holdings:   Holding[]
  start_date: string // YYYY-MM-DD
  end_date?:  string // YYYY-MM-DD, defaults to today
}

export interface DataPoint {
  date:  string // YYYY-MM-DD
  value: number // portfolio value indexed to $10,000
}

/** Matches compute_metrics() output — keys are exactly what Python returns */
export interface PortfolioMetrics {
  total_return: number
  cagr:         number
  sharpe:       number
  max_drawdown: number
  calmar:       number
  n_periods:    number
  win_rate:     number
  // benchmark fields — present when SPY data is available
  benchmark_cagr?:      number
  alpha?:               number
  beta?:                number
  information_ratio?:   number
}

export interface PortfolioResponse {
  portfolio_returns:  DataPoint[]
  benchmark_returns:  DataPoint[]
  metrics:            PortfolioMetrics
  on_demand_tickers:  string[]
}

// ---------------------------------------------------------------------------
// POST /api/simulate
// ---------------------------------------------------------------------------

export interface SimulateRequest {
  holdings:             Holding[]
  years?:               number  // 1–30, default 10
  n_simulations?:       number  // 100–10000, default 1000
  initial_value?:       number  // default 10000
  start_date?:          string  // default "2015-01-01"
  monthly_contribution?: number // dollars added each month, default 0
  use_regime?:          boolean // condition bootstrap on current market regime
}

export interface SimulateResponse {
  time_axis:       string[]           // ["2026-03", "2026-04", …]
  percentiles: {
    p10: number[]
    p50: number[]
    p90: number[]
  }
  disclaimer:          string
  backtest_metrics:    PortfolioMetrics
  current_regime?:     "expansion" | "contraction" | null
  regime_fallback?:    boolean
}

// ---------------------------------------------------------------------------
// POST /api/predict
// ---------------------------------------------------------------------------

export interface PredictRequest {
  tickers?: string[] // null = all preloaded tickers
}

export interface PredictionItem {
  ticker:               string
  predicted_return_21d: number // NOTE: 21d, not 20d
  rank:                 number
}

export interface PredictResponse {
  predictions:     PredictionItem[]
  as_of_date:      string
  model_version:   string
  model_available: boolean
  disclaimer:      string
}

// ---------------------------------------------------------------------------
// GET /api/model/health
// ---------------------------------------------------------------------------

export interface ModelHealthResponse {
  gate_status:          "HEALTHY" | "DEGRADED" | "CIRCUIT_BREAKER" | "NO_DATA"
  rolling_ic:           number | null   // live Spearman IC from prediction_log
  training_ic:          number | null   // IC at training time from registry
  n_samples:            number | null
  consecutive_failures: number
  retrain_result:       "DEPLOYED" | "REJECTED" | null
  model_version:        string | null
  model_artifact:       string | null
  last_checked:         string | null   // ISO datetime
}

// ---------------------------------------------------------------------------
// POST /api/optimize
// ---------------------------------------------------------------------------

export type OptimizeGoal = "sharpe" | "min_vol" | "max_cagr"

export interface OptimizeRequest {
  tickers:        string[]
  goal:           OptimizeGoal
  lookback_years: number
}

export interface OptimizeResult {
  weights: Record<string, number>   // { "AAPL": 0.25, ... }
  metrics: {
    sharpe:       number
    cagr:         number
    max_drawdown: number
    volatility:   number
  }
  goal: OptimizeGoal
}

// ---------------------------------------------------------------------------
// GET /api/health
// ---------------------------------------------------------------------------

export interface HealthResponse {
  status:    "ok" | "error"
  db:        "connected" | "error"
  scheduler: "running" | "stopped"
}
