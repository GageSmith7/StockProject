# Portfolio Simulator — Full Architecture

> Feed this document to Claude Code to scaffold the project. Each section contains the
> purpose, file structure, implementation details, and acceptance criteria for that layer.
> Follow the build order exactly — do not skip ahead.

---

## Project Overview

A web application that allows users to build a stock portfolio, view historical performance
against the S&P 500, and run Monte Carlo simulations to project a range of future outcomes.
The product is honest about uncertainty — it shows confidence intervals, not point predictions.

**Core philosophy:** Clean data → rigorous feature engineering → simple validated models → honest UI

---

## Tech Stack

| Layer | Technology | Reason |
|---|---|---|
| Data collection | yFinance, FRED API | Free, reliable, no auth required for MVP |
| Data storage | PostgreSQL | Open source, ACID compliant, runs on every cloud |
| Task scheduling | APScheduler | Lightweight, no Celery overhead for MVP |
| Python backend | FastAPI (Python 3.11) | Async, fast, auto-generates API docs |
| Data processing | pandas, numpy | Standard financial data stack |
| ML | scikit-learn, statsmodels | Interpretable, walk-forward validated |
| API proxy (BFF) | Next.js API routes (TypeScript) | Thin layer between UI and Python service |
| Frontend | Next.js 14 + TypeScript + Recharts | Type safety, SSR, good charting |
| Styling | Tailwind CSS | Rapid UI development |
| Containerization | Docker + docker-compose | Same image runs locally and on AWS |
| Frontend hosting | Vercel | Native Next.js, free tier, one-command deploy |
| Backend hosting | AWS EC2 or ECS Fargate | Resume value, cost effective |
| Database hosting | AWS RDS (PostgreSQL) | Managed, free tier 12 months |
| Model storage | AWS S3 | Versioned model artifact storage |

---

## Architecture Overview

```
Browser
  │
  ▼
Next.js Frontend (TypeScript)        ← Vercel
  │  UI components, state, charts
  │
  ▼
Next.js API Routes /api/*            ← Vercel (BFF layer — thin proxy only)
  │  TypeScript — no business logic here
  │
  ▼
FastAPI Service (Python)             ← AWS EC2 / ECS Fargate
  │  Data pipeline, analytics, ML
  │
  ├──► PostgreSQL                    ← AWS RDS (local: Docker container)
  │    prices, macro_data, symbols
  │
  └──► AWS S3                        ← Model artifact storage
       trained model files

External sources (outbound only):
  FastAPI ──► yFinance   (price data)
  FastAPI ──► FRED API   (macro data)
```

**Key separation:** Next.js API routes are a thin proxy only — they forward requests to
FastAPI and return the response. All business logic, data access, and ML lives in Python.

---

## Local vs Production

The application runs identically in both environments. Only environment variables change —
zero code changes required to go from local to AWS.

| Service | Local Dev | Production |
|---|---|---|
| Next.js frontend | localhost:3000 | Vercel (auto-deploy on git push) |
| FastAPI backend | localhost:8000 (Docker) | AWS EC2 / ECS Fargate |
| PostgreSQL | Docker container | AWS RDS free tier |
| Model files | ./models/ directory | AWS S3 bucket |
| Env config | .env file | AWS Secrets Manager |

**The bridge is Docker.** Build the FastAPI image locally, test it, push the same image
to AWS. "Works on my machine" is not a problem when you ship the machine.

---

## Ticker Data Strategy

Tickers are served from three tiers. This balances storage cost, response speed, and coverage.

```
Tier 1 — Pre-loaded (fast, always ready)
  S&P 500 (~503 tickers) + major ETFs: SPY, QQQ, VTI, IWM, GLD
  Loaded on first run via symbols.py
  Refreshed daily by scheduler at 6pm ET

Tier 2 — On-demand, then cached (slow first load, instant after)
  Any ticker outside Tier 1 that yFinance recognises
  First search: fetch → clean → store → return (10–30 seconds, show loading state)
  Subsequent searches: already in DB, returns instantly
  data_source = 'on_demand' in symbols table

Tier 3 — Not found
  yFinance returns no data
  Return clean 404 error to UI: "Ticker not found or insufficient history"
```

**Frontend loading state for on-demand tickers:**
When a Tier 2 fetch is in progress, show:
"First time loading [TICKER] — fetching and processing historical data.
This may take up to 30 seconds."
This is honest and acceptable. Users understand cold starts.

---

## Directory Structure

```
portfolio-simulator/
│
├── backend/                              # Python service — all data + ML logic
│   ├── app/
│   │   ├── main.py                       # FastAPI entry point, lifespan, routers
│   │   ├── config.py                     # Settings via pydantic-settings, reads .env
│   │   ├── database.py                   # SQLAlchemy engine, session factory
│   │   │
│   │   ├── data/
│   │   │   ├── fetcher.py                # yFinance + FRED API calls
│   │   │   ├── cleaner.py                # Adjust, gap-fill, flag outliers
│   │   │   ├── store.py                  # Upsert to DB, get_prices(), get_or_fetch()
│   │   │   ├── scheduler.py              # APScheduler daily refresh job
│   │   │   └── symbols.py                # Load S&P 500 list, manage symbol registry
│   │   │
│   │   ├── analytics/
│   │   │   ├── features.py               # Feature engineering — returns, vol, momentum
│   │   │   ├── portfolio.py              # Weighted returns, Sharpe, drawdown, CAGR
│   │   │   ├── benchmark.py              # SPY fetch + comparison
│   │   │   ├── montecarlo.py             # Bootstrap simulation, percentile cone
│   │   │   ├── metrics.py                # Beta, correlation, Calmar ratio
│   │   │   └── validation.py             # Walk-forward backtesting framework
│   │   │
│   │   ├── models/
│   │   │   ├── predictor.py              # Train Ridge model, walk-forward CV, inference
│   │   │   └── registry.py               # Save/load model artifacts, versioning
│   │   │
│   │   └── routes/
│   │       ├── portfolio.py              # POST /portfolio
│   │       ├── simulate.py               # POST /simulate
│   │       ├── search.py                 # GET /search
│   │       ├── predict.py                # POST /predict
│   │       └── health.py                 # GET /health
│   │
│   ├── tests/
│   │   ├── test_fetcher.py
│   │   ├── test_cleaner.py
│   │   ├── test_features.py              # Critical — verify no lookahead bias
│   │   ├── test_portfolio.py
│   │   ├── test_montecarlo.py
│   │   └── test_validation.py
│   │
│   ├── models/                           # Saved model artifacts (gitignored, synced to S3)
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/                             # Next.js app — TypeScript throughout
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx                  # Home page
│   │   │   ├── layout.tsx                # Root layout
│   │   │   └── api/                      # Next.js API routes (BFF proxy layer)
│   │   │       ├── portfolio/route.ts    # Proxies to FastAPI /portfolio
│   │   │       ├── simulate/route.ts     # Proxies to FastAPI /simulate
│   │   │       ├── search/route.ts       # Proxies to FastAPI /search
│   │   │       └── predict/route.ts      # Proxies to FastAPI /predict
│   │   │
│   │   ├── components/
│   │   │   ├── PortfolioBuilder.tsx      # Ticker search + weight sliders
│   │   │   ├── HistoricalChart.tsx       # Portfolio vs SPY line chart
│   │   │   ├── ProjectionCone.tsx        # Monte Carlo cone (p10/p50/p90)
│   │   │   ├── MetricsPanel.tsx          # Sharpe, CAGR, drawdown stat cards
│   │   │   └── TickerSearch.tsx          # Debounced autocomplete input
│   │   │
│   │   ├── hooks/
│   │   │   ├── usePortfolio.ts           # Portfolio state + fetch logic
│   │   │   └── useSimulation.ts          # Simulation fetch + loading state
│   │   │
│   │   ├── types/
│   │   │   └── api.ts                    # TypeScript interfaces for all API responses
│   │   │
│   │   └── lib/
│   │       └── apiClient.ts              # Typed fetch wrapper, base URL from env
│   │
│   ├── package.json
│   ├── tsconfig.json
│   └── Dockerfile
│
├── .github/
│   └── workflows/
│       └── deploy.yml                    # CI/CD pipeline — built last
│
├── docker-compose.yml                    # Local dev: Postgres + FastAPI
├── .env.example                          # All required env vars documented
└── README.md
```

---

## Build Order

**Follow this strictly. Each step must pass its acceptance criteria before moving forward.**

```
── PYTHON / DATA / ML ──────────────────────────────────────────────────────────
Step  1   data/fetcher.py          Verify AAPL downloads, FRED series pulls
Step  2   data/cleaner.py          Verify no NaN, quality flags on every row
Step  3   data/store.py + schema   Verify upsert, no duplicates, get_prices() works
Step  4   data/symbols.py          Verify S&P 500 list loads and seeds DB on first run
Step  5   data/store.py            Add get_or_fetch() — on-demand tier logic
Step  6   data/scheduler.py        Verify daily job runs, failures do not stop the run
Step  7   analytics/features.py    Verify no lookahead bias, zero NaN after warmup
Step  8   analytics/portfolio.py   Verify metrics match manual calculation
Step  9   analytics/montecarlo.py  Verify p10 < p50 < p90 at all time points
Step 10   models/predictor.py      Verify walk-forward CV, IC > 0 on held-out data
Step 11   routes/ (all)            Verify all endpoints return correct JSON shape

── DOCKER ──────────────────────────────────────────────────────────────────────
Step 12   Dockerfile + compose     docker-compose up starts everything cleanly
                                   API responds at localhost:8000/health

── FRONTEND ────────────────────────────────────────────────────────────────────
Step 13   types/api.ts             Define all TypeScript interfaces before any components
Step 14   lib/apiClient.ts         Typed fetch wrapper returning stub data
Step 15   TickerSearch.tsx         Autocomplete works against stub search results
Step 16   PortfolioBuilder.tsx     Weight assignment, validation, submit works with stubs
Step 17   HistoricalChart.tsx      Chart renders correctly with stub portfolio data
Step 18   ProjectionCone.tsx       Cone renders, disclaimer always visible
Step 19   MetricsPanel.tsx         All stat cards render, colour coding correct

── CONNECT ─────────────────────────────────────────────────────────────────────
Step 20   Next.js API routes       Swap stubs for real FastAPI proxy calls
Step 21   Integration test         Full flow with real tickers, edge cases, bad inputs
Step 22   Loading + error states   On-demand ticker UX, errors, empty states

── DEPLOY ──────────────────────────────────────────────────────────────────────
Step 23   AWS RDS                  Provision Postgres instance, update DATABASE_URL
Step 24   AWS EC2 / ECS            Deploy FastAPI Docker image, verify /health responds
Step 25   Vercel                   Connect GitHub repo, set FASTAPI_URL env var
Step 26   CI/CD pipeline           GitHub Actions: test → build → deploy on push to main
```

**Why stubs before real API (Steps 13–19):**
Build the entire frontend against hardcoded JSON that matches the API response shape.
UI work is never blocked waiting on the Python layer, and schema mismatches surface
at Step 20 in one place rather than being scattered across frontend debugging.

---

## Layer 1 — Data Collection

### File: `backend/app/data/fetcher.py`

**Responsibilities:**
- Fetch daily OHLCV data for a given ticker using yFinance
- Fetch macro series from FRED API
- Retry on failure with exponential backoff
- Never raise on network failure — log and return empty DataFrame

**Implementation notes:**
- Use `yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)`
- `auto_adjust=True` handles splits and dividends — do not implement manually
- Retry up to 3 times with 2s, 4s, 8s delays before giving up
- FRED series for MVP: `DFF` (Fed Funds Rate), `CPIAUCSL` (CPI), `UNRATE` (Unemployment)
- Output columns: `date`, `open`, `high`, `low`, `close`, `volume`, `ticker`

**Acceptance criteria:**
- `fetch_ticker("AAPL", "2020-01-01", "2024-01-01")` returns DataFrame, zero NaN in close
- Failed ticker logs warning and returns empty DataFrame without raising
- FRED fetch returns DataFrame with `date` and `value` columns

---

### File: `backend/app/data/cleaner.py`

**Responsibilities:**
- Forward-fill gaps up to 3 consecutive missing trading days
- Flag (do not remove) single-day price moves greater than 15%
- Attach `data_quality_flag` to every row

**Implementation notes:**
- `data_quality_flag` values: `clean`, `forward_filled`, `outlier_flagged`
- Gaps longer than 3 days: log warning with ticker + date range, do not fill
- Never delete rows — flag and let the model decide
- Splits and dividends already handled by yFinance `auto_adjust=True`

**Acceptance criteria:**
- Zero NaN values in `close` column after cleaning
- `data_quality_flag` present on every row
- Outlier rows are present with flag, not removed

---

### File: `backend/app/data/store.py`

**Responsibilities:**
- Upsert cleaned price data — insert new rows, update existing, never duplicate
- Expose `get_prices(ticker, start, end)` for the analytics layer
- Expose `get_or_fetch(ticker)` for on-demand tier logic

**`get_or_fetch()` implementation:**
```python
def get_or_fetch(ticker: str) -> pd.DataFrame:
    """
    Check DB first. If found, return immediately (fast path).
    If not found: fetch from yFinance, clean, store, return (slow path).
    If yFinance returns nothing: raise TickerNotFoundError.
    Mark on-demand tickers with data_source = 'on_demand' in symbols table.
    """
```

**Database schema:**
```sql
CREATE TABLE prices (
    date          DATE          NOT NULL,
    ticker        VARCHAR(10)   NOT NULL,
    open          NUMERIC(12,4),
    high          NUMERIC(12,4),
    low           NUMERIC(12,4),
    close         NUMERIC(12,4) NOT NULL,
    volume        BIGINT,
    quality_flag  VARCHAR(20),
    updated_at    TIMESTAMP     DEFAULT NOW(),
    PRIMARY KEY (date, ticker)
);

CREATE TABLE macro_data (
    date          DATE         NOT NULL,
    series_id     VARCHAR(20)  NOT NULL,
    value         NUMERIC(12,4),
    PRIMARY KEY (date, series_id)
);

CREATE TABLE symbols (
    ticker        VARCHAR(10)   PRIMARY KEY,
    name          VARCHAR(200),
    sector        VARCHAR(100),
    active        BOOLEAN       DEFAULT TRUE,
    data_source   VARCHAR(20)   DEFAULT 'preloaded',   -- 'preloaded' | 'on_demand'
    last_fetched  TIMESTAMP,
    added_at      TIMESTAMP     DEFAULT NOW()
);
```

**Acceptance criteria:**
- Upserting same ticker + date twice produces one row, not two
- `get_prices("AAPL", "2020-01-01", "2024-01-01")` returns date-sorted DataFrame
- `get_or_fetch("HIMS")` hits yFinance on first call, DB on second call
- 500 rows inserted in under 2 seconds

---

### File: `backend/app/data/symbols.py`

**Responsibilities:**
- Load S&P 500 ticker list on first run and seed the symbols table
- Expose `search_symbols(query)` for the autocomplete endpoint
- Manage Tier 1 vs Tier 2 symbol classification

**Implementation notes:**
- S&P 500 list: `pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")`
  — reliable, free, no API key needed
- Pre-load these ETFs alongside S&P 500: `SPY`, `QQQ`, `VTI`, `IWM`, `GLD`
- `search_symbols(query)` searches both `ticker` and `name` columns, returns top 10
- On-demand tickers are inserted into symbols table with `data_source = 'on_demand'`
  automatically when first fetched via `get_or_fetch()`

**Acceptance criteria:**
- First run seeds ~508 symbols (503 S&P 500 + 5 ETFs)
- `search_symbols("APP")` returns AAPL, APPF, and other matches
- Re-running seed does not create duplicates

---

### File: `backend/app/data/scheduler.py`

**Responsibilities:**
- Daily job at 6pm ET to refresh all active symbols (Tier 1 and any cached Tier 2)
- Fetch only last 5 trading days per refresh — not full history
- Log per-ticker success/failure without stopping the run

**Implementation notes:**
- Use APScheduler `BackgroundScheduler`
- Start in `main.py` lifespan context
- If more than 10 tickers fail in one run, log a CRITICAL alert
- Manual trigger available for development

**Acceptance criteria:**
- Scheduler starts when FastAPI starts
- One ticker failure does not abort the remaining tickers
- Manual trigger available via `POST /admin/refresh`

---

## Layer 2 — Feature Engineering

### File: `backend/app/analytics/features.py`

**Purpose:** Transform raw prices into model-ready features. Model quality is determined
here — not in the ML code. This is the most important file in the project.

**Features to engineer:**

| Feature | Calculation | Window |
|---|---|---|
| `return_1d` | Daily log return | 1 day |
| `return_5d` | 5-day cumulative return | 5 days |
| `return_20d` | 20-day cumulative return | 20 days |
| `return_60d` | 60-day cumulative return | 60 days |
| `volatility_20d` | Rolling std of daily returns | 20 days |
| `volatility_60d` | Rolling std of daily returns | 60 days |
| `momentum_12_1` | 12-month return minus last 1 month | 252 − 21 days |
| `ma_cross_50_200` | 50-day MA / 200-day MA ratio | 200 days |
| `volume_ratio` | Today volume / 20-day avg volume | 20 days |
| `relative_strength` | Ticker return / SPY return | 20 days |

**Target variable:**
```python
# Forward 20-day return — what the model predicts
target_return_20d = close.shift(-20) / close - 1
# NaN for the last 20 rows — correct, future is unknown
```

**Critical — lookahead bias prevention:**
- Every feature must use `.shift(1)` so features only contain information available
  at prediction time. Features reflect yesterday's data, not today's.
- Drop the first 200 rows after calculation — rolling windows produce NaN during warmup
- Calculate features per ticker independently

**Acceptance criteria:**
- No future information in any feature column (verified by `.shift(1)` on every feature)
- Zero NaN in feature matrix after dropping warmup period
- `target_return_20d` is NaN for the final 20 rows of each ticker (correct behaviour)

---

## Layer 3 — Portfolio Analytics

### File: `backend/app/analytics/portfolio.py`

```python
def portfolio_returns(
    prices: dict[str, pd.Series],
    weights: dict[str, float]
) -> pd.Series:
    """
    Weighted daily portfolio returns.
    Weights must sum to 1.0 — raise ValueError if not.
    Align all series to common trading days before weighting.
    """

def portfolio_metrics(returns: pd.Series) -> dict:
    """
    Returns:
      cagr                annualised compound growth rate
      volatility          annualised (daily_vol * sqrt(252))
      sharpe_ratio        (mean_daily * 252 - risk_free) / (vol * sqrt(252))
      max_drawdown        negative — worst peak-to-trough
      calmar_ratio        cagr / abs(max_drawdown)
      total_return        total % gain over full period
    risk_free rate: pull latest DFF value from macro_data table
    """

def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown at each point in time — used for chart overlay"""
```

---

### File: `backend/app/analytics/montecarlo.py`

```python
def run_simulation(
    historical_returns: pd.Series,
    years: int,
    n_simulations: int = 1000,
    initial_value: float = 10000
) -> dict:
    """
    Bootstrap sampling — sample with replacement from historical daily returns.
    Do NOT assume normality. Real returns are fat-tailed.
    1000 simulations is sufficient — 10,000 is overkill for MVP.
    Return monthly snapshots only — daily is too many data points for the chart.

    Returns:
      time_axis       list[str]    monthly labels e.g. ["2024-01", "2024-02"]
      percentiles:
        p10           list[float]  pessimistic path
        p50           list[float]  median path
        p90           list[float]  optimistic path
      disclaimer      str          always included, always shown in UI
    """
```

**Acceptance criteria:**
- p10 < p50 < p90 at every time point
- p50 roughly matches historical CAGR compounded forward
- 1000 simulations over 10 years completes in under 2 seconds
- Disclaimer string always present in response — UI must always render it

---

## Layer 4 — ML Model

### File: `backend/app/models/predictor.py`

**Model for MVP: Ridge Regression**
Interpretable coefficients map directly to feature importance. A quant interviewer can
ask "why does your model weight momentum?" and you can answer concretely.
Add Random Forest and XGBoost post-MVP via the model registry.

**Training scope: cross-sectional**
Each training row is one ticker on one date. The model learns what features predict
returns across all stocks simultaneously. This enables cross-ticker ranking (top quintile)
and requires only one model artifact. Features must be sector-normalized before training
to prevent sector membership from dominating the signal:
```python
# In features.py — apply before passing to train_model()
features[col] = features.groupby("sector")[col].transform(
    lambda x: (x - x.mean()) / x.std()
)
```

**Training pipeline:**
```python
def train_model(features: pd.DataFrame, target: pd.Series) -> tuple[Ridge, dict]:
    """
    Cross-sectional training — rows are (ticker, date) pairs across all tickers.
    Walk-forward cross-validation. NEVER use random k-fold on time series.
    sklearn cross_val_score with shuffle=True causes lookahead bias — do not use it.
    Use TimeSeriesSplit or custom walk-forward loop.

    Split strategy:
      Minimum 2 years training data
      Validate on next 6 months
      Slide 6 months forward, repeat
      Report IC per fold and average IC
      IC is calculated cross-sectionally: corr(predicted_rank, actual_rank) per date

    Returns: (fitted_model, validation_metrics_dict)
    """

def predict(model: Ridge, features: pd.DataFrame) -> pd.Series:
    """
    Predicted 20-day forward return per ticker.
    Input features must be sector-normalized (same transform as training).
    Output is used to rank all tickers — higher score = more favorable outlook.
    Tier 2 tickers get predictions immediately; no per-ticker training required.
    """
```

**Validation metrics:**
- IC (Information Coefficient): correlation of predicted vs actual returns
  IC > 0.05 meaningful. IC < 0 means model is worse than random — retrain.
- Directional accuracy: % of predictions where direction matches actual
- Sharpe of long-top-quintile strategy: does the ranking produce real returns?

**IC gate:** Do not expose model via API if average IC < 0.03. Return a
`model_unavailable` flag in the predict response instead of bad predictions.

**Acceptance criteria:**
- Model never trained on data after validation period start date
- IC reported per fold
- Model saved to `models/` with timestamp and metrics JSON alongside it

---

### File: `backend/app/models/registry.py`

**Responsibilities:**
- Save trained model + metrics to `models/` directory with ISO timestamp in filename
- Load latest valid model on FastAPI startup
- Post-MVP: sync to and load from AWS S3

---

## Layer 5 — API Layer

### File: `backend/app/main.py`

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.data.scheduler import start_scheduler
from app.routes import portfolio, simulate, search, predict, health
from app.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    start_scheduler()
    yield

app = FastAPI(title="Portfolio Simulator API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(portfolio.router, prefix="/api")
app.include_router(simulate.router, prefix="/api")
app.include_router(search.router, prefix="/api")
app.include_router(predict.router, prefix="/api")
app.include_router(health.router, prefix="/api")
```

---

### Route Specifications

Define TypeScript interfaces in `types/api.ts` first. Python Pydantic models must
match those interfaces exactly — this is the contract between frontend and backend.

**GET /api/health**
```json
{ "status": "ok", "db": "connected", "scheduler": "running" }
```

**GET /api/search?q=APP**
```json
{
  "results": [
    { "ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology", "tier": "preloaded" },
    { "ticker": "APPF", "name": "AppFolio Inc.", "sector": "Technology", "tier": "preloaded" }
  ]
}
```

**POST /api/portfolio**
```json
// Request
{
  "holdings": [
    { "ticker": "AAPL", "weight": 0.6 },
    { "ticker": "MSFT", "weight": 0.4 }
  ],
  "start_date": "2018-01-01",
  "end_date": "2024-01-01"
}

// Response
{
  "portfolio_returns": [{ "date": "2018-01-02", "value": 10000.00 }],
  "benchmark_returns": [{ "date": "2018-01-02", "value": 10000.00 }],
  "metrics": {
    "cagr": 0.142,
    "volatility": 0.187,
    "sharpe_ratio": 0.89,
    "max_drawdown": -0.312,
    "total_return": 1.24
  },
  "on_demand_tickers": []
}
```

**POST /api/simulate**
```json
// Request
{
  "holdings": [{ "ticker": "AAPL", "weight": 0.6 }, { "ticker": "MSFT", "weight": 0.4 }],
  "years": 10,
  "initial_value": 10000
}

// Response
{
  "time_axis": ["2024-01", "2024-02"],
  "percentiles": {
    "p10": [10000, 9800],
    "p50": [10000, 10120],
    "p90": [10000, 10450]
  },
  "disclaimer": "Projections based on historical return distribution. Past performance does not predict future results."
}
```

**POST /api/predict**
```json
// Request
{ "tickers": ["AAPL", "MSFT", "GOOGL"] }

// Response
{
  "predictions": [
    { "ticker": "AAPL", "predicted_return_20d": 0.034 },
    { "ticker": "MSFT", "predicted_return_20d": 0.021 }
  ],
  "model_version": "ridge_v1_20240315",
  "model_available": true,
  "disclaimer": "ML predictions are experimental and not financial advice."
}
```

---

## Layer 6 — Frontend (TypeScript)

### Define types first — before any components

**`frontend/src/types/api.ts`**
```typescript
export interface Holding {
  ticker: string
  weight: number        // 0.0 to 1.0
}

export interface PortfolioRequest {
  holdings: Holding[]
  start_date: string
  end_date: string
}

export interface DataPoint {
  date: string
  value: number
}

export interface PortfolioMetrics {
  cagr: number
  volatility: number
  sharpe_ratio: number
  max_drawdown: number
  total_return: number
}

export interface PortfolioResponse {
  portfolio_returns: DataPoint[]
  benchmark_returns: DataPoint[]
  metrics: PortfolioMetrics
  on_demand_tickers: string[]
}

export interface SimulationResponse {
  time_axis: string[]
  percentiles: {
    p10: number[]
    p50: number[]
    p90: number[]
  }
  disclaimer: string
}

export interface SearchResult {
  ticker: string
  name: string
  sector: string
  tier: 'preloaded' | 'on_demand'
}

export interface PredictionResponse {
  predictions: Array<{ ticker: string; predicted_return_20d: number }>
  model_version: string
  model_available: boolean
  disclaimer: string
}
```

---

### Component Specifications

**`TickerSearch.tsx`**
- Debounced input (300ms) calling `/api/search`
- Show sector alongside each result
- Results with `tier: 'on_demand'` show "⚡ Live fetch" badge to set expectations
- On select: add ticker to portfolio holdings list

**`PortfolioBuilder.tsx`**
- List of selected holdings with weight input per ticker (0–100, integer)
- Running weight total shown as progress bar — red if not 100%, green if exactly 100%
- "Run Analysis" button disabled until weights sum to exactly 100
- If `on_demand_tickers` in response is non-empty, show loading banner:
  "Fetching [TICKER] for the first time — this may take up to 30 seconds"

**`HistoricalChart.tsx`**
- Recharts `LineChart` — portfolio (`#00D4FF`) vs SPY benchmark (`#666`)
- Y-axis: portfolio value indexed to $10,000 at start date
- Date range selector: 1Y, 3Y, 5Y, 10Y, MAX
- Hover tooltip shows both values on same date

**`ProjectionCone.tsx`**
- Recharts `AreaChart` — shaded band between p10 and p90, solid line for p50
- Time horizon selector: 5Y, 10Y, 20Y, 30Y
- p50 line labeled "Median outcome"
- Disclaimer always rendered below chart — never collapsible, never hidden

**`MetricsPanel.tsx`**
- Stat cards: CAGR, Volatility, Sharpe Ratio, Max Drawdown
- Sharpe colour coding: green >= 1.0, yellow 0.5–1.0, red < 0.5
- Max drawdown shown in red as negative percentage
- Each card has a tooltip with plain-English explanation of the metric

---

## Docker Setup

```yaml
# docker-compose.yml — local development only
version: '3.8'

services:
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: portfolio_sim
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d portfolio_sim"]
      interval: 5s
      retries: 5

  backend:
    build: ./backend
    ports:
      - "8000:8000"
    env_file: .env
    depends_on:
      db:
        condition: service_healthy
    volumes:
      - ./backend:/app
      - ./backend/models:/app/models

volumes:
  pgdata:
```

**Note:** Run Next.js separately with `npm run dev`. Only Postgres and FastAPI
run in Docker during local development.

---

## Environment Variables

```bash
# .env.example — copy to .env, never commit .env

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/portfolio_sim

# External APIs
FRED_API_KEY=your_key_here          # free at fred.stlouisfed.org

# App config
ENVIRONMENT=development             # development | production
CORS_ORIGINS=http://localhost:3000
MODEL_DIR=./models
LOG_LEVEL=INFO

# AWS — production only, use AWS Secrets Manager not .env
AWS_REGION=us-east-1
S3_BUCKET_MODELS=portfolio-sim-models

# Next.js frontend (.env.local)
NEXT_PUBLIC_API_URL=http://localhost:3000   # points to Next.js API routes
FASTAPI_URL=http://localhost:8000           # server-side only, never exposed to browser
```

---

## CI/CD Pipeline (Step 26 — build last)

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Python tests
        run: |
          pip install -r backend/requirements.txt
          pytest backend/tests/
      - name: Run TypeScript type check
        run: |
          cd frontend && npm ci && npx tsc --noEmit

  deploy-backend:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Build Docker image
      - name: Push to AWS ECR
      - name: Deploy to ECS / restart EC2 service

  deploy-frontend:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Vercel
        run: vercel --prod
```

Vercel can also connect directly to GitHub and auto-deploy on push to main
without a pipeline step — either approach works.

---

## Known Risks and Mitigations

| Risk | Mitigation |
|---|---|
| yFinance rate limiting | Exponential backoff, cache aggressively, never re-fetch if data is in DB |
| Lookahead bias in features | `.shift(1)` on all features, `TimeSeriesSplit` for CV, dedicated test file |
| Survivorship bias | **Known MVP limitation.** symbols.py loads today's S&P 500 list, which excludes delisted stocks. Model trains only on survivors, which inflates predicted returns. Post-MVP fix: use historical membership list. |
| Overfitting | Ridge regularisation + walk-forward CV + IC gate (don't serve model if IC < 0.03) |
| On-demand fetch timeout | 30s timeout, loading state in UI, cache result immediately on success |
| CORS issues | CORSMiddleware in main.py, CORS_ORIGINS read from env var |
| TypeScript / Python schema mismatch | Define types/api.ts first, Pydantic models must match exactly |
| Docker DB not ready on startup | healthcheck on db service, depends_on condition: service_healthy |

---

## Post-MVP Extensions (do not build now)

- User accounts and saved portfolios — NextAuth.js, users + saved_portfolios tables
- Stress testing — simulate 2008, 2020, dot-com crash scenarios on the portfolio
- Factor decomposition — quantify how much return is market beta vs stock selection
- Multi-ticker ML ranking — rank all S&P 500 stocks by predicted 20-day return
- FRED macro overlay — plot rate hike cycles on the historical chart
- Export to CSV — download full performance history
- Premium tier — longer history, more benchmarks, unlimited saved portfolios

---

*Architecture version 2.0 — TypeScript + Next.js BFF, FastAPI Python service,
PostgreSQL, Docker-first local dev, AWS deployment, hybrid ticker strategy (pre-loaded
S&P 500 + on-demand cache), stub-first frontend build order*
