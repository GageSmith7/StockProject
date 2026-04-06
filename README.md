# Portfolio Analytics & Risk Engine

A full-stack quantitative finance platform built to demonstrate production-grade data engineering, statistical risk modeling, and machine learning validation. Covers the complete data lifecycle — collection, cleaning, feature engineering, model training, simulation, and monitoring across 500+ equities.

> **Audience note**: This project is structured to highlight skills in data pipelines, statistical inference, uncertainty quantification, and model governance.

---

## Table of Contents

- [What This Project Demonstrates](#what-this-project-demonstrates)
- [System Architecture](#system-architecture)
- [Data Engineering Pipeline](#data-engineering-pipeline)
- [Statistical & Actuarial Methods](#statistical--actuarial-methods)
- [Machine Learning & Model Governance](#machine-learning--model-governance)
- [Risk Metrics & Portfolio Analytics](#risk-metrics--portfolio-analytics)
- [Database Design](#database-design)
- [API Design](#api-design)
- [Tech Stack](#tech-stack)
- [Local Setup](#local-setup)
- [Testing](#testing)

---

## What This Project Demonstrates

### For Data Analytics Roles
- End-to-end ETL pipeline with automated scheduling, idempotent upserts, and tiered data sourcing
- Cross-sectional feature engineering (18 signals) across 500+ securities with sector-relative normalization
- Typed REST API layer (FastAPI + Pydantic) integrated with a typed frontend (TypeScript + Next.js)
- Audit-trail database design with quality flags on every observation

### For Actuarial Roles
- Block-bootstrap Monte Carlo simulation preserving volatility clustering — more realistic than IID approaches
- Regime-aware scenario generation (expansion vs. contraction environments)
- Percentile fan chart (p10 / p50 / p90) analogous to reserve range and loss distribution outputs
- Information Coefficient (IC) tracking as a model validation metric — analogous to Gini/lift in predictive models

### For Underwriting & Risk Roles
- Comprehensive risk metric suite: CAGR, Sharpe ratio, Calmar ratio, max drawdown, alpha, beta, information ratio
- Walk-forward backtesting that mirrors out-of-sample held-period testing used in pricing model validation
- Gated model deployment — model is rejected if IC falls below threshold (same principle as model risk management gates)
- Macro regime indicators (yield curve, CPI, Fed Funds rate) integrated as risk signals

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        External Data Sources                        │
│   yFinance (OHLCV)    FRED API (DFF, CPI, T10Y2Y)    Wikipedia     │
└────────────┬─────────────────────────┬──────────────────┬──────────┘
             │                         │                  │
             ▼                         ▼                  ▼
┌────────────────────────────────────────────────────────────────────┐
│                    Data Ingestion Layer (Python)                    │
│   fetcher.py — tenacity retry   •   scheduler.py — APScheduler     │
│   cleaner.py — gap fill / outlier flag   •   store.py — upsert     │
└─────────────────────────────────┬──────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────┐
│                     PostgreSQL 15 (Docker / RDS)                   │
│  prices  •  macro_data  •  symbols  •  prediction_log              │
│  model_metrics  •  constituent_history                              │
└──────┬──────────────┬───────────────────────┬──────────────────────┘
       │              │                       │
       ▼              ▼                       ▼
┌──────────┐  ┌──────────────────┐  ┌─────────────────────────────┐
│ features │  │ portfolio /      │  │  models/                    │
│ .py      │  │ benchmark /      │  │  predictor.py  LightGBM     │
│ 18 sigs  │  │ montecarlo /     │  │  registry.py   versioning   │
│ z-scored │  │ optimizer.py     │  │  health.py     IC tracking  │
└────┬─────┘  └──────┬───────────┘  └──────────┬──────────────────┘
     │               │                         │
     └───────────────┴─────────────────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │  FastAPI REST API (8 routes) │
              └──────────────┬───────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │  Next.js 14 + TypeScript     │
              │  Recharts · Tailwind CSS     │
              └──────────────────────────────┘
```

---

## Data Engineering Pipeline

### 1. Data Collection (`backend/app/data/fetcher.py`)

**Price Data — yFinance**
- `auto_adjust=True` automatically corrects for splits and dividends upstream, ensuring adjusted closing prices without manual reconstruction
- Returns empty DataFrame (never raises) — downstream validation handles missing tickers gracefully
- Retries: 3 attempts at 2s → 4s → 8s exponential backoff via `tenacity`

**Macro Data — FRED API**
- Series collected: `DFF` (Fed Funds Rate), `CPIAUCSL` (CPI), `UNRATE` (Unemployment), `T10Y2Y` (Yield Curve Spread)
- Sparse series (e.g., monthly CPI) are stored as-is and forward-filled during feature engineering, not during ingestion — preserving point-in-time accuracy

**S&P 500 Constituent Registry**
- Scraped from Wikipedia on startup using `httpx`
- Stored in `symbols` table with `data_source` tiering

### 2. Data Cleaning (`backend/app/data/cleaner.py`)

| Rule | Behavior | Rationale |
|------|----------|-----------|
| Missing trading days | Forward-fill up to 3 consecutive gaps | Stale-price assumption; beyond 3 days likely a data error |
| Single-day moves ≥ 15% | Flagged as `outlier_flagged`, retained | Preserves real market events (e.g., earnings shocks); model decides relevance |
| Any remaining NaN in `close` | Raise `DataQualityError` | Hard stop — downstream calculations require complete price series |

Every row written to the database carries a `quality_flag` column (`clean` / `forward_filled` / `outlier_flagged`). This audit trail allows downstream queries to optionally exclude or weight flagged observations.

### 3. Storage — Idempotent Upsert (`backend/app/data/store.py`)

PostgreSQL `ON CONFLICT (date, ticker) DO UPDATE` pattern ensures:
- Daily refresh jobs can be re-run without creating duplicate rows
- Partial failures are safe to retry (no phantom data)
- Last-write-wins semantics on price updates

**Tiered Ticker Strategy**

| Tier | Coverage | Behavior |
|------|----------|----------|
| Preloaded | S&P 500 + 5 ETFs (~510 symbols) | Seeded on container startup; refreshed daily |
| On-demand | Any valid ticker | Fetched on first user request (full history since 2010); cached permanently |
| Not found | Invalid symbols | yFinance returns empty → 404 returned to client |

### 4. Automated Scheduling (`backend/app/data/scheduler.py`)

Runs via `APScheduler` `BackgroundScheduler` inside the FastAPI process:

| Job | Schedule | Action |
|-----|----------|--------|
| Daily price refresh | Mon–Fri 6 PM ET | Fetches 10-day window per ticker, upserts, stamps `last_fetched` |
| FRED macro refresh | Mon–Fri 6 PM ET | Fetches 30-day window for all 4 macro series |
| Startup seed | 10s after app start | Idempotent: seeds symbols + backfills history if DB is empty |
| Weekly IC check | Sunday midnight ET | Computes rolling Spearman IC; triggers gated retrain if degraded |

Individual ticker failures are isolated — one bad API response does not abort the full refresh.

---

## Statistical & Actuarial Methods

### Feature Engineering — 18 Signals (`backend/app/analytics/features.py`)

Features fall into three categories, each with distinct normalization:

**Momentum (sector-normalized z-scores)**

| Feature | Lookback | Concept |
|---------|----------|---------|
| `ret_1d` | 1 day | Short-term reversal / noise |
| `ret_5d` | 5 days (1 week) | Short-term drift |
| `ret_21d` | 21 days (1 month) | Monthly momentum |
| `ret_63d` | 63 days (1 quarter) | Medium-term trend |
| `ret_126d` | 126 days (6 months) | Fama-French momentum gap |
| `ret_252d` | 252 days (1 year) | Jegadeesh-Titman annual momentum |
| `high_52w_ratio` | 252 days | Proximity to 52-week high — continuation signal |

**Volatility & Volume (sector-normalized z-scores)**

| Feature | Calculation | Purpose |
|---------|------------|---------|
| `vol_21d` | Rolling σ of daily returns, 21d | Short-term realized volatility |
| `vol_63d` | Rolling σ of daily returns, 63d | Medium-term baseline |
| `vol_ratio` | Today's volume / 21d avg volume | Abnormal volume flag |
| `vol_ratio_63d` | Today's volume / 63d avg volume | Smoothed volume signal |

**Technical Oscillators (sector-normalized)**

| Feature | Formula | Notes |
|---------|---------|-------|
| `rsi_14` | Wilder's RSI: `100 - 100 / (1 + RS)` | RS = EWM avg gain / EWM avg loss |
| `macd` | `(EMA₁₂ - EMA₂₆) / close` | Price-normalized momentum oscillator |
| `macd_signal` | EMA₉ of MACD | Divergence from signal line |
| `bb_position` | `(close - SMA₂₀) / (2 × σ₂₀)` | Bollinger band position; ±1 = 1 std dev |

**Cross-Sectional Relative Signals (not sector-normalized)**

| Feature | Formula |
|---------|---------|
| `ret_5d_vs_spy` | Stock 5d return − SPY 5d return |
| `ret_21d_vs_spy` | Stock 21d return − SPY 21d return |

**Macro Regime Signals (market-wide, StandardScaler only)**

| Feature | Source | Economic Meaning |
|---------|--------|-----------------|
| `log_vix` | Yahoo Finance (^VIX) | Fear gauge / volatility regime |
| `yield_curve` | FRED T10Y2Y | Negative = inverted (recession signal) |
| `cpi_yoy` | FRED CPIAUCSL, 12m % Δ | Inflation regime (raises/compresses multiples) |
| `dff_delta_63d` | FRED DFF, 63d Δ | Rate tightening/easing direction |

**Lookahead Bias Prevention**: All features apply `.shift(1)` (yesterday's data only). First 252 rows (warmup) and last 21 rows (unknown future) are dropped before training.

---

### Monte Carlo Simulation (`backend/app/analytics/montecarlo.py`)

The simulation uses **block bootstrap** rather than naive IID resampling — a statistically sounder approach that actuaries will recognize as more realistic for financial time series:

**Why Block Bootstrap?**

Financial returns exhibit:
1. **Volatility clustering** — high-vol periods cluster together (GARCH-like behavior)
2. **Short-run autocorrelation** — momentum persists over days-to-weeks

IID resampling destroys both properties. Block bootstrap resamples consecutive 3-month chunks with replacement, preserving intra-block dependencies.

**Regime-Aware Sampling (optional)**

The user can enable regime-conditional simulation:
- Current regime detected via 12-month rolling cumulative return
- Bootstrap pool restricted to historically matching regime (expansion or contraction)
- Result: tighter cone in bear regimes, wider in bull regimes — directionally correct uncertainty

**Simulation Output**

```
For each of N=1,000 simulations:
  W[0] = initial_value
  W[t+1] = W[t] × (1 + r_block[t]) + monthly_contribution
  Adjust for inflation: W_real[t] = W[t] / (1 + annual_inflation)^(t/12)

Output percentiles: p10 (pessimistic), p50 (median), p90 (optimistic)
```

This is architecturally equivalent to an actuarial **stochastic projection model** — a distribution of outcomes rather than a single deterministic path.

---

### Walk-Forward Cross-Validation (`backend/app/analytics/validation.py`)

Model validation uses **expanding-window walk-forward CV** — the standard for time series to prevent data leakage:

```
Training window (expanding)        Validation window (rolling)
│──────────────────────────────│   │──────────────────────────│
Jan 2014 ─────────── Jul 2018      Jul 2018 ─── Jan 2019       Fold 1
Jan 2014 ─────────────────────── Jan 2019      Jan 2019 ─── Jul 2019  Fold 2
Jan 2014 ──────────────────────────────── Jul 2019      Jul 2019 ─── Jan 2020  Fold 3
...
```

- Minimum 2 years training data before first validation fold
- 6-month validation windows; 6-month step forward
- Spearman IC computed per fold; averaged across all folds
- Model accepted only if mean IC ≥ 0.05

This mirrors **held-period testing** in actuarial pricing models and **out-of-time validation** in underwriting scorecards.

---

## Machine Learning & Model Governance

### LightGBM Cross-Sectional Regressor (`backend/app/models/predictor.py`)

**Target**: 21-day forward return — `target_21d = close.pct_change(21).shift(-21)`

**Training Data Construction**:
- Each row = (ticker, date) pair; all tickers treated as samples on a given date
- Winsorize target at 1st/99th percentiles (removes return outliers that would dominate MSE)
- Sector normalization of per-ticker features (z-score within date × sector group)

**Hyperparameters selected for generalization, not accuracy**:
```
max_depth=5, num_leaves=31   → shallow trees, low variance
min_child_samples=100        → prevents overfit to thin sectors
subsample=0.8, colsample_bytree=0.8  → stochastic regularization
reg_alpha=0.1, reg_lambda=1.0        → L1 + L2 penalty
early_stopping=50            → stop when val loss plateaus
```

### Information Coefficient (IC) as Validation Metric

The IC is the **Spearman rank correlation** between predicted 21d returns and actual 21d returns across all tickers on a given date.

| IC Range | Interpretation |
|----------|---------------|
| > 0.10 | Exceptional signal |
| 0.05 – 0.10 | Meaningful — model accepted |
| 0.03 – 0.05 | Marginal — monitor closely |
| < 0.03 | Insufficient — trigger retrain |
| < 0.00 | Worse than random — circuit breaker |

IC is rank-based (not Pearson), making it robust to outlier returns and non-normality — appropriate for heavy-tailed financial distributions.

### Model Health Monitoring (`backend/app/models/health.py`)

A continuous health-check loop runs independently of any user request:

```
Weekly IC check (Sunday midnight):
  1. Pull last 130 calendar days of predictions from prediction_log
  2. Join with realized returns
  3. Compute rolling Spearman IC
  4. Evaluate vs. gate threshold
  5. Log gate_status + consecutive_failures to model_metrics table
  6. If 3 consecutive failures → circuit_breaker = True, block inference
  7. If degraded → trigger gated retrain
```

**Gated Retrain Logic**:
- `predictor.train()` runs the full pipeline
- If walk-forward mean IC ≥ 0.05 → save model artifact to disk + S3
- If IC < 0.05 → `registry.delete_latest()` rolls back to previous version
- Model is **never silently degraded** — it either improves or reverts

This implements a **model risk management gate** analogous to validation standards in SR 11-7 (model risk guidance) and actuarial ASOP 56 (modeling).

### Model Artifact Versioning (`backend/app/models/registry.py`)

- Artifacts saved as `lgbm_YYYYMMDD_HHMMSS.{pkl, json}` with metadata
- Synced to AWS S3 on every save
- Restored from S3 on container restart (stateless deployment)
- All training runs logged to `model_metrics` table (append-only audit trail)

---

## Risk Metrics & Portfolio Analytics

### Performance Metrics (`backend/app/analytics/portfolio.py`)

All metrics computed on daily return series, annualized to 252 trading days:

| Metric | Formula | Relevance |
|--------|---------|-----------|
| **CAGR** | `(V_T / V_0)^(1/years) − 1` | Actuarial: equivalent to IRR |
| **Sharpe Ratio** | `(μ_r − r_f) / σ_r × √252` | Risk-adjusted return; standard in investment analytics |
| **Max Drawdown** | `min(V_t / max(V_s, s≤t)) − 1` | Downside risk; peak-to-trough loss |
| **Calmar Ratio** | `CAGR / |MaxDrawdown|` | Return per unit of tail risk |
| **Alpha** | `portfolio_return − benchmark_return` (annualized) | Active return vs. SPY |
| **Beta** | `Cov(r_port, r_spy) / Var(r_spy)` | Systematic risk exposure |
| **Information Ratio** | `active_return / tracking_error × √(periods/year)` | Consistency of outperformance |
| **Win Rate** | `% periods with positive return` | Directional accuracy |

### Markowitz Optimization (`backend/app/analytics/optimizer.py`)

Three objective modes:
- **Max Sharpe**: `maximize (μ_w − r_f) / σ_w` via SLSQP
- **Min Volatility**: `minimize σ_w = √(w^T Σ w)`
- **Max CAGR**: maximize compound growth accounting for variance drag

Constraints: long-only (`w_i ≥ 0`), fully-invested (`Σ w_i = 1.0`). Lookback window: 3–10 years (user-selectable).

### Benchmark Comparison (`backend/app/analytics/benchmark.py`)

- Portfolio and SPY prices aligned via inner join on common trading dates (eliminates calendar mismatch)
- All return series inflation-adjusted on demand
- Outputs same metric set for both portfolio and benchmark for side-by-side comparison

---

## Database Design

Six tables designed for auditability and time-series efficiency:

### `prices`
```sql
PRIMARY KEY (date, ticker)
-- Columns: open, high, low, close, volume, quality_flag, updated_at
-- quality_flag: 'clean' | 'forward_filled' | 'outlier_flagged'
```
Every price observation carries a data provenance flag. Analysts can filter by quality tier.

### `macro_data`
```sql
PRIMARY KEY (date, series_id)
-- series_id: 'DFF' | 'CPIAUCSL' | 'UNRATE' | 'T10Y2Y'
-- Refreshed daily; sparse series stored as-is
```

### `prediction_log`
```sql
UNIQUE (as_of_date, ticker)
INDEX ON as_of_date  -- for rolling IC window queries
-- Columns: predicted_return_21d, model_version
```
Append-only log of every daily inference run. Enables retrospective IC computation across any date range.

### `model_metrics` (append-only audit trail)
```sql
-- Columns: computed_at, rolling_ic, n_samples,
--          gate_status, consecutive_failures,
--          retrain_triggered, retrain_result, model_version
```
Complete history of model health events. Never updated, only inserted — preserves full audit trail.

### `constituent_history`
```sql
PRIMARY KEY (ticker)
-- Columns: ticker, date_added
-- Used for point-in-time S&P 500 membership filtering
```
Mitigates **survivorship bias** in model training by filtering to stocks that were actually in the index on each training date.

### `symbols`
```sql
PRIMARY KEY (ticker)
-- Columns: name, sector, data_source ('preloaded' | 'on_demand'),
--          active, last_fetched
```

---

## API Design

Eight REST endpoints, all with Pydantic-validated request/response schemas:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | DB + scheduler liveness check |
| `GET` | `/api/search?q=` | Ticker autocomplete (debounced, sector-tagged) |
| `POST` | `/api/portfolio` | Backtest holdings vs. SPY; returns daily values + 9 metrics |
| `POST` | `/api/simulate` | Monte Carlo projection; returns p10/p50/p90 cone |
| `POST` | `/api/predict` | LightGBM 21d return predictions for given tickers |
| `POST` | `/api/optimize` | Markowitz optimization (3 objective modes) |
| `GET` | `/api/model/health` | Rolling IC, gate status, retrain history |
| `GET` | `/api/health` | System health check |

All endpoints return typed JSON matching TypeScript interfaces in `frontend/types/api.ts`.

---

## Tech Stack

### Backend
| Component | Technology | Version |
|-----------|-----------|---------|
| API framework | FastAPI + Uvicorn | 0.111.0 |
| Database | PostgreSQL | 15 |
| ORM | SQLAlchemy | 2.0.30 |
| ML | LightGBM | 4.3.0 |
| Data processing | pandas + numpy | 2.2.2 / 1.26.4 |
| Statistics | scipy + statsmodels | 1.13.0 / 0.14.2 |
| Price data | yFinance | 1.2.0 |
| Macro data | pandas-datareader (FRED) | 0.10.0 |
| Scheduling | APScheduler | 3.10.4 |
| Retries | tenacity | 8.3.0 |
| Cloud storage | boto3 (S3) | 1.34.84 |
| Config | pydantic-settings | 2.2.1 |
| Testing | pytest + pytest-asyncio | 8.2.0 |

### Frontend
| Component | Technology | Version |
|-----------|-----------|---------|
| Framework | Next.js | 14.2.35 |
| Language | TypeScript | 5 |
| Charts | Recharts | 3.8.0 |
| Styling | Tailwind CSS | 3.4.1 |

### Infrastructure
- **Local**: Docker Compose (PostgreSQL + FastAPI)
- **Production target**: AWS ECS Fargate + RDS PostgreSQL + S3 (model artifacts) + Vercel (frontend)
- **Secrets**: AWS Secrets Manager (no `.env` in production)

---

## Local Setup

**Prerequisites**: Docker Desktop, Node.js 18+, Python 3.11+

```bash
# 1. Clone and configure
git clone <repo>
cd Stock_Project
cp .env.example .env
# Add FRED_API_KEY from https://fred.stlouisfed.org/docs/api/api_key.html

# 2. Start database + backend
docker compose up --build

# 3. Start frontend
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The backend seeds S&P 500 data automatically on first start (allow ~60s).

**Environment Variables**
```bash
DATABASE_URL=postgresql://user:password@localhost:5433/portfolio_sim
FRED_API_KEY=your_key_here
ENVIRONMENT=development
CORS_ORIGINS=http://localhost:3000
MODEL_DIR=./models

# Production only
AWS_REGION=us-east-1
S3_BUCKET_MODELS=portfolio-sim-models
```

---

## Testing

```bash
cd backend
pip install -r requirements.txt
pytest tests/ -v
```

Test suite covers the full data and modeling pipeline:

| Test File | What It Validates |
|-----------|------------------|
| `test_fetcher.py` | yFinance + FRED API calls, retry behavior |
| `test_cleaner.py` | Gap-fill logic, outlier flagging, NaN rejection |
| `test_features.py` | **Lookahead bias prevention** (critical — `.shift(1)` verified) |
| `test_portfolio.py` | Metric calculations vs. manual formulas (CAGR, Sharpe, drawdown) |
| `test_montecarlo.py` | p10 < p50 < p90 ordering; simulation convergence |
| `test_validation.py` | Walk-forward CV; no data leakage across fold boundaries |
| `test_predictor.py` | Model training; IC computation; gate logic |
| `test_symbols.py` | S&P 500 seed; on-demand tier fetch |
| `test_store.py` | Upsert idempotency; no duplicate rows |

---

## Key Design Decisions

**Why block bootstrap over IID?** Financial returns exhibit volatility clustering (GARCH-like). IID bootstrap underestimates tail risk in stress periods. Block bootstrap preserves consecutive return dependencies, producing more realistic loss scenarios — consistent with actuarial stochastic simulation best practices.

**Why Spearman IC over RMSE?** The model's value is in *ranking* stocks (which to overweight), not in predicting exact return magnitudes. Spearman rank correlation captures this correctly and is robust to non-normal, heavy-tailed return distributions.

**Why LightGBM over linear models?** The feature interaction between momentum and volatility regime is non-linear. LightGBM handles this naturally while remaining regularized and interpretable via feature importance. A Ridge model was the MVP predecessor and remains as a fallback.

**Why upsert over insert?** Scheduled jobs run daily. Idempotent upserts allow re-runs after partial failures without corrupting the price history or creating duplicates.

**Why append-only `model_metrics`?** Model health events must be auditable. An append-only log ensures no historical IC or retrain record can be overwritten — analogous to actuarial sign-off documentation.
