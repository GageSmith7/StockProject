"""
Portfolio construction, backtesting, and performance metrics.

Strategy: equal-weight long portfolio of the top quintile by predicted 21d return.
Backtest: walk-forward, rebalancing every rebalance_days trading days.
No transaction costs or slippage (MVP limitation).
"""
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from app.analytics.features import (
    FEATURE_COLS,
    TARGET_COL,
    build_cross_sectional_dataset,
)
from app.data.symbols import get_all_tickers
from app.models import predictor

logger = logging.getLogger(__name__)

REBALANCE_DAYS  = 21     # trading days between rebalances (= prediction horizon)
QUINTILE        = 5      # divide universe into this many buckets; long the top bucket
RISK_FREE_RATE  = 0.0    # annualized; 0 for simplicity (MVP)
TRADING_DAYS_PA = 252    # trading days per year


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------

def construct_portfolio(
    predictions: pd.Series,
    n_holdings: int | None = None,
) -> pd.DataFrame:
    """
    Select the top quintile of tickers by predicted return, equal-weighted.

    Args:
        predictions : pd.Series — index=ticker, values=predicted 21d return.
        n_holdings  : override the quintile size (default = len(predictions) // QUINTILE).

    Returns:
        DataFrame with columns ['ticker', 'weight'].
        Weights sum to 1.0.
        Returns empty DataFrame if predictions is empty.
    """
    if predictions.empty:
        return pd.DataFrame(columns=["ticker", "weight"])

    n = n_holdings if n_holdings is not None else max(1, len(predictions) // QUINTILE)
    top = predictions.nlargest(n)
    weight = 1.0 / len(top)

    return pd.DataFrame(
        {"ticker": top.index.tolist(), "weight": [weight] * len(top)}
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    period_returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    periods_per_year: float = TRADING_DAYS_PA / REBALANCE_DAYS,
) -> dict:
    """
    Compute standard performance metrics from a series of period returns.

    Args:
        period_returns   : pd.Series of per-period (e.g. 21-day) portfolio returns.
        benchmark_returns: pd.Series of per-period benchmark returns (optional).
        periods_per_year : number of periods in a year (default ≈ 12 for 21-day).

    Returns dict with:
        total_return, cagr, sharpe, max_drawdown, calmar,
        n_periods, win_rate,
        and if benchmark provided: benchmark_total_return, alpha, beta, information_ratio.
    """
    r = period_returns.dropna()
    n = len(r)

    if n == 0:
        return {"error": "no valid return periods"}

    # --- Core metrics ---
    total_return = float((1 + r).prod() - 1)
    n_years      = n / periods_per_year
    cagr         = float((1 + total_return) ** (1 / n_years) - 1) if n_years > 0 else 0.0

    excess   = r - RISK_FREE_RATE / periods_per_year
    sharpe   = float(excess.mean() / excess.std() * np.sqrt(periods_per_year)) \
               if excess.std() > 0 else 0.0

    # Max drawdown
    cum_ret    = (1 + r).cumprod()
    rolling_hi = cum_ret.cummax()
    drawdown   = (cum_ret - rolling_hi) / rolling_hi
    max_dd     = float(drawdown.min())

    calmar     = float(cagr / abs(max_dd)) if max_dd != 0 else 0.0
    win_rate   = float((r > 0).mean())

    metrics = {
        "total_return"  : round(total_return, 6),
        "cagr"          : round(cagr, 6),
        "sharpe"        : round(sharpe, 4),
        "max_drawdown"  : round(max_dd, 6),
        "calmar"        : round(calmar, 4),
        "n_periods"     : n,
        "win_rate"      : round(win_rate, 4),
    }

    # --- Benchmark comparison ---
    if benchmark_returns is not None:
        bm = benchmark_returns.dropna()
        bm, r_aligned = bm.align(r, join="inner")

        bm_total   = float((1 + bm).prod() - 1)
        bm_years   = len(bm) / periods_per_year
        bm_cagr    = float((1 + bm_total) ** (1 / bm_years) - 1) if bm_years > 0 else 0.0
        alpha_raw  = float(r_aligned.mean() - bm.mean())

        # Beta via OLS
        if bm.std() > 0:
            beta = float(np.cov(r_aligned, bm)[0][1] / bm.var())
        else:
            beta = 0.0

        # Information ratio: active return / tracking error
        active = r_aligned - bm
        ir     = float(active.mean() / active.std() * np.sqrt(periods_per_year)) \
                 if active.std() > 0 else 0.0

        metrics.update({
            "benchmark_total_return" : round(bm_total, 6),
            "benchmark_cagr"         : round(bm_cagr, 6),
            "alpha"                  : round(alpha_raw * periods_per_year, 6),
            "beta"                   : round(beta, 4),
            "information_ratio"      : round(ir, 4),
        })

    return metrics


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def backtest(
    start_date: str,
    end_date: str | None = None,
    rebalance_days: int = REBALANCE_DAYS,
    tickers: list[str] | None = None,
) -> dict:
    """
    Walk-forward backtest of the top-quintile equal-weight long strategy.

    Process every rebalance_days trading days:
      1. Use features as of that date to generate predictions.
      2. Construct equal-weight top-quintile portfolio.
      3. Record the actual 21-day return (target_21d) for each held stock.
      4. Portfolio period return = mean of held stocks' actual returns.

    No transaction costs or slippage (MVP).

    Returns:
        {
          "portfolio_returns"  : pd.Series (indexed by rebalance date),
          "benchmark_returns"  : pd.Series (SPY returns for same periods),
          "metrics"            : dict from compute_metrics(),
          "holdings_history"   : list[dict] — one entry per rebalance period,
        }
    """
    if end_date is None:
        end_date = datetime.utcnow().date().isoformat()

    if tickers is None:
        tickers = get_all_tickers(source="preloaded")

    logger.info("Building cross-sectional dataset for backtest %s → %s", start_date, end_date)
    dataset = build_cross_sectional_dataset(tickers, start_date, end_date)

    if dataset.empty:
        logger.warning("backtest: empty dataset — no results")
        return {}

    dataset["date"] = pd.to_datetime(dataset["date"])

    # Load model once
    model, scaler = predictor.registry.load_latest()

    # Sorted unique trading dates
    all_dates = sorted(dataset["date"].unique())

    # Sample rebalance dates every rebalance_days steps
    rebalance_dates = all_dates[::rebalance_days]

    portfolio_returns  = {}
    benchmark_returns  = {}
    holdings_history   = []

    for r_date in rebalance_dates:
        # Snapshot: latest feature row per ticker as of r_date
        snapshot = (
            dataset[dataset["date"] == r_date]
            .dropna(subset=FEATURE_COLS + [TARGET_COL])
        )

        if len(snapshot) < 10:   # need enough stocks to form a quintile
            continue

        # Predict
        from sklearn.preprocessing import StandardScaler
        X = snapshot[FEATURE_COLS].values
        X_s = scaler.transform(X)
        preds = model.predict(X_s)
        pred_series = pd.Series(preds, index=snapshot["ticker"].values)

        # Construct portfolio
        portfolio = construct_portfolio(pred_series)
        held = portfolio["ticker"].tolist()

        # Actual realized returns for held tickers on this date
        actual = snapshot[snapshot["ticker"].isin(held)].set_index("ticker")[TARGET_COL]
        period_ret = float(actual.mean()) if not actual.empty else 0.0

        # SPY benchmark for same period
        spy_row = snapshot[snapshot["ticker"] == "SPY"][TARGET_COL]
        spy_ret = float(spy_row.values[0]) if not spy_row.empty else np.nan

        portfolio_returns[r_date] = period_ret
        benchmark_returns[r_date] = spy_ret

        holdings_history.append({
            "date"             : r_date.date().isoformat(),
            "tickers"          : held,
            "n_holdings"       : len(held),
            "period_return"    : round(period_ret, 6),
            "benchmark_return" : round(spy_ret, 6) if not np.isnan(spy_ret) else None,
        })

    port_s = pd.Series(portfolio_returns).sort_index()
    bm_s   = pd.Series(benchmark_returns).sort_index()

    if port_s.empty:
        logger.warning("backtest: no rebalance periods produced results")
        return {}

    metrics = compute_metrics(port_s, bm_s)

    logger.info(
        "Backtest %s → %s | periods=%d | CAGR=%.1f%% | Sharpe=%.2f | MaxDD=%.1f%%",
        start_date, end_date,
        metrics.get("n_periods", 0),
        metrics.get("cagr", 0) * 100,
        metrics.get("sharpe", 0),
        metrics.get("max_drawdown", 0) * 100,
    )

    return {
        "portfolio_returns" : port_s,
        "benchmark_returns" : bm_s,
        "metrics"           : metrics,
        "holdings_history"  : holdings_history,
    }
