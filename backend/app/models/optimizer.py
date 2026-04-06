"""
Markowitz portfolio optimizer — SLSQP with multiple random restarts.

Constraints:
  - weights sum to 1.0
  - each weight >= WEIGHT_MIN (5%)
  - each weight <= WEIGHT_MAX (50%)

Goals:
  sharpe   — maximize risk-adjusted return (Sharpe ratio)
  min_vol  — minimize portfolio volatility
  max_cagr — maximize expected annualized return
"""
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from app.data.store import get_prices

logger = logging.getLogger(__name__)

RISK_FREE_RATE = 0.045   # annualized ~T-bill yield used in Sharpe calculation
TRADING_DAYS   = 252
WEIGHT_MIN     = 0.05
WEIGHT_MAX     = 0.50
N_RESTARTS     = 50


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_returns(tickers: list[str], start: str) -> pd.DataFrame:
    """
    Fetch daily closing prices for each ticker from start date, compute
    simple daily returns, and align on the intersection of trading days.

    Raises ValueError if any ticker has too little data or the aligned
    window is too short to be meaningful.
    """
    closes = {}
    for ticker in tickers:
        df = get_prices(ticker, start=start)
        if df.empty or len(df) < 60:
            raise ValueError(
                f"{ticker} has insufficient price history for the selected "
                "lookback period. Try a shorter lookback or a different ticker."
            )
        closes[ticker] = df.set_index("date")["close"]

    price_df = pd.DataFrame(closes).dropna()
    if len(price_df) < 60:
        raise ValueError(
            "Not enough overlapping trading days across all selected tickers. "
            "Try a shorter lookback or tickers with longer shared history."
        )

    return price_df.pct_change().dropna()


def _compute_metrics(
    weights: np.ndarray,
    mean_arr: np.ndarray,
    cov_arr: np.ndarray,
    returns_df: pd.DataFrame,
) -> dict:
    """
    Annualized portfolio metrics for a given weight vector.
    Max drawdown is computed from the actual daily return series
    (not a parametric estimate) so it reflects real historical behaviour.
    """
    port_return = float(weights @ mean_arr)
    port_vol    = float(np.sqrt(weights @ cov_arr @ weights))
    sharpe      = (port_return - RISK_FREE_RATE) / port_vol if port_vol > 0 else 0.0

    daily_port  = (returns_df @ weights)
    cum         = (1 + daily_port).cumprod()
    rolling_max = cum.cummax()
    max_dd      = float(((cum - rolling_max) / rolling_max).min())

    return {
        "sharpe":       round(sharpe,      4),
        "cagr":         round(port_return, 4),
        "max_drawdown": round(max_dd,      4),
        "volatility":   round(port_vol,    4),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def optimize(tickers: list[str], goal: str, lookback_years: int) -> dict:
    """
    Run Markowitz portfolio optimization for the given tickers and goal.

    Returns:
        {
          "weights": {"AAPL": 0.25, ...},
          "metrics": {"sharpe": 1.2, "cagr": 0.18, "max_drawdown": -0.12, "volatility": 0.14},
          "goal":    "sharpe"
        }

    Raises:
        ValueError  — bad inputs or insufficient data (caller returns 422)
        RuntimeError — optimizer failed to converge (caller returns 500)
    """
    if len(tickers) < 5:
        raise ValueError("At least 5 tickers are required.")

    start = (date.today() - timedelta(days=lookback_years * 365)).isoformat()

    returns_df  = _build_returns(tickers, start)
    mean_arr    = (returns_df.mean() * TRADING_DAYS).values
    cov_arr     = (returns_df.cov()  * TRADING_DAYS).values
    n           = len(tickers)

    # ── Objective functions (all minimized — negate for max goals) ──────────
    def neg_sharpe(w):
        ret = w @ mean_arr
        vol = np.sqrt(w @ cov_arr @ w)
        return -((ret - RISK_FREE_RATE) / vol) if vol > 0 else 0.0

    def portfolio_vol(w):
        return float(np.sqrt(w @ cov_arr @ w))

    def neg_cagr(w):
        return -(w @ mean_arr)

    objectives = {
        "sharpe":   neg_sharpe,
        "min_vol":  portfolio_vol,
        "max_cagr": neg_cagr,
    }
    if goal not in objectives:
        raise ValueError(f"Invalid goal '{goal}'. Must be one of: {list(objectives)}")

    objective   = objectives[goal]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds      = [(WEIGHT_MIN, WEIGHT_MAX)] * n

    best_result = None
    best_value  = np.inf

    rng = np.random.default_rng(42)
    for _ in range(N_RESTARTS):
        # Dirichlet gives a random point on the simplex; clip + renorm enforces bounds
        w0  = rng.dirichlet(np.ones(n))
        w0  = np.clip(w0, WEIGHT_MIN, WEIGHT_MAX)
        w0 /= w0.sum()

        result = minimize(
            objective, w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )

        if result.success and result.fun < best_value:
            best_value  = result.fun
            best_result = result

    if best_result is None:
        raise RuntimeError(
            "Optimizer failed to converge. Try different tickers or a longer lookback period."
        )

    weights  = np.clip(best_result.x, WEIGHT_MIN, WEIGHT_MAX)
    weights /= weights.sum()

    return {
        "weights": {t: round(float(w), 4) for t, w in zip(tickers, weights)},
        "metrics": _compute_metrics(weights, mean_arr, cov_arr, returns_df),
        "goal":    goal,
    }
