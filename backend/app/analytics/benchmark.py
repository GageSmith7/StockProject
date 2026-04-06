"""
User-defined weighted portfolio returns and SPY benchmark comparison.

Handles the portfolio builder use case: user picks tickers + weights,
we compute weighted daily returns and compare against SPY.

Distinct from analytics/portfolio.py (ML top-quintile strategy).
This module is purely about user-specified holdings.
"""
import logging
from datetime import datetime

import pandas as pd

from app.analytics.portfolio import TRADING_DAYS_PA, compute_metrics
from app.data.store import TickerNotFoundError, get_or_fetch, get_prices

logger = logging.getLogger(__name__)


def weighted_daily_returns(
    holdings: dict[str, float],
    start_date: str,
    end_date: str,
) -> tuple[pd.Series, list[str]]:
    """
    Compute daily weighted portfolio returns for user-specified holdings.

    Fetches each ticker from DB (fast path) or yFinance (slow path).
    Aligns all price series on common trading days before computing returns.

    Args:
        holdings  : {ticker: weight} — caller must ensure weights sum to 1.0.
        start_date: YYYY-MM-DD.
        end_date  : YYYY-MM-DD.

    Returns:
        (daily_returns, on_demand_tickers)
        daily_returns     : pd.Series indexed by date, values = daily portfolio return.
        on_demand_tickers : tickers fetched live from yFinance on this call.
    """
    close_frames: dict[str, pd.Series] = {}
    on_demand: list[str] = []

    for ticker in holdings:
        prices = get_prices(ticker, start=start_date, end=end_date)

        if prices.empty:
            try:
                # Tier 2: fetch full history then filter to range
                get_or_fetch(ticker)
                prices = get_prices(ticker, start=start_date, end=end_date)
                on_demand.append(ticker)
            except TickerNotFoundError:
                logger.warning("weighted_daily_returns: %s not found — skipping", ticker)
                continue

        if prices.empty:
            logger.warning("weighted_daily_returns: no data for %s in range — skipping", ticker)
            continue

        prices["date"] = pd.to_datetime(prices["date"])
        close_frames[ticker] = (
            prices.set_index("date")["close"].sort_index().astype(float)
        )

    if not close_frames:
        return pd.Series(dtype=float), on_demand

    # Inner join: only dates where every holding has a price
    close_df = pd.DataFrame(close_frames).dropna()

    if len(close_df) < 2:
        logger.warning("weighted_daily_returns: fewer than 2 common trading days")
        return pd.Series(dtype=float), on_demand

    daily_ret_df = close_df.pct_change().iloc[1:]   # drop first NaN row

    # Weighted sum
    port_returns = sum(
        daily_ret_df[ticker] * weight
        for ticker, weight in holdings.items()
        if ticker in daily_ret_df.columns
    )

    return port_returns, on_demand


def compare(
    holdings: dict[str, float],
    start_date: str,
    end_date: str | None = None,
) -> dict:
    """
    Compare a user-defined weighted portfolio against SPY.

    Args:
        holdings  : {ticker: weight} — weights must sum to 1.0.
        start_date: YYYY-MM-DD.
        end_date  : YYYY-MM-DD (defaults to today).

    Returns dict with:
        portfolio_returns : pd.Series — daily portfolio returns, date-indexed.
        benchmark_returns : pd.Series — SPY daily returns, date-indexed.
        metrics           : dict from compute_metrics() (CAGR, Sharpe, drawdown, …).
        on_demand_tickers : list[str] — tickers fetched live from yFinance.
    """
    if end_date is None:
        end_date = datetime.utcnow().date().isoformat()

    port_returns, on_demand = weighted_daily_returns(holdings, start_date, end_date)

    if port_returns.empty:
        logger.warning("compare: no portfolio returns produced")
        return {}

    # --- SPY benchmark ---
    spy_returns = pd.Series(dtype=float)
    spy_prices = get_prices("SPY", start=start_date, end=end_date)

    if spy_prices.empty:
        try:
            get_or_fetch("SPY")
            spy_prices = get_prices("SPY", start=start_date, end=end_date)
        except TickerNotFoundError:
            logger.warning("compare: SPY unavailable — returning without benchmark")

    if not spy_prices.empty:
        spy_prices["date"] = pd.to_datetime(spy_prices["date"])
        spy_close = spy_prices.set_index("date")["close"].sort_index().astype(float)
        spy_daily = spy_close.pct_change().iloc[1:]
        # Align to portfolio's trading calendar
        spy_returns = spy_daily.reindex(port_returns.index)

    metrics = compute_metrics(
        port_returns,
        benchmark_returns=spy_returns if not spy_returns.empty else None,
        periods_per_year=float(TRADING_DAYS_PA),   # daily returns → 252 periods/year
    )

    logger.info(
        "compare: %s → %s | holdings=%d | CAGR=%.1f%% | Sharpe=%.2f | MaxDD=%.1f%%",
        start_date, end_date,
        len(holdings),
        metrics.get("cagr", 0) * 100,
        metrics.get("sharpe", 0),
        metrics.get("max_drawdown", 0) * 100,
    )

    return {
        "portfolio_returns" : port_returns,
        "benchmark_returns" : spy_returns,
        "metrics"           : metrics,
        "on_demand_tickers" : on_demand,
    }
