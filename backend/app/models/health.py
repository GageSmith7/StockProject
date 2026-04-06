"""
Model health monitoring — IC tracking, gate checks, gated auto-retrain.

Flow:
  1. Daily:  log_daily_predictions() → writes to prediction_log
  2. Weekly: run_weekly_ic_check()   → reads prediction_log, computes rolling IC,
             gates retrain, writes to model_metrics

Gated retrain logic:
  predictor.train() always saves the new model to disk.
  If the new model fails the IC gate, registry.delete_latest() removes it so
  the previous model remains the active artifact.
"""
import logging
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats

from app.data.store import (
    get_latest_model_metrics,
    get_predictions_window,
    get_prices,
    upsert_model_metrics,
    upsert_predictions,
)
from app.models import predictor, registry

logger = logging.getLogger(__name__)

# Must match predictor.IC_GATE — kept here as the live-monitoring threshold
IC_GATE                    = 0.05
CIRCUIT_BREAKER_THRESHOLD  = 3      # consecutive failures before circuit breaker fires
MIN_IC_SAMPLES             = 50     # minimum valid (ticker, date) pairs for a reliable IC
# Window for pulling historical predictions — 90 calendar days ≈ 63 trading days,
# plus 32 calendar days as a proxy for the 21-trading-day lookforward.
IC_WINDOW_CALENDAR_DAYS    = 130


# ---------------------------------------------------------------------------
# Daily: log predictions
# ---------------------------------------------------------------------------

def log_daily_predictions() -> int:
    """
    Generate predictions for all preloaded tickers using the current model
    and persist them to prediction_log with today's date.

    Called at the end of _daily_refresh() so predictions are always based on
    the freshest price data. Returns number of predictions logged.
    """
    from datetime import datetime, timedelta

    from app.analytics.features import build_cross_sectional_dataset
    from app.data.symbols import get_all_tickers

    try:
        end_date   = datetime.utcnow().date().isoformat()
        start_date = (datetime.utcnow().date() - timedelta(days=730)).isoformat()
        tickers    = get_all_tickers(source="preloaded")

        dataset = build_cross_sectional_dataset(tickers, start_date, end_date)
        if dataset.empty:
            logger.warning("log_daily_predictions: no feature data available")
            return 0

        latest_date = dataset["date"].max()
        snapshot    = dataset[dataset["date"] == latest_date].copy()

        preds = predictor.predict(snapshot)
        if preds.empty:
            logger.warning("log_daily_predictions: no predictions generated")
            return 0

        meta          = registry.load_latest_metadata()
        model_version = meta.get("timestamp", "unknown")
        as_of_date    = latest_date.date() if hasattr(latest_date, "date") else latest_date

        records = [
            {
                "as_of_date":           as_of_date,
                "ticker":               ticker,
                "predicted_return_21d": float(pred),
                "model_version":        model_version,
            }
            for ticker, pred in preds.items()
        ]

        upsert_predictions(records)
        logger.info(
            "log_daily_predictions: logged %d predictions for %s",
            len(records), as_of_date,
        )
        return len(records)

    except Exception as e:
        logger.error("log_daily_predictions failed: %s", e)
        return 0


# ---------------------------------------------------------------------------
# IC computation
# ---------------------------------------------------------------------------

def compute_rolling_ic(
    pred_df: pd.DataFrame,
    prices_dict: dict[str, pd.DataFrame],
) -> tuple[float, int]:
    """
    Compute Spearman IC between predicted 21d returns and actual returns.

    pred_df:      columns [as_of_date, ticker, predicted_return_21d]
    prices_dict:  {ticker: DataFrame with columns [date, close]} — pre-fetched

    For each prediction we find the close on as_of_date and the close 21
    trading days later, compute the actual return, then compute the
    cross-sectional Spearman rank correlation.

    Returns (ic, n_samples).
    """
    actuals = []

    for _, row in pred_df.iterrows():
        ticker = row["ticker"]
        as_of  = row["as_of_date"]

        if ticker not in prices_dict:
            actuals.append(None)
            continue

        price_df     = prices_dict[ticker]
        future_prices = price_df[price_df["date"] >= as_of].sort_values("date")

        # Need the close on as_of_date (index 0) and 21 trading days later (index 21)
        if len(future_prices) < 22:
            actuals.append(None)
            continue

        close_start = future_prices.iloc[0]["close"]
        close_end   = future_prices.iloc[21]["close"]

        if pd.isna(close_start) or pd.isna(close_end) or close_start <= 0:
            actuals.append(None)
            continue

        actuals.append((close_end - close_start) / close_start)

    pred_df          = pred_df.copy()
    pred_df["actual"] = actuals
    valid            = pred_df.dropna(subset=["actual"])
    n                = len(valid)

    if n < MIN_IC_SAMPLES:
        logger.warning(
            "compute_rolling_ic: only %d valid samples (need %d) — IC unreliable",
            n, MIN_IC_SAMPLES,
        )
        return 0.0, n

    ic, _ = stats.spearmanr(valid["predicted_return_21d"], valid["actual"])
    ic    = float(ic) if not np.isnan(ic) else 0.0

    logger.info("Rolling IC: %.4f over %d samples", ic, n)
    return ic, n


# ---------------------------------------------------------------------------
# Weekly: full IC check + gated retrain
# ---------------------------------------------------------------------------

def run_weekly_ic_check() -> dict:
    """
    Full health check orchestration:
      1. Pull prediction_log rows from the IC window
      2. Pre-fetch prices for all tickers in the window
      3. Compute rolling Spearman IC
      4. Gate check + consecutive failure tracking
      5. Trigger gated retrain if degraded (but not circuit-broken)
      6. Write a ModelMetrics row

    Returns the metrics dict that was written (empty dict on data shortage).
    """
    today    = date.today()
    ic_end   = today - timedelta(days=32)   # 32 cal days ≈ 21 trading days lookforward
    ic_start = today - timedelta(days=IC_WINDOW_CALENDAR_DAYS)

    pred_df = get_predictions_window(ic_start.isoformat(), ic_end.isoformat())
    if pred_df.empty:
        logger.warning(
            "run_weekly_ic_check: no predictions in window %s → %s", ic_start, ic_end
        )
        return {}

    # Pre-fetch prices for every ticker in the window (one DB query per ticker)
    tickers     = pred_df["ticker"].unique().tolist()
    prices_dict = {}
    for ticker in tickers:
        df = get_prices(ticker, start=ic_start.isoformat(), end=today.isoformat())
        if not df.empty:
            prices_dict[ticker] = df

    rolling_ic, n_samples = compute_rolling_ic(pred_df, prices_dict)

    # Consecutive failure tracking
    latest        = get_latest_model_metrics()
    prev_failures = latest.get("consecutive_failures", 0) if latest else 0
    is_healthy    = rolling_ic >= IC_GATE

    if is_healthy:
        consecutive_failures = 0
        gate_status          = "HEALTHY"
        retrain_triggered    = False
        retrain_result       = None
        new_model_ic         = None

    elif prev_failures + 1 >= CIRCUIT_BREAKER_THRESHOLD:
        consecutive_failures = prev_failures + 1
        gate_status          = "CIRCUIT_BREAKER"
        retrain_triggered    = False
        retrain_result       = None
        new_model_ic         = None
        logger.error(
            "CIRCUIT BREAKER FIRED after %d consecutive failures "
            "(rolling IC=%.4f). Falling back to equal-weight. "
            "Manual feature engineering intervention required.",
            consecutive_failures, rolling_ic,
        )

    else:
        consecutive_failures = prev_failures + 1
        gate_status          = "DEGRADED"
        retrain_triggered    = True
        logger.warning(
            "Model degraded — rolling IC=%.4f (failure %d). Triggering gated retrain.",
            rolling_ic, consecutive_failures,
        )
        retrain_result, new_model_ic = _gated_retrain()

    metrics_row = {
        "computed_at":          datetime.utcnow(),
        "rolling_ic":           round(rolling_ic, 6),
        "n_samples":            n_samples,
        "gate_status":          gate_status,
        "consecutive_failures": consecutive_failures,
        "retrain_triggered":    retrain_triggered,
        "retrain_result":       retrain_result,
        "new_model_ic":         round(new_model_ic, 6) if new_model_ic is not None else None,
    }

    upsert_model_metrics(metrics_row)
    logger.info("Weekly IC check complete: %s", gate_status)
    return metrics_row


# ---------------------------------------------------------------------------
# Gated retrain
# ---------------------------------------------------------------------------

def _gated_retrain() -> tuple[str, float]:
    """
    Train a new model and keep it only if it passes the IC gate.

    predictor.train() always saves the artifact on completion. If the new
    model fails the gate, registry.delete_latest() removes the artifact so
    the previous model remains active.

    Returns (result, new_model_ic) where result is 'DEPLOYED' or 'REJECTED'.
    """
    try:
        metrics = predictor.train()
        new_ic  = float(metrics.get("mean_ic", 0.0))

        if new_ic >= IC_GATE:
            logger.info("Gated retrain DEPLOYED — new training IC=%.4f", new_ic)
            return "DEPLOYED", new_ic
        else:
            registry.delete_latest()
            logger.info(
                "Gated retrain REJECTED — new IC=%.4f < gate=%.4f. Old model retained.",
                new_ic, IC_GATE,
            )
            return "REJECTED", new_ic

    except Exception as e:
        logger.error("Gated retrain failed with exception: %s", e)
        return "REJECTED", 0.0
