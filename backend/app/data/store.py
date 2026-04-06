import logging
from datetime import date, datetime

import pandas as pd
from sqlalchemy.dialects.postgresql import insert

from app.database import MacroData, ModelMetrics, PredictionLog, Price, SessionLocal, Symbol, create_tables
from app.data.cleaner import clean
from app.data.fetcher import fetch_ticker

logger = logging.getLogger(__name__)

# How far back to fetch when a Tier 2 ticker is first requested
ON_DEMAND_START_DATE = "2010-01-01"


class TickerNotFoundError(Exception):
    pass


# ---------------------------------------------------------------------------
# Table setup
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create tables on first run. Called from main.py lifespan."""
    create_tables()
    logger.info("Database tables verified / created")


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def upsert_prices(df: pd.DataFrame) -> int:
    """
    Batch upsert cleaned OHLCV DataFrame into prices table.
    ON CONFLICT (date, ticker) DO UPDATE — never creates duplicates.
    Returns number of rows processed.
    """
    if df.empty:
        return 0

    rows = _price_df_to_records(df)

    stmt = insert(Price).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["date", "ticker"],
        set_={
            "open": stmt.excluded.open,
            "high": stmt.excluded.high,
            "low": stmt.excluded.low,
            "close": stmt.excluded.close,
            "volume": stmt.excluded.volume,
            "quality_flag": stmt.excluded.quality_flag,
            "updated_at": stmt.excluded.updated_at,
        },
    )

    with SessionLocal() as session:
        session.execute(stmt)
        session.commit()

    return len(rows)


def upsert_macro(df: pd.DataFrame) -> int:
    """
    Batch upsert FRED macro DataFrame into macro_data table.
    Returns number of rows processed.
    """
    if df.empty:
        return 0

    rows = _macro_df_to_records(df)

    stmt = insert(MacroData).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["date", "series_id"],
        set_={"value": stmt.excluded.value},
    )

    with SessionLocal() as session:
        session.execute(stmt)
        session.commit()

    return len(rows)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def get_prices(ticker: str, start: str = None, end: str = None) -> pd.DataFrame:
    """
    Query prices table for a ticker, optionally filtered by date range.
    Returns date-sorted DataFrame. Returns empty DataFrame if ticker not found.
    """
    with SessionLocal() as session:
        query = session.query(Price).filter(Price.ticker == ticker.upper())
        if start:
            query = query.filter(Price.date >= start)
        if end:
            query = query.filter(Price.date <= end)
        rows = query.order_by(Price.date).all()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame([
        {
            "date": r.date,
            "open": float(r.open) if r.open is not None else None,
            "high": float(r.high) if r.high is not None else None,
            "low": float(r.low) if r.low is not None else None,
            "close": float(r.close),
            "volume": int(r.volume) if r.volume is not None else None,
            "ticker": r.ticker,
            "quality_flag": r.quality_flag,
        }
        for r in rows
    ])


# ---------------------------------------------------------------------------
# Prediction log
# ---------------------------------------------------------------------------

def upsert_predictions(records: list[dict]) -> int:
    """
    Batch upsert prediction_log rows.
    ON CONFLICT (as_of_date, ticker) DO NOTHING — predictions for a given date
    are stable; we never overwrite a logged prediction.
    Returns number of rows processed.
    """
    if not records:
        return 0

    stmt = insert(PredictionLog).values(records)
    stmt = stmt.on_conflict_do_nothing(
        index_elements=["as_of_date", "ticker"],
    )

    with SessionLocal() as session:
        session.execute(stmt)
        session.commit()

    return len(records)


def get_predictions_window(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch prediction_log rows within [start_date, end_date].
    Returns DataFrame with columns: as_of_date, ticker, predicted_return_21d.
    """
    with SessionLocal() as session:
        rows = (
            session.query(PredictionLog)
            .filter(
                PredictionLog.as_of_date >= start_date,
                PredictionLog.as_of_date <= end_date,
            )
            .all()
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame([
        {
            "as_of_date":           r.as_of_date,
            "ticker":               r.ticker,
            "predicted_return_21d": float(r.predicted_return_21d),
        }
        for r in rows
    ])


# ---------------------------------------------------------------------------
# Model metrics
# ---------------------------------------------------------------------------

def upsert_model_metrics(row: dict) -> None:
    """Append a model health check result to model_metrics."""
    with SessionLocal() as session:
        session.add(ModelMetrics(**row))
        session.commit()


def get_latest_model_metrics() -> dict | None:
    """Return the most recent model_metrics row as a dict, or None if empty."""
    with SessionLocal() as session:
        row = (
            session.query(ModelMetrics)
            .order_by(ModelMetrics.computed_at.desc())
            .first()
        )
    if row is None:
        return None
    return {
        "computed_at":          row.computed_at.isoformat(),
        "rolling_ic":           float(row.rolling_ic) if row.rolling_ic is not None else None,
        "n_samples":            row.n_samples,
        "gate_status":          row.gate_status,
        "consecutive_failures": row.consecutive_failures,
        "retrain_triggered":    row.retrain_triggered,
        "retrain_result":       row.retrain_result,
        "new_model_ic":         float(row.new_model_ic) if row.new_model_ic is not None else None,
    }


# ---------------------------------------------------------------------------
# On-demand tier
# ---------------------------------------------------------------------------

def get_or_fetch(ticker: str) -> pd.DataFrame:
    """
    Tier 2 on-demand fetch logic.

    Fast path:  ticker already in DB → return immediately.
    Slow path:  not in DB → fetch from yFinance, clean, store, return.
                Marks symbol as data_source='on_demand' in symbols table.

    Raises TickerNotFoundError if yFinance returns no data.
    """
    ticker = ticker.upper()

    existing = get_prices(ticker)
    if not existing.empty:
        logger.debug("get_or_fetch: %s served from DB", ticker)
        return existing

    logger.info("get_or_fetch: %s not in DB — fetching from yFinance", ticker)
    today = date.today().isoformat()
    raw = fetch_ticker(ticker, ON_DEMAND_START_DATE, today)

    if raw.empty:
        raise TickerNotFoundError(
            f"Ticker not found or insufficient history: {ticker}"
        )

    cleaned = clean(raw)
    upsert_prices(cleaned)
    _register_on_demand_symbol(ticker)

    logger.info("get_or_fetch: stored %d rows for %s", len(cleaned), ticker)
    return get_prices(ticker)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _price_df_to_records(df: pd.DataFrame) -> list[dict]:
    now = datetime.utcnow()
    records = []
    for row in df.itertuples(index=False):
        records.append({
            "date": row.date,
            "ticker": row.ticker,
            "open": _to_float(row.open),
            "high": _to_float(row.high),
            "low": _to_float(row.low),
            "close": float(row.close),
            "volume": _to_int(row.volume),
            "quality_flag": getattr(row, "data_quality_flag", "clean"),
            "updated_at": now,
        })
    return records


def _macro_df_to_records(df: pd.DataFrame) -> list[dict]:
    return [
        {
            "date": row.date,
            "series_id": row.series_id,
            "value": _to_float(row.value),
        }
        for row in df.itertuples(index=False)
    ]


def _register_on_demand_symbol(ticker: str) -> None:
    stmt = insert(Symbol).values(
        ticker=ticker,
        data_source="on_demand",
        last_fetched=datetime.utcnow(),
    ).on_conflict_do_update(
        index_elements=["ticker"],
        set_={
            "data_source": "on_demand",
            "last_fetched": datetime.utcnow(),
        },
    )
    with SessionLocal() as session:
        session.execute(stmt)
        session.commit()


def _to_float(val) -> float | None:
    try:
        return None if val is None or (val != val) else float(val)
    except (TypeError, ValueError):
        return None


def _to_int(val) -> int | None:
    try:
        return None if val is None or (val != val) else int(val)
    except (TypeError, ValueError):
        return None
