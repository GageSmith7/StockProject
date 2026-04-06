import logging
import os

from dotenv import load_dotenv
import httpx

load_dotenv()
import pandas as pd
import yfinance as yf
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

FRED_SERIES_MVP = ["DFF", "CPIAUCSL", "UNRATE", "T10Y2Y"]
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


# ---------------------------------------------------------------------------
# Ticker (yFinance)
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=8),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _yf_download(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df.empty:
        raise ValueError(f"yFinance returned no data for {ticker}")
    return df


def _normalize_ohlcv(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Flatten yFinance MultiIndex output to standard schema."""
    # yfinance 0.2.x returns MultiIndex columns: (Price, Ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "date"
    df = df.reset_index()
    df["ticker"] = ticker.upper()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df[["date", "open", "high", "low", "close", "volume", "ticker"]]


def fetch_ticker(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch daily OHLCV data for a ticker via yFinance.

    auto_adjust=True handles splits and dividends — do not adjust manually.
    Retries up to 3 times with 2s / 4s / 8s exponential backoff.
    Returns empty DataFrame on failure — never raises.

    Output columns: date, open, high, low, close, volume, ticker
    """
    try:
        raw = _yf_download(ticker, start_date, end_date)
        return _normalize_ohlcv(raw, ticker)
    except Exception as e:
        logger.warning("fetch_ticker failed for %s: %s", ticker, e)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Macro data (FRED)
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=8),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _fred_get(series_id: str, start_date: str, end_date: str, api_key: str) -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "observation_start": start_date,
        "observation_end": end_date,
        "file_type": "json",
    }
    response = httpx.get(FRED_BASE_URL, params=params, timeout=30)
    response.raise_for_status()

    observations = response.json()["observations"]
    if not observations:
        raise ValueError(f"FRED returned no observations for {series_id}")

    df = pd.DataFrame(observations)[["date", "value"]]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def fetch_fred_series(series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch a single FRED macro time series.

    Requires FRED_API_KEY in environment (free at fred.stlouisfed.org).
    Retries up to 3 times with 2s / 4s / 8s exponential backoff.
    Returns empty DataFrame on failure — never raises.

    Output columns: date, series_id, value
    """
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        logger.error("FRED_API_KEY not set in environment — cannot fetch %s", series_id)
        return pd.DataFrame()

    try:
        df = _fred_get(series_id, start_date, end_date, api_key)
        df.insert(1, "series_id", series_id)
        return df[["date", "series_id", "value"]]
    except Exception as e:
        logger.warning("fetch_fred_series failed for %s: %s", series_id, e)
        return pd.DataFrame()


def fetch_all_fred_series(start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
    """
    Fetch all MVP macro series (DFF, CPIAUCSL, UNRATE).
    Returns dict of series_id -> DataFrame.
    Failures return empty DataFrames — partial results are acceptable.
    """
    return {
        series_id: fetch_fred_series(series_id, start_date, end_date)
        for series_id in FRED_SERIES_MVP
    }
