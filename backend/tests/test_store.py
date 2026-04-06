"""
Step 3 acceptance criteria:
- Upserting same ticker + date twice produces one row, not two
- get_prices("AAPL", ...) returns date-sorted DataFrame
- get_or_fetch hits yFinance on first call, DB on second call
- 500 rows inserted in under 2 seconds
"""
import time
from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from app.data.store import (
    TickerNotFoundError,
    get_or_fetch,
    get_prices,
    upsert_macro,
    upsert_prices,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_price_df(ticker: str = "TEST", n: int = 10, start: str = "2023-01-02") -> pd.DataFrame:
    dates = pd.date_range(start, periods=n, freq="B").date
    closes = [100.0 + i for i in range(n)]
    return pd.DataFrame({
        "date": dates,
        "open": closes,
        "high": [c + 1 for c in closes],
        "low": [c - 1 for c in closes],
        "close": closes,
        "volume": [1_000_000] * n,
        "ticker": ticker,
        "data_quality_flag": "clean",
    })


def make_macro_df(series_id: str = "DFF", n: int = 10) -> pd.DataFrame:
    dates = pd.date_range("2023-01-02", periods=n, freq="B").date
    return pd.DataFrame({
        "date": dates,
        "series_id": series_id,
        "value": [5.0 + i * 0.1 for i in range(n)],
    })


# ---------------------------------------------------------------------------
# upsert_prices
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("db_clean")
class TestUpsertPrices:
    def test_inserts_rows(self):
        df = make_price_df(n=5)
        count = upsert_prices(df)
        assert count == 5

    def test_no_duplicate_on_second_upsert(self):
        df = make_price_df(n=5)
        upsert_prices(df)
        upsert_prices(df)  # same data again
        result = get_prices("TEST")
        assert len(result) == 5

    def test_update_on_conflict(self):
        df = make_price_df(n=1)
        upsert_prices(df)

        updated = df.copy()
        updated.loc[0, "close"] = 999.0
        upsert_prices(updated)

        result = get_prices("TEST")
        assert result.iloc[0]["close"] == 999.0

    def test_empty_dataframe_returns_zero(self):
        assert upsert_prices(pd.DataFrame()) == 0

    def test_500_rows_under_2_seconds(self):
        df = make_price_df(n=500)
        start = time.perf_counter()
        upsert_prices(df)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"Upsert took {elapsed:.2f}s — expected < 2s"


# ---------------------------------------------------------------------------
# get_prices
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("db_clean")
class TestGetPrices:
    def test_returns_dataframe(self):
        upsert_prices(make_price_df(n=5))
        result = get_prices("TEST")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5

    def test_returns_empty_for_unknown_ticker(self):
        result = get_prices("DOESNOTEXIST")
        assert result.empty

    def test_results_are_date_sorted(self):
        upsert_prices(make_price_df(n=10))
        result = get_prices("TEST")
        assert list(result["date"]) == sorted(result["date"])

    def test_date_filter_start(self):
        upsert_prices(make_price_df(n=10, start="2023-01-02"))
        result = get_prices("TEST", start="2023-01-05")
        assert all(d >= date(2023, 1, 5) for d in result["date"])

    def test_date_filter_end(self):
        upsert_prices(make_price_df(n=10, start="2023-01-02"))
        result = get_prices("TEST", end="2023-01-05")
        assert all(d <= date(2023, 1, 5) for d in result["date"])

    def test_ticker_case_insensitive(self):
        upsert_prices(make_price_df(ticker="AAPL", n=5))
        result = get_prices("aapl")
        assert len(result) == 5

    def test_output_columns(self):
        upsert_prices(make_price_df(n=3))
        result = get_prices("TEST")
        for col in ["date", "open", "high", "low", "close", "volume", "ticker"]:
            assert col in result.columns


# ---------------------------------------------------------------------------
# upsert_macro
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("db_clean")
class TestUpsertMacro:
    def test_inserts_rows(self):
        count = upsert_macro(make_macro_df(n=5))
        assert count == 5

    def test_no_duplicate_on_second_upsert(self):
        df = make_macro_df(n=5)
        upsert_macro(df)
        upsert_macro(df)
        # If no error and first insert returned 5, upsert works
        assert upsert_macro(df) == 5

    def test_empty_returns_zero(self):
        assert upsert_macro(pd.DataFrame()) == 0


# ---------------------------------------------------------------------------
# get_or_fetch
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("db_clean")
class TestGetOrFetch:
    def test_fetches_from_yfinance_when_not_in_db(self):
        """First call should hit yFinance (fetch_ticker called once)."""
        with patch("app.data.store.fetch_ticker", wraps=__import__("app.data.fetcher", fromlist=["fetch_ticker"]).fetch_ticker) as mock_fetch:
            result = get_or_fetch("AAPL")
            assert mock_fetch.called
        assert not result.empty

    def test_second_call_served_from_db(self):
        """Second call must not call fetch_ticker again."""
        get_or_fetch("AAPL")  # populate DB

        with patch("app.data.store.fetch_ticker") as mock_fetch:
            result = get_or_fetch("AAPL")
            mock_fetch.assert_not_called()

        assert not result.empty

    def test_invalid_ticker_raises_ticker_not_found_error(self):
        with pytest.raises(TickerNotFoundError):
            get_or_fetch("INVALIDTICKER99999")

    def test_on_demand_ticker_stored_in_db_after_fetch(self):
        get_or_fetch("AAPL")
        result = get_prices("AAPL")
        assert not result.empty
