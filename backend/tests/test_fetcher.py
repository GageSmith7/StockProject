"""
Step 1 acceptance criteria:
- fetch_ticker("AAPL", "2020-01-01", "2024-01-01") returns DataFrame, zero NaN in close
- Failed ticker logs warning and returns empty DataFrame without raising
- fetch_fred_series returns DataFrame with date, series_id, value columns
"""
import os

import pandas as pd
import pytest

from app.data.fetcher import fetch_ticker, fetch_fred_series, fetch_all_fred_series


class TestFetchTicker:
    def test_returns_dataframe_for_valid_ticker(self):
        df = fetch_ticker("AAPL", "2020-01-01", "2024-01-01")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_output_columns(self):
        df = fetch_ticker("AAPL", "2020-01-01", "2024-01-01")
        assert list(df.columns) == ["date", "open", "high", "low", "close", "volume", "ticker"]

    def test_zero_nan_in_close(self):
        df = fetch_ticker("AAPL", "2020-01-01", "2024-01-01")
        assert df["close"].isna().sum() == 0

    def test_ticker_column_is_uppercase(self):
        df = fetch_ticker("aapl", "2023-01-01", "2024-01-01")
        assert (df["ticker"] == "AAPL").all()

    def test_invalid_ticker_returns_empty_dataframe(self):
        df = fetch_ticker("INVALIDTICKER99999", "2020-01-01", "2024-01-01")
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_invalid_ticker_does_not_raise(self):
        # Must not raise under any circumstances
        try:
            fetch_ticker("INVALIDTICKER99999", "2020-01-01", "2024-01-01")
        except Exception as e:
            pytest.fail(f"fetch_ticker raised an exception: {e}")


class TestFetchFredSeries:
    def test_returns_dataframe_when_api_key_set(self):
        if not os.environ.get("FRED_API_KEY"):
            pytest.skip("FRED_API_KEY not set")
        df = fetch_fred_series("DFF", "2020-01-01", "2024-01-01")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_output_columns(self):
        if not os.environ.get("FRED_API_KEY"):
            pytest.skip("FRED_API_KEY not set")
        df = fetch_fred_series("DFF", "2020-01-01", "2024-01-01")
        assert list(df.columns) == ["date", "series_id", "value"]

    def test_series_id_column_populated(self):
        if not os.environ.get("FRED_API_KEY"):
            pytest.skip("FRED_API_KEY not set")
        df = fetch_fred_series("UNRATE", "2020-01-01", "2024-01-01")
        assert (df["series_id"] == "UNRATE").all()

    def test_missing_api_key_returns_empty_dataframe(self, monkeypatch):
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        df = fetch_fred_series("DFF", "2020-01-01", "2024-01-01")
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_missing_api_key_does_not_raise(self, monkeypatch):
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        try:
            fetch_fred_series("DFF", "2020-01-01", "2024-01-01")
        except Exception as e:
            pytest.fail(f"fetch_fred_series raised an exception: {e}")
