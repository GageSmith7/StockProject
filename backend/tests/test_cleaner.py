"""
Step 2 acceptance criteria:
- Zero NaN values in close column after cleaning (for gaps <= MAX_FILL_DAYS)
- data_quality_flag present on every row
- Outlier rows are present with flag, not removed
"""
import numpy as np
import pandas as pd
import pytest

from app.data.cleaner import clean, MAX_FILL_DAYS, OUTLIER_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ohlcv(n: int = 10, base_close: float = 100.0, ticker: str = "TEST") -> pd.DataFrame:
    """Build a clean synthetic OHLCV DataFrame with n rows."""
    dates = pd.date_range("2023-01-02", periods=n, freq="B")  # business days
    closes = [base_close + i for i in range(n)]
    return pd.DataFrame({
        "date": dates,
        "open":   closes,
        "high":   [c + 1 for c in closes],
        "low":    [c - 1 for c in closes],
        "close":  closes,
        "volume": [1_000_000] * n,
        "ticker": ticker,
    })


# ---------------------------------------------------------------------------
# Flag column
# ---------------------------------------------------------------------------

class TestFlagColumn:
    def test_flag_present_on_every_row(self):
        df = clean(make_ohlcv())
        assert "data_quality_flag" in df.columns
        assert df["data_quality_flag"].notna().all()

    def test_clean_data_all_flagged_clean(self):
        df = clean(make_ohlcv())
        assert (df["data_quality_flag"] == "clean").all()

    def test_empty_dataframe_returns_empty(self):
        result = clean(pd.DataFrame())
        assert result.empty


# ---------------------------------------------------------------------------
# Forward-fill — short gaps (<= MAX_FILL_DAYS)
# ---------------------------------------------------------------------------

class TestForwardFill:
    def test_single_nan_is_filled(self):
        raw = make_ohlcv(10)
        raw.loc[3, "close"] = np.nan
        result = clean(raw)
        assert result["close"].isna().sum() == 0

    def test_filled_row_flagged_forward_filled(self):
        raw = make_ohlcv(10)
        raw.loc[3, "close"] = np.nan
        result = clean(raw)
        assert result.loc[3, "data_quality_flag"] == "forward_filled"

    def test_max_fill_days_gap_is_filled(self):
        raw = make_ohlcv(15)
        raw.loc[4:4 + MAX_FILL_DAYS - 1, "close"] = np.nan
        result = clean(raw)
        assert result["close"].isna().sum() == 0

    def test_filled_value_matches_previous_close(self):
        raw = make_ohlcv(10)
        prev_close = raw.loc[3, "close"]
        raw.loc[4, "close"] = np.nan
        result = clean(raw)
        assert result.loc[4, "close"] == prev_close

    def test_non_gap_rows_remain_clean(self):
        raw = make_ohlcv(10)
        raw.loc[4, "close"] = np.nan
        result = clean(raw)
        non_gap = result[result["data_quality_flag"] != "forward_filled"]
        assert (non_gap["data_quality_flag"] == "clean").all()

    def test_all_price_cols_are_forward_filled(self):
        raw = make_ohlcv(10)
        for col in ["open", "high", "low", "close", "volume"]:
            raw.loc[4, col] = np.nan
        result = clean(raw)
        for col in ["open", "high", "low", "close", "volume"]:
            assert not pd.isna(result.loc[4, col]), f"{col} was not filled"


# ---------------------------------------------------------------------------
# Long gaps (> MAX_FILL_DAYS) — not filled, warning logged
# ---------------------------------------------------------------------------

class TestLongGaps:
    def test_long_gap_not_filled(self):
        raw = make_ohlcv(20)
        gap_slice = slice(5, 5 + MAX_FILL_DAYS + 1)  # one more than limit
        raw.loc[gap_slice, "close"] = np.nan
        result = clean(raw)
        assert result.loc[gap_slice, "close"].isna().any()

    def test_long_gap_logs_warning(self, caplog):
        import logging
        raw = make_ohlcv(20)
        raw.loc[5:5 + MAX_FILL_DAYS, "close"] = np.nan  # MAX_FILL_DAYS + 1 rows
        with caplog.at_level(logging.WARNING, logger="app.data.cleaner"):
            clean(raw)
        assert any("not filled" in msg for msg in caplog.messages)

    def test_short_gap_does_not_log_warning(self, caplog):
        import logging
        raw = make_ohlcv(10)
        raw.loc[4, "close"] = np.nan  # only 1 missing — within limit
        with caplog.at_level(logging.WARNING, logger="app.data.cleaner"):
            clean(raw)
        assert not any("not filled" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# Outlier flagging
# ---------------------------------------------------------------------------

class TestOutlierFlagging:
    def _inject_spike(self, df: pd.DataFrame, idx: int, multiplier: float = 1.20) -> pd.DataFrame:
        """Multiply close at idx by multiplier to simulate a price spike."""
        df = df.copy()
        df.loc[idx, "close"] = df.loc[idx - 1, "close"] * multiplier
        return df

    def test_outlier_row_is_flagged(self):
        raw = self._inject_spike(make_ohlcv(10), idx=5)
        result = clean(raw)
        assert result.loc[5, "data_quality_flag"] == "outlier_flagged"

    def test_outlier_row_is_not_removed(self):
        raw = self._inject_spike(make_ohlcv(10), idx=5)
        result = clean(raw)
        assert len(result) == len(raw)

    def test_outlier_close_value_unchanged(self):
        raw = self._inject_spike(make_ohlcv(10), idx=5)
        original_close = raw.loc[5, "close"]
        result = clean(raw)
        assert result.loc[5, "close"] == original_close

    def test_below_threshold_not_flagged(self):
        raw = make_ohlcv(10)
        # Inject a move just below threshold
        raw.loc[5, "close"] = raw.loc[4, "close"] * (1 + OUTLIER_THRESHOLD - 0.01)
        result = clean(raw)
        assert result.loc[5, "data_quality_flag"] == "clean"

    def test_forward_filled_row_takes_priority_over_outlier(self):
        """A forward-filled row should stay 'forward_filled', not 'outlier_flagged'."""
        raw = make_ohlcv(10)
        # Set a NaN that when filled will look like a big jump
        raw.loc[4, "close"] = np.nan
        raw.loc[3, "close"] = 100.0
        raw.loc[5, "close"] = 200.0  # big jump after gap
        result = clean(raw)
        assert result.loc[4, "data_quality_flag"] == "forward_filled"


# ---------------------------------------------------------------------------
# Input is not mutated
# ---------------------------------------------------------------------------

class TestImmutability:
    def test_original_dataframe_not_modified(self):
        raw = make_ohlcv(10)
        raw.loc[4, "close"] = np.nan
        original_copy = raw.copy()
        clean(raw)
        pd.testing.assert_frame_equal(raw, original_copy)
