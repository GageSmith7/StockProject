"""
Lookahead bias tests for analytics/features.py

Core invariant: a feature value on date T must not change if all data
after T is removed from the input.  If any feature uses future data,
truncating the history at T would produce a different value — caught here.

All tests use synthetic OHLCV data (no DB required).
"""
import numpy as np
import pandas as pd
import pytest

from app.analytics.features import (
    TARGET_COL,
    TARGET_HORIZON,
    WARMUP_DAYS,
    FEATURE_COLS,
    PER_TICKER_COLS,
    build_features,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prices(n: int, ticker: str = "TEST", seed: int = 42) -> pd.DataFrame:
    """Return n business days of synthetic OHLCV with realistic price path."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n)
    close = 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    return pd.DataFrame({
        "date":   dates,
        "ticker": ticker,
        "open":   close * 0.999,
        "high":   close * 1.005,
        "low":    close * 0.995,
        "close":  close,
        "volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
    })


# ---------------------------------------------------------------------------
# Lookahead bias
# ---------------------------------------------------------------------------

class TestNoLookaheadBias:
    """
    Verify that no feature in PER_TICKER_COLS depends on future data.

    Method: build features on a full price series, then truncate the series
    at an arbitrary cutoff and rebuild.  Any row that appears in both outputs
    must have identical feature values — if future data leaks in, it won't.
    """

    def test_features_unchanged_after_truncation(self):
        """Feature values on shared dates must be identical before and after cutoff."""
        n = 600
        prices_full = _make_prices(n)

        cutoff = n - 50                                      # drop last 50 rows
        prices_trunc = prices_full.iloc[:cutoff].copy()

        feats_full  = build_features(prices_full)
        feats_trunc = build_features(prices_trunc)

        feats_full["date"]  = pd.to_datetime(feats_full["date"])
        feats_trunc["date"] = pd.to_datetime(feats_trunc["date"])

        shared_dates = set(feats_full["date"]) & set(feats_trunc["date"])
        assert len(shared_dates) > 100, "Too few shared dates to be a meaningful test"

        full_sub  = feats_full[feats_full["date"].isin(shared_dates)].set_index("date").sort_index()
        trunc_sub = feats_trunc[feats_trunc["date"].isin(shared_dates)].set_index("date").sort_index()

        for col in PER_TICKER_COLS:
            pd.testing.assert_series_equal(
                full_sub[col],
                trunc_sub[col],
                check_names=False,
                rtol=1e-5,
                obj=f"Feature '{col}'",
            )

    def test_features_unchanged_at_single_day_boundary(self):
        """
        Stricter version: remove exactly ONE future day and verify no feature changes.
        Catches off-by-one errors in rolling windows.
        """
        n = 500
        prices_full  = _make_prices(n)
        prices_minus1 = prices_full.iloc[:-1].copy()

        feats_full   = build_features(prices_full)
        feats_minus1 = build_features(prices_minus1)

        feats_full["date"]   = pd.to_datetime(feats_full["date"])
        feats_minus1["date"] = pd.to_datetime(feats_minus1["date"])

        shared_dates = set(feats_full["date"]) & set(feats_minus1["date"])
        assert len(shared_dates) > 0

        full_sub   = feats_full[feats_full["date"].isin(shared_dates)].set_index("date").sort_index()
        minus1_sub = feats_minus1[feats_minus1["date"].isin(shared_dates)].set_index("date").sort_index()

        for col in PER_TICKER_COLS:
            pd.testing.assert_series_equal(
                full_sub[col],
                minus1_sub[col],
                check_names=False,
                rtol=1e-5,
                obj=f"Feature '{col}' (single-day boundary)",
            )


# ---------------------------------------------------------------------------
# Target label correctness
# ---------------------------------------------------------------------------

class TestTargetLabel:
    """
    target_21d is intentionally future data — it's the prediction label.
    These tests verify it's computed correctly and kept out of FEATURE_COLS.
    """

    def test_target_equals_21d_forward_return(self):
        """target_21d on date T must equal (close[T+21] - close[T]) / close[T]."""
        n = 500
        prices = _make_prices(n)
        prices["date"] = pd.to_datetime(prices["date"])
        feats = build_features(prices)
        feats["date"] = pd.to_datetime(feats["date"])

        # Check 5 rows spread across the output
        indices = [0, 10, 50, 100, len(feats) - 1]
        for i in indices:
            sample   = feats.iloc[i]
            t_date   = pd.Timestamp(sample["date"])
            t_idx    = prices.index[prices["date"] == t_date].item()

            close_t        = prices.iloc[t_idx]["close"]
            close_t_plus21 = prices.iloc[t_idx + TARGET_HORIZON]["close"]
            expected       = (close_t_plus21 - close_t) / close_t

            assert abs(sample[TARGET_COL] - expected) < 1e-6, (
                f"target_21d mismatch at row {i} ({t_date}): "
                f"got {sample[TARGET_COL]:.8f}, expected {expected:.8f}"
            )

    def test_target_not_in_feature_cols(self):
        """target_21d must never appear in FEATURE_COLS — it would leak the label."""
        assert TARGET_COL not in FEATURE_COLS, (
            f"'{TARGET_COL}' is in FEATURE_COLS. "
            "This leaks the prediction target into training features."
        )


# ---------------------------------------------------------------------------
# Row-dropping correctness
# ---------------------------------------------------------------------------

class TestRowDropping:
    """
    build_features() must drop warmup rows (insufficient history) and
    tail rows (no forward return available).  Verify both.
    """

    def test_warmup_rows_dropped(self):
        """
        First output row must correspond to prices.iloc[WARMUP_DAYS],
        i.e. the first row with a valid 252-day lookback window.
        """
        n = 500
        prices = _make_prices(n)
        prices["date"] = pd.to_datetime(prices["date"])
        feats = build_features(prices)
        feats["date"] = pd.to_datetime(feats["date"])

        expected_first_date = prices.iloc[WARMUP_DAYS]["date"]
        actual_first_date   = feats["date"].min()

        assert actual_first_date == expected_first_date, (
            f"Expected first output date {expected_first_date}, got {actual_first_date}. "
            f"Warmup rows ({WARMUP_DAYS}) may not be correctly dropped."
        )

    def test_tail_rows_dropped_no_forward_return(self):
        """
        The last TARGET_HORIZON rows have no forward return — they must be absent.
        Last output date must be strictly before last input date.
        """
        n = 500
        prices = _make_prices(n)
        prices["date"] = pd.to_datetime(prices["date"])
        feats = build_features(prices)
        feats["date"] = pd.to_datetime(feats["date"])

        last_output_date = feats["date"].max()
        last_input_date  = prices["date"].max()

        assert last_output_date < last_input_date, (
            f"Last output date ({last_output_date}) == last input date ({last_input_date}). "
            f"The final {TARGET_HORIZON} rows should be dropped (no target available)."
        )

    def test_output_row_count(self):
        """Output row count must equal n - WARMUP_DAYS - TARGET_HORIZON."""
        n = 500
        prices = _make_prices(n)
        feats = build_features(prices)

        expected = n - WARMUP_DAYS - TARGET_HORIZON
        assert len(feats) == expected, (
            f"Expected {expected} rows, got {len(feats)}. "
            "Check warmup drop and tail drop counts."
        )

    def test_insufficient_history_returns_empty(self):
        """Fewer than WARMUP_DAYS + TARGET_HORIZON + 1 rows must return an empty DataFrame."""
        prices = _make_prices(WARMUP_DAYS + TARGET_HORIZON - 1)
        feats  = build_features(prices)
        assert feats.empty, (
            f"Expected empty DataFrame for {WARMUP_DAYS + TARGET_HORIZON - 1} input rows, "
            f"but got {len(feats)} rows."
        )

    def test_empty_input_returns_empty(self):
        """Empty input DataFrame must return empty output without raising."""
        empty = pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])
        feats = build_features(empty)
        assert feats.empty


# ---------------------------------------------------------------------------
# Feature quality
# ---------------------------------------------------------------------------

class TestFeatureQuality:
    """Sanity checks on feature values after build_features()."""

    def test_no_nan_in_per_ticker_cols(self):
        """All PER_TICKER_COLS must be NaN-free after warmup rows are dropped."""
        n = 500
        prices = _make_prices(n)
        feats  = build_features(prices)

        for col in PER_TICKER_COLS:
            nan_count = feats[col].isna().sum()
            assert nan_count == 0, (
                f"Feature '{col}' has {nan_count} NaN rows. "
                "Rolling window or warmup logic may be incorrect."
            )

    def test_rsi_bounded(self):
        """RSI must be in [0, 100]."""
        n = 500
        prices = _make_prices(n)
        feats  = build_features(prices)

        assert feats["rsi_14"].between(0, 100).all(), (
            f"RSI out of [0, 100]: min={feats['rsi_14'].min():.2f}, "
            f"max={feats['rsi_14'].max():.2f}"
        )

    def test_vol_ratio_positive(self):
        """Volume ratio must be strictly positive."""
        n = 500
        prices = _make_prices(n)
        feats  = build_features(prices)

        assert (feats["vol_ratio"] > 0).all(), "vol_ratio has non-positive values"

    def test_volatility_positive(self):
        """Rolling volatility must be positive."""
        n = 500
        prices = _make_prices(n)
        feats  = build_features(prices)

        assert (feats["vol_21d"] > 0).all(), "vol_21d has non-positive values"
        assert (feats["vol_63d"] > 0).all(), "vol_63d has non-positive values"
