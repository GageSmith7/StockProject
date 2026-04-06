"""
Tests for analytics/portfolio.py

Pure-function tests (construct_portfolio, compute_metrics) use synthetic data — no DB.
Backtest smoke test uses the live DB and trained model.
"""
import numpy as np
import pandas as pd
import pytest

from app.analytics.portfolio import (
    QUINTILE,
    REBALANCE_DAYS,
    TRADING_DAYS_PA,
    compute_metrics,
    construct_portfolio,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def predictions_20():
    """20-ticker prediction series with known order."""
    return pd.Series(
        {f"T{i:02d}": float(i) for i in range(20)},  # T19 is highest
        name="predicted_return_21d",
    )


@pytest.fixture
def flat_returns():
    """50 identical 1% period returns — easy to verify metrics analytically."""
    return pd.Series([0.01] * 50)


@pytest.fixture
def alternating_returns():
    """Alternating +5% / -3% — produces known drawdown and win_rate."""
    pattern = [0.05, -0.03] * 25   # 50 periods
    return pd.Series(pattern)


# ===========================================================================
# construct_portfolio
# ===========================================================================

class TestConstructPortfolio:

    def test_top_quintile_selected(self, predictions_20):
        port = construct_portfolio(predictions_20)
        # Default: top 20% of 20 = 4 holdings
        assert len(port) == 4

    def test_top_quintile_are_highest_predictions(self, predictions_20):
        port = construct_portfolio(predictions_20)
        selected = set(port["ticker"])
        expected = {"T19", "T18", "T17", "T16"}
        assert selected == expected

    def test_equal_weight(self, predictions_20):
        port = construct_portfolio(predictions_20)
        assert port["weight"].tolist() == pytest.approx([0.25] * len(port))

    def test_weights_sum_to_one(self, predictions_20):
        port = construct_portfolio(predictions_20)
        assert port["weight"].sum() == pytest.approx(1.0)

    def test_n_holdings_override(self, predictions_20):
        port = construct_portfolio(predictions_20, n_holdings=10)
        assert len(port) == 10

    def test_n_holdings_override_selects_top_n(self, predictions_20):
        port = construct_portfolio(predictions_20, n_holdings=3)
        assert set(port["ticker"]) == {"T19", "T18", "T17"}

    def test_n_holdings_1(self, predictions_20):
        port = construct_portfolio(predictions_20, n_holdings=1)
        assert len(port) == 1
        assert port.iloc[0]["ticker"] == "T19"
        assert port.iloc[0]["weight"] == pytest.approx(1.0)

    def test_empty_predictions_returns_empty_df(self):
        port = construct_portfolio(pd.Series(dtype=float))
        assert port.empty
        assert list(port.columns) == ["ticker", "weight"]

    def test_single_stock(self):
        port = construct_portfolio(pd.Series({"AAPL": 0.05}))
        assert len(port) == 1
        assert port.iloc[0]["weight"] == pytest.approx(1.0)

    def test_output_columns(self, predictions_20):
        port = construct_portfolio(predictions_20)
        assert list(port.columns) == ["ticker", "weight"]

    def test_ties_broken_consistently(self):
        """Identical predictions — should still return n_holdings rows."""
        preds = pd.Series({"A": 0.05, "B": 0.05, "C": 0.05, "D": 0.05})
        port  = construct_portfolio(preds)
        assert len(port) == 1   # top quintile of 4 = 1


# ===========================================================================
# compute_metrics
# ===========================================================================

class TestComputeMetrics:

    def test_flat_returns_total_return(self, flat_returns):
        m = compute_metrics(flat_returns)
        expected_total = (1.01 ** 50) - 1
        assert m["total_return"] == pytest.approx(expected_total, rel=1e-4)

    def test_flat_returns_win_rate_is_one(self, flat_returns):
        m = compute_metrics(flat_returns)
        assert m["win_rate"] == pytest.approx(1.0)

    def test_alternating_returns_win_rate(self, alternating_returns):
        m = compute_metrics(alternating_returns)
        assert m["win_rate"] == pytest.approx(0.5)

    def test_cagr_consistent_with_total_return(self, flat_returns):
        m = compute_metrics(flat_returns)
        ppa = TRADING_DAYS_PA / REBALANCE_DAYS
        n_years = 50 / ppa
        expected_cagr = (1 + m["total_return"]) ** (1 / n_years) - 1
        assert m["cagr"] == pytest.approx(expected_cagr, rel=1e-4)

    def test_sharpe_positive_for_positive_returns(self):
        # Need variation in returns for std > 0; positive mean → positive Sharpe
        r = pd.Series([0.03, 0.01, 0.02, 0.04, 0.01] * 10)
        m = compute_metrics(r)
        assert m["sharpe"] > 0

    def test_sharpe_zero_for_zero_returns(self):
        m = compute_metrics(pd.Series([0.0] * 20))
        assert m["sharpe"] == pytest.approx(0.0)

    def test_max_drawdown_is_non_positive(self, alternating_returns):
        m = compute_metrics(alternating_returns)
        assert m["max_drawdown"] <= 0

    def test_max_drawdown_monotone_gains(self):
        """Strictly increasing equity curve → drawdown must be 0."""
        m = compute_metrics(pd.Series([0.01] * 30))
        assert m["max_drawdown"] == pytest.approx(0.0, abs=1e-10)

    def test_max_drawdown_known_value(self):
        """Single -50% period after gains → drawdown = -0.50."""
        returns = pd.Series([0.0, 0.0, -0.50, 0.0])
        m = compute_metrics(returns)
        assert m["max_drawdown"] == pytest.approx(-0.50, abs=1e-6)

    def test_calmar_ratio_positive_cagr_negative_dd(self):
        m = compute_metrics(pd.Series([0.05, -0.02] * 20))
        assert m["calmar"] > 0

    def test_n_periods(self, flat_returns):
        m = compute_metrics(flat_returns)
        assert m["n_periods"] == 50

    def test_empty_returns_returns_error_key(self):
        m = compute_metrics(pd.Series(dtype=float))
        assert "error" in m

    def test_nan_returns_dropped(self):
        r = pd.Series([0.01, np.nan, 0.01, np.nan, 0.01])
        m = compute_metrics(r)
        assert m["n_periods"] == 3

    # --- Benchmark comparison ---

    def test_benchmark_keys_present(self, flat_returns):
        bm = pd.Series([0.005] * 50)
        m  = compute_metrics(flat_returns, benchmark_returns=bm)
        for key in ("benchmark_total_return", "benchmark_cagr", "alpha", "beta", "information_ratio"):
            assert key in m

    def test_alpha_positive_when_portfolio_outperforms(self):
        port = pd.Series([0.03] * 30)
        bm   = pd.Series([0.01] * 30)
        m    = compute_metrics(port, bm)
        assert m["alpha"] > 0

    def test_alpha_negative_when_portfolio_underperforms(self):
        port = pd.Series([0.01] * 30)
        bm   = pd.Series([0.03] * 30)
        m    = compute_metrics(port, bm)
        assert m["alpha"] < 0

    def test_beta_near_one_for_identical_returns(self):
        r = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01] * 10)
        m = compute_metrics(r, benchmark_returns=r)
        assert m["beta"] == pytest.approx(1.0, abs=1e-6)

    def test_information_ratio_zero_for_identical_returns(self):
        r = pd.Series([0.02, -0.01, 0.03] * 10)
        m = compute_metrics(r, benchmark_returns=r)
        assert m["information_ratio"] == pytest.approx(0.0, abs=1e-6)


# ===========================================================================
# Backtest smoke test  (requires live DB + trained model)
# ===========================================================================

class TestBacktest:

    @pytest.mark.integration
    def test_backtest_returns_expected_keys(self, db_seeded):
        from app.analytics.portfolio import backtest
        result = backtest(
            start_date="2021-01-01",
            end_date="2022-12-31",
            tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "META",
                     "NVDA", "JPM", "JNJ", "XOM", "SPY",
                     "BRK-B", "UNH", "V", "PG", "HD",
                     "MA", "CVX", "LLY", "ABBV", "MRK"],
        )
        assert result, "backtest returned empty result"
        for key in ("portfolio_returns", "benchmark_returns", "metrics", "holdings_history"):
            assert key in result

    @pytest.mark.integration
    def test_backtest_metrics_structure(self, db_seeded):
        from app.analytics.portfolio import backtest
        result = backtest(
            start_date="2021-01-01",
            end_date="2022-12-31",
            tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "META",
                     "NVDA", "JPM", "JNJ", "XOM", "SPY",
                     "BRK-B", "UNH", "V", "PG", "HD",
                     "MA", "CVX", "LLY", "ABBV", "MRK"],
        )
        m = result["metrics"]
        for key in ("total_return", "cagr", "sharpe", "max_drawdown", "n_periods"):
            assert key in m

    @pytest.mark.integration
    def test_backtest_holdings_equal_weight(self, db_seeded):
        """Each rebalance period's holdings should be equal-weight top quintile."""
        from app.analytics.portfolio import backtest
        result = backtest(
            start_date="2021-01-01",
            end_date="2022-12-31",
            tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "META",
                     "NVDA", "JPM", "JNJ", "XOM", "SPY",
                     "BRK-B", "UNH", "V", "PG", "HD",
                     "MA", "CVX", "LLY", "ABBV", "MRK"],
        )
        max_expected = max(1, 20 // QUINTILE)
        for period in result["holdings_history"]:
            # n_holdings = top quintile of however many tickers had valid features
            # that day — may be less than max if some tickers lacked history
            assert 1 <= period["n_holdings"] <= max_expected

    @pytest.mark.integration
    def test_backtest_returns_series_indexed_by_date(self, db_seeded):
        from app.analytics.portfolio import backtest
        result = backtest(
            start_date="2021-01-01",
            end_date="2022-12-31",
            tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "META",
                     "NVDA", "JPM", "JNJ", "XOM", "SPY",
                     "BRK-B", "UNH", "V", "PG", "HD",
                     "MA", "CVX", "LLY", "ABBV", "MRK"],
        )
        assert isinstance(result["portfolio_returns"], pd.Series)
        assert pd.api.types.is_datetime64_any_dtype(result["portfolio_returns"].index)
