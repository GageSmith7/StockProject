# Tests for analytics/montecarlo.py
import numpy as np
import pandas as pd
import pytest

from app.analytics.montecarlo import _block_sample, _label_regimes, simulate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def flat_returns():
    """Monthly returns that are all exactly 1 % — deterministic paths."""
    return pd.Series([0.01] * 60)


@pytest.fixture
def mixed_returns():
    """60 monthly returns with alternating +5 % / -3 % to test block structure."""
    vals = [0.05 if i % 2 == 0 else -0.03 for i in range(60)]
    return pd.Series(vals)


@pytest.fixture
def volatile_returns():
    """Random but seeded returns for statistical tests."""
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0.007, 0.04, 120))


# ---------------------------------------------------------------------------
# _block_sample
# ---------------------------------------------------------------------------

class TestBlockSample:
    def test_output_shape(self, volatile_returns):
        rng = np.random.default_rng(1)
        r = volatile_returns.values
        out = _block_sample(rng, r, block_size=3, n_simulations=200, horizon_periods=24)
        assert out.shape == (200, 24)

    def test_all_values_come_from_input(self, volatile_returns):
        rng = np.random.default_rng(2)
        r = volatile_returns.values
        out = _block_sample(rng, r, block_size=3, n_simulations=50, horizon_periods=12)
        assert np.all(np.isin(out, r))

    def test_fallback_to_iid_when_too_few_observations(self):
        """When len(r) < block_size, must still return correct shape (IID fallback)."""
        rng = np.random.default_rng(3)
        r = np.array([0.01, 0.02])
        out = _block_sample(rng, r, block_size=5, n_simulations=10, horizon_periods=8)
        assert out.shape == (10, 8)

    def test_iid_equals_block_size_one(self, volatile_returns):
        """block_size=1 should reproduce IID sampling shape and values-in-r."""
        rng = np.random.default_rng(4)
        r = volatile_returns.values
        out = _block_sample(rng, r, block_size=1, n_simulations=100, horizon_periods=36)
        assert out.shape == (100, 36)
        assert np.all(np.isin(out, r))

    def test_horizon_not_multiple_of_block_size(self, volatile_returns):
        """Horizon=25 with block_size=3 must still produce exactly 25 columns."""
        rng = np.random.default_rng(5)
        r = volatile_returns.values
        out = _block_sample(rng, r, block_size=3, n_simulations=50, horizon_periods=25)
        assert out.shape == (50, 25)


# ---------------------------------------------------------------------------
# simulate — basic structure
# ---------------------------------------------------------------------------

class TestSimulateStructure:
    def test_returns_expected_keys(self, flat_returns):
        result = simulate(flat_returns, horizon_periods=12, n_simulations=100, seed=0)
        for key in ("cone", "final", "horizon_periods", "n_simulations",
                    "n_historical", "inflation_rate", "monthly_contribution"):
            assert key in result

    def test_cone_length(self, flat_returns):
        result = simulate(flat_returns, horizon_periods=24, n_simulations=100, seed=0)
        assert len(result["cone"]) == 25  # 0..24

    def test_cone_period_zero_equals_initial_value(self, flat_returns):
        result = simulate(flat_returns, horizon_periods=12, initial_value=5000, seed=0)
        assert result["cone"][0]["p50"] == 5000.0

    def test_empty_series_raises(self):
        with pytest.raises(ValueError, match="empty"):
            simulate(pd.Series([], dtype=float), horizon_periods=12)

    def test_non_positive_horizon_raises(self, flat_returns):
        with pytest.raises(ValueError):
            simulate(flat_returns, horizon_periods=0)

    def test_echoed_metadata(self, flat_returns):
        result = simulate(
            flat_returns,
            horizon_periods=12,
            n_simulations=200,
            inflation_rate=0.025,
            monthly_contribution=100.0,
            seed=0,
        )
        assert result["n_simulations"] == 200
        assert result["inflation_rate"] == 0.025
        assert result["monthly_contribution"] == 100.0


# ---------------------------------------------------------------------------
# simulate — deterministic flat-return checks
# ---------------------------------------------------------------------------

class TestSimulateDeterministic:
    def test_flat_return_p50_near_expected(self, flat_returns):
        """
        With all returns = 1 %, p50 after 12 months should be ~1.01^12 ≈ 1.1268
        (no contributions, no inflation, starting value = 1).
        """
        result = simulate(flat_returns, horizon_periods=12, initial_value=1.0, seed=0)
        expected = 1.01 ** 12
        # final values are rounded to 4 decimal places, so tolerance = 1e-4
        assert abs(result["final"]["p50"] - expected) < 1e-4

    def test_p10_le_p50_le_p90(self, volatile_returns):
        result = simulate(volatile_returns, horizon_periods=36, n_simulations=500, seed=0)
        assert result["final"]["p10"] <= result["final"]["p50"] <= result["final"]["p90"]

    def test_seeded_reproducible(self, volatile_returns):
        r1 = simulate(volatile_returns, horizon_periods=24, n_simulations=200, seed=42)
        r2 = simulate(volatile_returns, horizon_periods=24, n_simulations=200, seed=42)
        assert r1["final"]["p50"] == r2["final"]["p50"]

    def test_different_seeds_differ(self, volatile_returns):
        r1 = simulate(volatile_returns, horizon_periods=24, n_simulations=200, seed=1)
        r2 = simulate(volatile_returns, horizon_periods=24, n_simulations=200, seed=2)
        assert r1["final"]["p50"] != r2["final"]["p50"]


# ---------------------------------------------------------------------------
# simulate — inflation
# ---------------------------------------------------------------------------

class TestSimulateInflation:
    def test_inflation_reduces_real_value(self, flat_returns):
        """Real (inflation-adjusted) p50 must be below nominal p50."""
        nominal  = simulate(flat_returns, horizon_periods=12, initial_value=1.0, seed=0)
        real     = simulate(flat_returns, horizon_periods=12, initial_value=1.0,
                            inflation_rate=0.03, seed=0)
        assert real["final"]["p50"] < nominal["final"]["p50"]

    def test_inflation_zero_matches_no_inflation(self, volatile_returns):
        r1 = simulate(volatile_returns, horizon_periods=24, n_simulations=200,
                      inflation_rate=0.0, seed=7)
        r2 = simulate(volatile_returns, horizon_periods=24, n_simulations=200, seed=7)
        assert r1["final"]["p50"] == r2["final"]["p50"]

    def test_flat_return_real_value_correct(self):
        """
        With 1 % monthly return and 2.4 % annual inflation, real value after
        12 months = (1.01^12) / (1.024^1) ≈ 1.1268 / 1.0240 ≈ 1.1004.
        """
        flat = pd.Series([0.01] * 120)
        result = simulate(flat, horizon_periods=12, initial_value=1.0,
                          inflation_rate=0.024, seed=0)
        expected = (1.01 ** 12) / (1.024 ** 1)
        assert abs(result["final"]["p50"] - expected) < 1e-4

    def test_inflation_cone_period0_unchanged(self, flat_returns):
        """t=0 deflator = (1+r)^0 = 1 → initial value should be unaffected."""
        result = simulate(flat_returns, horizon_periods=12, initial_value=5000.0,
                          inflation_rate=0.03, seed=0)
        assert result["cone"][0]["p50"] == 5000.0


# ---------------------------------------------------------------------------
# simulate — monthly contributions
# ---------------------------------------------------------------------------

class TestSimulateContributions:
    def test_positive_contribution_increases_final_value(self, flat_returns):
        base    = simulate(flat_returns, horizon_periods=12, initial_value=1000.0, seed=0)
        contrib = simulate(flat_returns, horizon_periods=12, initial_value=1000.0,
                           monthly_contribution=100.0, seed=0)
        assert contrib["final"]["p50"] > base["final"]["p50"]

    def test_negative_contribution_decreases_final_value(self, flat_returns):
        base     = simulate(flat_returns, horizon_periods=12, initial_value=10000.0, seed=0)
        withdraw = simulate(flat_returns, horizon_periods=12, initial_value=10000.0,
                            monthly_contribution=-200.0, seed=0)
        assert withdraw["final"]["p50"] < base["final"]["p50"]

    def test_zero_contribution_matches_baseline(self, volatile_returns):
        base = simulate(volatile_returns, horizon_periods=24, n_simulations=200,
                        monthly_contribution=0.0, seed=5)
        same = simulate(volatile_returns, horizon_periods=24, n_simulations=200, seed=5)
        assert base["final"]["p50"] == same["final"]["p50"]

    def test_flat_return_with_contribution_deterministic(self):
        """
        With 1 % monthly return and $100/month contribution starting from $1000:
          W[1] = 1000 * 1.01 + 100 = 1110
          W[2] = 1110 * 1.01 + 100 = 1221.10
          ...
        p50 should be exactly equal to the analytic closed-form after 3 months.
        """
        flat = pd.Series([0.01] * 120)
        result = simulate(flat, horizon_periods=3, initial_value=1000.0,
                          monthly_contribution=100.0, seed=0)
        w = 1000.0
        for _ in range(3):
            w = w * 1.01 + 100.0
        assert abs(result["final"]["p50"] - w) < 1e-6

    def test_contribution_and_inflation_combined(self, flat_returns):
        """Contributions + inflation should both apply; real value must be finite and positive."""
        result = simulate(flat_returns, horizon_periods=12, initial_value=5000.0,
                          monthly_contribution=200.0, inflation_rate=0.025, seed=0)
        assert result["final"]["p50"] > 0
        # Sanity: real value with contributions should exceed uninflated initial value
        assert result["final"]["p50"] > 5000.0 / (1.025 ** 1)


# ---------------------------------------------------------------------------
# simulate — block bootstrap statistical property
# ---------------------------------------------------------------------------

class TestBlockBootstrapVsIID:
    def test_block_and_iid_similar_median(self, volatile_returns):
        """
        With enough paths, block and IID medians should be within a few percent.
        Both use the same empirical distribution, just different correlation structure.
        """
        iid   = simulate(volatile_returns, horizon_periods=60, n_simulations=2000,
                         block_size=1, seed=10)
        block = simulate(volatile_returns, horizon_periods=60, n_simulations=2000,
                         block_size=3, seed=10)
        ratio = block["final"]["p50"] / iid["final"]["p50"]
        assert 0.90 < ratio < 1.10   # within 10 %

    def test_block_size_one_behaves_as_iid(self, volatile_returns):
        """block_size=1 should produce same result as the IID path."""
        r1 = simulate(volatile_returns, horizon_periods=24, n_simulations=500,
                      block_size=1, seed=99)
        r2 = simulate(volatile_returns, horizon_periods=24, n_simulations=500,
                      block_size=1, seed=99)
        assert r1["final"]["p50"] == r2["final"]["p50"]


# ---------------------------------------------------------------------------
# _label_regimes
# ---------------------------------------------------------------------------

class TestLabelRegimes:
    def test_all_positive_returns_expansion(self):
        s = pd.Series([0.01] * 24)
        labels = _label_regimes(s)
        # After the first 11 warm-up months every label should be expansion (1)
        assert all(labels[11:] == 1)

    def test_all_negative_returns_contraction(self):
        s = pd.Series([-0.02] * 24)
        labels = _label_regimes(s)
        assert all(labels[11:] == 0)

    def test_warm_up_months_default_to_expansion(self):
        """First 11 months have no full window → should be labeled 1 (expansion)."""
        s = pd.Series([-0.05] * 24)
        labels = _label_regimes(s)
        assert all(labels[:11] == 1)

    def test_output_length_matches_input(self):
        s = pd.Series([0.01] * 36)
        assert len(_label_regimes(s)) == 36

    def test_transition_detected(self):
        """12 months of +1 % then 12 months of -5 % should end in contraction."""
        s = pd.Series([0.01] * 12 + [-0.05] * 12)
        labels = _label_regimes(s)
        assert labels[-1] == 0  # most recent month is contraction

    def test_values_are_binary(self):
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(0.005, 0.04, 60))
        labels = _label_regimes(s)
        assert set(labels).issubset({0, 1})


# ---------------------------------------------------------------------------
# simulate — regime-aware sampling
# ---------------------------------------------------------------------------

class TestRegimeAwareSampling:
    def test_regime_off_returns_none(self, volatile_returns):
        result = simulate(volatile_returns, horizon_periods=12, n_simulations=100,
                          use_regime=False, seed=0)
        assert result["current_regime"] is None
        assert result["regime_fallback"] is False

    def test_regime_on_returns_regime_label(self, volatile_returns):
        result = simulate(volatile_returns, horizon_periods=12, n_simulations=100,
                          use_regime=True, seed=0)
        assert result["current_regime"] in ("expansion", "contraction")

    def test_bull_regime_detected_on_rising_series(self):
        """Series that ends on a strong uptrend should detect expansion."""
        s = pd.Series([0.03] * 60)
        result = simulate(s, horizon_periods=12, n_simulations=100,
                          use_regime=True, seed=0)
        assert result["current_regime"] == "expansion"

    def test_bear_regime_detected_on_falling_series(self):
        """Series that ends on a strong downtrend should detect contraction."""
        # 12 months of gains to warm up, then 12 months of steep losses
        s = pd.Series([0.02] * 12 + [-0.04] * 24)
        result = simulate(s, horizon_periods=12, n_simulations=100,
                          use_regime=True, seed=0)
        assert result["current_regime"] == "contraction"

    def test_regime_pool_size_reported(self):
        """n_historical should reflect the regime-filtered pool, not full history."""
        # All positive → expansion regime; pool = all months
        s = pd.Series([0.02] * 60)
        result_regime = simulate(s, horizon_periods=12, n_simulations=100,
                                 use_regime=True, seed=0)
        result_full   = simulate(s, horizon_periods=12, n_simulations=100,
                                 use_regime=False, seed=0)
        # expansion-only pool on all-positive series ≤ full pool
        assert result_regime["n_historical"] <= result_full["n_historical"]

    def test_fallback_when_too_few_regime_samples(self):
        """
        If there are fewer than _REGIME_MIN_SAMPLES months in the detected regime,
        the simulation should fall back to full history and set regime_fallback=True.

        Guaranteed scenario: 12 all-negative months.
          - Months 0–10: expansion (warm-up default, no rolling window yet)
          - Month 11:    contraction (first full 12-month rolling sum = -0.60)
        Current regime = contraction, pool = 1 month  < _REGIME_MIN_SAMPLES (12) → fallback.
        """
        from app.analytics.montecarlo import _REGIME_MIN_SAMPLES
        s = pd.Series([-0.05] * _REGIME_MIN_SAMPLES)
        result = simulate(s, horizon_periods=6, n_simulations=100,
                          use_regime=True, seed=0)
        assert result["current_regime"] == "contraction"
        assert result["regime_fallback"] is True

    def test_regime_contraction_cone_lower_than_expansion(self):
        """
        On an all-positive history split into bear/bull, conditioning on
        contraction months should give a lower p50 than conditioning on
        expansion months.
        """
        rng_np = np.random.default_rng(7)
        # Two-segment history: first 30 months strong gains, last 30 months losses
        bull_months = list(rng_np.normal(0.04, 0.02, 30))
        bear_months = list(rng_np.normal(-0.02, 0.02, 30))

        s_bull_end  = pd.Series(bull_months + bear_months[::-1])  # ends with gains
        s_bear_end  = pd.Series(bull_months + bear_months)         # ends with losses

        res_bull = simulate(s_bull_end, horizon_periods=36, n_simulations=1000,
                            use_regime=True, seed=0)
        res_bear = simulate(s_bear_end, horizon_periods=36, n_simulations=1000,
                            use_regime=True, seed=0)

        if (res_bull["current_regime"] == "expansion" and
                res_bear["current_regime"] == "contraction" and
                not res_bull["regime_fallback"] and
                not res_bear["regime_fallback"]):
            assert res_bull["final"]["p50"] > res_bear["final"]["p50"]

    def test_regime_reproducible_with_seed(self, volatile_returns):
        r1 = simulate(volatile_returns, horizon_periods=24, n_simulations=200,
                      use_regime=True, seed=11)
        r2 = simulate(volatile_returns, horizon_periods=24, n_simulations=200,
                      use_regime=True, seed=11)
        assert r1["final"]["p50"] == r2["final"]["p50"]
        assert r1["current_regime"] == r2["current_regime"]

    def test_regime_off_matches_non_regime_call(self, volatile_returns):
        """use_regime=False should be identical to not passing the flag."""
        r1 = simulate(volatile_returns, horizon_periods=24, n_simulations=200,
                      use_regime=False, seed=3)
        r2 = simulate(volatile_returns, horizon_periods=24, n_simulations=200, seed=3)
        assert r1["final"]["p50"] == r2["final"]["p50"]
