"""
Block-bootstrap Monte Carlo simulation — percentile cone (p10 / p50 / p90).

Improvements over the naive IID bootstrap:
  1. Block bootstrap (default block_size=3 months): resamples consecutive
     chunks of returns to preserve short-run autocorrelation and volatility
     clustering rather than treating each month as independent.
  2. Inflation adjustment: optionally deflates the wealth path to real (CPI-
     adjusted) dollars using a user-supplied annual inflation rate.
  3. Contributions: a fixed monthly cash deposit is applied each period,
     allowing DCA scenarios.
  4. Regime-aware sampling (use_regime=True): detects whether the portfolio
     is currently in an expansion or contraction regime (trailing 12-month
     cumulative return >= 0 = expansion, < 0 = contraction), then restricts
     the bootstrap pool to months that shared that regime.  This means a
     bear-market run produces a tighter, lower cone while a bull-market run
     produces a wider, higher one — more honest than drawing from all history
     uniformly.  Falls back to full history if the regime subset is too thin.

Limitations acknowledged:
  - Block bootstrap assumes stationarity within each regime.
  - Inflation is a flat constant rate, not stochastic.
  - Contributions are fixed in nominal terms (not inflation-scaled).
  - Regime model is binary (no transitions) — simple but not perfect.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_BLOCK_SIZE  = 3   # months; large enough to capture vol clustering
_REGIME_MIN_SAMPLES  = 12  # minimum months required to use regime-filtered pool
_REGIME_WINDOW       = 12  # rolling window (months) for expansion/contraction signal


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _block_sample(
    rng: np.random.Generator,
    r: np.ndarray,
    block_size: int,
    n_simulations: int,
    horizon_periods: int,
) -> np.ndarray:
    """
    Vectorised block bootstrap.

    Draws ceil(horizon / block_size) non-overlapping block starts per path,
    concatenates into (n_simulations, n_blocks * block_size), then truncates
    to exactly (n_simulations, horizon_periods).

    Falls back to IID sampling when len(r) < block_size.
    """
    n_obs = len(r)

    if n_obs < block_size:
        # not enough history for blocks — degrade gracefully to IID
        return rng.choice(r, size=(n_simulations, horizon_periods), replace=True)

    n_blocks  = -(-horizon_periods // block_size)   # ceil division
    max_start = n_obs - block_size                  # last valid start index

    # (n_simulations, n_blocks) random block start indices
    starts = rng.integers(0, max_start + 1, size=(n_simulations, n_blocks))

    # Offset array: [0, 1, ..., block_size-1]
    offsets = np.arange(block_size)

    # (n_simulations, n_blocks, block_size) → index into r
    indices = starts[:, :, np.newaxis] + offsets[np.newaxis, np.newaxis, :]
    sampled = r[indices].reshape(n_simulations, n_blocks * block_size)

    return sampled[:, :horizon_periods]


def _label_regimes(period_returns: pd.Series) -> np.ndarray:
    """
    Assign a binary regime label to each observed month.

    Signal: trailing _REGIME_WINDOW-month cumulative return.
      >= 0  →  expansion   (1)
      <  0  →  contraction (0)

    The first (_REGIME_WINDOW - 1) months lack a full rolling window and
    are conservatively labeled expansion (1).

    Returns an int array of shape (len(period_returns),).
    """
    rolling_cum = period_returns.rolling(_REGIME_WINDOW, min_periods=_REGIME_WINDOW).sum()
    # fillna *before* the comparison — NaN comparisons in pandas yield False,
    # so filling with 0.0 makes warm-up months (0.0 >= 0) evaluate to True (expansion).
    labels = (rolling_cum.fillna(0.0) >= 0).astype(int)
    return labels.values


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate(
    period_returns: pd.Series,
    horizon_periods: int,
    n_simulations: int = 1000,
    initial_value: float = 1.0,
    block_size: int = _DEFAULT_BLOCK_SIZE,
    inflation_rate: float = 0.0,
    monthly_contribution: float = 0.0,
    use_regime: bool = False,
    seed: int | None = None,
) -> dict:
    """
    Block-bootstrap simulation of portfolio value over a forward horizon.

    Args:
        period_returns       : observed per-period returns (monthly).
        horizon_periods      : number of future months to project.
        n_simulations        : number of Monte Carlo paths.
        initial_value        : starting portfolio value (nominal dollars).
        block_size           : consecutive-month block length for bootstrap.
                               1 = IID (old behaviour).
        inflation_rate       : annual inflation rate (e.g. 0.025 = 2.5 %).
                               0.0 = no adjustment (nominal path).
        monthly_contribution : fixed cash deposited each month in nominal dollars.
        use_regime           : if True, detect the current market regime from
                               the trailing 12-month return and restrict the
                               bootstrap pool to months that shared that regime.
        seed                 : random seed for reproducibility.

    Returns dict with:
        cone                 : list of {period, p10, p50, p90} for t=0..horizon.
        final                : {p10, p50, p90} at t=horizon.
        horizon_periods      : int (echoed).
        n_simulations        : int (echoed).
        n_historical         : number of monthly returns actually sampled from.
        inflation_rate       : float (echoed).
        monthly_contribution : float (echoed).
        current_regime       : "expansion" | "contraction" | None
        regime_fallback      : True if regime pool was too thin → full history used.
    """
    clean = period_returns.dropna()
    r     = clean.values

    if len(r) == 0:
        raise ValueError("period_returns is empty — cannot run simulation.")
    if horizon_periods <= 0:
        raise ValueError(f"horizon_periods must be > 0, got {horizon_periods}")

    rng = np.random.default_rng(seed)

    # --- 1. Regime detection & pool selection ---
    current_regime: str | None = None
    regime_fallback             = False
    r_pool                      = r   # default: full history

    if use_regime and len(r) >= _REGIME_WINDOW:
        regime_labels  = _label_regimes(clean)
        regime_int     = int(regime_labels[-1])
        current_regime = "expansion" if regime_int == 1 else "contraction"

        r_regime = r[regime_labels == regime_int]

        if len(r_regime) >= _REGIME_MIN_SAMPLES:
            r_pool = r_regime
            logger.info(
                "Regime-aware sampling: regime=%s | pool=%d/%d months",
                current_regime, len(r_pool), len(r),
            )
        else:
            regime_fallback = True
            logger.warning(
                "Regime pool too thin (%d samples < %d minimum) — using full history",
                len(r_regime), _REGIME_MIN_SAMPLES,
            )

    # --- 2. Sample return paths via block bootstrap ---
    sampled = _block_sample(rng, r_pool, block_size, n_simulations, horizon_periods)
    # sampled shape: (n_simulations, horizon_periods)

    # --- 2. Build wealth paths with optional monthly contributions ---
    #
    # W[t+1] = W[t] * (1 + r[t]) + monthly_contribution
    #
    # This requires iterating over time rather than a single cumprod because
    # each period's contribution base changes with the running balance.
    wealth = np.empty((n_simulations, horizon_periods + 1))
    wealth[:, 0] = initial_value

    for t in range(horizon_periods):
        wealth[:, t + 1] = wealth[:, t] * (1.0 + sampled[:, t]) + monthly_contribution

    # --- 3. Optionally deflate to real (inflation-adjusted) dollars ---
    if inflation_rate > 0.0:
        # deflators[t] = cumulative price level at month t
        deflators = (1.0 + inflation_rate) ** (np.arange(horizon_periods + 1) / 12.0)
        wealth /= deflators[np.newaxis, :]

    # --- 4. Percentile cone ---
    p10 = np.percentile(wealth, 10, axis=0)
    p50 = np.percentile(wealth, 50, axis=0)
    p90 = np.percentile(wealth, 90, axis=0)

    cone = [
        {
            "period": int(t),
            "p10":    round(float(p10[t]), 4),
            "p50":    round(float(p50[t]), 4),
            "p90":    round(float(p90[t]), 4),
        }
        for t in range(horizon_periods + 1)
    ]

    logger.info(
        "Monte Carlo: %d paths × %d periods | block_size=%d | regime=%s | "
        "inflation=%.1f%% | contrib=%.0f | final p10=%.2f p50=%.2f p90=%.2f",
        n_simulations, horizon_periods, block_size,
        current_regime or "off",
        inflation_rate * 100, monthly_contribution,
        p10[-1], p50[-1], p90[-1],
    )

    return {
        "cone":                  cone,
        "final": {
            "p10": round(float(p10[-1]), 4),
            "p50": round(float(p50[-1]), 4),
            "p90": round(float(p90[-1]), 4),
        },
        "horizon_periods":       horizon_periods,
        "n_simulations":         n_simulations,
        "n_historical":          int(len(r_pool)),
        "inflation_rate":        inflation_rate,
        "monthly_contribution":  monthly_contribution,
        "current_regime":        current_regime,
        "regime_fallback":       regime_fallback,
    }
