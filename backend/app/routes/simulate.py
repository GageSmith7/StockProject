from datetime import datetime

import pandas as pd
from dateutil.relativedelta import relativedelta
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from app.analytics.benchmark import compare
from app.analytics.montecarlo import simulate as mc_simulate
from app.routes.portfolio import Holding

router = APIRouter(tags=["simulate"])

_DISCLAIMER = (
    "Projections are based on the historical return distribution of your portfolio. "
    "Past performance does not predict future results. "
    "This is not financial advice."
)
_DEFAULT_START = "2015-01-01"   # aligns with model training history


class SimulateRequest(BaseModel):
    holdings:             list[Holding] = Field(..., min_length=1, max_length=50)
    years:                float = Field(default=10.0, ge=1.0, le=30.0)
    n_simulations:        int   = Field(default=1000, ge=100, le=10000)
    initial_value:        float = Field(default=10000.0, gt=0)
    start_date:           str   = _DEFAULT_START
    inflation_rate:       float = Field(default=0.0, ge=0.0, le=0.20,
                                        description="Annual inflation rate (0 = nominal, 0.025 = 2.5 %)")
    monthly_contribution: float = Field(default=0.0,
                                        description="Monthly cash deposit in dollars")
    use_regime:           bool  = Field(default=False,
                                        description="Condition bootstrap on current expansion/contraction regime")

    @model_validator(mode="after")
    def weights_sum_to_one(self):
        total = sum(h.weight for h in self.holdings)
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Holdings weights must sum to 1.0 (got {total:.4f}).")
        return self


def _to_monthly_returns(daily_returns: pd.Series) -> pd.Series:
    """Compound daily returns within each calendar month into a monthly return."""
    s = daily_returns.copy()
    s.index = pd.to_datetime(s.index)
    return s.groupby(s.index.to_period("M")).apply(lambda x: (1 + x).prod() - 1)


def _future_month_labels(n: int) -> list[str]:
    """Return n+1 month labels starting from the current month: ['2026-03', ...]"""
    base = datetime.utcnow().date().replace(day=1)
    return [
        (base + relativedelta(months=i)).strftime("%Y-%m")
        for i in range(n + 1)
    ]


@router.post("/simulate")
def run_simulate(req: SimulateRequest):
    holdings = {h.ticker.upper(): h.weight for h in req.holdings}
    end_date = datetime.utcnow().date().isoformat()

    result = compare(holdings, req.start_date, end_date)

    if not result or result["portfolio_returns"].empty:
        raise HTTPException(
            status_code=422,
            detail="Could not compute historical returns. Check holdings and ensure DB is seeded.",
        )

    monthly = _to_monthly_returns(result["portfolio_returns"])

    if len(monthly) < 12:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Insufficient history for simulation "
                f"(need ≥ 12 months, got {len(monthly)})."
            ),
        )

    horizon_periods = max(1, int(req.years * 12))

    mc = mc_simulate(
        monthly,
        horizon_periods=horizon_periods,
        n_simulations=req.n_simulations,
        initial_value=req.initial_value,
        inflation_rate=req.inflation_rate,
        monthly_contribution=req.monthly_contribution,
        use_regime=req.use_regime,
    )

    cone = mc["cone"]

    return {
        "time_axis": _future_month_labels(len(cone) - 1),
        "percentiles": {
            "p10": [c["p10"] for c in cone],
            "p50": [c["p50"] for c in cone],
            "p90": [c["p90"] for c in cone],
        },
        "disclaimer":            _DISCLAIMER,
        "backtest_metrics":      result["metrics"],
        "inflation_rate":        req.inflation_rate,
        "monthly_contribution":  req.monthly_contribution,
        "current_regime":        mc.get("current_regime"),
        "regime_fallback":       mc.get("regime_fallback", False),
    }
