import math

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from app.analytics.benchmark import compare

router = APIRouter(tags=["portfolio"])

INITIAL_VALUE = 10_000.0   # portfolio indexed to $10,000 at start date


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class Holding(BaseModel):
    ticker: str
    weight: float = Field(..., gt=0, le=1.0)


class PortfolioRequest(BaseModel):
    holdings:   list[Holding] = Field(..., min_length=1, max_length=50)
    start_date: str
    end_date:   str | None = None

    @model_validator(mode="after")
    def weights_sum_to_one(self):
        total = sum(h.weight for h in self.holdings)
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Holdings weights must sum to 1.0 (got {total:.4f}). "
                "Adjust weights before submitting."
            )
        return self


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(v) -> float | None:
    if v is None:
        return None
    f = float(v)
    return None if math.isnan(f) else round(f, 6)


def _to_value_series(returns, initial: float = INITIAL_VALUE) -> list[dict]:
    """Convert a daily returns Series to a cumulative wealth list [{date, value}]."""
    if returns is None or (hasattr(returns, "empty") and returns.empty):
        return []
    cum = (1 + returns).cumprod() * initial
    return [
        {
            "date":  d.date().isoformat() if hasattr(d, "date") else str(d),
            "value": round(float(v), 2),
        }
        for d, v in cum.items()
        if not math.isnan(float(v))
    ]


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("/portfolio")
def run_portfolio(req: PortfolioRequest):
    holdings = {h.ticker.upper(): h.weight for h in req.holdings}

    result = compare(holdings, req.start_date, req.end_date)

    if not result:
        raise HTTPException(
            status_code=422,
            detail=(
                "No results produced. Check that tickers are valid, the date range "
                "is at least 30 days, and the DB is seeded."
            ),
        )

    return {
        "portfolio_returns" : _to_value_series(result["portfolio_returns"]),
        "benchmark_returns" : _to_value_series(result["benchmark_returns"]),
        "metrics"           : result["metrics"],
        "on_demand_tickers" : result["on_demand_tickers"],
    }
