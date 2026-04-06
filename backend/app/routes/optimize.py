from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.models.optimizer import optimize

router = APIRouter(tags=["optimize"])


class OptimizeRequest(BaseModel):
    tickers:        list[str] = Field(..., min_length=5)
    goal:           str       = Field("sharpe", pattern="^(sharpe|min_vol|max_cagr)$")
    lookback_years: int       = Field(5, ge=3, le=10)


@router.post("/optimize")
def run_optimize(req: OptimizeRequest):
    """
    Run Markowitz portfolio optimization.

    Returns optimal weights + annualized metrics (Sharpe, CAGR, Max Drawdown, Volatility).
    """
    try:
        return optimize(
            tickers=[t.upper() for t in req.tickers],
            goal=req.goal,
            lookback_years=req.lookback_years,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
