from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.analytics.features import build_cross_sectional_dataset
from app.data.symbols import get_all_tickers
from app.models import predictor
from app.models.registry import list_artifacts

router = APIRouter(tags=["predict"])

_DISCLAIMER  = "ML predictions are experimental and not financial advice."
_IC_GATE     = 0.03


class PredictRequest(BaseModel):
    tickers: list[str] | None = None


@router.post("/predict")
def run_predict(req: PredictRequest):
    end_date   = datetime.utcnow().date().isoformat()
    start_date = (datetime.utcnow().date() - timedelta(days=730)).isoformat()

    tickers = req.tickers if req.tickers else get_all_tickers(source="preloaded")
    if not tickers:
        raise HTTPException(status_code=422, detail="No tickers available.")

    dataset = build_cross_sectional_dataset(tickers, start_date, end_date)
    if dataset.empty:
        raise HTTPException(
            status_code=422,
            detail="No feature data available. Ensure DB is seeded.",
        )

    latest_date = dataset["date"].max()
    snapshot    = dataset[dataset["date"] == latest_date].copy()

    preds = predictor.predict(snapshot)
    if preds.empty:
        raise HTTPException(status_code=422, detail="Model produced no predictions.")

    ranked = preds.sort_values(ascending=False).reset_index()
    # columns: ["ticker", "predicted_return_21d"]
    ranked["predicted_return_21d"] = ranked["predicted_return_21d"].round(6)
    ranked["rank"] = range(1, len(ranked) + 1)

    artifacts  = list_artifacts()
    model_meta = artifacts[0] if artifacts else {}

    as_of = latest_date
    as_of = as_of.date().isoformat() if hasattr(as_of, "date") else str(as_of)

    mean_ic = model_meta.get("metrics", {}).get("mean_ic")
    model_available = mean_ic is None or float(mean_ic) >= _IC_GATE

    return {
        "predictions":     ranked.to_dict("records"),
        "as_of_date":      as_of,
        "model_version":   model_meta.get("timestamp", "unknown"),
        "model_available": model_available,
        "disclaimer":      _DISCLAIMER,
    }
