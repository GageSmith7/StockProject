from fastapi import APIRouter

from app.data.store import get_latest_model_metrics
from app.models.registry import load_latest_metadata

router = APIRouter(tags=["model_health"])


@router.get("/model/health")
def get_model_health():
    """
    Returns the current model health status for the frontend.

    Combines:
      - Latest IC check result from model_metrics table (live performance)
      - Latest model artifact metadata from registry (training-time IC)

    gate_status values:
      HEALTHY        — rolling IC >= 0.05, model performing as expected
      DEGRADED       — rolling IC below gate, retrain attempted
      CIRCUIT_BREAKER — 3 consecutive failures, equal-weight fallback active
      NO_DATA        — not enough prediction history yet to compute IC
    """
    latest_metrics  = get_latest_model_metrics()
    latest_artifact = load_latest_metadata()

    training_ic      = None
    model_version    = None
    model_artifact   = latest_artifact.get("artifact")

    if latest_artifact:
        m             = latest_artifact.get("metrics", {})
        training_ic   = m.get("mean_ic")
        model_version = latest_artifact.get("timestamp")

    if latest_metrics is None:
        return {
            "gate_status":          "NO_DATA",
            "rolling_ic":           None,
            "training_ic":          training_ic,
            "n_samples":            None,
            "consecutive_failures": 0,
            "retrain_result":       None,
            "model_version":        model_version,
            "model_artifact":       model_artifact,
            "last_checked":         None,
        }

    return {
        "gate_status":          latest_metrics["gate_status"],
        "rolling_ic":           latest_metrics["rolling_ic"],
        "training_ic":          training_ic,
        "n_samples":            latest_metrics["n_samples"],
        "consecutive_failures": latest_metrics["consecutive_failures"],
        "retrain_result":       latest_metrics["retrain_result"],
        "model_version":        model_version,
        "model_artifact":       model_artifact,
        "last_checked":         latest_metrics["computed_at"],
    }
