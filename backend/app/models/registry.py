"""
Model registry — save and load LightGBM artifacts with timestamp + metrics.
Artifacts are stored in backend/models/ (gitignored) and synced to S3 when
S3_BUCKET_MODELS is set, so models survive container restarts.

Naming convention: lgbm_YYYYMMDD_HHMMSS.{pkl,json}
Backward compat:   ridge_*.pkl artifacts from the previous Ridge implementation
                   are still loadable via load_latest() when no lgbm_* files exist.
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)

# Resolve models/ directory relative to this file: backend/app/models/ → backend/models/
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODELS_DIR.mkdir(exist_ok=True)

S3_BUCKET = os.getenv("S3_BUCKET_MODELS")
S3_PREFIX  = "artifacts"


def _s3_client():
    import boto3
    return boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))


def _s3_upload(local_path: Path) -> None:
    if not S3_BUCKET:
        return
    try:
        key = f"{S3_PREFIX}/{local_path.name}"
        _s3_client().upload_file(str(local_path), S3_BUCKET, key)
        logger.info("S3 upload → s3://%s/%s", S3_BUCKET, key)
    except Exception as e:
        logger.warning("S3 upload failed for %s: %s", local_path.name, e)


def restore_from_s3() -> None:
    """
    Called at startup: if the local models/ directory is empty and S3 is configured,
    download the two most recent artifact pairs (pkl + json) from S3.
    This ensures the model survives container restarts.
    """
    if not S3_BUCKET:
        return
    if any(MODELS_DIR.glob("lgbm_*.pkl")) or any(MODELS_DIR.glob("ridge_*.pkl")):
        return  # already have local artifacts

    try:
        s3   = _s3_client()
        resp = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=f"{S3_PREFIX}/")
        # Include both lgbm_* and ridge_* keys
        objects = sorted(
            [
                o["Key"] for o in resp.get("Contents", [])
                if Path(o["Key"]).name.startswith(("lgbm_", "ridge_"))
            ],
            reverse=True,
        )
        downloaded = 0
        for key in objects:
            fname = Path(key).name
            local = MODELS_DIR / fname
            s3.download_file(S3_BUCKET, key, str(local))
            logger.info("S3 restore ← s3://%s/%s", S3_BUCKET, key)
            downloaded += 1
            if downloaded >= 2:  # one pkl + one json
                break
    except Exception as e:
        logger.warning("S3 restore failed: %s", e)


def save(model, scaler, metrics: dict) -> Path:
    """
    Persist a trained model + scaler (may be None for LightGBM) + metrics to disk.
    Filename: lgbm_YYYYMMDD_HHMMSS.pkl
    Metrics saved alongside as lgbm_YYYYMMDD_HHMMSS.json

    Returns path to the saved .pkl file.
    """
    ts        = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    pkl_path  = MODELS_DIR / f"lgbm_{ts}.pkl"
    meta_path = MODELS_DIR / f"lgbm_{ts}.json"

    joblib.dump({"model": model, "scaler": scaler}, pkl_path)

    with open(meta_path, "w") as f:
        json.dump(
            {
                "timestamp": ts,
                "metrics":   metrics,
                "artifact":  pkl_path.name,
            },
            f,
            indent=2,
        )

    logger.info("Model saved → %s  |  metrics: %s", pkl_path.name, metrics)

    _s3_upload(pkl_path)
    _s3_upload(meta_path)

    return pkl_path


def load_latest():
    """
    Load the most recently saved model artifact.
    Prefers lgbm_* artifacts; falls back to ridge_* for backward compatibility.
    Returns (model, scaler) tuple — scaler is None for LightGBM models.
    Raises FileNotFoundError if no artifacts exist.
    """
    pkls = sorted(MODELS_DIR.glob("lgbm_*.pkl"))
    if not pkls:
        pkls = sorted(MODELS_DIR.glob("ridge_*.pkl"))
    if not pkls:
        raise FileNotFoundError(
            f"No trained model found in {MODELS_DIR}. Run predictor.train() first."
        )
    latest = pkls[-1]
    logger.info("Loading model ← %s", latest.name)
    bundle = joblib.load(latest)
    return bundle["model"], bundle.get("scaler")


def load_latest_metadata() -> dict:
    """
    Return the metrics dict from the most recent model JSON without loading the pkl.
    Used by the health monitor to read current model IC.
    Returns empty dict if no artifacts exist.
    """
    metas = sorted(MODELS_DIR.glob("lgbm_*.json"))
    if not metas:
        metas = sorted(MODELS_DIR.glob("ridge_*.json"))
    if not metas:
        return {}
    with open(metas[-1]) as f:
        return json.load(f)


def delete_latest() -> bool:
    """
    Delete the most recently saved model artifacts (pkl + json).
    Used by the gated retrain to roll back a newly trained model that failed
    the IC gate — the previous model remains as the latest artifact.
    Returns True if artifacts were deleted, False if none existed.
    """
    pkls = sorted(MODELS_DIR.glob("lgbm_*.pkl"))
    if not pkls:
        pkls = sorted(MODELS_DIR.glob("ridge_*.pkl"))
    if not pkls:
        return False
    latest_pkl  = pkls[-1]
    latest_json = latest_pkl.with_suffix(".json")
    latest_pkl.unlink(missing_ok=True)
    latest_json.unlink(missing_ok=True)
    logger.info("Deleted rejected model artifacts: %s", latest_pkl.name)
    return True


def list_artifacts() -> list[dict]:
    """Return metadata for all saved artifacts, newest first (lgbm first, then ridge)."""
    lgbm_metas  = sorted(MODELS_DIR.glob("lgbm_*.json"),  reverse=True)
    ridge_metas = sorted(MODELS_DIR.glob("ridge_*.json"), reverse=True)
    results = []
    for path in lgbm_metas + ridge_metas:
        with open(path) as f:
            results.append(json.load(f))
    return results
