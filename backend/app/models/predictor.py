"""
Cross-sectional LightGBM regressor — train, walk-forward CV, predict.

Replaces Ridge regression to:
  - Capture non-linear interactions between the 18 features (macro × momentum, etc.)
  - Fully utilise all macro features (yield_curve, cpi_yoy, dff_delta_63d, log_vix)
    that Ridge largely ignored due to linear constraints.

Survivorship-bias fix: training data is built with point-in-time constituent filtering
via get_constituent_dates() → build_cross_sectional_dataset(constituent_dates=...).

Training scope: cross-sectional (one model, all tickers as samples).
Each row = one ticker on one date.
Target = 21-day forward return (already computed in features.py).
"""
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

from app.analytics.features import (
    FEATURE_COLS,
    TARGET_COL,
    build_cross_sectional_dataset,
)
from app.data.symbols import get_all_tickers, get_constituent_dates
from app.models import registry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# LightGBM base params — tree structure kept shallow to stay cross-sectional
# (we want factors that generalise across thousands of (ticker, date) rows).
LGB_PARAMS: dict = {
    "objective":         "regression",
    "metric":            "rmse",
    "learning_rate":     0.05,
    "max_depth":         5,
    "num_leaves":        31,
    "min_child_samples": 100,   # prevents overfitting to thin sector/date slices
    "subsample":         0.8,
    "subsample_freq":    1,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,   # L1
    "reg_lambda":        1.0,   # L2
    "n_jobs":            -1,
    "random_state":      42,
    "verbose":           -1,
}

N_ESTIMATORS          = 1000   # upper bound; early stopping finds the true optimum per fold
EARLY_STOPPING_ROUNDS = 50
MIN_TRAIN_YEARS       = 2      # minimum history before first validation fold
VAL_MONTHS            = 6      # validation window per fold
WINSOR_CLIP           = 0.01   # clip target at 1st / 99th percentile
IC_GATE               = 0.05   # minimum mean Spearman IC to pass the quality gate


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _winsorize(series: pd.Series, pct: float = WINSOR_CLIP) -> pd.Series:
    lo = series.quantile(pct)
    hi = series.quantile(1 - pct)
    return series.clip(lo, hi)


def _compute_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank IC between predicted and actual 21d returns."""
    ic, _ = stats.spearmanr(y_pred, y_true)
    return float(ic) if not np.isnan(ic) else 0.0


def _fit_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[float, lgb.Booster, int]:
    """
    Fit LightGBM on training split with early stopping on validation RMSE.
    Returns (ic_on_val, fitted_booster, best_n_estimators).
    """
    train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    val_set   = lgb.Dataset(X_val,   label=y_val,   reference=train_set, free_raw_data=False)

    callbacks = [
        lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
        lgb.log_evaluation(period=-1),   # silence per-round output
    ]

    booster = lgb.train(
        LGB_PARAMS,
        train_set,
        num_boost_round=N_ESTIMATORS,
        valid_sets=[val_set],
        callbacks=callbacks,
    )

    y_pred = booster.predict(X_val)
    ic     = _compute_ic(y_val, y_pred)
    return ic, booster, booster.best_iteration


# ---------------------------------------------------------------------------
# Walk-forward cross-validation
# ---------------------------------------------------------------------------

def walk_forward_cv(dataset: pd.DataFrame) -> dict:
    """
    Run expanding-window walk-forward CV on the cross-sectional dataset.

    Fold structure:
      Train: dataset start → fold_end
      Val  : fold_end     → fold_end + VAL_MONTHS

    Minimum training window: MIN_TRAIN_YEARS.
    Step size: VAL_MONTHS.

    Returns dict with mean_ic, fold_ics, best_n_estimators (average across folds).
    """
    dataset = dataset.copy()
    dataset["date"] = pd.to_datetime(dataset["date"])

    start_date      = dataset["date"].min()
    end_date        = dataset["date"].max()
    first_val_start = start_date + relativedelta(years=MIN_TRAIN_YEARS)

    fold_ics          = []
    best_n_per_fold   = []
    fold_num          = 0

    val_start = first_val_start
    while True:
        val_end = val_start + relativedelta(months=VAL_MONTHS)
        if val_end > end_date:
            break

        train_mask = dataset["date"] < val_start
        val_mask   = (dataset["date"] >= val_start) & (dataset["date"] < val_end)

        X_train = dataset.loc[train_mask, FEATURE_COLS].values
        y_train = _winsorize(dataset.loc[train_mask, TARGET_COL]).values
        X_val   = dataset.loc[val_mask,   FEATURE_COLS].values
        y_val   = dataset.loc[val_mask,   TARGET_COL].values

        if len(X_train) < 500 or len(X_val) < 50:
            val_start += relativedelta(months=VAL_MONTHS)
            continue

        fold_num += 1
        ic, _, best_n = _fit_fold(X_train, y_train, X_val, y_val)
        fold_ics.append(ic)
        best_n_per_fold.append(best_n)

        logger.info(
            "Fold %2d | train %-10s → %-10s | val %-10s → %-10s | IC=%.4f | n_trees=%d",
            fold_num,
            start_date.date(),
            val_start.date(),
            val_start.date(),
            val_end.date(),
            ic,
            best_n,
        )

        val_start += relativedelta(months=VAL_MONTHS)

    mean_ic    = float(np.mean(fold_ics))    if fold_ics    else 0.0
    mean_n     = int(np.mean(best_n_per_fold)) if best_n_per_fold else N_ESTIMATORS
    gate       = "PASS" if mean_ic >= IC_GATE else "BELOW GATE"

    logger.info(
        "Walk-forward CV complete | folds=%d | mean_IC=%.4f | mean_n_trees=%d | %s",
        len(fold_ics), mean_ic, mean_n, gate,
    )

    return {
        "mean_ic":          mean_ic,
        "fold_ics":         fold_ics,
        "mean_n_estimators": mean_n,
        "n_folds":          len(fold_ics),
        "gate":             gate,
    }


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(
    start_date: str = "2015-01-01",
    end_date: str | None = None,
    run_cv: bool = True,
) -> dict:
    """
    Build point-in-time-filtered cross-sectional dataset, run walk-forward CV,
    fit final LightGBM model, save to registry.

    Args:
        start_date: history start for training data
        end_date:   history end (defaults to today)
        run_cv:     set False to skip CV and go straight to final fit (faster, less safe)

    Returns metrics dict including mean_ic and artifact path.
    """
    if end_date is None:
        end_date = datetime.utcnow().date().isoformat()

    logger.info("Building cross-sectional dataset %s → %s", start_date, end_date)
    tickers           = get_all_tickers(source="preloaded")
    constituent_dates = get_constituent_dates()

    if constituent_dates:
        tracked = sum(1 for v in constituent_dates.values() if v is not None)
        logger.info(
            "Point-in-time filtering enabled: %d/%d tickers have a known add date",
            tracked, len(constituent_dates),
        )
    else:
        logger.warning(
            "constituent_history table is empty — training without point-in-time filtering. "
            "Run seed_constituent_history() first."
        )

    dataset = build_cross_sectional_dataset(
        tickers,
        start_date,
        end_date,
        constituent_dates=constituent_dates if constituent_dates else None,
    )
    logger.info("Dataset shape: %s", dataset.shape)

    cv_metrics: dict = {}
    n_estimators      = N_ESTIMATORS

    if run_cv:
        cv_metrics   = walk_forward_cv(dataset)
        n_estimators = cv_metrics.get("mean_n_estimators", N_ESTIMATORS)

    # --- Final fit on full dataset using CV-tuned n_estimators ---
    X = dataset[FEATURE_COLS].values
    y = _winsorize(dataset[TARGET_COL]).values

    train_set = lgb.Dataset(X, label=y)

    final_model = lgb.train(
        LGB_PARAMS,
        train_set,
        num_boost_round=n_estimators,
    )

    logger.info("Final LightGBM model fitted | n_trees=%d", n_estimators)

    metrics = {
        **cv_metrics,
        "n_estimators": n_estimators,
        "n_samples":    len(dataset),
        "n_features":   len(FEATURE_COLS),
        "train_start":  start_date,
        "train_end":    end_date,
        "point_in_time_filtering": bool(constituent_dates),
    }

    # scaler=None — LightGBM is scale-invariant; stored as None for interface compat
    artifact_path   = registry.save(final_model, None, metrics)
    metrics["artifact"] = str(artifact_path)
    return metrics


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

def predict(feature_df: pd.DataFrame) -> pd.Series:
    """
    Generate 21-day return forecasts for a set of tickers.

    Args:
        feature_df: DataFrame with columns = FEATURE_COLS, index or 'ticker' column.
                    Typically the latest row per ticker from build_cross_sectional_dataset.
                    constituent_dates is NOT passed here — live prediction uses the full
                    recent window, not training-time point-in-time filtering.

    Returns:
        pd.Series indexed by ticker, values = predicted 21d return (raw, not ranked).
        Caller is responsible for ranking / quintile assignment.
    """
    model, scaler = registry.load_latest()

    df = feature_df.copy()
    if "ticker" in df.columns:
        df = df.set_index("ticker")

    # Drop rows with NaN features — can occur when SPY is absent or DB partially seeded
    before = len(df)
    df = df.dropna(subset=FEATURE_COLS)
    if len(df) < before:
        logger.warning("predict: dropped %d rows with NaN features", before - len(df))

    if df.empty:
        return pd.Series(dtype=float, name="predicted_return_21d")

    X = df[FEATURE_COLS].values

    # Backward compat: old Ridge artifacts have a StandardScaler; LightGBM models set scaler=None
    if scaler is not None:
        X = scaler.transform(X)

    preds = model.predict(X)
    return pd.Series(preds, index=df.index, name="predicted_return_21d")
