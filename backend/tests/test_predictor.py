"""
Tests for app/models/predictor.py and app/models/registry.py

Pure-function / unit tests:
  - _winsorize           — clipping behavior
  - _compute_ic          — Spearman IC edge cases
  - walk_forward_cv      — fold structure on synthetic data
  - predict              — output shape, no NaN, NaN-row guard

Registry tests:
  - save / load_latest   — round-trip on disk
  - list_artifacts       — ordering
  - backward compat      — ridge_* artifacts still loadable

Integration tests (require live DB + trained model artifact):
  - train                — end-to-end train on real data; mean_ic >= IC_GATE
  - predict (live)       — live snapshot predictions are finite and correctly indexed
"""
import numpy as np
import pandas as pd
import pytest

import lightgbm as lgb

from app.analytics.features import FEATURE_COLS, TARGET_COL
from app.models import registry
from app.models.predictor import (
    IC_GATE,
    LGB_PARAMS,
    MIN_TRAIN_YEARS,
    N_ESTIMATORS,
    VAL_MONTHS,
    WINSOR_CLIP,
    _compute_ic,
    _winsorize,
    predict,
    walk_forward_cv,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_dataset(n_tickers: int = 20, n_dates: int = 300) -> pd.DataFrame:
    """
    Synthetic cross-sectional dataset with the right columns.
    Dates span n_dates trading days from 2015-01-02.
    Each (date, ticker) row has random features and a random target.
    """
    rng     = np.random.default_rng(42)
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    dates   = pd.date_range("2015-01-02", periods=n_dates, freq="B")

    rows = []
    for date in dates:
        for ticker in tickers:
            row            = {col: rng.normal() for col in FEATURE_COLS}
            row[TARGET_COL] = rng.normal(0, 0.02)
            row["date"]    = date
            row["ticker"]  = ticker
            rows.append(row)

    return pd.DataFrame(rows)


def _make_registered_lgbm_model(tmp_path, monkeypatch):
    """Save a tiny trained LightGBM model into a temp registry dir, patch MODELS_DIR."""
    monkeypatch.setattr(registry, "MODELS_DIR", tmp_path)
    tmp_path.mkdir(exist_ok=True)

    rng = np.random.default_rng(0)
    X   = rng.standard_normal((200, len(FEATURE_COLS))).astype(np.float32)
    y   = rng.standard_normal(200).astype(np.float32)

    ds      = lgb.Dataset(X, label=y)
    params  = {**LGB_PARAMS, "num_leaves": 4, "n_estimators": 10}
    booster = lgb.train(params, ds, num_boost_round=10)

    metrics = {"mean_ic": 0.08, "n_estimators": 10}
    registry.save(booster, None, metrics)
    return booster


# ===========================================================================
# _winsorize
# ===========================================================================

class TestWinsorize:

    def test_clips_top_tail(self):
        s       = pd.Series(list(range(100)) + [1_000_000])
        clipped = _winsorize(s)
        assert clipped.max() < 1_000_000

    def test_clips_bottom_tail(self):
        s       = pd.Series([-1_000_000] + list(range(100)))
        clipped = _winsorize(s)
        assert clipped.min() > -1_000_000

    def test_length_unchanged(self):
        s = pd.Series(np.random.randn(200))
        assert len(_winsorize(s)) == len(s)

    def test_middle_values_unchanged(self):
        """Values nowhere near the tails should not be clipped."""
        s       = pd.Series([-1_000, 0.0, 1_000])
        clipped = _winsorize(s)
        assert clipped.iloc[1] == pytest.approx(0.0)

    def test_constant_series_unchanged(self):
        s      = pd.Series([5.0] * 50)
        result = _winsorize(s)
        assert (result == 5.0).all()


# ===========================================================================
# _compute_ic
# ===========================================================================

class TestComputeIC:

    def test_perfect_rank_correlation(self):
        y  = np.arange(10, dtype=float)
        ic = _compute_ic(y, y)
        assert ic == pytest.approx(1.0, abs=1e-6)

    def test_perfect_negative_correlation(self):
        y  = np.arange(10, dtype=float)
        ic = _compute_ic(y, -y)
        assert ic == pytest.approx(-1.0, abs=1e-6)

    def test_uncorrelated_returns_near_zero(self):
        rng    = np.random.default_rng(99)
        y_true = rng.standard_normal(500)
        y_pred = rng.standard_normal(500)
        ic     = _compute_ic(y_true, y_pred)
        assert abs(ic) < 0.15

    def test_constant_prediction_returns_zero(self):
        """Constant predictions → zero variance → Spearman undefined → returns 0."""
        y_true = np.arange(10, dtype=float)
        y_pred = np.ones(10)
        ic     = _compute_ic(y_true, y_pred)
        assert ic == pytest.approx(0.0)

    def test_output_is_float(self):
        y = np.array([1.0, 2.0, 3.0])
        assert isinstance(_compute_ic(y, y), float)


# ===========================================================================
# walk_forward_cv
# ===========================================================================

class TestWalkForwardCV:

    def test_returns_expected_keys(self):
        ds     = _make_dataset(n_tickers=20, n_dates=300)
        result = walk_forward_cv(ds)
        for key in ("mean_ic", "fold_ics", "mean_n_estimators", "n_folds", "gate"):
            assert key in result

    def test_n_folds_positive(self):
        # 700 trading days ≈ 2.8 years — enough for MIN_TRAIN_YEARS=2 + one VAL_MONTHS fold
        ds     = _make_dataset(n_tickers=20, n_dates=700)
        result = walk_forward_cv(ds)
        assert result["n_folds"] >= 1

    def test_fold_ics_length_matches_n_folds(self):
        ds     = _make_dataset(n_tickers=20, n_dates=700)
        result = walk_forward_cv(ds)
        assert len(result["fold_ics"]) == result["n_folds"]

    def test_mean_ic_is_mean_of_fold_ics(self):
        ds     = _make_dataset(n_tickers=20, n_dates=700)
        result = walk_forward_cv(ds)
        assert result["n_folds"] >= 1, "need at least one fold"
        assert result["mean_ic"] == pytest.approx(np.mean(result["fold_ics"]), abs=1e-9)

    def test_gate_pass_or_below(self):
        ds     = _make_dataset(n_tickers=20, n_dates=700)
        result = walk_forward_cv(ds)
        assert result["gate"] in ("PASS", "BELOW GATE")

    def test_gate_reflects_ic_gate_constant(self):
        ds     = _make_dataset(n_tickers=20, n_dates=700)
        result = walk_forward_cv(ds)
        if result["mean_ic"] >= IC_GATE:
            assert result["gate"] == "PASS"
        else:
            assert result["gate"] == "BELOW GATE"

    def test_insufficient_data_returns_zero_mean_ic(self):
        """Dataset too small to form any valid fold → mean_ic=0."""
        ds     = _make_dataset(n_tickers=5, n_dates=10)
        result = walk_forward_cv(ds)
        assert result["mean_ic"] == pytest.approx(0.0)
        assert result["n_folds"] == 0

    def test_mean_n_estimators_positive_when_folds_exist(self):
        ds     = _make_dataset(n_tickers=20, n_dates=700)
        result = walk_forward_cv(ds)
        if result["n_folds"] > 0:
            assert result["mean_n_estimators"] > 0


# ===========================================================================
# Registry — save / load_latest / list_artifacts / backward compat
# ===========================================================================

class TestRegistry:

    def test_save_creates_lgbm_pkl_and_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr(registry, "MODELS_DIR", tmp_path)
        rng     = np.random.default_rng(0)
        X       = rng.standard_normal((100, len(FEATURE_COLS))).astype(np.float32)
        y       = rng.standard_normal(100).astype(np.float32)
        ds      = lgb.Dataset(X, label=y)
        booster = lgb.train({**LGB_PARAMS, "num_leaves": 4}, ds, num_boost_round=5)

        path = registry.save(booster, None, {"mean_ic": 0.1})
        assert path.name.startswith("lgbm_")
        assert path.suffix == ".pkl"
        assert path.exists()
        assert path.with_suffix(".json").exists()

    def test_load_latest_returns_booster_and_none_scaler(self, tmp_path, monkeypatch):
        _make_registered_lgbm_model(tmp_path, monkeypatch)
        model, scaler = registry.load_latest()
        assert hasattr(model, "predict")    # lgb.Booster has predict
        assert scaler is None               # LightGBM doesn't need a scaler

    def test_load_latest_model_predicts_correct_shape(self, tmp_path, monkeypatch):
        _make_registered_lgbm_model(tmp_path, monkeypatch)
        model, _ = registry.load_latest()
        X        = np.random.randn(5, len(FEATURE_COLS)).astype(np.float32)
        preds    = model.predict(X)
        assert preds.shape == (5,)

    def test_load_latest_raises_when_no_artifacts(self, tmp_path, monkeypatch):
        monkeypatch.setattr(registry, "MODELS_DIR", tmp_path)
        with pytest.raises(FileNotFoundError):
            registry.load_latest()

    def test_list_artifacts_newest_first(self, tmp_path, monkeypatch):
        monkeypatch.setattr(registry, "MODELS_DIR", tmp_path)
        rng = np.random.default_rng(0)
        X   = rng.standard_normal((100, len(FEATURE_COLS))).astype(np.float32)
        y   = rng.standard_normal(100).astype(np.float32)

        import time
        for i in range(3):
            ds      = lgb.Dataset(X, label=y)
            booster = lgb.train({**LGB_PARAMS, "num_leaves": 4}, ds, num_boost_round=5)
            registry.save(booster, None, {"run": i})
            time.sleep(1.1)   # ensure distinct timestamps

        artifacts  = registry.list_artifacts()
        timestamps = [a["timestamp"] for a in artifacts]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_list_artifacts_metadata_keys(self, tmp_path, monkeypatch):
        _make_registered_lgbm_model(tmp_path, monkeypatch)
        artifacts = registry.list_artifacts()
        assert len(artifacts) >= 1
        a = artifacts[0]
        assert "timestamp" in a
        assert "metrics"   in a
        assert "artifact"  in a

    def test_backward_compat_loads_ridge_when_no_lgbm(self, tmp_path, monkeypatch):
        """
        If only a ridge_*.pkl exists (old artifact), load_latest() should still
        return it rather than raising FileNotFoundError.
        """
        import joblib
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        monkeypatch.setattr(registry, "MODELS_DIR", tmp_path)
        tmp_path.mkdir(exist_ok=True)

        rng    = np.random.default_rng(7)
        X      = rng.standard_normal((50, len(FEATURE_COLS)))
        y      = rng.standard_normal(50)
        scaler = StandardScaler()
        model  = Ridge(alpha=1.0)
        model.fit(scaler.fit_transform(X), y)

        # Write directly as a ridge_ artifact (bypassing registry.save)
        pkl_path  = tmp_path / "ridge_20230101_120000.pkl"
        meta_path = tmp_path / "ridge_20230101_120000.json"
        joblib.dump({"model": model, "scaler": scaler}, pkl_path)
        import json
        with open(meta_path, "w") as f:
            json.dump({"timestamp": "20230101_120000", "metrics": {}, "artifact": pkl_path.name}, f)

        loaded_model, loaded_scaler = registry.load_latest()
        assert hasattr(loaded_model, "predict")
        assert loaded_scaler is not None    # Ridge bundle has a real scaler


# ===========================================================================
# predict (unit — uses temp registry with LightGBM model)
# ===========================================================================

class TestPredict:

    def test_returns_series(self, tmp_path, monkeypatch):
        _make_registered_lgbm_model(tmp_path, monkeypatch)
        rng = np.random.default_rng(1)
        df  = pd.DataFrame(
            rng.standard_normal((10, len(FEATURE_COLS))),
            columns=FEATURE_COLS,
            index=[f"T{i}" for i in range(10)],
        )
        result = predict(df)
        assert isinstance(result, pd.Series)

    def test_output_length_matches_input(self, tmp_path, monkeypatch):
        _make_registered_lgbm_model(tmp_path, monkeypatch)
        rng = np.random.default_rng(2)
        df  = pd.DataFrame(
            rng.standard_normal((15, len(FEATURE_COLS))),
            columns=FEATURE_COLS,
            index=[f"T{i}" for i in range(15)],
        )
        result = predict(df)
        assert len(result) == 15

    def test_no_nan_in_output(self, tmp_path, monkeypatch):
        _make_registered_lgbm_model(tmp_path, monkeypatch)
        rng = np.random.default_rng(3)
        df  = pd.DataFrame(
            rng.standard_normal((10, len(FEATURE_COLS))),
            columns=FEATURE_COLS,
            index=[f"T{i}" for i in range(10)],
        )
        result = predict(df)
        assert result.isna().sum() == 0

    def test_ticker_column_accepted(self, tmp_path, monkeypatch):
        """predict() should handle a 'ticker' column as well as a ticker index."""
        _make_registered_lgbm_model(tmp_path, monkeypatch)
        rng = np.random.default_rng(4)
        df  = pd.DataFrame(rng.standard_normal((5, len(FEATURE_COLS))), columns=FEATURE_COLS)
        df.insert(0, "ticker", [f"T{i}" for i in range(5)])
        result = predict(df)
        assert list(result.index) == [f"T{i}" for i in range(5)]

    def test_nan_rows_dropped_silently(self, tmp_path, monkeypatch):
        """Rows with NaN in any feature column should be dropped, not raise."""
        _make_registered_lgbm_model(tmp_path, monkeypatch)
        rng  = np.random.default_rng(5)
        data = rng.standard_normal((10, len(FEATURE_COLS)))
        data[2, 0] = np.nan   # inject NaN into row 2
        df   = pd.DataFrame(data, columns=FEATURE_COLS, index=[f"T{i}" for i in range(10)])
        result = predict(df)
        assert len(result) == 9
        assert "T2" not in result.index

    def test_empty_input_returns_empty_series(self, tmp_path, monkeypatch):
        _make_registered_lgbm_model(tmp_path, monkeypatch)
        df     = pd.DataFrame(columns=FEATURE_COLS)
        result = predict(df)
        assert result.empty
        assert result.name == "predicted_return_21d"

    def test_series_name_is_predicted_return(self, tmp_path, monkeypatch):
        _make_registered_lgbm_model(tmp_path, monkeypatch)
        rng    = np.random.default_rng(6)
        df     = pd.DataFrame(rng.standard_normal((5, len(FEATURE_COLS))), columns=FEATURE_COLS)
        result = predict(df)
        assert result.name == "predicted_return_21d"

    def test_predict_uses_scaler_for_ridge_artifact(self, tmp_path, monkeypatch):
        """
        Backward compat: if a Ridge model with a real scaler is loaded,
        predict() should apply the scaler before calling model.predict().
        """
        import joblib
        import json
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        monkeypatch.setattr(registry, "MODELS_DIR", tmp_path)
        tmp_path.mkdir(exist_ok=True)

        rng    = np.random.default_rng(8)
        X      = rng.standard_normal((100, len(FEATURE_COLS)))
        y      = rng.standard_normal(100)
        scaler = StandardScaler()
        model  = Ridge(alpha=1.0)
        model.fit(scaler.fit_transform(X), y)

        pkl_path  = tmp_path / "ridge_20230101_120000.pkl"
        meta_path = tmp_path / "ridge_20230101_120000.json"
        joblib.dump({"model": model, "scaler": scaler}, pkl_path)
        with open(meta_path, "w") as f:
            json.dump({"timestamp": "20230101_120000", "metrics": {}, "artifact": pkl_path.name}, f)

        df     = pd.DataFrame(rng.standard_normal((5, len(FEATURE_COLS))), columns=FEATURE_COLS)
        result = predict(df)
        assert len(result) == 5
        assert result.isna().sum() == 0


# ===========================================================================
# Integration — train + predict on live DB
# ===========================================================================

class TestTrainIntegration:

    @pytest.mark.integration
    def test_train_returns_pass_gate(self, db_seeded):
        """Full train on real data must achieve mean_ic >= IC_GATE."""
        from app.models.predictor import train
        metrics = train(start_date="2015-01-01", run_cv=True)
        assert metrics["gate"] == "PASS", (
            f"mean_ic={metrics.get('mean_ic', '?'):.4f} below IC_GATE={IC_GATE}"
        )

    @pytest.mark.integration
    def test_train_artifact_saved(self, db_seeded):
        from app.models.predictor import train
        metrics = train(start_date="2015-01-01", run_cv=False)
        assert "artifact" in metrics
        from pathlib import Path
        assert Path(metrics["artifact"]).exists()

    @pytest.mark.integration
    def test_train_artifact_is_lgbm(self, db_seeded):
        """New artifacts must use the lgbm_ prefix."""
        from app.models.predictor import train
        from pathlib import Path
        metrics = train(start_date="2015-01-01", run_cv=False)
        assert Path(metrics["artifact"]).name.startswith("lgbm_")

    @pytest.mark.integration
    def test_train_n_features_matches_feature_cols(self, db_seeded):
        from app.models.predictor import train
        metrics = train(start_date="2015-01-01", run_cv=False)
        assert metrics["n_features"] == len(FEATURE_COLS)

    @pytest.mark.integration
    def test_train_records_point_in_time_flag(self, db_seeded):
        from app.models.predictor import train
        metrics = train(start_date="2015-01-01", run_cv=False)
        assert "point_in_time_filtering" in metrics

    @pytest.mark.integration
    def test_live_predict_all_finite(self, db_seeded):
        """Live snapshot predictions must be finite for all returned tickers."""
        from app.analytics.features import build_cross_sectional_dataset
        from app.data.symbols import get_all_tickers

        tickers = get_all_tickers(source="preloaded")[:30]
        dataset = build_cross_sectional_dataset(tickers, "2024-01-01", "2026-03-16")
        latest  = dataset.sort_values("date").groupby("ticker").tail(1)

        result = predict(latest)
        assert len(result) > 0
        assert np.isfinite(result.values).all()

    @pytest.mark.integration
    def test_live_predict_index_are_tickers(self, db_seeded):
        from app.analytics.features import build_cross_sectional_dataset
        from app.data.symbols import get_all_tickers

        tickers = get_all_tickers(source="preloaded")[:20]
        dataset = build_cross_sectional_dataset(tickers, "2024-01-01", "2026-03-16")
        latest  = dataset.sort_values("date").groupby("ticker").tail(1)

        result = predict(latest)
        assert all(isinstance(t, str) for t in result.index)
