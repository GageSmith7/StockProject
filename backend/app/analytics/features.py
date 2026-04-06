import logging

import numpy as np
import pandas as pd

from app.data.fetcher import fetch_ticker
from app.data.store import get_prices
from app.data.symbols import get_all_tickers
from app.database import MacroData, SessionLocal, Symbol

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Features produced by build_features() for a single ticker.
PER_TICKER_COLS = [
    "ret_1d",
    "ret_5d",
    "ret_21d",
    "ret_63d",
    "ret_126d",    # 6-month momentum — fills gap between 63d and 252d (Fama-French)
    "ret_252d",    # 1-year momentum (Jegadeesh-Titman factor)
    "vol_21d",
    "vol_63d",
    "vol_ratio",
    "vol_ratio_63d",  # 63-day volume ratio — less noisy than 21d
    "rsi_14",
    "macd",
    "macd_signal",
    "bb_position",
    "high_52w_ratio",  # close / 252d rolling max — nearness to 52-week high (George & Hwang 2004)
]

# All stock-level features that get z-scored per (date × sector) group.
# Includes PER_TICKER_COLS plus SPY-relative features added in
# build_cross_sectional_dataset (still stock-level signals, just market-adjusted).
SECTOR_NORM_COLS = PER_TICKER_COLS + [
    "ret_5d_vs_spy",   # market-neutral 5d momentum — still sector-normalized
    "ret_21d_vs_spy",  # market-neutral 21d momentum — still sector-normalized
]

# Market-wide features — identical for every stock on a given date.
# MUST NOT be sector-normalized (sector mean = the value → would zero out).
# StandardScaler in predictor.py handles their scaling instead.
MARKET_FEATURES = [
    "log_vix",         # log(VIX) — fear/regime indicator
    "yield_curve",     # T10Y2Y spread — recession signal (negative = inverted)
    "cpi_yoy",         # CPI 12-month % change — inflation regime
    "dff_delta_63d",   # Fed funds rate change over 63 days — tightening/easing direction
]

FEATURE_COLS = SECTOR_NORM_COLS + MARKET_FEATURES   # 21 total
TARGET_COL   = "target_21d"

# Rows dropped from the front of each ticker's history before features are valid.
# Driven by the longest backward-looking window: 252-day return.
WARMUP_DAYS    = 252
TARGET_HORIZON = 21   # forward days for prediction target


# ---------------------------------------------------------------------------
# Per-ticker feature builder
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical features for a single ticker.

    Input : cleaned OHLCV DataFrame (date, open, high, low, close, volume, ticker).
    Output: feature DataFrame — one row per trading day with SECTOR_NORM_COLS + target_21d.
            SPY-relative and VIX features are added later in build_cross_sectional_dataset.

    Drops:
      - First WARMUP_DAYS rows (insufficient history for 252-day windows).
      - Last TARGET_HORIZON rows (no forward return available yet).

    IMPORTANT: target_21d is a label — never pass it to the model as a feature.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy().sort_values("date").reset_index(drop=True)

    close  = df["close"].astype(float)
    volume = df["volume"].astype(float)

    # --- Momentum ---
    df["ret_1d"]   = close.pct_change(1)
    df["ret_5d"]   = close.pct_change(5)
    df["ret_21d"]  = close.pct_change(21)
    df["ret_63d"]  = close.pct_change(63)
    df["ret_126d"] = close.pct_change(126)   # 6-month momentum
    df["ret_252d"] = close.pct_change(252)   # 1-year momentum

    # --- 52-week high ratio: where close sits relative to its 1-year peak ---
    # Values near 1.0 = near the high (momentum continuation signal).
    # Requires 252-day window — same warmup as ret_252d.
    rolling_max_252        = close.rolling(252).max()
    df["high_52w_ratio"]   = close / rolling_max_252.replace(0, np.nan)

    # --- Volatility: rolling std of daily returns ---
    daily_ret     = close.pct_change(1)
    df["vol_21d"] = daily_ret.rolling(21).std()
    df["vol_63d"] = daily_ret.rolling(63).std()

    # --- Volume ratio: today vs rolling mean ---
    df["vol_ratio"]    = volume / volume.rolling(21).mean()
    df["vol_ratio_63d"] = volume / volume.rolling(63).mean()  # smoother long-window version

    # --- RSI (14-day, Wilder smoothing) ---
    df["rsi_14"] = _rsi(close, 14)

    # --- MACD (normalized by close — price-scale-independent) ---
    ema12         = close.ewm(span=12, adjust=False).mean()
    ema26         = close.ewm(span=26, adjust=False).mean()
    macd_line     = (ema12 - ema26) / close
    signal        = macd_line.ewm(span=9, adjust=False).mean()
    df["macd"]        = macd_line
    df["macd_signal"] = signal

    # --- Bollinger Band position: where close sits within the band ---
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["bb_position"] = (close - sma20) / (2 * std20.replace(0, np.nan))

    # --- Target: 21-day forward return (label only — never a feature) ---
    df[TARGET_COL] = close.pct_change(TARGET_HORIZON).shift(-TARGET_HORIZON)

    # --- Drop warmup rows and rows with no target ---
    df = df.iloc[WARMUP_DAYS:].copy()
    df = df.dropna(subset=[TARGET_COL])

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# RSI helper
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder's RSI.  Returns values in [0, 100].
    Uses EWM with alpha = 1/period (equivalent to Wilder's smoothing).
    """
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------------------------
# Cross-sectional dataset builder
# ---------------------------------------------------------------------------

def build_cross_sectional_dataset(
    tickers: list[str],
    start_date: str,
    end_date: str,
    constituent_dates: dict[str, str | None] | None = None,
) -> pd.DataFrame:
    """
    Build the full feature matrix used for ML training.

    For each ticker:
      1. Pull prices from DB (start_date → end_date).
      2. Apply point-in-time constituent filter: if constituent_dates is provided,
         only keep price rows from the date the ticker actually joined the S&P 500.
         This eliminates survivorship bias — we never train on a company's history
         from before it was selected into the index.
      3. Run build_features() — produces SECTOR_NORM_COLS + target.

    Then across all tickers:
      4. Attach sector from symbols table.
      5. Add SPY-relative momentum (ret_5d_vs_spy, ret_21d_vs_spy).
      6. Add log(VIX) regime indicator.
      7. Add FRED macro features.
      8. Sector-normalize SECTOR_NORM_COLS only (market-wide features skip this).

    Args:
        constituent_dates: {ticker: date_added_iso_or_None} from get_constituent_dates().
                           None disables the filter (e.g. for live predictions where we
                           want the full recent window, not training-time filtering).

    Returns DataFrame with: date, ticker, sector, FEATURE_COLS, target_21d.
    Rows with any NaN in FEATURE_COLS are dropped.
    """
    sector_map = _get_sector_map()
    frames     = []
    skipped    = 0

    # Per-ticker base features
    for ticker in tickers:
        prices = get_prices(ticker, start=start_date, end=end_date)
        if prices.empty:
            skipped += 1
            continue

        # Point-in-time constituent filter — core survivorship bias fix.
        # Only use price history from when this ticker actually entered the index.
        # Tickers with None date have been in the index since before recorded history.
        if constituent_dates is not None:
            added_date = constituent_dates.get(ticker)
            if added_date:
                prices = prices[prices["date"] >= pd.Timestamp(added_date).date()]
                if prices.empty:
                    skipped += 1
                    continue

        feat = build_features(prices)
        if feat.empty:
            skipped += 1
            continue

        feat["sector"] = sector_map.get(ticker, "Unknown")
        base_cols = ["date", "ticker", "sector"] + PER_TICKER_COLS + [TARGET_COL]
        frames.append(feat[base_cols])

    if not frames:
        logger.warning("build_cross_sectional_dataset: no data for any ticker")
        return pd.DataFrame()

    if skipped:
        logger.info("build_cross_sectional_dataset: skipped %d tickers (no data)", skipped)

    dataset = pd.concat(frames, ignore_index=True)
    dataset["date"] = pd.to_datetime(dataset["date"])

    # --- SPY-relative momentum ---
    dataset = _add_spy_relative(dataset)

    # --- VIX regime indicator ---
    dataset = _add_vix(dataset, start_date, end_date)

    # --- FRED macro features (yield curve, CPI, Fed rate direction) ---
    dataset = _add_macro_features(dataset)

    # Drop rows where any feature is NaN before normalization
    dataset = dataset.dropna(subset=FEATURE_COLS)

    # Sector-normalize stock-level features only (not log_vix)
    dataset = _normalize_cross_sectional(dataset, SECTOR_NORM_COLS)

    logger.info(
        "build_cross_sectional_dataset: %d rows, %d tickers, %s → %s",
        len(dataset),
        dataset["ticker"].nunique(),
        start_date,
        end_date,
    )

    return dataset.sort_values(["date", "ticker"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# SPY-relative momentum
# ---------------------------------------------------------------------------

def _add_spy_relative(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Subtract SPY's ret_5d and ret_21d from every ticker's equivalent.
    Result: market-neutral momentum — 'did this stock outperform the market?'

    SPY must be present in the dataset. If missing, columns default to 0.
    """
    spy = dataset[dataset["ticker"] == "SPY"][["date", "ret_5d", "ret_21d"]].copy()

    if spy.empty:
        logger.warning("SPY not found in dataset — ret_*_vs_spy will be 0")
        dataset["ret_5d_vs_spy"]  = 0.0
        dataset["ret_21d_vs_spy"] = 0.0
        return dataset

    spy = spy.rename(columns={"ret_5d": "spy_ret_5d", "ret_21d": "spy_ret_21d"})
    dataset = dataset.merge(spy, on="date", how="left")
    dataset["ret_5d_vs_spy"]  = dataset["ret_5d"]  - dataset["spy_ret_5d"]
    dataset["ret_21d_vs_spy"] = dataset["ret_21d"] - dataset["spy_ret_21d"]
    dataset = dataset.drop(columns=["spy_ret_5d", "spy_ret_21d"])
    return dataset


# ---------------------------------------------------------------------------
# VIX regime indicator
# ---------------------------------------------------------------------------

def _add_vix(dataset: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch ^VIX from yFinance, compute log(VIX), join by date.
    log-transform makes the relationship more linear for Ridge:
      VIX 10→20 (normal) and VIX 40→80 (crisis) are treated proportionally.

    Falls back to log(18) — median long-run VIX — if fetch fails.
    """
    FALLBACK_LOG_VIX = float(np.log(18))

    vix_df = fetch_ticker("^VIX", start_date, end_date)

    if vix_df.empty:
        logger.warning("^VIX fetch failed — using fallback log_vix=%.3f", FALLBACK_LOG_VIX)
        dataset["log_vix"] = FALLBACK_LOG_VIX
        return dataset

    vix_df["date"]    = pd.to_datetime(vix_df["date"])
    vix_df["log_vix"] = np.log(vix_df["close"].clip(lower=1))  # clip avoids log(0)
    vix_lookup        = vix_df[["date", "log_vix"]].drop_duplicates("date")

    dataset = dataset.merge(vix_lookup, on="date", how="left")

    # Forward-fill up to 5 days for holidays / weekends where VIX has no entry
    dataset["log_vix"] = dataset.sort_values("date")["log_vix"].ffill(limit=5)
    dataset["log_vix"] = dataset["log_vix"].fillna(FALLBACK_LOG_VIX)

    return dataset


# ---------------------------------------------------------------------------
# FRED macro features
# ---------------------------------------------------------------------------

def _add_macro_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Join three market-regime signals from the macro_data table onto the dataset.
    All three are market-wide — the same value for every ticker on a given date.
    MUST NOT be sector-normalized (they would zero out).

      yield_curve   : T10Y2Y spread in percentage points.
                      Negative = inverted curve = recession signal.
      cpi_yoy       : CPI 12-month % change (inflation regime).
                      Computed from CPIAUCSL monthly values; forward-filled daily.
      dff_delta_63d : Change in federal funds rate over 63 trading days (~3 months).
                      Captures Fed tightening (+) vs easing (-) direction.

    Falls back to 0.0 for any series not yet in the DB, so training is never
    blocked by missing data — the feature simply has no signal that period.
    """
    with SessionLocal() as session:
        rows = session.query(MacroData.date, MacroData.series_id, MacroData.value).all()

    if not rows:
        logger.warning("_add_macro_features: macro_data table is empty — all macro features set to 0")
        for col in ["yield_curve", "cpi_yoy", "dff_delta_63d"]:
            dataset[col] = 0.0
        return dataset

    macro = pd.DataFrame(rows, columns=["date", "series_id", "value"])
    macro["date"] = pd.to_datetime(macro["date"])
    macro["value"] = macro["value"].astype(float)

    dates = dataset["date"].drop_duplicates().sort_values()
    date_index = pd.DataFrame({"date": dates})

    # ── T10Y2Y — yield curve spread ─────────────────────────────────────────
    t10y2y = macro[macro["series_id"] == "T10Y2Y"][["date", "value"]].copy()
    t10y2y = t10y2y.rename(columns={"value": "yield_curve"})
    t10y2y = date_index.merge(t10y2y, on="date", how="left").sort_values("date")
    t10y2y["yield_curve"] = t10y2y["yield_curve"].ffill(limit=5).fillna(0.0)

    # ── CPIAUCSL — CPI 12-month % change ────────────────────────────────────
    cpi = macro[macro["series_id"] == "CPIAUCSL"][["date", "value"]].copy()
    cpi = cpi.sort_values("date").set_index("date")
    # pct_change(12) on monthly data = 12-month YoY change
    cpi["cpi_yoy"] = cpi["value"].pct_change(12, fill_method=None)
    cpi = cpi[["cpi_yoy"]].reset_index()
    cpi = date_index.merge(cpi, on="date", how="left").sort_values("date")
    cpi["cpi_yoy"] = cpi["cpi_yoy"].ffill(limit=35).fillna(0.0)  # monthly → ffill up to 35 days

    # ── DFF — 63-day rate change ─────────────────────────────────────────────
    dff = macro[macro["series_id"] == "DFF"][["date", "value"]].copy()
    dff = dff.sort_values("date").set_index("date")
    dff["dff_delta_63d"] = dff["value"].diff(63)
    dff = dff[["dff_delta_63d"]].reset_index()
    dff = date_index.merge(dff, on="date", how="left").sort_values("date")
    dff["dff_delta_63d"] = dff["dff_delta_63d"].ffill(limit=5).fillna(0.0)

    # ── Merge all three onto dataset ─────────────────────────────────────────
    dataset = dataset.merge(t10y2y[["date", "yield_curve"]], on="date", how="left")
    dataset = dataset.merge(cpi[["date", "cpi_yoy"]],        on="date", how="left")
    dataset = dataset.merge(dff[["date", "dff_delta_63d"]],  on="date", how="left")

    for col in ["yield_curve", "cpi_yoy", "dff_delta_63d"]:
        dataset[col] = dataset[col].fillna(0.0)

    logger.info(
        "_add_macro_features: yield_curve=%.2f cpi_yoy=%.3f dff_delta=%.2f (latest values)",
        dataset["yield_curve"].iloc[-1],
        dataset["cpi_yoy"].iloc[-1],
        dataset["dff_delta_63d"].iloc[-1],
    )

    return dataset


# ---------------------------------------------------------------------------
# Sector normalization
# ---------------------------------------------------------------------------

def _normalize_cross_sectional(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    For each (date, sector) group, z-score each column in cols.
    Groups with only one member (std = 0) are set to 0.
    Columns not in cols (e.g. log_vix) are left untouched.
    """
    df = df.copy()
    for col in cols:
        df[col] = df.groupby(["date", "sector"])[col].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0
        )
    return df


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _get_sector_map() -> dict[str, str]:
    """Return {ticker: sector} for all active symbols."""
    with SessionLocal() as session:
        rows = session.query(Symbol.ticker, Symbol.sector).filter(
            Symbol.active == True  # noqa: E712
        ).all()
    return {r.ticker: r.sector for r in rows}
