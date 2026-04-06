import logging

import pandas as pd

logger = logging.getLogger(__name__)

OUTLIER_THRESHOLD = 0.15  # 15% single-day close move
MAX_FILL_DAYS = 3

PRICE_COLS = ["open", "high", "low", "close", "volume"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a raw OHLCV DataFrame from fetcher.py.

    - Forward-fills NaN closes up to MAX_FILL_DAYS consecutive missing days.
      Longer runs are left as NaN and logged as warnings.
    - Flags single-day close moves > 15% as outliers. Does NOT remove them.
    - Attaches data_quality_flag to every row: clean | forward_filled | outlier_flagged

    Never removes rows. Splits and dividends are already handled upstream
    by yFinance auto_adjust=True — do not re-adjust here.

    Returns a cleaned copy. Input DataFrame is not modified.
    """
    if df.empty:
        return df.copy()

    df = df.copy()
    ticker = df["ticker"].iloc[0] if "ticker" in df.columns else "unknown"

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Initialize every row as clean
    df["data_quality_flag"] = "clean"

    originally_missing = df["close"].isna()

    if originally_missing.any():
        _log_long_gaps(df["close"], ticker)

        for col in PRICE_COLS:
            if col in df.columns:
                df[col] = df[col].ffill(limit=MAX_FILL_DAYS)

        forward_filled_mask = originally_missing & df["close"].notna()
        df.loc[forward_filled_mask, "data_quality_flag"] = "forward_filled"
    else:
        forward_filled_mask = pd.Series(False, index=df.index)

    # Flag outliers — rows not already marked as forward_filled take priority
    daily_return = df["close"].pct_change(fill_method=None).abs()
    outlier_mask = daily_return > OUTLIER_THRESHOLD
    df.loc[outlier_mask & ~forward_filled_mask, "data_quality_flag"] = "outlier_flagged"

    df["date"] = df["date"].dt.date

    return df


def _log_long_gaps(close: pd.Series, ticker: str) -> None:
    """Log a warning for each consecutive NaN run longer than MAX_FILL_DAYS."""
    null_mask = close.isna()
    if not null_mask.any():
        return

    # Label each consecutive group of NaN/non-NaN
    groups = (null_mask != null_mask.shift()).cumsum()
    for _, group in close[null_mask].groupby(groups[null_mask]):
        if len(group) > MAX_FILL_DAYS:
            logger.warning(
                "Gap > %d trading days for %s: %s to %s (%d days) — not filled",
                MAX_FILL_DAYS,
                ticker,
                group.index[0],
                group.index[-1],
                len(group),
            )
