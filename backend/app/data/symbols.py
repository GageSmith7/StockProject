import concurrent.futures
import logging
import re
import time
from datetime import datetime

import httpx
import pandas as pd
from sqlalchemy import case, or_
from sqlalchemy.dialects.postgresql import insert

from app.database import ConstituentHistory, SessionLocal, Symbol
from app.data.cleaner import clean
from app.data.fetcher import fetch_ticker
from app.data.store import get_prices, upsert_prices

logger = logging.getLogger(__name__)

EXTRA_ETFS = [
    {"ticker": "SPY", "name": "SPDR S&P 500 ETF Trust", "sector": "ETF"},
    {"ticker": "QQQ", "name": "Invesco QQQ Trust", "sector": "ETF"},
    {"ticker": "VTI", "name": "Vanguard Total Stock Market ETF", "sector": "ETF"},
    {"ticker": "IWM", "name": "iShares Russell 2000 ETF", "sector": "ETF"},
    {"ticker": "GLD", "name": "SPDR Gold Shares", "sector": "ETF"},
]

WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def _fetch_sp500_wikipedia() -> pd.DataFrame:
    """
    Scrape current S&P 500 constituents from Wikipedia.
    Known limitation: returns today's list — delisted stocks are excluded
    (survivorship bias — documented MVP limitation).
    """
    # urllib's default user-agent is blocked by Wikipedia — use httpx with a real one
    headers = {"User-Agent": "Mozilla/5.0 (compatible; portfolio-simulator/1.0)"}
    response = httpx.get(WIKIPEDIA_SP500_URL, headers=headers, follow_redirects=True)
    response.raise_for_status()
    tables = pd.read_html(pd.io.common.StringIO(response.text), attrs={"id": "constituents"})
    df = tables[0].rename(columns={
        "Symbol": "ticker",
        "Security": "name",
        "GICS Sector": "sector",
    })
    return df[["ticker", "name", "sector"]].copy()


def seed_symbols() -> int:
    """
    Upsert S&P 500 + 5 ETFs into the symbols table.
    Idempotent — safe to call on every startup.
    Returns total number of symbols upserted.
    """
    sp500 = _fetch_sp500_wikipedia()
    # yFinance uses dashes not dots: BRK.B → BRK-B
    sp500["ticker"] = sp500["ticker"].str.replace(".", "-", regex=False)
    etfs = pd.DataFrame(EXTRA_ETFS)
    all_symbols = pd.concat([sp500, etfs], ignore_index=True)
    all_symbols["data_source"] = "preloaded"

    records = all_symbols[["ticker", "name", "sector", "data_source"]].to_dict("records")

    stmt = insert(Symbol).values(records)
    stmt = stmt.on_conflict_do_update(
        index_elements=["ticker"],
        set_={
            "name": stmt.excluded.name,
            "sector": stmt.excluded.sector,
            # Do not overwrite data_source — on_demand tickers stay on_demand
        },
    )

    with SessionLocal() as session:
        session.execute(stmt)
        session.commit()

    logger.info("seed_symbols: upserted %d symbols", len(records))
    return len(records)


def _fetch_sp500_changes() -> dict[str, str]:
    """
    Scrape the S&P 500 changes table from Wikipedia (id="changes").
    Returns {ticker: date_added_iso} — the most recent recorded addition date per ticker.
    Tickers absent from this table have been in the index since before tracked history.
    Falls back to an empty dict on parse failure so the caller degrades gracefully.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; portfolio-simulator/1.0)"}
    response = httpx.get(WIKIPEDIA_SP500_URL, headers=headers, follow_redirects=True)
    response.raise_for_status()

    try:
        tables = pd.read_html(pd.io.common.StringIO(response.text), attrs={"id": "changes"})
        changes = tables[0]
    except Exception as exc:
        logger.warning("_fetch_sp500_changes: could not parse changes table (%s) — no point-in-time filtering", exc)
        return {}

    # Wikipedia uses multi-level headers: ('Date',''), ('Added','Symbol'), ('Added','Security'), ...
    # Flatten them to single strings so we can search by keyword.
    if isinstance(changes.columns, pd.MultiIndex):
        changes.columns = ["_".join(str(part).strip() for part in col if str(part).strip()).lower()
                           for col in changes.columns]
    else:
        changes.columns = [str(c).strip().lower() for c in changes.columns]

    # Locate date column and "Added" ticker column regardless of exact header wording.
    date_col  = next((c for c in changes.columns if "date" in c), None)
    added_col = next(
        (c for c in changes.columns if "added" in c and ("ticker" in c or "symbol" in c)),
        None,
    )

    if not date_col or not added_col:
        logger.warning(
            "_fetch_sp500_changes: unexpected column structure %s — skipping point-in-time filtering",
            list(changes.columns),
        )
        return {}

    result: dict[str, str] = {}
    for _, row in changes.iterrows():
        raw_ticker = str(row.get(added_col, "")).strip()
        raw_date   = str(row.get(date_col,  "")).strip()

        if not raw_ticker or raw_ticker in ("nan", "—", "-", ""):
            continue

        try:
            date_added = pd.to_datetime(raw_date).date().isoformat()
        except Exception:
            continue

        # Normalize dot → dash to match yfinance convention (BRK.B → BRK-B)
        ticker = raw_ticker.replace(".", "-")

        # Keep the most recent addition date for tickers that were removed and re-added
        if ticker not in result or date_added > result[ticker]:
            result[ticker] = date_added

    logger.info("_fetch_sp500_changes: parsed add dates for %d tickers", len(result))
    return result


def seed_constituent_history() -> int:
    """
    Populate constituent_history with the most recent S&P 500 addition date for every
    current preloaded ticker.  date_added = NULL means the ticker predates our recorded
    history — use all available price data for it during training.

    Idempotent — safe to call on every startup.
    Returns number of rows upserted.
    """
    changes = _fetch_sp500_changes()
    tickers = get_all_tickers(source="preloaded")

    if not tickers:
        logger.warning("seed_constituent_history: no preloaded tickers found — skipping")
        return 0

    records = [
        {"ticker": t, "date_added": changes.get(t)}   # None if not in changes table
        for t in tickers
    ]

    stmt = insert(ConstituentHistory).values(records)
    stmt = stmt.on_conflict_do_update(
        index_elements=["ticker"],
        set_={"date_added": stmt.excluded.date_added},
    )

    with SessionLocal() as session:
        session.execute(stmt)
        session.commit()

    tracked = sum(1 for r in records if r["date_added"] is not None)
    logger.info(
        "seed_constituent_history: upserted %d tickers (%d with known add date, %d pre-history)",
        len(records), tracked, len(records) - tracked,
    )
    return len(records)


def get_constituent_dates() -> dict[str, str | None]:
    """
    Return {ticker: date_added_iso_or_None} for all rows in constituent_history.
    None means the ticker has been in the index since before our recorded history —
    use all available price history for it during ML training.
    """
    with SessionLocal() as session:
        rows = session.query(
            ConstituentHistory.ticker, ConstituentHistory.date_added
        ).all()
    return {r.ticker: str(r.date_added) if r.date_added else None for r in rows}


def seed_prices(start_date: str = "2010-01-01", delay: float = 0.3) -> None:
    """
    Fetch full price history for all Tier 1 symbols not yet in prices table.
    Idempotent — skips tickers that already have price data.
    Logs progress every 25 tickers.

    delay: seconds between yFinance calls to avoid rate limiting.
    """
    tickers = get_all_tickers(source="preloaded")
    today = datetime.utcnow().date().isoformat()
    total = len(tickers)
    fetched = skipped = failed = 0

    logger.info("seed_prices: starting for %d tickers from %s", total, start_date)

    for i, ticker in enumerate(tickers, 1):
        if not get_prices(ticker).empty:
            skipped += 1
        else:
            try:
                raw = fetch_ticker(ticker, start_date, today)
                if raw.empty:
                    logger.warning("seed_prices: no data returned for %s", ticker)
                    failed += 1
                else:
                    upsert_prices(clean(raw))
                    fetched += 1
                    if delay > 0:
                        time.sleep(delay)
            except Exception as e:
                logger.warning("seed_prices: failed for %s: %s", ticker, e)
                failed += 1

        if i % 25 == 0 or i == total:
            logger.info(
                "seed_prices: %d/%d — fetched=%d skipped=%d failed=%d",
                i, total, fetched, skipped, failed,
            )

    logger.info(
        "seed_prices complete: fetched=%d skipped=%d failed=%d",
        fetched, skipped, failed,
    )


def run_initial_seed(start_date: str = "2010-01-01") -> None:
    """
    Full initial seed: symbol registry + price history.
    Safe to call on every startup — skips what's already populated.
    Called from main.py lifespan.
    """
    logger.info("run_initial_seed: starting")
    count = seed_symbols()
    logger.info("run_initial_seed: symbol registry ready (%d symbols)", count)
    seed_constituent_history()
    seed_prices(start_date=start_date)
    logger.info("run_initial_seed: complete")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

_TICKER_RE = re.compile(r"^[A-Z0-9.\-]{1,12}$")


def _validate_ticker_yfinance(ticker: str) -> dict | None:
    """
    Lightweight yfinance check for tickers not in the DB.
    Times out after 6 seconds so slow lookups don't block the search response.
    Returns a minimal SearchResult-shaped dict or None if the ticker is invalid.
    """
    def _fetch() -> dict | None:
        import yfinance as yf  # local import — avoids startup cost
        t = yf.Ticker(ticker)

        # fast_info is a lightweight property that doesn't require a full .info call.
        # last_price being a positive number is enough to confirm the ticker exists.
        try:
            last_price = t.fast_info.last_price
            if not last_price or last_price <= 0:
                return None
        except Exception:
            return None

        # Best-effort name/sector — fall back gracefully if .info is slow or incomplete
        name = sector = None
        try:
            info   = t.info
            name   = info.get("longName") or info.get("shortName")
            sector = info.get("sector")
        except Exception:
            pass

        return {
            "ticker": ticker,
            "name":   name or ticker,
            "sector": sector,
            "tier":   "on_demand",
        }

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(_fetch).result(timeout=6)
    except Exception:
        return None


def search_symbols(query: str, limit: int = 10) -> list[dict]:
    """
    Search symbols table by ticker or name (case-insensitive).
    Returns up to `limit` results, ranked: exact ticker → starts-with → contains.

    Fallback: if no DB results and the query looks like a ticker symbol,
    validates it against yfinance and returns it as an on_demand result so
    the user can add any valid ticker (e.g. BTC-USD, TSM, individual ETFs).
    """
    if not query or not query.strip():
        return []

    q = query.strip().upper()

    with SessionLocal() as session:
        results = (
            session.query(Symbol)
            .filter(
                Symbol.active == True,  # noqa: E712
                or_(
                    Symbol.ticker.ilike(f"%{q}%"),
                    Symbol.name.ilike(f"%{q}%"),
                ),
            )
            .order_by(
                case(
                    (Symbol.ticker == q, 0),
                    (Symbol.ticker.ilike(f"{q}%"), 1),
                    else_=2,
                ),
                Symbol.ticker,
            )
            .limit(limit)
            .all()
        )

    db_results = [
        {
            "ticker": r.ticker,
            "name":   r.name,
            "sector": r.sector,
            "tier":   r.data_source,
        }
        for r in results
    ]

    # If nothing came back from the DB and the query looks like a ticker,
    # try yfinance as a fallback so users can add arbitrary valid tickers.
    # Also try the "{query}-USD" form for crypto (BTC → BTC-USD, ETH → ETH-USD).
    if not db_results and _TICKER_RE.match(q):
        candidates = [q]
        if "-" not in q and "." not in q:
            candidates.append(f"{q}-USD")

        for candidate in candidates:
            validated = _validate_ticker_yfinance(candidate)
            if validated:
                logger.info("search_symbols: yfinance validated on-demand ticker %s", candidate)
                return [validated]

    return db_results


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_all_tickers(source: str = None) -> list[str]:
    """
    Return all active ticker symbols, optionally filtered by data_source.
    source: None (all), 'preloaded', or 'on_demand'
    """
    with SessionLocal() as session:
        query = session.query(Symbol.ticker).filter(Symbol.active == True)  # noqa: E712
        if source:
            query = query.filter(Symbol.data_source == source)
        return [row.ticker for row in query.all()]
