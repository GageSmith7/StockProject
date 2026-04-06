import logging
from datetime import datetime, timedelta

from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_MISSED
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app.data.cleaner import clean
from app.data.fetcher import FRED_SERIES_MVP, fetch_fred_series, fetch_ticker
from app.data.store import upsert_macro, upsert_prices
from app.data.symbols import get_all_tickers, run_initial_seed
from app.database import MacroData, SessionLocal, Symbol
from app.models.health import log_daily_predictions, run_weekly_ic_check

logger = logging.getLogger(__name__)

# Days of calendar history to fetch on each refresh.
# 10 covers weekends + a single-day holiday buffer.
REFRESH_WINDOW_DAYS = 10

# Days of FRED history to refresh (some series are monthly).
FRED_REFRESH_WINDOW_DAYS = 30

_scheduler = BackgroundScheduler(timezone="America/New_York")


# ---------------------------------------------------------------------------
# Refresh job
# ---------------------------------------------------------------------------

def _on_job_event(event) -> None:
    """Log job errors and missed firings so they're visible in docker logs."""
    if event.exception:
        logger.error("scheduler: job %s raised an exception: %s", event.job_id, event.exception)
    else:
        logger.warning("scheduler: job %s was missed (fired too late)", event.job_id)


def _daily_refresh() -> None:
    """
    Fetch recent price data for all Tier 1 tickers + FRED macro series.
    Runs Mon–Fri at 18:00 ET. Safe to trigger manually via trigger_refresh().
    One failed ticker does not stop the rest.
    """
    today = datetime.utcnow().date()
    price_start = (today - timedelta(days=REFRESH_WINDOW_DAYS)).isoformat()
    fred_start = (today - timedelta(days=FRED_REFRESH_WINDOW_DAYS)).isoformat()
    end = today.isoformat()

    logger.info("daily_refresh: starting — window %s → %s", price_start, end)

    # All active tickers — preloaded S&P 500/ETFs + any on_demand tickers users have added
    tickers = get_all_tickers()
    fetched_tickers: list[str] = []
    fetched = skipped = failed = 0

    for ticker in tickers:
        try:
            raw = fetch_ticker(ticker, price_start, end)
            if raw.empty:
                skipped += 1
                continue
            upsert_prices(clean(raw))
            fetched_tickers.append(ticker)
            fetched += 1
        except Exception as e:
            logger.warning("daily_refresh: failed for %s: %s", ticker, e)
            failed += 1

    # Stamp last_fetched on every successfully refreshed symbol
    if fetched_tickers:
        now = datetime.utcnow()
        with SessionLocal() as session:
            session.query(Symbol).filter(
                Symbol.ticker.in_(fetched_tickers)
            ).update({"last_fetched": now}, synchronize_session=False)
            session.commit()

    logger.info(
        "daily_refresh: prices complete — fetched=%d skipped=%d failed=%d",
        fetched, skipped, failed,
    )

    # --- FRED macro series ---
    macro_updated = 0
    for series_id in FRED_SERIES_MVP:
        try:
            df = fetch_fred_series(series_id, fred_start, end)
            if not df.empty:
                df["series_id"] = series_id
                upsert_macro(df)
                macro_updated += 1
        except Exception as e:
            logger.warning("daily_refresh: FRED %s failed: %s", series_id, e)

    logger.info("daily_refresh: macro complete — updated=%d series", macro_updated)

    # Log today's predictions after fresh prices are in — needed for rolling IC tracking
    logged = log_daily_predictions()
    logger.info("daily_refresh: prediction log — %d predictions stored", logged)


# ---------------------------------------------------------------------------
# Startup seed
# ---------------------------------------------------------------------------

def _seed_macro() -> None:
    """
    Backfill FRED macro series from 2014-01-01 to today if not already present.
    2014 start gives a full year of CPI history before the 2015 model training window,
    which is needed to compute the first CPI YoY value.
    Idempotent — checks earliest date in DB before fetching.
    """
    BACKFILL_START = "2014-01-01"
    today = datetime.utcnow().date().isoformat()

    with SessionLocal() as session:
        earliest = session.query(MacroData.date).order_by(MacroData.date.asc()).first()

    if earliest and str(earliest[0]) <= BACKFILL_START:
        logger.info("_seed_macro: macro_data already backfilled to %s — skipping", earliest[0])
        return

    logger.info("_seed_macro: backfilling FRED series from %s → %s", BACKFILL_START, today)
    for series_id in FRED_SERIES_MVP:
        try:
            df = fetch_fred_series(series_id, BACKFILL_START, today)
            if not df.empty:
                df["series_id"] = series_id
                upsert_macro(df)
                logger.info("_seed_macro: %s — %d rows", series_id, len(df))
        except Exception as e:
            logger.warning("_seed_macro: FRED %s failed: %s", series_id, e)


def _startup_seed() -> None:
    """
    Run the initial symbol registry + full price history seed on startup.
    Runs in the scheduler's background thread so it never blocks the API.
    Idempotent — skips symbols and prices already in the DB, so it completes
    in seconds on subsequent startups once the DB is populated.
    """
    logger.info("startup_seed: running initial seed (skips already-populated data)")
    run_initial_seed()
    _seed_macro()
    logger.info("startup_seed: done")


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def start() -> None:
    """
    Start the background scheduler.
    Called from main.py lifespan on startup.
    """
    # One-time startup seed — fires 10 seconds after app start to let the
    # server finish initializing first. Idempotent, safe on every restart.
    _scheduler.add_job(
        _startup_seed,
        "date",
        run_date=datetime.now() + timedelta(seconds=10),
        id="startup_seed",
        replace_existing=True,
    )

    _scheduler.add_job(
        _daily_refresh,
        CronTrigger(
            day_of_week="mon-fri",
            hour=18,
            minute=0,
            timezone="America/New_York",
        ),
        id="daily_refresh",
        replace_existing=True,
        max_instances=1,  # skip if previous run is still in progress
        misfire_grace_time=86400,  # fire up to 24 hours late (covers weekend restarts)
    )

    # Weekly IC health check — Sunday midnight ET.
    # Computes rolling Spearman IC, gates retrain if degraded.
    _scheduler.add_job(
        run_weekly_ic_check,
        CronTrigger(
            day_of_week="sun",
            hour=0,
            minute=0,
            timezone="America/New_York",
        ),
        id="weekly_ic_check",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=86400,
    )

    _scheduler.add_listener(_on_job_event, EVENT_JOB_ERROR | EVENT_JOB_MISSED)
    _scheduler.start()
    logger.info(
        "Scheduler started — startup seed in 10s, "
        "daily refresh at 18:00 ET Mon–Fri, "
        "IC check Sundays at 00:00 ET"
    )


def shutdown() -> None:
    """
    Graceful shutdown. Called from main.py lifespan on teardown.
    wait=False so the app doesn't hang waiting for a running job.
    """
    if _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")


def trigger_refresh() -> None:
    """
    Run the daily refresh immediately (manual trigger / admin use).
    Runs in the calling thread — blocks until complete.
    """
    logger.info("trigger_refresh: manual run started")
    _daily_refresh()
    logger.info("trigger_refresh: done")
