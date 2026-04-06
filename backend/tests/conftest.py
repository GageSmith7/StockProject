"""
Shared pytest fixtures.

Database isolation strategy:
  - Unit tests use `portfolio_sim_test` — isolated, safe to TRUNCATE freely.
    The dev DB (portfolio_sim) is never touched or wiped by unit tests.
  - Integration tests use `portfolio_sim` (dev DB) via the `db_seeded` fixture,
    which temporarily patches all app SessionLocal imports to point to the dev DB.

To seed the dev DB (one-time / after Docker reset):
    python -c "from app.data.symbols import seed_symbols, seed_prices; seed_symbols(); seed_prices()"
"""
import os
from unittest.mock import patch

from dotenv import load_dotenv

load_dotenv()

# Redirect DATABASE_URL to the isolated test database BEFORE any app modules
# are imported.  Every `from app.database import SessionLocal` in app code will
# therefore bind to portfolio_sim_test, leaving the dev DB untouched.
_DEV_DB_URL  = os.environ.get("DATABASE_URL",      "postgresql://user:password@localhost:5433/portfolio_sim")
_TEST_DB_URL = os.environ.get("TEST_DATABASE_URL",  "postgresql://user:password@localhost:5433/portfolio_sim_test")
os.environ["DATABASE_URL"] = _TEST_DB_URL

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.database import Base, engine, SessionLocal   # bound to portfolio_sim_test


# ---------------------------------------------------------------------------
# Session-scoped: create tables in the test DB once per run
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def create_tables():
    """Create all tables once in portfolio_sim_test for the test session."""
    Base.metadata.create_all(engine)
    yield
    # Leave tables intact — easier to inspect failures


# ---------------------------------------------------------------------------
# Unit test isolation
# ---------------------------------------------------------------------------

@pytest.fixture
def db_clean():
    """
    Truncate all data tables in the TEST database before a test.

    Apply to unit / store / symbols tests that need a blank slate.
    The dev database (portfolio_sim) is completely unaffected.
    """
    with SessionLocal() as session:
        session.execute(text("TRUNCATE prices, macro_data, symbols, constituent_history RESTART IDENTITY CASCADE"))
        session.commit()
    yield


# ---------------------------------------------------------------------------
# Integration test support
# ---------------------------------------------------------------------------

@pytest.fixture
def db_seeded():
    """
    Redirect all app DB calls to the dev database (portfolio_sim) for this test.
    Skips the test automatically if the dev DB prices table is empty.

    Works by patching the `SessionLocal` name in every app module that imported
    it directly, for the duration of the test only.  Unit test isolation is
    unaffected — the test DB is left as-is.
    """
    dev_engine = create_engine(_DEV_DB_URL)
    DevSession  = sessionmaker(bind=dev_engine, autocommit=False, autoflush=False)

    with DevSession() as s:
        count = s.execute(text("SELECT COUNT(*) FROM prices")).scalar()

    if count == 0:
        dev_engine.dispose()
        pytest.skip("dev DB prices table is empty — run seed_prices() first")

    # Patch every module that did `from app.database import SessionLocal`
    _patches = [
        patch("app.database.SessionLocal",           DevSession),
        patch("app.data.store.SessionLocal",          DevSession),
        patch("app.data.symbols.SessionLocal",        DevSession),
        patch("app.analytics.features.SessionLocal",  DevSession),
    ]
    for p in _patches:
        p.start()

    yield

    for p in _patches:
        p.stop()
    dev_engine.dispose()
