import os

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    create_engine,
    func,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://user:password@localhost:5433/portfolio_sim",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


class Price(Base):
    __tablename__ = "prices"

    date = Column(Date, primary_key=True)
    ticker = Column(String(10), primary_key=True)
    open = Column(Numeric(12, 4))
    high = Column(Numeric(12, 4))
    low = Column(Numeric(12, 4))
    close = Column(Numeric(12, 4), nullable=False)
    volume = Column(BigInteger)
    quality_flag = Column(String(20))
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class MacroData(Base):
    __tablename__ = "macro_data"

    date = Column(Date, primary_key=True)
    series_id = Column(String(20), primary_key=True)
    value = Column(Numeric(12, 4))


class Symbol(Base):
    __tablename__ = "symbols"

    ticker = Column(String(10), primary_key=True)
    name = Column(String(200))
    sector = Column(String(100))
    active = Column(Boolean, default=True)
    data_source = Column(String(20), default="preloaded")
    last_fetched = Column(DateTime)
    added_at = Column(DateTime, server_default=func.now())


class PredictionLog(Base):
    """
    Stores daily model predictions per ticker so we can compute live IC
    21 trading days later by comparing predicted vs actual returns.
    One row per (as_of_date, ticker) — upserted daily after the price refresh.
    """
    __tablename__ = "prediction_log"

    id                   = Column(Integer, primary_key=True, autoincrement=True)
    as_of_date           = Column(Date,        nullable=False)
    ticker               = Column(String(10),  nullable=False)
    predicted_return_21d = Column(Numeric(10, 6), nullable=False)
    model_version        = Column(String(50))

    __table_args__ = (
        UniqueConstraint("as_of_date", "ticker", name="uq_pred_log_date_ticker"),
        Index("ix_pred_log_date", "as_of_date"),
    )


class ModelMetrics(Base):
    """
    Stores the result of each weekly IC health check.
    One row per check — append-only audit trail.
    """
    __tablename__ = "model_metrics"

    id                   = Column(Integer,     primary_key=True, autoincrement=True)
    computed_at          = Column(DateTime,    nullable=False)
    rolling_ic           = Column(Numeric(8, 6))
    n_samples            = Column(Integer)
    # HEALTHY | DEGRADED | CIRCUIT_BREAKER
    gate_status          = Column(String(20),  nullable=False)
    consecutive_failures = Column(Integer,     default=0)
    retrain_triggered    = Column(Boolean,     default=False)
    # DEPLOYED | REJECTED | null
    retrain_result       = Column(String(20))
    new_model_ic         = Column(Numeric(8, 6))


class ConstituentHistory(Base):
    """
    Stores the most recent S&P 500 addition date for each current constituent.
    Used for point-in-time filtering during ML training to eliminate survivorship bias:
    a ticker's price history is only used from the date it actually joined the index.

    date_added = NULL means the ticker has been in the index since before our recorded
    history (~2000) — treat it as present for the full training window.
    """
    __tablename__ = "constituent_history"

    ticker     = Column(String(10), primary_key=True)
    date_added = Column(Date,       nullable=True)  # NULL = pre-history, use all data
    updated_at = Column(DateTime,   server_default=func.now(), onupdate=func.now())


def create_tables() -> None:
    """Create all tables if they don't exist. Safe to call on every startup."""
    Base.metadata.create_all(engine)
