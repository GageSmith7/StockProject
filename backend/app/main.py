import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import create_tables
from app.data.scheduler import shutdown as shutdown_scheduler
from app.data.scheduler import start as start_scheduler
from app.models.registry import restore_from_s3
from app.routes import health, model_health, optimize, portfolio, predict, search, simulate

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    restore_from_s3()   # no-op locally; restores model artifacts after container restart
    start_scheduler()
    yield
    shutdown_scheduler()


app = FastAPI(
    title="Portfolio Simulator API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router,        prefix="/api")
app.include_router(model_health.router,  prefix="/api")
app.include_router(search.router,     prefix="/api")
app.include_router(portfolio.router,  prefix="/api")
app.include_router(simulate.router,   prefix="/api")
app.include_router(predict.router,    prefix="/api")
app.include_router(optimize.router,   prefix="/api")
