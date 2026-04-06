from fastapi import APIRouter
from sqlalchemy import text

from app.data.scheduler import _scheduler
from app.database import SessionLocal

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check():
    try:
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception:
        db_status = "error"

    return {
        "status": "ok",
        "db": db_status,
        "scheduler": "running" if _scheduler.running else "stopped",
    }
