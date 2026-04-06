from fastapi import APIRouter, Query

from app.data.symbols import search_symbols

router = APIRouter(tags=["search"])


@router.get("/search")
def search(
    q: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50),
):
    return {"results": search_symbols(q, limit=limit)}
