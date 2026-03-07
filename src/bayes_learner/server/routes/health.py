"""Health check route."""
from fastapi import APIRouter

router = APIRouter(prefix="/api/health", tags=["health"])

@router.get("")
async def health():
    return {
        "status": "ok",
        "service": "bayes-learner",
        "version": "0.1.0",
    }