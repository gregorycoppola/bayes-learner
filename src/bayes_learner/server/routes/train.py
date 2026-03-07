"""Train route."""
from fastapi import APIRouter
from pydantic import BaseModel
from bayes_learner.core.trainer import train

router = APIRouter(prefix="/api/train", tags=["train"])

class TrainRequest(BaseModel):
    n_graphs: int = 5000
    n_vars: int = 5
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3

@router.post("")
async def run_train(req: TrainRequest):
    results = train(
        n_graphs=req.n_graphs,
        n_vars=req.n_vars,
        epochs=req.epochs,
        batch_size=req.batch_size,
        lr=req.lr,
    )
    final_mae = results["val_mae"][-1]
    return {
        "status": "done",
        "epochs": req.epochs,
        "final_val_mae": final_mae,
        "curve": results,
    }