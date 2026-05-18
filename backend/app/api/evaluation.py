from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.evaluation import Evaluation as EvaluationModel
from app.services.evaluation_service import run_evaluation

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


class EvalRequest(BaseModel):
    queries: list[str]
    has_docs: bool = False


@router.post("/run")
async def run_eval(request: EvalRequest, db: Session = Depends(get_db)):
    results = run_evaluation(request.queries, request.has_docs)

    for r in results:
        record = EvaluationModel(
            query=r["query"],
            answer=r["answer"],
            context=r["context"],
            faithfulness=r["faithfulness"],
            answer_relevancy=r["answer_relevancy"],
        )
        db.add(record)
    db.commit()

    return {"results": results}


@router.get("/history")
async def get_eval_history(db: Session = Depends(get_db)):
    records = db.query(EvaluationModel).order_by(EvaluationModel.created_at.desc()).limit(50).all()
    return [
        {
            "id": r.id,
            "query": r.query,
            "answer": r.answer[:200],
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "created_at": r.created_at.isoformat(),
        }
        for r in records
    ]
