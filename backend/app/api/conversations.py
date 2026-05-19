from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.services.conversation_service import (
    list_conversations,
    get_conversation_with_context,
    delete_conversation,
)

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.get("")
def api_list_conversations(db: Session = Depends(get_db)):
    return list_conversations(db)


@router.get("/{conversation_id}")
def api_get_conversation(conversation_id: int, db: Session = Depends(get_db)):
    return get_conversation_with_context(db, conversation_id)


@router.delete("/{conversation_id}")
def api_delete_conversation(conversation_id: int, db: Session = Depends(get_db)):
    delete_conversation(db, conversation_id)
    db.commit()
    return {"ok": True}
