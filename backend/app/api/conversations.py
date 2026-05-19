from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.services.conversation_service import (
    list_conversations,
    get_conversation_with_context,
    rename_conversation,
    delete_conversation,
)

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


class RenameRequest(BaseModel):
    title: str


@router.get("")
def api_list_conversations(db: Session = Depends(get_db)):
    return list_conversations(db)


@router.get("/{conversation_id}")
def api_get_conversation(conversation_id: int, db: Session = Depends(get_db)):
    try:
        return get_conversation_with_context(db, conversation_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Conversation not found")


@router.patch("/{conversation_id}")
def api_rename_conversation(conversation_id: int, body: RenameRequest, db: Session = Depends(get_db)):
    try:
        rename_conversation(db, conversation_id, body.title)
        db.commit()
        return {"ok": True}
    except ValueError:
        raise HTTPException(status_code=404, detail="Conversation not found")


@router.delete("/{conversation_id}")
def api_delete_conversation(conversation_id: int, db: Session = Depends(get_db)):
    delete_conversation(db, conversation_id)
    db.commit()
    return {"ok": True}
