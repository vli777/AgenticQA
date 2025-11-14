# backend/models.py

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    question: str
    namespace: str = "default"
    history: List[Dict[str, str]] = Field(default_factory=list)
    conversation_id: Optional[str] = None
