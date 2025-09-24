# backend/models.py

from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str
    namespace: str = "default"
