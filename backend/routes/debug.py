# backend/routes/debug.py

import json
import re
from fastapi import APIRouter
from pinecone_client import pc, index

router = APIRouter()


@router.get("/indexes")
def list_indexes():
    indexes = pc.list_indexes()
    cleaned = []
    for i in indexes:
        s = str(i)
        s = s.replace("'", '"')
        s = re.sub(r"\n\s*", "", s)
        s = s.replace("True", "true").replace("False", "false").replace("None", "null")
        try:
            obj = json.loads(s)
            cleaned.append(
                {
                    "name": obj.get("name"),
                    "dimension": obj.get("dimension"),
                    "metric": obj.get("metric"),
                    "status": obj.get("status"),
                    "model": obj.get("embed", {}).get("model"),
                    "cloud": obj.get("spec", {}).get("serverless", {}).get("cloud"),
                    "region": obj.get("spec", {}).get("serverless", {}).get("region"),
                }
            )
        except Exception as e:
            cleaned.append({"raw": s, "error": str(e)})
    return {"indexes": cleaned}


@router.delete("/namespace/{namespace}")
def clear_namespace(namespace: str):
    """Delete all vectors within a Pinecone namespace."""
    index.delete(delete_all=True, namespace=namespace)
    return {"namespace": namespace, "status": "cleared"}
