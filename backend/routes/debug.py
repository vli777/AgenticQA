# backend/routes/debug.py

import json
import re
from fastapi import APIRouter

from config import ENABLE_CACHING
from pinecone_client import pc, index
from logger import logger

if ENABLE_CACHING:
    from cache import clear_all_caches
else:
    clear_all_caches = None

router = APIRouter()

@router.get("/indexes")
def list_indexes():
    indexes = pc.list_indexes()
    cleaned = []
    for i in indexes:
        s = str(i)
        s = s.replace("'", '"')
        s = re.sub(r'\n\s*', '', s)
        s = s.replace('True', 'true').replace('False', 'false').replace('None', 'null')
        try:
            obj = json.loads(s)
            cleaned.append({
                "name": obj.get("name"),
                "dimension": obj.get("dimension"),
                "metric": obj.get("metric"),
                "status": obj.get("status"),
                "model": obj.get("embed", {}).get("model"),
                "cloud": obj.get("spec", {}).get("serverless", {}).get("cloud"),
                "region": obj.get("spec", {}).get("serverless", {}).get("region"),
            })
        except Exception as e:
            cleaned.append({"raw": s, "error": str(e)})
    return {"indexes": cleaned}


@router.delete("/namespace/{namespace}")
async def clear_namespace(namespace: str):
    """Delete all vectors within a Pinecone namespace and reset caches."""
    logger.info("Clearing namespace '%s' via debug endpoint", namespace)

    vectors_deleted = True
    delete_warning = None
    try:
        index.delete(delete_all=True, namespace=namespace)
    except Exception as exc:
        vectors_deleted = False
        delete_warning = str(exc)
        logger.warning(
            "Pinecone delete failed for namespace '%s' (treating as already empty): %s",
            namespace,
            exc,
        )

    caches_cleared = False
    if clear_all_caches is not None:
        try:
            await clear_all_caches()
            caches_cleared = True
        except Exception as exc:
            logger.warning("Failed to clear caches after namespace delete: %s", exc)

    payload = {
        "namespace": namespace,
        "status": "cleared",
        "caches_cleared": caches_cleared,
        "vectors_deleted": vectors_deleted,
    }
    if delete_warning:
        payload["warning"] = delete_warning
    return payload
