# backend/logger.py

import logging

# Configure a single “agenticqa” logger here
logger = logging.getLogger("agenticqa")
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
