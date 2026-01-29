# backend/agent/__init__.py

"""QA agent: orchestrates tools, planning, composition, and verification."""

from .orchestrator import get_agent, get_streaming_agent

__all__ = ["get_agent", "get_streaming_agent"]
