"""
main.py
───────
FastAPI service exposing:
  GET  /health  →  {"status": "ok"}
  POST /chat    →  agent reply + recommendations

The service is stateless: every /chat call carries the full conversation history.
The catalog and FAISS index are loaded once at startup.
"""

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from agent import SHLAgent
from catalog import SHLCatalog

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

# ── Global singletons (loaded once at startup) ────────────────────────────────
_catalog: SHLCatalog | None = None
_agent: SHLAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _catalog, _agent
    log.info("Loading SHL catalog and building FAISS index …")
    t0 = time.time()
    _catalog = SHLCatalog()
    _agent = SHLAgent(_catalog)
    log.info("Ready in %.1fs — %d assessments indexed", time.time() - t0, len(_catalog.items))
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="SHL Assessment Advisor API",
    description="Conversational agent for SHL assessment selection",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Request / Response schemas ────────────────────────────────────────────────


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Message content cannot be empty")
        return v.strip()


class ChatRequest(BaseModel):
    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def messages_not_empty(cls, v: list) -> list:
        if not v:
            raise ValueError("messages list cannot be empty")
        if len(v) > 20:
            raise ValueError("Too many messages (max 20 per call)")
        return v


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    """Readiness probe. Returns 200 once the catalog is loaded."""
    if _agent is None:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Stateless chat endpoint. Send the full conversation history on every call.

    The agent will:
    - Ask for clarification if context is too vague
    - Return 1–10 grounded recommendations when ready
    - Refine recommendations when constraints change
    - Compare assessments when asked
    - Refuse off-topic requests
    """
    if _agent is None:
        raise HTTPException(status_code=503, detail="Service is still starting up")

    # Convert pydantic models to plain dicts for the agent
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    try:
        result = _agent.chat(messages)
    except Exception as e:
        log.exception("Agent error: %s", e)
        raise HTTPException(status_code=500, detail="Internal agent error") from e

    return ChatResponse(
        reply=result["reply"],
        recommendations=[Recommendation(**r) for r in result["recommendations"]],
        end_of_conversation=result["end_of_conversation"],
    )


# ── Error handlers ────────────────────────────────────────────────────────────


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    log.exception("Unhandled error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )