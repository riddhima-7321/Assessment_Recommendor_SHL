"""
main.py

FastAPI service for:
GET  /health
POST /chat

catalog + faiss loads once when app starts
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

from fastapi.middleware.cors import CORSMiddleware

from agent import SHLAgent
from catalog import SHLCatalog


# logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

log = logging.getLogger(__name__)


# global vars loaded once
_catalog: SHLCatalog | None = None
_agent: SHLAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):

    global _catalog, _agent

    log.info("Loading SHL catalog and building FAISS index …")

    t0 = time.time()

    _catalog = SHLCatalog()

    _agent = SHLAgent(_catalog)

    log.info(
        "Ready in %.1fs — %d assessments indexed",
        time.time() - t0,
        len(_catalog.items)
    )

    yield

    log.info("Shutting down.")


app = FastAPI(
    title="SHL Assessment Advisor API",
    description="Conversational agent for SHL assessment selection",
    version="1.0.0",
    lifespan=lifespan,
)


# cors middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# request response schemas


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


# endpoints


@app.get("/health")
async def health():

    """checks if service is ready"""

    if _agent is None:
        return JSONResponse(
            status_code=503,
            content={"status": "loading"}
        )

    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    """
    stateless chat endpoint
    full history comes every req
    """

    if _agent is None:
        raise HTTPException(
            status_code=503,
            detail="Service is still starting up"
        )

    # converting pydantic objs
    messages = [
        {
            "role": m.role,
            "content": m.content
        }
        for m in request.messages
    ]

    try:

        result = _agent.chat(messages)

    except Exception as e:

        log.exception("Agent error: %s", e)

        raise HTTPException(
            status_code=500,
            detail="Internal agent error"
        ) from e

    return ChatResponse(
        reply=result["reply"],
        recommendations=[
            Recommendation(**r)
            for r in result["recommendations"]
        ],
        end_of_conversation=result["end_of_conversation"],
    )


# error handlers


@app.exception_handler(Exception)
async def generic_error_handler(
    request: Request,
    exc: Exception
):

    log.exception(
        "Unhandled error on %s: %s",
        request.url.path,
        exc
    )

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )