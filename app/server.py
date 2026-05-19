"""
Legal RAG — FastAPI Server
==========================
Endpoints:
    POST /chat          → one conversation turn
    POST /ingest        → load documents into the vector store
    GET  /health        → health check
    GET  /config        → current engine config
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
# Find .env relative to this file, not the working directory
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from itertools import groupby
from langchain_core.documents import Document

from fastapi import Request
from fastapi.responses import JSONResponse



from rag_engine import LegalRAGEngine


#___helpers ______________________________________________________
import re


def is_arabic(text: str) -> bool:
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    return arabic_chars > 10  # more than 10 arabic chars = arabic response
def fix_bidi(text: str) -> str:
    if not is_arabic(text):
        return text  # French/Latin response → return as-is

    LRI = "\u2066"  # Left-to-Right Isolate
    PDI = "\u2069"  # Pop Directional Isolate

    return re.sub(
        r'([A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9\s\-\.]*)',
        lambda m: f"{LRI}{m.group(1)}{PDI}",
        text
    )




# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Legal RAG API",
    description="Moroccan Legal Assistant ",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singleton engine (initialised on startup from env vars) ──────────────────
engine: LegalRAGEngine | None = None



@app.on_event("startup")
def startup():
    global engine
    engine = LegalRAGEngine(
        provider        = os.getenv("LLM_PROVIDER", "gemini"),
        model           = os.getenv("LLM_MODEL", "gemini-1.5-flash"),
        temperature     = float(os.getenv("LLM_TEMPERATURE", "0.1")),
        api_key         = None,  # each provider reads its own env var (GOOGLE_API_KEY / OPENAI_API_KEY)
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        vector_store_dir= os.getenv("VECTOR_STORE_DIR", "./data/vector_stores"),
        k_results       = int(os.getenv("K_RESULTS", "4")),
    )
    print("[API] Engine ready.")


# ── Schemas ───────────────────────────────────────────────────────────────────
class ContextMessage(BaseModel):
    role: str = Field(..., examples=["user", "assistant"])
    content: str

class ChatRequest(BaseModel):
    user_prompt: str
    system_prompt: str | None = None
    context: list[ContextMessage] = Field(default_factory=list)

class ChatResponse(BaseModel):
    response_to: str
    response: str
    query: str
    context: list[ContextMessage]

class IngestRequest(BaseModel):
    source_dir: str = Field(..., examples=["./data/tmp"])
    chunk_size: int = 400    # short chunks suit legal articles well
    chunk_overlap: int = 50

class IngestResponse(BaseModel):
    status: str
    chunks_created: int

class EngineConfig(BaseModel):
    provider: str
    model: str
    temperature: float
    vector_store_dir: str
    k_results: int
    vector_store_loaded: bool


# ── Routes ────────────────────────────────────────────────────────────────────



@app.options("/{rest_of_path:path}")
async def preflight_handler(rest_of_path: str, request: Request):
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )






@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/config", response_model=EngineConfig)
def get_config():
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised")
    return EngineConfig(
        provider          = engine.provider,
        model             = engine.model,
        temperature       = engine.temperature,
        vector_store_dir  = str(engine.vector_store_dir),
        k_results         = engine.k_results,
        vector_store_loaded = engine.vectorstore is not None,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised")

    payload = {
        "user_prompt":   req.user_prompt,
        "context":       [m.model_dump() for m in req.context],
    }


    result = engine.chat(payload)
    return ChatResponse(
        response_to = result["response_to"],
        response    = fix_bidi(result["response"]),
        query       = result["query"],
        context     = [ContextMessage(**m) for m in result["context"]],
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised")
    try:
        n = engine.ingest_documents(req.source_dir, req.chunk_size, req.chunk_overlap)
        return IngestResponse(status="success", chunks_created=n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Dev entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)