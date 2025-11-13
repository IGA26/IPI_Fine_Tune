#!/usr/bin/env python3
"""
sil_api.py

FastAPI application exposing SIL inference for a simple web demo.
Run with:
    uvicorn sil_api:app --reload
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sil_inference import GibberishInputError, SILInferenceService, _parse_env_flag

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="SIL Tester",
    version="0.1.0",
    description="Minimal API for testing SIL classifications via a web frontend.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class InferRequest(BaseModel):
    text: str


class InferResponse(BaseModel):
    input_text: str
    normalized_text: str
    quality: dict
    predictions: dict
    timing: dict


class GenerateResponse(BaseModel):
    text: str


_service: Optional[SILInferenceService] = None


def get_service() -> SILInferenceService:
    global _service
    if _service is None:
        model_dir = Path(os.environ.get("SIL_MODEL_DIR", "./models"))
        use_fake_env = os.environ.get("USE_FAKE_MODELS")
        use_fake = None if use_fake_env is None else _parse_env_flag(use_fake_env)
        _service = SILInferenceService(model_dir=model_dir, use_fake_models=use_fake)
    return _service


SAMPLE_PROMPTS = [
    "How can I increase my ISA contributions this year?",
    "What happens if I miss a mortgage payment?",
    "Can you explain the difference between a SIPP and a workplace pension?",
    "I want to transfer money from my savings to pay off my loan.",
    "Do you offer any investment products with low risk?",
    "How do I check my account balance?",
]


@app.get("/", response_class=HTMLResponse)
async def index():
    if not STATIC_DIR.exists():
        raise HTTPException(status_code=404, detail="Static frontend not found.")
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html not found.")
    return index_file.read_text(encoding="utf-8")


@app.get("/healthz")
async def healthz():
    try:
        service = get_service()
        return {
            "status": "ok",
            "models_loaded": list(service.models.keys()),
            "device": service.device,
        }
    except Exception as exc:  # pragma: no cover - startup failures
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/sil/infer", response_model=InferResponse)
async def infer(request: InferRequest):
    service = get_service()
    try:
        result = service.classify(request.text)
    except GibberishInputError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": str(exc),
                "quality": {
                    "word_count": exc.quality.word_count,
                    "valid_word_ratio": exc.quality.valid_word_ratio,
                    "non_alnum_ratio": exc.quality.non_alnum_ratio,
                    "repeated_char_ratio": exc.quality.repeated_char_ratio,
                },
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result


@app.post("/sil/generate", response_model=GenerateResponse)
async def generate_example():
    return {"text": random.choice(SAMPLE_PROMPTS)}

