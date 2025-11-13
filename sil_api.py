#!/usr/bin/env python3
"""
sil_api.py

FastAPI application exposing SIL inference for a simple web demo.
Run with:
    uvicorn sil_api:app --reload
"""

from __future__ import annotations

import json
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

try:
    import vertexai
    from vertexai.preview.generative_models import GenerativeModel
except ImportError:  # pragma: no cover - optional dependency
    vertexai = None  # type: ignore
    GenerativeModel = None  # type: ignore

STATIC_DIR = Path(__file__).parent / "static"

DEFAULT_VERTEX_PROJECT = "playpen-c84caa"
DEFAULT_VERTEX_LOCATION = "us-central1"
DEFAULT_VERTEX_MODEL = "gemini-2.5-flash"

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
    topic: Optional[str] = None


_service: Optional[SILInferenceService] = None


def get_service() -> SILInferenceService:
    global _service
    if _service is None:
        model_dir = Path(os.environ.get("SIL_MODEL_DIR", "./models"))
        use_fake_env = os.environ.get("USE_FAKE_MODELS")
        use_fake = None if use_fake_env is None else _parse_env_flag(use_fake_env)
        _service = SILInferenceService(model_dir=model_dir, use_fake_models=use_fake)
    return _service


SIL_TOPICS = [
    "savings",
    "investments",
    "pensions",
    "mortgages",
    "banking",
    "loans",
    "debt",
    "insurance",
    "taxation",
    "general",
    "off_topic",
]

SAMPLE_PROMPTS = [
    "How can I increase my ISA contributions this year?",
    "What happens if I miss a mortgage payment?",
    "Can you explain the difference between a SIPP and a workplace pension?",
    "I want to transfer money from my savings to pay off my loan.",
    "Do you offer any investment products with low risk?",
    "How do I check my account balance?",
    "What is the best way to care for tomato plants on a balcony?",
]

VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", DEFAULT_VERTEX_PROJECT)
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", DEFAULT_VERTEX_LOCATION)
VERTEX_MODEL = os.environ.get("VERTEX_MODEL", DEFAULT_VERTEX_MODEL)
VERTEX_ENABLED = bool(VERTEX_PROJECT and vertexai and GenerativeModel)

_vertex_model: Optional[GenerativeModel] = None


def _get_vertex_model() -> GenerativeModel:
    global _vertex_model
    if _vertex_model is None:
        vertexai.init(project=VERTEX_PROJECT, location=VERTEX_LOCATION)
        _vertex_model = GenerativeModel(VERTEX_MODEL)
    return _vertex_model


GENERATE_PROMPT = f"""
You are creating short UK retail-finance utterances for testing a classification system.

Instructions:
1. Choose exactly one topic at random from this list:
   {", ".join(SIL_TOPICS)}
2. Write a concise first-person message (5-20 words) that matches the chosen topic.
   • If the topic is "off_topic", ensure the message is clearly unrelated to money or finance.
   • For financial topics, keep the language conversational and realistic (UK context).
3. Respond strictly as JSON on a single line using this schema:
   {{"topic": "<chosen_topic>", "text": "<user_message>"}}

Do not include any additional commentary.
""".strip()


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
    if VERTEX_ENABLED:
        try:
            model = _get_vertex_model()
            response = model.generate_content(
                GENERATE_PROMPT,
                generation_config={
                    "temperature": float(os.environ.get("VERTEX_TEMPERATURE", "0.8")),
                    "max_output_tokens": int(os.environ.get("VERTEX_MAX_TOKENS", "64")),
                },
            )
            raw_text = response.text.strip()
            try:
                payload = json.loads(raw_text)
            except json.JSONDecodeError:
                # Attempt to extract JSON between braces if model added formatting
                start = raw_text.find("{")
                end = raw_text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    payload = json.loads(raw_text[start : end + 1])
                else:
                    raise

            topic = str(payload.get("topic", "")).strip().lower()
            text = str(payload.get("text", "")).strip()

            if topic not in SIL_TOPICS or not text:
                raise ValueError("Invalid topic or empty text from Vertex response")

            return {"text": text, "topic": topic}
        except Exception as exc:
            # Fall back to static prompts if Vertex generation fails
            print(f"⚠️  Vertex generation failed: {exc}", flush=True)

    fallback_topic = random.choice(SIL_TOPICS)
    fallback_text = random.choice(SAMPLE_PROMPTS)
    return {"text": fallback_text, "topic": fallback_topic}

