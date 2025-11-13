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
    from vertexai.preview.generative_models import GenerativeModel, SafetySetting, HarmCategory, HarmBlockThreshold
except ImportError:  # pragma: no cover - optional dependency
    vertexai = None  # type: ignore
    GenerativeModel = None  # type: ignore
    SafetySetting = None  # type: ignore
    HarmCategory = None  # type: ignore
    HarmBlockThreshold = None  # type: ignore

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

# Static files will be mounted after all API routes are defined


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
Create a short UK banking customer question for testing.

1. Pick one topic: {", ".join(SIL_TOPICS)}
2. Write a simple question (5-15 words) about that topic.
3. Return JSON only: {{"topic": "topic_name", "text": "the question"}}

Example: {{"topic": "savings", "text": "How do I open an ISA?"}}
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
        all_models = list(service.sil_models.keys()) + list(service.emotion_models.keys())
        return {
            "status": "ok",
            "models_loaded": all_models,
            "sil_models": list(service.sil_models.keys()),
            "emotion_models": list(service.emotion_models.keys()),
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
    print(f"🔍 Generate request received. VERTEX_ENABLED: {VERTEX_ENABLED}", flush=True)
    
    if VERTEX_ENABLED:
        try:
            print(f"📞 Calling Vertex AI (model: {VERTEX_MODEL}, project: {VERTEX_PROJECT})", flush=True)
            model = _get_vertex_model()
            
            # Configure safety settings to allow financial content
            # Use SafetySetting objects (not dictionaries)
            if SafetySetting and HarmCategory and HarmBlockThreshold:
                safety_settings = [
                    SafetySetting(
                        category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    ),
                    SafetySetting(
                        category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    ),
                    SafetySetting(
                        category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    ),
                    SafetySetting(
                        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    ),
                ]
            else:
                # Fallback: no safety settings if imports failed
                safety_settings = None
            
            print(f"📤 Sending request to Vertex AI...", flush=True)
            # Build generate_content arguments
            generate_kwargs = {
                "generation_config": {
                    "temperature": float(os.environ.get("VERTEX_TEMPERATURE", "0.8")),
                    "max_output_tokens": int(os.environ.get("VERTEX_MAX_TOKENS", "128")),  # Increased from 64
                },
            }
            # Only add safety_settings if we have them
            if safety_settings is not None:
                generate_kwargs["safety_settings"] = safety_settings
            
            response = model.generate_content(
                GENERATE_PROMPT,
                **generate_kwargs,
            )
            print(f"✅ Vertex AI response received", flush=True)
            
            # Check if response has text content
            if not hasattr(response, 'text') or not response.text:
                # Check finish_reason to understand why
                finish_reason = None
                if hasattr(response, 'candidates') and response.candidates:
                    finish_reason = getattr(response.candidates[0], 'finish_reason', None)
                
                if finish_reason == "SAFETY":
                    raise ValueError("Response blocked by safety filters. Try a different prompt.")
                elif finish_reason == "MAX_TOKENS":
                    raise ValueError("Response hit token limit. Increase max_output_tokens.")
                else:
                    raise ValueError(f"Response has no text content. Finish reason: {finish_reason}")
            
            raw_text = response.text.strip()
            
            # Debug: log what Vertex returned
            print(f"🔍 Vertex raw response: {raw_text[:200]}...", flush=True)
            
            # Try to clean and parse JSON
            payload = None
            
            # Step 1: Remove markdown code blocks if present
            if raw_text.startswith("```"):
                # Extract content between ```json and ```
                lines = raw_text.split("\n")
                json_lines = []
                in_code_block = False
                for line in lines:
                    if line.strip().startswith("```"):
                        in_code_block = not in_code_block
                        continue
                    if in_code_block:
                        json_lines.append(line)
                if json_lines:
                    raw_text = "\n".join(json_lines).strip()
            
            # Step 2: Try direct JSON parse
            try:
                payload = json.loads(raw_text)
            except json.JSONDecodeError:
                # Step 3: Extract JSON between braces
                start = raw_text.find("{")
                end = raw_text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    json_str = raw_text[start : end + 1]
                    try:
                        payload = json.loads(json_str)
                    except json.JSONDecodeError as e:
                        print(f"⚠️  JSON extraction failed: {e}", flush=True)
                        print(f"   Extracted JSON string: {json_str[:200]}...", flush=True)
                        raise
                else:
                    raise ValueError(f"No JSON object found in response: {raw_text[:200]}")

            topic = str(payload.get("topic", "")).strip().lower()
            text = str(payload.get("text", "")).strip()

            if topic not in SIL_TOPICS or not text:
                raise ValueError("Invalid topic or empty text from Vertex response")

            print(f"✅ Successfully generated query from Vertex AI: {text[:50]}...", flush=True)
            return {"text": text, "topic": topic}
        except Exception as exc:
            # Fall back to static prompts if Vertex generation fails
            print(f"⚠️  Vertex generation failed: {exc}", flush=True)
            import traceback
            print(f"   Traceback: {traceback.format_exc()}", flush=True)
    else:
        print(f"⚠️  Vertex AI is disabled. Using fallback prompts.", flush=True)
        print(f"   VERTEX_PROJECT: {VERTEX_PROJECT}, vertexai module: {vertexai is not None}, GenerativeModel: {GenerativeModel is not None}", flush=True)

    fallback_topic = random.choice(SIL_TOPICS)
    fallback_text = random.choice(SAMPLE_PROMPTS)
    print(f"📝 Using fallback prompt: {fallback_text[:50]}...", flush=True)
    return {"text": fallback_text, "topic": fallback_topic}


# Mount static files last, after all API routes are defined
# This ensures API routes take precedence over static file serving
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

