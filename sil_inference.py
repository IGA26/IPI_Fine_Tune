#!/usr/bin/env python3
"""
sil_inference.py

Shared SIL inference utilities for both CLI scripts and the FastAPI demo service.
Supports real fine-tuned checkpoints or the fake transformers used for local testing.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from text_normalizer import normalize_spelling
from text_quality import TextQualityChecker, TextQualityResult

# Label mappings (mirrors test_sil_model.py)
TOPIC_LABELS = [
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

INTENT_TYPE_LABELS = [
    "fact_seeking",
    "advice_seeking",
    "account_action",
    "goal_expression",
    "guidance",
    "off_topic",
]

QUERY_TYPE_LABELS = [
    "what_is",
    "eligibility",
    "recommendation",
    "account_action",
    "goal_expression",
    "comparison",
]

STAGE_LABELS = [
    "goal_setup",
    "accumulation",
    "understanding",
    "optimisation",
    "withdrawal",
    "goal_definition",
    "execution",
    "enrolment",
    "decumulation",
    "application",
    "repayment",
    "remortgage",
    "awareness",
    "action",
    "planning",
    "management",
    "claim",
]

DOMAIN_SCOPE_LABELS = ["general", "bank_specific"]

ADVICE_RISK_LABELS = ["low", "medium", "high"]

LABEL_MAPPINGS: Dict[str, List[str]] = {
    "topic": TOPIC_LABELS,
    "intent": INTENT_TYPE_LABELS,
    "query": QUERY_TYPE_LABELS,
    "stage": STAGE_LABELS,
    "domain": DOMAIN_SCOPE_LABELS,
    "advice_risk": ADVICE_RISK_LABELS,
}

EMOTION_LABELS = ["positive", "neutral", "negative"]
BINARY_LABELS = ["false", "true"]

EMOTION_LABEL_MAPPINGS: Dict[str, List[str]] = {
    "emotion": EMOTION_LABELS,
    "distress": BINARY_LABELS,
    "vulnerability": BINARY_LABELS,
    "handover": BINARY_LABELS,
}

EMOTION_MODEL_ORDER = ["emotion", "handover", "distress", "vulnerability", "severity"]

PARAMETER_ORDER = [
    "topic",
    "stage",
    "intent",
    "query",
    "advice_risk",
    "domain",
    "emotion",
    "handover",
    "distress",
    "vulnerability",
    "severity",
]


def _parse_env_flag(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class GibberishInputError(ValueError):
    """Raised when input fails basic quality checks."""

    def __init__(self, message: str, quality: TextQualityResult):
        super().__init__(message)
        self.quality = quality


@dataclass
class PredictionResult:
    predicted_label: str
    confidence: float
    top3: List[Dict[str, float]]
    inference_time_ms: float


class SILInferenceService:
    def __init__(
        self,
        model_dir: Path,
        use_fake_models: Optional[bool] = None,
        text_quality_checker: Optional[TextQualityChecker] = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.use_fake_models = (
            _parse_env_flag(os.environ.get("USE_FAKE_MODELS"))
            if use_fake_models is None
            else use_fake_models
        )
        self.text_quality_checker = text_quality_checker or TextQualityChecker()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sil_models: Dict[str, torch.nn.Module] = {}
        self.sil_tokenizers: Dict[str, object] = {}
        self.emotion_models: Dict[str, torch.nn.Module] = {}
        self.emotion_tokenizers: Dict[str, object] = {}
        self._load_models()

    def _resolve_transformers(self):
        if self.use_fake_models:
            from fake_transformers import (  # type: ignore
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        else:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        return AutoTokenizer, AutoModelForSequenceClassification

    def _load_models(self) -> None:
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        print(f"🔄 Loading models from: {self.model_dir}", flush=True)
        print(f"   Device: {self.device.upper()}", flush=True)
        AutoTokenizer, AutoModel = self._resolve_transformers()

        print("\n📦 Loading SIL models...", flush=True)
        for label_type in LABEL_MAPPINGS.keys():
            model_path = self.model_dir / label_type
            if not model_path.exists():
                print(f"   ⚠️  {label_type}: not found (skipping)", flush=True)
                continue
            try:
                print(f"   📥 Loading {label_type}...", end=" ", flush=True)
                tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                model = AutoModel.from_pretrained(str(model_path))
                model.eval()
                model = model.to(self.device)
                self.sil_models[label_type] = model
                self.sil_tokenizers[label_type] = tokenizer
                print(f"✅", flush=True)
            except Exception as exc:  # pragma: no cover - surface during runtime
                print(f"❌ Failed: {exc}", flush=True)

        print("\n📦 Loading emotion models...", flush=True)
        emotion_base = self.model_dir / "emotion"
        for label_type in EMOTION_MODEL_ORDER:
            if label_type == "emotion":
                model_path = emotion_base
            else:
                model_path = emotion_base / label_type

            if not model_path.exists():
                print(f"   ⚠️  {label_type}: not found (skipping)", flush=True)
                continue

            try:
                print(f"   📥 Loading {label_type}...", end=" ", flush=True)
                tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                if label_type == "severity" and not self.use_fake_models:
                    model = AutoModel.from_pretrained(
                        str(model_path),
                        num_labels=1,
                        problem_type="regression",
                        ignore_mismatched_sizes=True,
                    )
                else:
                    model = AutoModel.from_pretrained(str(model_path))
                model.eval()
                model = model.to(self.device)
                self.emotion_models[label_type] = model
                self.emotion_tokenizers[label_type] = tokenizer
                print(f"✅", flush=True)
            except Exception as exc:
                print(f"❌ Failed: {exc}", flush=True)

        if not self.sil_models and not self.emotion_models:
            raise RuntimeError(
                f"No models were loaded. Checked directory: {self.model_dir}"
            )
        
        print(f"\n✅ Models loaded successfully!")
        print(f"   SIL models: {len(self.sil_models)} ({', '.join(self.sil_models.keys())})")
        print(f"   Emotion models: {len(self.emotion_models)} ({', '.join(self.emotion_models.keys())})")
        print(f"   Total: {len(self.sil_models) + len(self.emotion_models)} models ready for inference\n", flush=True)

    def _predict_classification(
        self, model: torch.nn.Module, tokenizer, text: str, labels: List[str]
    ) -> PredictionResult:
        start = time.time()
        inputs = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        predicted_label = labels[pred_idx] if pred_idx < len(labels) else "unknown"
        top_indices = np.argsort(probs)[-3:][::-1]
        top3 = [
            {"label": labels[idx] if idx < len(labels) else "unknown", "confidence": float(probs[idx])}
            for idx in top_indices
        ]
        inference_time_ms = (time.time() - start) * 1000.0
        return PredictionResult(
            predicted_label=predicted_label,
            confidence=float(probs[pred_idx]),
            top3=top3,
            inference_time_ms=inference_time_ms,
        )

    def _predict_regression(self, model: torch.nn.Module, tokenizer, text: str) -> Tuple[float, float, float]:
        start = time.time()
        inputs = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        value = float(logits.view(-1)[0].item())
        value = max(0.0, min(1.0, value))
        confidence = 1.0 - abs(value - 0.5) * 2.0
        confidence = max(0.0, min(1.0, confidence))
        inference_time_ms = (time.time() - start) * 1000.0
        return value, confidence, inference_time_ms

    def _predict_wrapper(self, task: Tuple) -> Tuple[Tuple[str, str], Optional[Dict]]:
        """Wrapper function for parallel execution of model predictions.
        Works well with GPU as CUDA operations release the GIL."""
        model_category, label_type, model, tokenizer, text, is_regression = task
        
        try:
            if is_regression:
                # Regression model (severity)
                value, confidence, inference_time = self._predict_regression(
                    model, tokenizer, text
                )
                return (model_category, label_type), {
                    "category": model_category,
                    "prediction": value,
                    "confidence": confidence,
                    "top3": [],
                    "inference_time_ms": inference_time,
                }
            else:
                # Classification model
                if model_category == "sil":
                    labels = LABEL_MAPPINGS[label_type]
                else:
                    labels = EMOTION_LABEL_MAPPINGS.get(label_type, [])
                
                prediction = self._predict_classification(model, tokenizer, text, labels)
                return (model_category, label_type), {
                    "category": model_category,
                    "prediction": prediction.predicted_label,
                    "confidence": prediction.confidence,
                    "top3": prediction.top3,
                    "inference_time_ms": prediction.inference_time_ms,
                }
        except Exception as exc:
            print(f"⚠️  Error in parallel prediction for {model_category}/{label_type}: {exc}", file=sys.stderr)
            return (model_category, label_type), None

    def classify(self, text: str) -> Dict:
        if not text or not text.strip():
            raise ValueError("Text must not be empty.")

        quality = self.text_quality_checker.score(text)
        if quality.is_gibberish:
            raise GibberishInputError("Input appears to be gibberish.", quality)

        normalized_text = normalize_spelling(text)

        results: Dict[str, Dict] = {}
        inference_times = {}

        overall_start = time.time()

        # Prepare all tasks for parallel execution
        all_tasks = []
        for label_type, model in self.sil_models.items():
            all_tasks.append(
                ("sil", label_type, model, self.sil_tokenizers[label_type], normalized_text, False)
            )
        for label_type, model in self.emotion_models.items():
            is_regression = (label_type == "severity")
            all_tasks.append(
                ("emotion", label_type, model, self.emotion_tokenizers[label_type], normalized_text, is_regression)
            )

        # Parallel execution - works well with GPU as CUDA operations release the GIL
        # Limit workers to avoid GPU memory issues (use all workers if on CPU, limit to 6-8 for GPU)
        max_workers = len(all_tasks) if self.device == "cpu" else min(len(all_tasks), 8)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._predict_wrapper, task): task for task in all_tasks}
            for future in as_completed(futures):
                try:
                    (model_category, label_type), prediction = future.result()
                    if prediction:
                        results[label_type] = prediction
                        inference_times[label_type] = prediction["inference_time_ms"]
                except Exception as exc:
                    task = futures[future]
                    print(f"⚠️  Error processing {task[0]}/{task[1]}: {exc}", file=sys.stderr)

        total_time_measured_ms = (time.time() - overall_start) * 1000.0
        total_time_ms = float(sum(inference_times.values())) if inference_times else 0.0
        avg_time_ms = (
            total_time_ms / len(inference_times) if inference_times else 0.0
        )

        ordered_predictions: Dict[str, Dict] = {}

        for param in PARAMETER_ORDER:
            if param in results:
                ordered_predictions[param] = results[param]

        for key, value in results.items():
            if key not in ordered_predictions:
                ordered_predictions[key] = value

        return {
            "input_text": text,
            "normalized_text": normalized_text,
            "quality": {
                "word_count": quality.word_count,
                "valid_word_ratio": quality.valid_word_ratio,
                "non_alnum_ratio": quality.non_alnum_ratio,
                "repeated_char_ratio": quality.repeated_char_ratio,
            },
            "predictions": ordered_predictions,
            "timing": {
                "total_inference_ms": total_time_ms,
                "measured_time_ms": total_time_measured_ms,
                "average_per_model_ms": avg_time_ms,
            },
            "models_loaded": {
                "sil": list(self.sil_models.keys()),
                "emotion": list(self.emotion_models.keys()),
            },
        }

