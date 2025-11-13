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

        AutoTokenizer, AutoModel = self._resolve_transformers()

        for label_type in LABEL_MAPPINGS.keys():
            model_path = self.model_dir / label_type
            if not model_path.exists():
                continue
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                model = AutoModel.from_pretrained(str(model_path))
                model.eval()
                model = model.to(self.device)
                self.sil_models[label_type] = model
                self.sil_tokenizers[label_type] = tokenizer
            except Exception as exc:  # pragma: no cover - surface during runtime
                print(f"⚠️  Failed to load {label_type} model: {exc}", file=sys.stderr)

        emotion_base = self.model_dir / "emotion"
        for label_type in EMOTION_MODEL_ORDER:
            if label_type == "emotion":
                model_path = emotion_base
            else:
                model_path = emotion_base / label_type

            if not model_path.exists():
                continue

            try:
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
            except Exception as exc:
                print(f"⚠️  Failed to load emotion {label_type} model: {exc}", file=sys.stderr)

        if not self.sil_models and not self.emotion_models:
            raise RuntimeError(
                f"No models were loaded. Checked directory: {self.model_dir}"
            )

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

    def classify(self, text: str) -> Dict:
        if not text or not text.strip():
            raise ValueError("Text must not be empty.")

        quality = self.text_quality_checker.score(text)
        if quality.is_gibberish:
            raise GibberishInputError("Input appears to be gibberish.", quality)

        normalized_text = normalize_spelling(text)

        sil_results: Dict[str, Dict] = {}
        emotion_results: Dict[str, Dict] = {}
        inference_times: List[float] = []

        overall_start = time.time()

        for label_type, model in self.sil_models.items():
            tokenizer = self.sil_tokenizers[label_type]
            prediction = self._predict_classification(
                model, tokenizer, normalized_text, LABEL_MAPPINGS[label_type]
            )
            sil_results[label_type] = {
                "category": "sil",
                "prediction": prediction.predicted_label,
                "confidence": prediction.confidence,
                "top3": prediction.top3,
                "inference_time_ms": prediction.inference_time_ms,
            }
            inference_times.append(prediction.inference_time_ms)

        for label_type, model in self.emotion_models.items():
            tokenizer = self.emotion_tokenizers[label_type]
            if label_type == "severity":
                value, confidence, inference_time = self._predict_regression(
                    model, tokenizer, normalized_text
                )
                emotion_results[label_type] = {
                    "category": "emotion",
                    "prediction": value,
                    "confidence": confidence,
                    "top3": [],
                    "inference_time_ms": inference_time,
                }
                inference_times.append(inference_time)
            else:
                labels = EMOTION_LABEL_MAPPINGS.get(label_type, [])
                prediction = self._predict_classification(
                    model, tokenizer, normalized_text, labels
                )
                emotion_results[label_type] = {
                    "category": "emotion",
                    "prediction": prediction.predicted_label,
                    "confidence": prediction.confidence,
                    "top3": prediction.top3,
                    "inference_time_ms": prediction.inference_time_ms,
                }
                inference_times.append(prediction.inference_time_ms)

        total_time_measured_ms = (time.time() - overall_start) * 1000.0
        total_time_ms = float(sum(inference_times))
        avg_time_ms = (
            total_time_ms / len(inference_times) if inference_times else 0.0
        )

        ordered_predictions: Dict[str, Dict] = {}
        combined_results = {**sil_results, **emotion_results}

        for param in PARAMETER_ORDER:
            if param in combined_results:
                ordered_predictions[param] = combined_results[param]

        for key, value in combined_results.items():
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

