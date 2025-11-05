#!/usr/bin/env python3
"""
compare_sil_vs_finetuned_with_emotion_parallel.py

Compare Gemini SIL prompt output with fine-tuned intent and emotion classifiers.
Runs intent and emotion models in parallel for performance.

Usage:
  python compare_sil_vs_finetuned_with_emotion_parallel.py \
      --sil-model-dir ./models/sil \
      --emotion-model-dir ./models/emotion \
      --test-data ./test_questions_100.json \
      --output-dir ./comparison_results \
      --project playpen-c84caa \
      --location us-central1
"""

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vertexai import init as vertex_init
from vertexai.generative_models import GenerativeModel

# Hardcoded defaults
DEFAULT_PROJECT = "playpen-c84caa"
DEFAULT_LOCATION = "us-central1"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('comparison_parallel.log')
    ]
)
logger = logging.getLogger(__name__)

# Label mappings (must match finetune_sil_model.py)
TOPIC_LABELS = [
    "savings", "investments", "pensions", "mortgages", "banking",
    "loans", "debt", "insurance", "taxation", "general", "off_topic"
]

INTENT_TYPE_LABELS = [
    "fact_seeking", "advice_seeking", "account_action", "goal_expression",
    "guidance", "off_topic"
]

QUERY_TYPE_LABELS = [
    "what_is", "eligibility", "recommendation", "account_action",
    "goal_expression", "comparison"
]

STAGE_LABELS = [
    "goal_setup", "accumulation", "understanding", "optimisation", "withdrawal",
    "goal_definition", "execution", "enrolment", "decumulation",
    "application", "repayment", "remortgage", "awareness", "action",
    "planning", "management", "claim"
]

DOMAIN_SCOPE_LABELS = ["general", "bank_specific"]
ADVICE_RISK_LABELS = ["low", "medium", "high"]
EMOTION_LABELS = ["positive", "neutral", "negative", "confused"]

LABEL_MAPPINGS = {
    "topic": TOPIC_LABELS,
    "intent": INTENT_TYPE_LABELS,
    "query": QUERY_TYPE_LABELS,
    "stage": STAGE_LABELS,
    "domain": DOMAIN_SCOPE_LABELS,
    "advice_risk": ADVICE_RISK_LABELS,
}


class FineTunedPredictor:
    """Load and run fine-tuned models for predictions."""
    
    def __init__(self, sil_model_dir: str, emotion_model_dir: str, device: str = "cpu"):
        self.device = device
        self.tokenizers = {}
        self.models = {}
        
        # Load SIL models
        logger.info(f"Loading SIL models from {sil_model_dir}")
        for label_type in ["topic", "intent", "query", "stage", "domain", "advice_risk"]:
            model_path = Path(sil_model_dir) / label_type
            if model_path.exists():
                try:
                    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
                    model.eval()
                    model.to(device)
                    self.tokenizers[label_type] = tokenizer
                    self.models[label_type] = model
                    logger.info(f"  Loaded {label_type} model")
                except Exception as e:
                    logger.warning(f"  Failed to load {label_type} model: {e}")
        
        # Load emotion models
        logger.info(f"Loading emotion models from {emotion_model_dir}")
        emotion_path = Path(emotion_model_dir)
        if emotion_path.exists():
            try:
                # Emotion classifier
                emotion_model_path = emotion_path / "emotion"
                if emotion_model_path.exists():
                    tokenizer = AutoTokenizer.from_pretrained(str(emotion_model_path))
                    model = AutoModelForSequenceClassification.from_pretrained(str(emotion_model_path))
                    model.eval()
                    model.to(device)
                    self.tokenizers["emotion"] = tokenizer
                    self.models["emotion"] = model
                    logger.info(f"  Loaded emotion classifier")
                
                # Distress classifier
                distress_model_path = emotion_path / "distress"
                if distress_model_path.exists():
                    tokenizer = AutoTokenizer.from_pretrained(str(distress_model_path))
                    model = AutoModelForSequenceClassification.from_pretrained(str(distress_model_path))
                    model.eval()
                    model.to(device)
                    self.tokenizers["distress"] = tokenizer
                    self.models["distress"] = model
                    logger.info(f"  Loaded distress classifier")
                
                # Vulnerability classifier
                vulnerability_model_path = emotion_path / "vulnerability"
                if vulnerability_model_path.exists():
                    tokenizer = AutoTokenizer.from_pretrained(str(vulnerability_model_path))
                    model = AutoModelForSequenceClassification.from_pretrained(str(vulnerability_model_path))
                    model.eval()
                    model.to(device)
                    self.tokenizers["vulnerability"] = tokenizer
                    self.models["vulnerability"] = model
                    logger.info(f"  Loaded vulnerability classifier")
            except Exception as e:
                logger.warning(f"  Failed to load emotion models: {e}")
    
    def predict_sil(self, text: str) -> Dict:
        """Predict all SIL labels (runs in parallel)."""
        results = {}
        
        def predict_single(label_type: str):
            if label_type not in self.models:
                return None
            tokenizer = self.tokenizers[label_type]
            model = self.models[label_type]
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                pred_idx = torch.argmax(logits, dim=-1).item()
                confidence = probs[0][pred_idx].item()
            
            labels = LABEL_MAPPINGS[label_type]
            return {
                "label": labels[pred_idx],
                "confidence": confidence,
                "all_probs": {labels[i]: probs[0][i].item() for i in range(len(labels))}
            }
        
        # Run all SIL predictions in parallel
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {executor.submit(predict_single, label_type): label_type 
                      for label_type in ["topic", "intent", "query", "stage", "domain", "advice_risk"]}
            
            for future in as_completed(futures):
                label_type = futures[future]
                try:
                    result = future.result()
                    if result:
                        results[label_type] = result
                except Exception as e:
                    logger.warning(f"Error predicting {label_type}: {e}")
        
        return results
    
    def predict_emotion(self, text: str) -> Dict:
        """Predict emotion, distress, and vulnerability (runs in parallel)."""
        results = {}
        
        def predict_single(model_name: str, label_type: str):
            if model_name not in self.models:
                return None
            tokenizer = self.tokenizers[model_name]
            model = self.models[model_name]
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                
                if label_type == "emotion":
                    probs = torch.softmax(logits, dim=-1)
                    pred_idx = torch.argmax(logits, dim=-1).item()
                    confidence = probs[0][pred_idx].item()
                    return {
                        "label": EMOTION_LABELS[pred_idx],
                        "confidence": confidence
                    }
                else:
                    # Binary classification (distress/vulnerability)
                    probs = torch.softmax(logits, dim=-1)
                    pred_idx = torch.argmax(logits, dim=-1).item()
                    confidence = probs[0][pred_idx].item()
                    return {
                        "flag": bool(pred_idx),
                        "confidence": confidence
                    }
        
        # Run all emotion predictions in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(predict_single, "emotion", "emotion"): "emotion",
                executor.submit(predict_single, "distress", "distress"): "distress",
                executor.submit(predict_single, "vulnerability", "vulnerability"): "vulnerability"
            }
            
            for future in as_completed(futures):
                label_type = futures[future]
                try:
                    result = future.result()
                    if result:
                        results[label_type] = result
                except Exception as e:
                    logger.warning(f"Error predicting {label_type}: {e}")
        
        return results
    
    def predict_all(self, text: str) -> Dict:
        """Predict both SIL and emotion in parallel."""
        # Run SIL and emotion predictions in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            sil_future = executor.submit(self.predict_sil, text)
            emotion_future = executor.submit(self.predict_emotion, text)
            
            sil_results = sil_future.result()
            emotion_results = emotion_future.result()
        
        return {
            "sil": sil_results,
            "emotion": emotion_results
        }


def call_sil_prompt(question: str, model: GenerativeModel) -> Dict:
    """Call Gemini SIL prompt and parse JSON response."""
    # Simplified SIL prompt (you may need to adjust this)
    sil_prompt = f"""Analyze this financial query and return JSON with:
- topic: one of {TOPIC_LABELS}
- intent_type: one of {INTENT_TYPE_LABELS}
- query_type: one of {QUERY_TYPE_LABELS}
- stage: one of {STAGE_LABELS}
- domain_scope: one of {DOMAIN_SCOPE_LABELS}
- advice_risk_score: 0.0-1.0
- emotion: one of {EMOTION_LABELS}
- sentiment_score: -1.0 to 1.0
- distress_flag: boolean
- vulnerability_flag: boolean

Query: {question}

Return JSON only (no markdown):"""
    
    try:
        response = model.generate_content(sil_prompt)
        text = response.text if hasattr(response, 'text') else str(response)
        
        # Try to extract JSON
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        text = text.strip()
        result = json.loads(text)
        return result
    except Exception as e:
        logger.warning(f"Error calling SIL prompt: {e}")
        return {}


def compare_question(question: str, predictor: FineTunedPredictor, gemini_model: GenerativeModel) -> Dict:
    """Compare fine-tuned models vs Gemini SIL for a single question."""
    start_time = time.time()
    
    # Run fine-tuned models and Gemini in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        finetuned_future = executor.submit(predictor.predict_all, question)
        gemini_future = executor.submit(call_sil_prompt, question, gemini_model)
        
        finetuned_result = finetuned_future.result()
        gemini_result = gemini_future.result()
    
    total_time = time.time() - start_time
    
    return {
        "question": question,
        "finetuned": finetuned_result,
        "gemini": gemini_result,
        "time_ms": total_time * 1000
    }


def main():
    parser = argparse.ArgumentParser(description="Compare SIL vs fine-tuned models with emotion")
    parser.add_argument("--sil-model-dir", required=True, help="Directory with fine-tuned SIL models")
    parser.add_argument("--emotion-model-dir", required=True, help="Directory with fine-tuned emotion models")
    parser.add_argument("--test-data", required=True, help="JSON file with test questions")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help=f"GCP project (default: {DEFAULT_PROJECT})")
    parser.add_argument("--location", default=DEFAULT_LOCATION, help=f"Vertex AI location (default: {DEFAULT_LOCATION})")
    parser.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL, help=f"Gemini model (default: {DEFAULT_GEMINI_MODEL})")
    parser.add_argument("--device", default="cpu", help="Device for models (cpu/cuda)")
    parser.add_argument("--max-questions", type=int, default=100, help="Maximum number of questions to process")
    
    args = parser.parse_args()
    
    # Initialize Vertex AI
    vertex_init(project=args.project, location=args.location)
    gemini_model = GenerativeModel(args.gemini_model)
    
    # Load test questions
    test_path = Path(args.test_data)
    if not test_path.exists():
        logger.error(f"Test data file not found: {test_path}")
        sys.exit(1)
    
    with open(test_path) as f:
        test_data = json.load(f)
    
    if isinstance(test_data, list):
        questions = [q.get("question", q.get("text", str(q))) for q in test_data[:args.max_questions]]
    elif isinstance(test_data, dict) and "questions" in test_data:
        questions = test_data["questions"][:args.max_questions]
    else:
        logger.error("Invalid test data format")
        sys.exit(1)
    
    logger.info(f"Loaded {len(questions)} questions")
    
    # Load fine-tuned models
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    predictor = FineTunedPredictor(args.sil_model_dir, args.emotion_model_dir, device)
    
    # Process questions
    logger.info("Processing questions...")
    results = []
    total_time = 0
    
    for i, question in enumerate(questions, 1):
        logger.info(f"Processing {i}/{len(questions)}: {question[:60]}...")
        result = compare_question(question, predictor, gemini_model)
        results.append(result)
        total_time += result["time_ms"]
        
        if i % 10 == 0:
            avg_time = total_time / i
            logger.info(f"  Progress: {i}/{len(questions)} (avg: {avg_time:.1f}ms per question)")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"comparison_results_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "metadata": {
                "total_questions": len(questions),
                "sil_model_dir": args.sil_model_dir,
                "emotion_model_dir": args.emotion_model_dir,
                "avg_time_ms": total_time / len(questions),
                "total_time_seconds": total_time / 1000
            },
            "results": results
        }, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    logger.info(f"Average time per question: {total_time / len(questions):.1f}ms")
    logger.info(f"Total time: {total_time / 1000:.1f}s")


if __name__ == "__main__":
    main()

