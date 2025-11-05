#!/usr/bin/env python3
"""
test_finetuned_models_parallel.py

Test fine-tuned SIL and emotion models in parallel.
Runs all models concurrently and outputs results in SIL format for validation.

Usage:
  python test_finetuned_models_parallel.py \
      --sil-model-dir ./models/sil \
      --emotion-model-dir ./models/emotion \
      --test-data ./test_questions_100.json \
      --output-file ./test_results.json
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_finetuned.log')
    ]
)
logger = logging.getLogger(__name__)

# Label mappings (must match finetune_sil_model.py and finetune_emotion_model.py)
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
    """Load and run fine-tuned models for predictions in parallel."""
    
    def __init__(self, sil_model_dir: str, emotion_model_dir: str, device: str = "cpu"):
        self.device = device
        self.tokenizers = {}
        self.models = {}
        self.model_load_times = {}
        
        # Pre-warm GPU if available (reduces first inference latency)
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Warm up with dummy tensor
            dummy = torch.zeros(1, 1).to(device)
            _ = dummy * 2
            logger.info("GPU warmed up")
        
        # Load SIL models
        logger.info(f"Loading SIL models from {sil_model_dir}")
        for label_type in ["topic", "intent", "query", "stage", "domain", "advice_risk"]:
            model_path = Path(sil_model_dir) / label_type
            if model_path.exists():
                try:
                    load_start = time.time()
                    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
                    model.eval()
                    model.to(device)
                    self.tokenizers[label_type] = tokenizer
                    self.models[label_type] = model
                    load_time = (time.time() - load_start) * 1000
                    self.model_load_times[label_type] = load_time
                    logger.info(f"  Loaded {label_type} model ({load_time:.1f}ms)")
                except Exception as e:
                    logger.warning(f"  Failed to load {label_type} model: {e}")
        
        # Load emotion models
        logger.info(f"Loading emotion models from {emotion_model_dir}")
        emotion_path = Path(emotion_model_dir)
        if emotion_path.exists():
            for model_name in ["emotion", "distress", "vulnerability"]:
                model_path = emotion_path / model_name
                if model_path.exists():
                    try:
                        load_start = time.time()
                        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
                        model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
                        model.eval()
                        model.to(device)
                        self.tokenizers[model_name] = tokenizer
                        self.models[model_name] = model
                        load_time = (time.time() - load_start) * 1000
                        self.model_load_times[model_name] = load_time
                        # Debug: Check model config
                        num_labels = model.config.num_labels
                        logger.info(f"  Loaded {model_name} model ({load_time:.1f}ms) - num_labels: {num_labels}")
                        if model_name == "emotion" and num_labels != len(EMOTION_LABELS):
                            logger.warning(f"  WARNING: Emotion model has {num_labels} labels but expected {len(EMOTION_LABELS)}")
                    except Exception as e:
                        logger.warning(f"  Failed to load {model_name} model: {e}")
                else:
                    logger.warning(f"  Emotion model path does not exist: {model_path}")
        else:
            logger.warning(f"  Emotion model directory does not exist: {emotion_path}")
    
    def predict_single_model(self, text: str, model_name: str, label_type: str, pre_tokenized: Optional[Dict] = None) -> Tuple[Dict, float]:
        """Predict using a single model and return result with timing.
        
        Args:
            text: Input text
            model_name: Model identifier
            label_type: Type of label being predicted
            pre_tokenized: Pre-tokenized inputs (optional, for optimization)
        """
        if model_name not in self.models:
            return None, 0.0
        
        start_time = time.time()
        tokenizer = self.tokenizers[model_name]
        model = self.models[model_name]
        
        try:
            # Use pre-tokenized if provided, otherwise tokenize
            if pre_tokenized:
                inputs = pre_tokenized
            else:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Move to device if not already (for pre-tokenized case)
            if pre_tokenized:
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                # Only compute softmax if needed (for confidence/all_probs)
                pred_idx = torch.argmax(logits, dim=-1).item()
                probs = torch.softmax(logits, dim=-1)
                confidence = probs[0][pred_idx].item()
            
            if label_type == "emotion":
                labels = EMOTION_LABELS
                # Debug: Check if we have the right number of labels
                if len(labels) != logits.shape[1]:
                    logger.warning(f"Emotion model output shape mismatch: expected {len(labels)} labels, got {logits.shape[1]} logits")
                result = {
                    "label": labels[pred_idx],
                    "confidence": round(confidence, 4),
                    "all_probs": {labels[i]: round(probs[0][i].item(), 4) for i in range(len(labels))},
                    "raw_logits": [round(logits[0][i].item(), 4) for i in range(len(labels))],  # Debug: show raw logits
                    "pred_idx": pred_idx  # Debug: show prediction index
                }
            elif label_type in ["distress", "vulnerability"]:
                # Binary classification
                result = {
                    "flag": bool(pred_idx),
                    "confidence": round(confidence, 4),
                    "false_prob": round(probs[0][0].item(), 4),
                    "true_prob": round(probs[0][1].item(), 4)
                }
            else:
                # SIL classification
                labels = LABEL_MAPPINGS[label_type]
                result = {
                    "label": labels[pred_idx],
                    "confidence": round(confidence, 4),
                    "all_probs": {labels[i]: round(probs[0][i].item(), 4) for i in range(len(labels))}
                }
            
            inference_time = (time.time() - start_time) * 1000
            return result, inference_time
            
        except Exception as e:
            logger.warning(f"Error predicting {model_name}: {e}")
            return None, 0.0
    
    def predict_all_parallel(self, text: str, optimize_tokenization: bool = True) -> Dict:
        """Run all models in parallel and return results with timing.
        
        Args:
            text: Input text
            optimize_tokenization: If True, tokenize once and reuse (faster but less accurate if models use different tokenizers)
        """
        results = {}
        timings = {}
        
        # OPTIMIZATION: Pre-tokenize with first tokenizer if all models use same tokenizer
        # (DistilBERT models typically use the same tokenizer)
        pre_tokenized = None
        if optimize_tokenization and len(self.tokenizers) > 0:
            first_tokenizer = next(iter(self.tokenizers.values()))
            # Check if all tokenizers are the same type (optimization)
            tokenizer_types = {type(t) for t in self.tokenizers.values()}
            if len(tokenizer_types) == 1:
                # All same tokenizer - tokenize once
                pre_tokenized = first_tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
        
        # All models to predict
        tasks = [
            ("topic", "topic", "topic"),
            ("intent", "intent", "intent"),
            ("query", "query", "query"),
            ("stage", "stage", "stage"),
            ("domain", "domain", "domain"),
            ("advice_risk", "advice_risk", "advice_risk"),
            ("emotion", "emotion", "emotion"),
            ("distress", "distress", "distress"),
            ("vulnerability", "vulnerability", "vulnerability"),
        ]
        
        # Run all predictions in parallel
        # Use max_workers based on device (GPU can handle more concurrent ops)
        max_workers = 16 if self.device == "cuda" else 9
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.predict_single_model, text, model_name, label_type, pre_tokenized): (model_name, label_type)
                for model_name, label_type, _ in tasks
            }
            
            for future in as_completed(futures):
                model_name, label_type = futures[future]
                try:
                    result, inference_time = future.result()
                    if result:
                        results[label_type] = result
                        timings[label_type] = round(inference_time, 2)
                except Exception as e:
                    logger.warning(f"Error in {model_name}: {e}")
                    timings[label_type] = 0.0
        
        return results, timings
    
    def format_sil_output(self, results: Dict, timings: Dict) -> Dict:
        """Format results into SIL output structure."""
        # Map fine-tuned results to SIL format
        sil_output = {
            "topic": results.get("topic", {}).get("label", "general"),
            "intent_type": results.get("intent", {}).get("label", "fact_seeking"),
            "query_type": results.get("query", {}).get("label", "what_is"),
            "stage": results.get("stage", {}).get("label", "understanding"),
            "domain_scope": results.get("domain", {}).get("label", "general"),
            "advice_risk_score": self._convert_advice_risk_to_score(results.get("advice_risk", {})),
            "detected_emotion": results.get("emotion", {}).get("label", "neutral"),
            "sentiment_score": self._emotion_to_sentiment(results.get("emotion", {})),
            "distress_flag": results.get("distress", {}).get("flag", False),
            "vulnerability_flag": results.get("vulnerability", {}).get("flag", False),
            "confidence": {
                "topic": results.get("topic", {}).get("confidence", 0.0),
                "intent_type": results.get("intent", {}).get("confidence", 0.0),
                "query_type": results.get("query", {}).get("confidence", 0.0),
                "stage": results.get("stage", {}).get("confidence", 0.0),
                "domain_scope": results.get("domain", {}).get("confidence", 0.0),
                "advice_risk": results.get("advice_risk", {}).get("confidence", 0.0),
                "emotion": results.get("emotion", {}).get("confidence", 0.0),
                "distress": results.get("distress", {}).get("confidence", 0.0),
                "vulnerability": results.get("vulnerability", {}).get("confidence", 0.0),
            },
            "timings_ms": timings,
            "all_predictions": results  # Full predictions for validation
        }
        return sil_output
    
    def _convert_advice_risk_to_score(self, advice_result: Dict) -> float:
        """Convert advice_risk label to score (0.0-1.0)."""
        label = advice_result.get("label", "low")
        if label == "low":
            return 0.15
        elif label == "medium":
            return 0.5
        elif label == "high":
            return 0.85
        return 0.15
    
    def _emotion_to_sentiment(self, emotion_result: Dict) -> float:
        """Convert emotion to sentiment score (-1.0 to 1.0)."""
        label = emotion_result.get("label", "neutral")
        if label == "positive":
            return 0.6
        elif label == "negative":
            return -0.6
        elif label == "confused":
            return -0.2
        return 0.0


def process_questions(questions: List[str], predictor: FineTunedPredictor) -> List[Dict]:
    """Process all questions and return results."""
    results = []
    
    logger.info(f"Processing {len(questions)} questions...")
    
    for i, question in enumerate(questions, 1):
        logger.info(f"\n[{i}/{len(questions)}] Processing: {question[:60]}...")
        
        question_start = time.time()
        
        # Run all models in parallel
        model_results, timings = predictor.predict_all_parallel(question)
        
        # Format as SIL output
        sil_output = predictor.format_sil_output(model_results, timings)
        
        question_time = (time.time() - question_start) * 1000
        
        result = {
            "question": question,
            "question_number": i,
            "sil_output": sil_output,
            "total_time_ms": round(question_time, 2),
            "timings_breakdown": timings
        }
        
        results.append(result)
        
        # Log summary
        logger.info(f"  Topic: {sil_output['topic']} ({sil_output['confidence']['topic']:.2%})")
        emotion_info = sil_output.get('all_predictions', {}).get('emotion', {})
        if emotion_info:
            all_probs = emotion_info.get('all_probs', {})
            logger.info(f"  Emotion: {sil_output['detected_emotion']} ({sil_output['confidence']['emotion']:.2%})")
            logger.info(f"    Emotion probabilities: {all_probs}")
            if 'raw_logits' in emotion_info:
                logger.info(f"    Raw logits: {emotion_info['raw_logits']}")
        else:
            logger.warning(f"  Emotion prediction missing or failed!")
        logger.info(f"  Distress: {sil_output['distress_flag']} | Vulnerability: {sil_output['vulnerability_flag']}")
        logger.info(f"  Total time: {question_time:.1f}ms | Avg model time: {np.mean(list(timings.values())):.1f}ms")
        
        if i % 10 == 0:
            avg_time = np.mean([r['total_time_ms'] for r in results])
            logger.info(f"\n  Progress: {i}/{len(questions)} | Avg time per question: {avg_time:.1f}ms")
    
    return results


def calculate_statistics(results: List[Dict]) -> Dict:
    """Calculate timing statistics."""
    all_times = [r['total_time_ms'] for r in results]
    all_model_times = defaultdict(list)
    
    for result in results:
        for model, time_ms in result['timings_breakdown'].items():
            all_model_times[model].append(time_ms)
    
    stats = {
        "total_questions": len(results),
        "total_time_seconds": round(sum(all_times) / 1000, 2),
        "avg_time_per_question_ms": round(np.mean(all_times), 2),
        "median_time_per_question_ms": round(np.median(all_times), 2),
        "min_time_ms": round(min(all_times), 2),
        "max_time_ms": round(max(all_times), 2),
        "model_timings": {
            model: {
                "avg_ms": round(np.mean(times), 2),
                "median_ms": round(np.median(times), 2),
                "min_ms": round(min(times), 2),
                "max_ms": round(max(times), 2),
            }
            for model, times in all_model_times.items()
        }
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned models in parallel")
    parser.add_argument("--sil-model-dir", required=True, help="Directory with fine-tuned SIL models")
    parser.add_argument("--emotion-model-dir", required=True, help="Directory with fine-tuned emotion models")
    parser.add_argument("--test-data", required=True, help="JSON file with test questions")
    parser.add_argument("--output-file", required=True, help="Output JSON file for results")
    # Auto-detect GPU availability
    cuda_available = torch.cuda.is_available()
    default_device = "cuda" if cuda_available else "cpu"
    
    parser.add_argument("--device", default=default_device, 
                       help=f"Device for models (cpu/cuda). Default: {default_device} (CUDA available: {cuda_available})")
    
    args = parser.parse_args()
    
    # Load test questions
    test_path = Path(args.test_data)
    if not test_path.exists():
        logger.error(f"Test data file not found: {test_path}")
        sys.exit(1)
    
    with open(test_path) as f:
        test_data = json.load(f)
    
    if isinstance(test_data, list):
        questions = [q.get("question", q.get("text", str(q))) for q in test_data]
    elif isinstance(test_data, dict) and "questions" in test_data:
        questions = test_data["questions"]
    else:
        logger.error("Invalid test data format")
        sys.exit(1)
    
    logger.info(f"Loaded {len(questions)} questions")
    
    # Load models - validate device choice
    if args.device == "cuda" and not cuda_available:
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    elif args.device == "cuda" and cuda_available:
        device = "cuda"
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
    
    logger.info(f"Using device: {device}")
    
    load_start = time.time()
    predictor = FineTunedPredictor(args.sil_model_dir, args.emotion_model_dir, device)
    load_time = (time.time() - load_start) * 1000
    logger.info(f"Model loading complete ({load_time:.1f}ms)")
    
    # Process questions
    process_start = time.time()
    results = process_questions(questions, predictor)
    process_time = time.time() - process_start
    
    # Calculate statistics
    stats = calculate_statistics(results)
    
    # Save results
    output_data = {
        "metadata": {
            "sil_model_dir": args.sil_model_dir,
            "emotion_model_dir": args.emotion_model_dir,
            "device": device,
            "model_load_time_ms": round(load_time, 2),
            "model_load_times": predictor.model_load_times,
            "statistics": stats
        },
        "results": results
    }
    
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Total questions: {stats['total_questions']}")
    logger.info(f"Total processing time: {stats['total_time_seconds']}s")
    logger.info(f"Average time per question: {stats['avg_time_per_question_ms']:.1f}ms")
    logger.info(f"Median time per question: {stats['median_time_per_question_ms']:.1f}ms")
    logger.info(f"Min/Max time: {stats['min_time_ms']:.1f}ms / {stats['max_time_ms']:.1f}ms")
    logger.info("\nModel timing averages:")
    for model, model_stats in stats['model_timings'].items():
        logger.info(f"  {model:15s}: {model_stats['avg_ms']:.1f}ms (min: {model_stats['min_ms']:.1f}ms, max: {model_stats['max_ms']:.1f}ms)")
    logger.info(f"\nResults saved to: {output_path}")
    logger.info("="*60)


if __name__ == "__main__":
    main()

