#!/usr/bin/env python3
"""
test_all_models.py

Test both SIL and emotion models with a sentence.
Tests all 11 models (6 SIL + 5 emotion) and categorizes predictions by confidence.

Usage:
  python test_all_models.py \
      --model-dir ./models \
      --text "What is my account balance?"

  # Interactive mode
  python test_all_models.py --model-dir ./models

  # Custom confidence thresholds
  python test_all_models.py --model-dir ./models --text "..." --high-threshold 0.85 --low-threshold 0.55
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# SIL Label mappings (must match finetune_sil_model.py)
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

# Emotion Label mappings (must match finetune_emotion_model.py)
EMOTION_LABELS = ["positive", "neutral", "negative"]
BINARY_LABELS = ["false", "true"]  # For distress, vulnerability, handover

SIL_LABEL_MAPPINGS = {
    "topic": TOPIC_LABELS,
    "intent": INTENT_TYPE_LABELS,
    "query": QUERY_TYPE_LABELS,
    "stage": STAGE_LABELS,
    "domain": DOMAIN_SCOPE_LABELS,
    "advice_risk": ADVICE_RISK_LABELS,
}

EMOTION_LABEL_MAPPINGS = {
    "emotion": EMOTION_LABELS,
    "distress": BINARY_LABELS,
    "vulnerability": BINARY_LABELS,
    "handover": BINARY_LABELS,
}

# Confidence thresholds (default values, can be overridden via CLI)
DEFAULT_HIGH_CONFIDENCE = 0.8  # Above this = confident
DEFAULT_LOW_CONFIDENCE = 0.6   # Below this = not confident


def load_sil_model(model_dir: Path, label_type: str):
    """Load a SIL fine-tuned model and tokenizer."""
    model_path = model_dir / label_type
    
    if not model_path.exists():
        return None, None
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        
        # Set to eval mode
        model.eval()
        
        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        return model, tokenizer
    except Exception as e:
        print(f"Error loading SIL {label_type} model: {e}", file=sys.stderr)
        return None, None


def load_emotion_model(model_dir: Path, model_name: str):
    """Load an emotion fine-tuned model and tokenizer."""
    if model_name == "emotion":
        # Emotion classification model is directly in emotion/ directory
        model_path = model_dir / "emotion"
    else:
        # Other emotion models are in subdirectories
        model_path = model_dir / "emotion" / model_name
    
    if not model_path.exists():
        return None, None
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        
        # Set to eval mode
        model.eval()
        
        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        return model, tokenizer
    except Exception as e:
        print(f"Error loading emotion {model_name} model: {e}", file=sys.stderr)
        return None, None


def predict_classification(model, tokenizer, text: str, label_type: str, label_mapping: dict, is_regression: bool = False):
    """Run inference on a single text for classification or regression."""
    if model is None or tokenizer is None:
        return None, None, None, None, None
    
    start_time = time.time()
    
    # Tokenize
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    
    # Move to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    if is_regression:
        # Regression model (severity) - returns continuous value
        prediction_value = float(logits.cpu().numpy()[0][0])
        # Clamp to valid range
        prediction_value = max(0.0, min(1.0, prediction_value))
        
        # For regression, we don't have confidence in the same way
        # Use a pseudo-confidence based on how close to boundaries
        confidence = 1.0 - abs(prediction_value - 0.5) * 2  # Higher confidence if closer to 0.5 (middle)
        confidence = max(0.0, min(1.0, confidence))
        
        inference_time = (time.time() - start_time) * 1000
        
        return prediction_value, confidence, None, inference_time, None
    else:
        # Classification model
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        
        # Get top prediction
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]
        
        # Get label - label_mapping is a dict like {"emotion": [...], "distress": [...]}
        # We need to find the right key (could be label_type or the first key)
        labels = []
        if label_type in label_mapping:
            labels = label_mapping[label_type]
        elif len(label_mapping) > 0:
            # Use first available label list
            labels = list(label_mapping.values())[0]
        
        predicted_label = labels[pred_idx] if pred_idx < len(labels) else "unknown"
        
        # Get top 3 predictions (or top 2 for binary)
        num_top = min(3, len(probs))
        top3_indices = np.argsort(probs)[-num_top:][::-1]
        top3_predictions = [
            {
                "label": labels[idx] if idx < len(labels) else "unknown",
                "confidence": float(probs[idx])
            }
            for idx in top3_indices
        ]
        
        # Calculate confidence gap (difference between top and 2nd prediction)
        confidence_gap = float(confidence - top3_predictions[1]['confidence']) if len(top3_predictions) > 1 else float(confidence)
        
        inference_time = (time.time() - start_time) * 1000
        
        return predicted_label, confidence, top3_predictions, inference_time, confidence_gap


def _predict_wrapper(args):
    """Wrapper function for parallel execution."""
    model, tokenizer, text, label_type, label_mapping, is_regression, model_category = args
    result = predict_classification(model, tokenizer, text, label_type, label_mapping, is_regression)
    return (model_category, label_type), result


def categorize_by_confidence(results: dict, high_threshold: float, low_threshold: float):
    """Categorize predictions by confidence level."""
    confident = {}
    uncertain = {}
    medium = {}
    
    for key, result in results.items():
        if result is None or result[0] is None:
            continue
        
        # Handle regression results differently
        if isinstance(result[0], float):
            # Regression result (severity)
            conf = result[1] if result[1] is not None else 0.5
        else:
            # Classification result
            conf = result[1] if result[1] is not None else 0.0
        
        if conf >= high_threshold:
            confident[key] = result
        elif conf < low_threshold:
            uncertain[key] = result
        else:
            medium[key] = result
    
    return confident, medium, uncertain


def load_all_models(model_dir: Path):
    """Load all SIL and emotion models once and return them."""
    sil_models = {}
    sil_tokenizers = {}
    emotion_models = {}
    emotion_tokenizers = {}
    
    print("Loading SIL models...", file=sys.stderr)
    for label_type in ["topic", "intent", "query", "stage", "domain", "advice_risk"]:
        model, tokenizer = load_sil_model(model_dir, label_type)
        if model is not None:
            sil_models[label_type] = model
            sil_tokenizers[label_type] = tokenizer
            print(f"✅ Loaded SIL {label_type} model", file=sys.stderr)
        else:
            print(f"⚠️  SIL {label_type} model not found, skipping", file=sys.stderr)
    
    print("\nLoading emotion models...", file=sys.stderr)
    # Emotion classification model (directly in emotion/ directory)
    model, tokenizer = load_emotion_model(model_dir, "emotion")
    if model is not None:
        emotion_models["emotion"] = model
        emotion_tokenizers["emotion"] = tokenizer
        print(f"✅ Loaded emotion classification model", file=sys.stderr)
    else:
        print(f"⚠️  Emotion classification model not found, skipping", file=sys.stderr)
    
    # Other emotion models (in subdirectories)
    for model_name in ["distress", "vulnerability", "handover", "severity"]:
        model, tokenizer = load_emotion_model(model_dir, model_name)
        if model is not None:
            emotion_models[model_name] = model
            emotion_tokenizers[model_name] = tokenizer
            print(f"✅ Loaded emotion {model_name} model", file=sys.stderr)
        else:
            print(f"⚠️  Emotion {model_name} model not found, skipping", file=sys.stderr)
    
    if not sil_models and not emotion_models:
        print("\n❌ No models found! Check --model-dir path.", file=sys.stderr)
        return None, None, None, None
    
    print(f"\nDevice: {'GPU' if torch.cuda.is_available() else 'CPU'}", file=sys.stderr)
    print(f"✅ All models loaded and ready for inference\n", file=sys.stderr)
    
    return sil_models, sil_tokenizers, emotion_models, emotion_tokenizers


def test_sentence(sil_models: dict, sil_tokenizers: dict, emotion_models: dict, emotion_tokenizers: dict,
                  text: str, high_threshold: float = DEFAULT_HIGH_CONFIDENCE,
                  low_threshold: float = DEFAULT_LOW_CONFIDENCE, parallel: bool = True):
    """Test a sentence with all models and categorize by confidence."""
    print(f"\n{'='*60}")
    print(f"Testing: \"{text}\"")
    print(f"{'='*60}\n")
    
    if not sil_models and not emotion_models:
        print("❌ No models loaded!")
        return None
    
    execution_mode = "PARALLEL" if parallel else "SEQUENTIAL"
    print(f"Execution mode: {execution_mode}")
    print(f"Confidence thresholds: High ≥{high_threshold:.0%}, Low <{low_threshold:.0%}\n")
    
    # Combine all models for unified processing
    all_models = {}
    all_tokenizers = {}
    all_label_mappings = {}
    all_is_regression = {}
    all_categories = {}
    
    # Add SIL models
    for label_type in sil_models.keys():
        all_models[("sil", label_type)] = sil_models[label_type]
        all_tokenizers[("sil", label_type)] = sil_tokenizers[label_type]
        all_label_mappings[("sil", label_type)] = SIL_LABEL_MAPPINGS
        all_is_regression[("sil", label_type)] = False
        all_categories[("sil", label_type)] = "SIL"
    
    # Add emotion models
    for model_name in emotion_models.keys():
        is_regression = (model_name == "severity")
        all_models[("emotion", model_name)] = emotion_models[model_name]
        all_tokenizers[("emotion", model_name)] = emotion_tokenizers[model_name]
        # Use appropriate label mapping for each emotion model
        if model_name == "emotion":
            label_mapping = {"emotion": EMOTION_LABELS}
        elif model_name in ["distress", "vulnerability", "handover"]:
            label_mapping = {model_name: BINARY_LABELS}
        else:
            label_mapping = {}
        all_label_mappings[("emotion", model_name)] = label_mapping
        all_is_regression[("emotion", model_name)] = is_regression
        all_categories[("emotion", model_name)] = "Emotion"
    
    # Run predictions with timing
    total_start = time.time()
    results = {}
    inference_times = {}
    
    if parallel:
        # Parallel execution using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(all_models)) as executor:
            # Submit all prediction tasks
            futures = {
                executor.submit(_predict_wrapper, (
                    all_models[key],
                    all_tokenizers[key],
                    text,
                    key[1],  # label_type
                    all_label_mappings[key],
                    all_is_regression[key],
                    all_categories[key]
                )): key
                for key in all_models.keys()
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    key, result = future.result()
                    if result and result[0] is not None:
                        results[key] = result
                        if result[3] is not None:  # inference_time
                            inference_times[key] = result[3]
                except Exception as e:
                    print(f"⚠️  Error in parallel prediction: {e}", file=sys.stderr)
    else:
        # Sequential execution
        for key in all_models.keys():
            result = predict_classification(
                all_models[key],
                all_tokenizers[key],
                text,
                key[1],  # label_type
                all_label_mappings[key],
                all_is_regression[key]
            )
            
            if result and result[0] is not None:
                results[key] = result
                if result[3] is not None:  # inference_time
                    inference_times[key] = result[3]
    
    total_time = (time.time() - total_start) * 1000  # Convert to milliseconds
    
    # Categorize by confidence
    confident, medium, uncertain = categorize_by_confidence(results, high_threshold, low_threshold)
    
    # Display results organized by category
    print(f"{'='*60}")
    print("SIL PREDICTIONS:")
    print(f"{'='*60}\n")
    
    sil_results = {k: v for k, v in results.items() if k[0] == "sil"}
    for (category, label_type), result in sil_results.items():
        if result is None or result[0] is None:
            continue
        
        predicted_label, confidence, top3, inference_time, confidence_gap = result
        
        # Determine confidence category
        if (category, label_type) in confident:
            status = "✅"
        elif (category, label_type) in uncertain:
            status = "❌"
        else:
            status = "⚠️"
        
        print(f"{status} {label_type.upper()}:")
        print(f"   Prediction: {predicted_label}")
        print(f"   Confidence: {confidence:.2%}")
        if confidence_gap is not None:
            print(f"   Confidence Gap: {confidence_gap:.2%} (vs 2nd place)")
        if inference_time is not None:
            print(f"   Inference Time: {inference_time:.2f} ms")
        if top3:
            print(f"   Top 3:")
            for i, top in enumerate(top3, 1):
                marker = "👈" if i == 1 else ""
                print(f"     {i}. {top['label']}: {top['confidence']:.2%} {marker}")
        print()
    
    print(f"{'='*60}")
    print("EMOTION PREDICTIONS:")
    print(f"{'='*60}\n")
    
    emotion_results = {k: v for k, v in results.items() if k[0] == "emotion"}
    for (category, model_name), result in emotion_results.items():
        if result is None or result[0] is None:
            continue
        
        predicted_value, confidence, top3, inference_time, confidence_gap = result
        
        # Determine confidence category
        if (category, model_name) in confident:
            status = "✅"
        elif (category, model_name) in uncertain:
            status = "❌"
        else:
            status = "⚠️"
        
        print(f"{status} {model_name.upper()}:")
        if model_name == "severity":
            # Regression output
            print(f"   Predicted Severity: {predicted_value:.3f} (0.0-1.0)")
            print(f"   Confidence: {confidence:.2%}")
        else:
            # Classification output
            print(f"   Prediction: {predicted_value}")
            print(f"   Confidence: {confidence:.2%}")
            if confidence_gap is not None:
                print(f"   Confidence Gap: {confidence_gap:.2%} (vs 2nd place)")
            if top3:
                print(f"   Top 3:")
                for i, top in enumerate(top3, 1):
                    marker = "👈" if i == 1 else ""
                    print(f"     {i}. {top['label']}: {top['confidence']:.2%} {marker}")
        if inference_time is not None:
            print(f"   Inference Time: {inference_time:.2f} ms")
        print()
    
    # Confidence summary
    print(f"{'='*60}")
    print("CONFIDENCE SUMMARY:")
    print(f"{'='*60}")
    print(f"✅ Confident: {len(confident)}/{len(results)} parameters")
    if medium:
        print(f"⚠️  Medium: {len(medium)}/{len(results)} parameters")
    print(f"❌ Uncertain: {len(uncertain)}/{len(results)} parameters")
    
    if uncertain:
        print(f"\n⚠️  ATTENTION REQUIRED:")
        print(f"   The following parameters have low confidence and may need:")
        print(f"   - Clarification from user")
        print(f"   - Human review")
        print(f"   - Model improvement")
        for (category, label_type) in uncertain.keys():
            print(f"   • {category.upper()} {label_type.upper()}")
    else:
        print(f"\n✅ All predictions are confident - can proceed with routing")
    print()
    
    # Display timing summary
    print(f"{'='*60}")
    print("TIMING SUMMARY:")
    print(f"{'='*60}")
    print(f"⏱️  TOTAL TIME: {total_time:.2f} ms ({total_time/1000:.3f} seconds)")
    if inference_times:
        sum_individual = sum(inference_times.values())
        print(f"   Sum of individual times: {sum_individual:.2f} ms")
        if parallel:
            speedup = sum_individual / total_time if total_time > 0 else 1.0
            print(f"   Speedup: {speedup:.2f}x (parallel execution)")
        print(f"   Average per model: {sum_individual / len(results):.2f} ms")
        print(f"   Fastest: {min(inference_times.values()):.2f} ms ({min(inference_times, key=inference_times.get)})")
        print(f"   Slowest: {max(inference_times.values()):.2f} ms ({max(inference_times, key=inference_times.get)})")
    print()
    
    # Add confidence categorization to results
    results["_confidence_categories"] = {
        "confident": [f"{cat}_{label}" for cat, label in confident.keys()],
        "medium": [f"{cat}_{label}" for cat, label in medium.keys()],
        "uncertain": [f"{cat}_{label}" for cat, label in uncertain.keys()]
    }
    
    # Add timing to results
    results["_timing"] = {
        "total_time_ms": float(total_time),
        "total_time_seconds": float(total_time / 1000),
        "sum_individual_times_ms": float(sum(inference_times.values())) if inference_times else 0.0,
        "average_per_model_ms": float(sum(inference_times.values()) / len(results)) if inference_times and results else 0.0,
        "speedup": float(sum(inference_times.values()) / total_time) if parallel and total_time > 0 and inference_times else 1.0,
    }
    
    # Create simplified final JSON output with all 11 parameters
    # Just parameter name and selected prediction value
    final_json = {}
    
    # Add SIL and emotion predictions
    # Only process keys that are tuples (category, label_type), skip metadata keys like "_confidence_categories"
    for key, result in results.items():
        # Skip metadata keys (strings starting with "_")
        if not isinstance(key, tuple) or len(key) != 2:
            continue
        
        category, label_type = key
        if result is None or result[0] is None:
            continue
        
        if category == "sil":
            predicted_label, confidence, top3, inference_time, confidence_gap = result
            final_json[label_type] = predicted_label
        elif category == "emotion":
            predicted_value, confidence, top3, inference_time, confidence_gap = result
            if label_type == "severity":
                # Regression - keep as float
                final_json[label_type] = float(predicted_value)
            else:
                # Classification - use the predicted label/value
                final_json[label_type] = predicted_value
    
    # Display final JSON output
    print(f"{'='*60}")
    print("FINAL JSON OUTPUT:")
    print(f"{'='*60}")
    print(json.dumps(final_json, indent=2))
    print()
    
    # Return structured output
    return results


def main():
    parser = argparse.ArgumentParser(description="Test all SIL and emotion models with confidence categorization")
    parser.add_argument("--model-dir", required=True, help="Directory containing trained models (should contain SIL models and emotion/ subdirectory)")
    parser.add_argument("--text", help="Text to classify (optional, will prompt if not provided)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--high-threshold", type=float, default=DEFAULT_HIGH_CONFIDENCE,
                        help=f"High confidence threshold (default: {DEFAULT_HIGH_CONFIDENCE:.0%})")
    parser.add_argument("--low-threshold", type=float, default=DEFAULT_LOW_CONFIDENCE,
                        help=f"Low confidence threshold (default: {DEFAULT_LOW_CONFIDENCE:.0%})")
    parser.add_argument("--parallel", action="store_true", default=True, help="Run predictions in parallel (default: True)")
    parser.add_argument("--sequential", action="store_false", dest="parallel", help="Run predictions sequentially")
    
    args = parser.parse_args()
    
    # Validate thresholds
    if not (0.0 < args.low_threshold < args.high_threshold <= 1.0):
        print("Error: Thresholds must satisfy 0 < low_threshold < high_threshold ≤ 1.0", file=sys.stderr)
        sys.exit(1)
    
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Load all models once at startup
    sil_models, sil_tokenizers, emotion_models, emotion_tokenizers = load_all_models(model_dir)
    if sil_models is None and emotion_models is None:
        sys.exit(1)
    
    # Get text input
    if args.text:
        text = args.text
    else:
        # Interactive mode
        print("Enter text to classify (or 'quit' to exit):")
        text = input("> ").strip()
        if text.lower() in ['quit', 'exit', 'q']:
            return
    
    # Run prediction
    results = test_sentence(sil_models, sil_tokenizers, emotion_models, emotion_tokenizers,
                           text, args.high_threshold, args.low_threshold, parallel=args.parallel)
    
    # JSON output if requested
    if args.json and results:
        print("\n" + "="*60)
        print("JSON OUTPUT:")
        print("="*60)
        # Convert results to JSON-serializable format
        json_results = {}
        for key, value in results.items():
            # Skip metadata keys (strings starting with "_") or non-tuple keys
            if not isinstance(key, tuple) or len(key) != 2:
                if isinstance(key, str) and key.startswith("_"):
                    json_results[key] = value
                continue
            
            category, label_type = key
            if value and value[0] is not None:
                    if isinstance(value[0], float):
                        # Regression
                        json_results[f"{category}_{label_type}"] = {
                            "prediction": value[0],
                            "confidence": value[1],
                            "inference_time_ms": value[3]
                        }
                    else:
                        # Classification
                        json_results[f"{category}_{label_type}"] = {
                            "prediction": value[0],
                            "confidence": value[1],
                            "confidence_gap": value[4],
                            "top3": value[2],
                            "inference_time_ms": value[3]
                        }
        print(json.dumps(json_results, indent=2))
    
    # Interactive loop
    if not args.text:
        while True:
            print("\n" + "-"*60)
            text = input("Enter another text (or 'quit' to exit): ").strip()
            if text.lower() in ['quit', 'exit', 'q']:
                break
            if text:
                results = test_sentence(sil_models, sil_tokenizers, emotion_models, emotion_tokenizers,
                                       text, args.high_threshold, args.low_threshold, parallel=args.parallel)
                if args.json and results:
                    print("\n" + "="*60)
                    print("JSON OUTPUT:")
                    print("="*60)
                    json_results = {}
                    for key, value in results.items():
                        # Skip metadata keys (strings starting with "_") or non-tuple keys
                        if not isinstance(key, tuple) or len(key) != 2:
                            if isinstance(key, str) and key.startswith("_"):
                                json_results[key] = value
                            continue
                        
                        category, label_type = key
                        if value and value[0] is not None:
                                if isinstance(value[0], float):
                                    json_results[f"{category}_{label_type}"] = {
                                        "prediction": value[0],
                                        "confidence": value[1],
                                        "inference_time_ms": value[3]
                                    }
                                else:
                                    json_results[f"{category}_{label_type}"] = {
                                        "prediction": value[0],
                                        "confidence": value[1],
                                        "confidence_gap": value[4],
                                        "top3": value[2],
                                        "inference_time_ms": value[3]
                                    }
                    print(json.dumps(json_results, indent=2))


if __name__ == "__main__":
    main()

