#!/usr/bin/env python3
"""
test_sil_model_confidence.py

Test the fine-tuned SIL classification models with a sentence.
Categorizes predictions by confidence level to identify which parameters
are certain vs uncertain.

Usage:
  python test_sil_model_confidence.py \
      --model-dir ./models \
      --text "What is an ISA?"
  
  # Interactive mode
  python test_sil_model_confidence.py --model-dir ./models
  
  # Custom confidence thresholds
  python test_sil_model_confidence.py --model-dir ./models --text "..." --high-threshold 0.85 --low-threshold 0.55
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


# Label mappings (must match training script)
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

LABEL_MAPPINGS = {
    "topic": TOPIC_LABELS,
    "intent": INTENT_TYPE_LABELS,
    "query": QUERY_TYPE_LABELS,
    "stage": STAGE_LABELS,
    "domain": DOMAIN_SCOPE_LABELS,
    "advice_risk": ADVICE_RISK_LABELS,
}

# Confidence thresholds (default values, can be overridden via CLI)
DEFAULT_HIGH_CONFIDENCE = 0.8  # Above this = confident
DEFAULT_LOW_CONFIDENCE = 0.6   # Below this = not confident


def load_model(model_dir: Path, label_type: str):
    """Load a fine-tuned model and tokenizer."""
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
        print(f"Error loading {label_type} model: {e}", file=sys.stderr)
        return None, None


def predict(model, tokenizer, text: str, label_type: str):
    """Run inference on a single text."""
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
    
    # Get probabilities
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    
    # Get top prediction
    pred_idx = np.argmax(probs)
    confidence = probs[pred_idx]
    
    # Get label
    labels = LABEL_MAPPINGS[label_type]
    predicted_label = labels[pred_idx] if pred_idx < len(labels) else "unknown"
    
    # Get top 3 predictions
    top3_indices = np.argsort(probs)[-3:][::-1]
    top3_predictions = [
        {
            "label": labels[idx] if idx < len(labels) else "unknown",
            "confidence": float(probs[idx])
        }
        for idx in top3_indices
    ]
    
    # Calculate confidence gap (difference between top and 2nd prediction)
    confidence_gap = float(confidence - top3_predictions[1]['confidence']) if len(top3_predictions) > 1 else float(confidence)
    
    inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    return predicted_label, confidence, top3_predictions, inference_time, confidence_gap


def _predict_wrapper(args):
    """Wrapper function for parallel execution."""
    model, tokenizer, text, label_type = args
    return label_type, predict(model, tokenizer, text, label_type)


def categorize_by_confidence(results: dict, high_threshold: float, low_threshold: float):
    """Categorize predictions by confidence level."""
    confident = {}
    uncertain = {}
    medium = {}
    
    for label_type, result in results.items():
        conf = result['confidence']
        if conf >= high_threshold:
            confident[label_type] = result
        elif conf < low_threshold:
            uncertain[label_type] = result
        else:
            medium[label_type] = result
    
    return confident, medium, uncertain


def load_all_models(model_dir: Path):
    """Load all models once and return them."""
    models = {}
    tokenizers = {}
    
    print("Loading models...")
    for label_type in ["topic", "intent", "query", "stage", "domain", "advice_risk"]:
        model, tokenizer = load_model(model_dir, label_type)
        if model is not None:
            models[label_type] = model
            tokenizers[label_type] = tokenizer
            print(f"✅ Loaded {label_type} model")
        else:
            print(f"⚠️  {label_type} model not found, skipping")
    
    if not models:
        print("\n❌ No models found! Check --model-dir path.")
        return None, None
    
    print(f"\nDevice: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"✅ All models loaded and ready for inference\n")
    
    return models, tokenizers


def test_sentence(models: dict, tokenizers: dict, text: str, high_threshold: float = DEFAULT_HIGH_CONFIDENCE, 
                  low_threshold: float = DEFAULT_LOW_CONFIDENCE, parallel: bool = True):
    """Test a sentence with pre-loaded models and categorize by confidence."""
    print(f"\n{'='*60}")
    print(f"Testing: \"{text}\"")
    print(f"{'='*60}\n")
    
    if not models or not tokenizers:
        print("❌ Models not loaded!")
        return None
    
    execution_mode = "PARALLEL" if parallel else "SEQUENTIAL"
    print(f"Execution mode: {execution_mode}")
    print(f"Confidence thresholds: High ≥{high_threshold:.0%}, Low <{low_threshold:.0%}\n")
    
    # Run predictions with timing
    total_start = time.time()
    results = {}
    inference_times = {}
    
    if parallel:
        # Parallel execution using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            # Submit all prediction tasks
            futures = {
                executor.submit(_predict_wrapper, (models[label_type], tokenizers[label_type], text, label_type)): label_type
                for label_type in models.keys()
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    label_type, result = future.result()
                    if result and result[0] is not None:  # predicted_label is not None
                        predicted_label, confidence, top3, inference_time, confidence_gap = result
                        results[label_type] = {
                            "prediction": predicted_label,
                            "confidence": float(confidence),
                            "confidence_gap": float(confidence_gap),
                            "top3": top3,
                            "inference_time_ms": inference_time
                        }
                        inference_times[label_type] = inference_time
                except Exception as e:
                    print(f"⚠️  Error in parallel prediction: {e}", file=sys.stderr)
    else:
        # Sequential execution (original behavior)
        for label_type in models.keys():
            predicted_label, confidence, top3, inference_time, confidence_gap = predict(
                models[label_type],
                tokenizers[label_type],
                text,
                label_type
            )
            
            if predicted_label:
                results[label_type] = {
                    "prediction": predicted_label,
                    "confidence": float(confidence),
                    "confidence_gap": float(confidence_gap),
                    "top3": top3,
                    "inference_time_ms": inference_time
                }
                inference_times[label_type] = inference_time
    
    total_time = (time.time() - total_start) * 1000  # Convert to milliseconds
    
    # Categorize by confidence
    confident, medium, uncertain = categorize_by_confidence(results, high_threshold, low_threshold)
    
    # Display CONFIDENT predictions
    print(f"{'='*60}")
    print("✅ CONFIDENT PREDICTIONS (High Confidence ≥{:.0%}):".format(high_threshold))
    print(f"{'='*60}\n")
    
    if confident:
        for label_type, result in confident.items():
            print(f"✅ {label_type.upper()}:")
            print(f"   Prediction: {result['prediction']}")
            print(f"   Confidence: {result['confidence']:.2%}")
            print(f"   Confidence Gap: {result['confidence_gap']:.2%} (vs 2nd place)")
            print(f"   Inference Time: {result['inference_time_ms']:.2f} ms")
            print()
    else:
        print("   (None)\n")
    
    # Display MEDIUM confidence predictions
    if medium:
        print(f"{'='*60}")
        print("⚠️  MEDIUM CONFIDENCE ({:.0%} ≤ confidence < {:.0%}):".format(low_threshold, high_threshold))
        print(f"{'='*60}\n")
        
        for label_type, result in medium.items():
            print(f"⚠️  {label_type.upper()}:")
            print(f"   Prediction: {result['prediction']}")
            print(f"   Confidence: {result['confidence']:.2%}")
            print(f"   Confidence Gap: {result['confidence_gap']:.2%} (vs 2nd place)")
            print(f"   Top 3:")
            for i, top in enumerate(result['top3'], 1):
                marker = "👈" if i == 1 else ""
                print(f"     {i}. {top['label']}: {top['confidence']:.2%} {marker}")
            print()
    
    # Display UNCERTAIN predictions
    print(f"{'='*60}")
    print("❌ UNCERTAIN PREDICTIONS (Low Confidence <{:.0%}):".format(low_threshold))
    print(f"{'='*60}\n")
    
    if uncertain:
        for label_type, result in uncertain.items():
            print(f"❌ {label_type.upper()}:")
            print(f"   Prediction: {result['prediction']}")
            print(f"   Confidence: {result['confidence']:.2%}")
            print(f"   Confidence Gap: {result['confidence_gap']:.2%} (vs 2nd place)")
            print(f"   ⚠️  WARNING: Low confidence - may need clarification")
            print(f"   Top 3:")
            for i, top in enumerate(result['top3'], 1):
                marker = "👈" if i == 1 else ""
                print(f"     {i}. {top['label']}: {top['confidence']:.2%} {marker}")
            print()
    else:
        print("   (None - all predictions are confident!)\n")
    
    # Summary
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
        for label_type in uncertain.keys():
            print(f"   • {label_type.upper()}")
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
        "confident": list(confident.keys()),
        "medium": list(medium.keys()),
        "uncertain": list(uncertain.keys())
    }
    
    # Return structured output
    return results


def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned SIL models with confidence categorization")
    parser.add_argument("--model-dir", required=True, help="Directory containing trained models")
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
    models, tokenizers = load_all_models(model_dir)
    if models is None:
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
    results = test_sentence(models, tokenizers, text, args.high_threshold, args.low_threshold, parallel=args.parallel)
    
    # JSON output if requested
    if args.json and results:
        print("\n" + "="*60)
        print("JSON OUTPUT:")
        print("="*60)
        print(json.dumps(results, indent=2))
    
    # Interactive loop
    if not args.text:
        while True:
            print("\n" + "-"*60)
            text = input("Enter another text (or 'quit' to exit): ").strip()
            if text.lower() in ['quit', 'exit', 'q']:
                break
            if text:
                results = test_sentence(models, tokenizers, text, args.high_threshold, args.low_threshold, parallel=args.parallel)
                if args.json and results:
                    print("\n" + "="*60)
                    print("JSON OUTPUT:")
                    print("="*60)
                    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

