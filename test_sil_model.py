#!/usr/bin/env python3
"""
test_sil_model.py

Test the fine-tuned SIL classification models with a sentence.

Usage:
  python test_sil_model.py \
      --model-dir ./models \
      --text "What is an ISA?"
  
  # Interactive mode
  python test_sil_model.py --model-dir ./models
"""

import argparse
import json
import sys
import time
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
        return None, None, None
    
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
    
    inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    return predicted_label, confidence, top3_predictions, inference_time


def test_sentence(model_dir: Path, text: str):
    """Test a sentence with all models."""
    print(f"\n{'='*60}")
    print(f"Testing: \"{text}\"")
    print(f"{'='*60}\n")
    
    # Load all models
    models = {}
    tokenizers = {}
    
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
        return
    
    print(f"\nDevice: {'GPU' if torch.cuda.is_available() else 'CPU'}\n")
    
    # Run predictions with timing
    total_start = time.time()
    results = {}
    inference_times = {}
    
    for label_type in models.keys():
        predicted_label, confidence, top3, inference_time = predict(
            models[label_type],
            tokenizers[label_type],
            text,
            label_type
        )
        
        if predicted_label:
            results[label_type] = {
                "prediction": predicted_label,
                "confidence": float(confidence),
                "top3": top3,
                "inference_time_ms": inference_time
            }
            inference_times[label_type] = inference_time
    
    total_time = (time.time() - total_start) * 1000  # Convert to milliseconds
    
    # Display results
    print(f"{'='*60}")
    print("PREDICTIONS:")
    print(f"{'='*60}\n")
    
    for label_type, result in results.items():
        print(f"{label_type.upper()}:")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Inference Time: {result['inference_time_ms']:.2f} ms")
        print(f"  Top 3:")
        for i, top in enumerate(result['top3'], 1):
            print(f"    {i}. {top['label']}: {top['confidence']:.2%}")
        print()
    
    # Display timing summary
    print(f"{'='*60}")
    print("TIMING SUMMARY:")
    print(f"{'='*60}")
    print(f"Total inference time: {total_time:.2f} ms")
    print(f"Average per model: {total_time / len(results):.2f} ms")
    if inference_times:
        print(f"Fastest: {min(inference_times.values()):.2f} ms ({min(inference_times, key=inference_times.get)})")
        print(f"Slowest: {max(inference_times.values()):.2f} ms ({max(inference_times, key=inference_times.get)})")
    print()
    
    # Return structured output
    return results


def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned SIL models")
    parser.add_argument("--model-dir", required=True, help="Directory containing trained models")
    parser.add_argument("--text", help="Text to classify (optional, will prompt if not provided)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}", file=sys.stderr)
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
    results = test_sentence(model_dir, text)
    
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
                results = test_sentence(model_dir, text)
                if args.json and results:
                    print("\n" + "="*60)
                    print("JSON OUTPUT:")
                    print("="*60)
                    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

