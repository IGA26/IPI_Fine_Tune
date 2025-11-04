#!/usr/bin/env python3
"""
compare_finetuned_vs_sil_prompt.py

Compare fine-tuned SIL models against Vertex AI Gemini SIL prompt baseline.
Runs on remote Vertex AI instance.

Usage:
  python compare_finetuned_vs_sil_prompt.py \
      --model-dir ./models \
      --test-data ./test_data.json \
      --output-dir ./comparison_results \
      --project playpen-c84caa \
      --location us-central1

  # Or with GCS paths
  python compare_finetuned_vs_sil_prompt.py \
      --model-dir gs://bucket/sil-models \
      --test-data gs://bucket/test_data.json \
      --output-dir gs://bucket/comparison_results \
      --project playpen-c84caa \
      --location us-central1
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from google.cloud import storage
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vertexai import init as vertex_init
from vertexai.generative_models import GenerativeModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('comparison.log')
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

LABEL_MAPPINGS = {
    "topic": TOPIC_LABELS,
    "intent": INTENT_TYPE_LABELS,
    "query": QUERY_TYPE_LABELS,
    "stage": STAGE_LABELS,
    "domain": DOMAIN_SCOPE_LABELS,
    "advice_risk": ADVICE_RISK_LABELS,
}


def load_from_gcs(gcs_path: str, local_path: str, gcs_client: Optional[storage.Client]):
    """Download file from GCS if needed."""
    if not gcs_path.startswith("gs://"):
        return Path(gcs_path) if Path(gcs_path).exists() else None
    
    if not gcs_client:
        raise ValueError("GCS client required for gs:// paths")
    
    bucket_name, *blob_parts = gcs_path.replace("gs://", "").split("/")
    blob_name = "/".join(blob_parts)
    
    local_file = Path(local_path)
    local_file.parent.mkdir(parents=True, exist_ok=True)
    
    bucket = gcs_client.bucket(bucket_name)
    bucket.blob(blob_name).download_to_filename(local_file)
    logger.info(f"Downloaded {gcs_path} to {local_file}")
    return local_file


def load_models(model_dir: str, gcs_client: Optional[storage.Client] = None) -> Dict[str, Tuple]:
    """Load all fine-tuned models."""
    models = {}
    tokenizers = {}
    
    # Check if GCS path
    is_gcs = model_dir.startswith("gs://")
    
    if is_gcs:
        # Download models to local temp directory
        local_model_dir = Path("/tmp/finetuned_models")
        local_model_dir.mkdir(parents=True, exist_ok=True)
        
        bucket_name, *blob_parts = model_dir.replace("gs://", "").split("/")
        bucket = gcs_client.bucket(bucket_name)
        prefix = "/".join(blob_parts) if blob_parts else ""
        
        # List and download all model directories
        for label_type in ["topic", "intent", "query", "stage", "domain", "advice_risk"]:
            blob_path = f"{prefix}/{label_type}" if prefix else label_type
            blobs = list(bucket.list_blobs(prefix=blob_path))
            if blobs:
                local_type_dir = local_model_dir / label_type
                local_type_dir.mkdir(parents=True, exist_ok=True)
                for blob in blobs:
                    local_path = local_model_dir / blob.name.replace(prefix + "/", "")
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    blob.download_to_filename(local_path)
        
        model_dir = str(local_model_dir)
    
    model_path = Path(model_dir)
    
    for label_type in ["topic", "intent", "query", "stage", "domain", "advice_risk"]:
        type_dir = model_path / label_type
        if not type_dir.exists():
            logger.warning(f"Model directory not found: {type_dir}")
            continue
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(type_dir))
            model = AutoModelForSequenceClassification.from_pretrained(str(type_dir))
            model.eval()
            
            models[label_type] = model
            tokenizers[label_type] = tokenizer
            logger.info(f"Loaded {label_type} model from {type_dir}")
        except Exception as e:
            logger.error(f"Failed to load {label_type} model: {e}")
    
    return models, tokenizers


def predict_finetuned(text: str, models: Dict, tokenizers: Dict, label_mappings: Dict) -> Dict:
    """Run inference with fine-tuned models."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    predictions = {}
    
    for label_type, model in models.items():
        tokenizer = tokenizers[label_type]
        
        # Tokenize
        inputs = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        ).to(device)
        
        model = model.to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_idx = torch.argmax(logits, dim=1).item()
            confidence = torch.softmax(logits, dim=1)[0][predicted_idx].item()
        
        # Map to label
        labels = label_mappings[label_type]
        if predicted_idx < len(labels):
            predicted_label = labels[predicted_idx]
        else:
            predicted_label = labels[0]  # Fallback
        
        predictions[label_type] = {
            "label": predicted_label,
            "confidence": confidence,
            "index": predicted_idx
        }
    
    return predictions


def call_sil_prompt(text: str, gemini_model: GenerativeModel) -> Dict:
    """Call Vertex AI Gemini with SIL prompt (same as production)."""
    # This is the SIL prompt from openai_sil_service.py, adapted for Gemini
    sil_prompt = f"""You are a financial services intent classification system. Analyze the user query and extract structured information.

CRITICAL VALIDATION RULES:
- ALL ISA types (Cash ISA, Stocks & Shares ISA, etc.) = savings topic (NEVER investments)
- Inheritance money questions = savings topic
- Empty/null stages are FORBIDDEN - always provide a valid stage
- Advice requests ("I lost my job", "should I") = advice_seeking (not account_action)
- How-to questions = fact_seeking (not account_action)
- High advice risk (0.7+) = explicit advice requests ("should I", "recommend", "help me decide")
- Low advice risk (0.0-0.3) = factual questions ("what is", "how does", "explain")

EXTRACT THE FOLLOWING FIELDS:

1. INTENT_TYPE: string
   - fact_seeking: "what is", "how does", "explain", "tell me about"
   - advice_seeking: "should I", "what should I", "recommend", "help me decide"
   - account_action: "open account", "apply for", "transfer money", "close account"
   - guidance: general financial guidance
   - goal_expression: "I want to save for", "saving for house"
   - off_topic: non-financial queries

2. TOPIC: string
   - savings, investments, pensions, mortgages, banking, loans, debt, insurance, taxation, general, off_topic

3. STAGE: string
   - For savings: goal_setup, accumulation, understanding, optimisation, withdrawal
   - For investments: goal_definition, execution
   - For pensions: enrolment, accumulation, decumulation
   - For mortgages: application, repayment, remortgage
   - For banking: awareness, action
   - For loans: planning, execution
   - For debt: management
   - For insurance: planning, claim
   - For taxation: planning
   - For general: planning, understanding
   - For off_topic: awareness

4. QUERY_TYPE: string
   - what_is: definitional questions
   - eligibility: "am I eligible", "can I qualify"
   - recommendation: seeking specific product/action advice
   - account_action: account operations
   - goal_expression: expressing goals
   - comparison: "compare", "difference between", "which is better"

5. DOMAIN_SCOPE: string
   - general: broad financial information, educational content
   - bank_specific: requires access to user's account data OR specific bank's product details

6. ADVICE_RISK_SCORE: number (0.0 to 1.0)
   - LOW (0.0-0.3): Factual questions ("what is", "how does", "explain")
   - MEDIUM (0.4-0.6): Comparative questions, suitability queries
   - HIGH (0.7-1.0): Explicit advice ("should I", "recommend", "help me decide")

RESPONSE FORMAT (JSON only, no markdown, no code blocks):
{{
  "intent_type": "<intent_type>",
  "topic": "<topic>", 
  "stage": "<stage>",
  "query_type": "<query_type>",
  "domain_scope": "<domain_scope>",
  "advice_risk_score": <number>
}}

User query: "{text}"

Return ONLY the JSON object, no markdown, no explanations."""
    try:
        response = gemini_model.generate_content(
            sil_prompt,
            generation_config={
                "temperature": 0.0,
                "max_output_tokens": 8192,
            },
            safety_settings={},
        )
        
        # Extract text from response
        result_text = None
        if hasattr(response, 'text') and response.text:
            result_text = response.text
        elif hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    text_parts = [p.text for p in candidate.content.parts if hasattr(p, 'text') and p.text]
                    if text_parts:
                        result_text = "".join(text_parts)
        
        if not result_text:
            logger.error("Empty response from Gemini")
            return None
        
        # Clean JSON (remove markdown code blocks if present)
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]
        
        result_text = result_text.strip()
        
        # Parse JSON
        result = json.loads(result_text)
        
        # Map advice_risk_score to bucket
        advice_risk_score = result.get("advice_risk_score", 0.0)
        if isinstance(advice_risk_score, (int, float)):
            if advice_risk_score <= 0.3:
                advice_risk_bucket = "low"
            elif advice_risk_score <= 0.6:
                advice_risk_bucket = "medium"
            else:
                advice_risk_bucket = "high"
        else:
            advice_risk_bucket = "low"  # Fallback
        
        return {
            "topic": result.get("topic", "general"),
            "intent": result.get("intent_type", "fact_seeking"),
            "query": result.get("query_type", "what_is"),
            "stage": result.get("stage", "planning"),
            "domain": result.get("domain_scope", "general"),
            "advice_risk": advice_risk_bucket,
            "raw": result
        }
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error from Gemini: {e}")
        logger.error(f"Response text: {result_text[:500] if 'result_text' in locals() else 'N/A'}")
        return None
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return None


def encode_label(value: str, label_type: str, label_mappings: Dict) -> int:
    """Encode label to index."""
    labels = label_mappings[label_type]
    try:
        return labels.index(value)
    except ValueError:
        return 0  # Unknown/fallback


def compare_predictions(
    test_data: List[Dict],
    models: Dict,
    tokenizers: Dict,
    gemini_model: GenerativeModel,
    label_mappings: Dict
) -> Dict:
    """Compare fine-tuned models vs SIL prompt baseline."""
    
    results = {
        "total_examples": len(test_data),
        "comparisons": [],
        "metrics": {},
        "agreement": {},
        "disagreements": defaultdict(list)
    }
    
    fine_tuned_times = []
    sil_prompt_times = []
    
    for idx, example in enumerate(test_data):
        text = example.get("text", "").strip()
        if not text:
            continue
        
        # Ground truth
        ground_truth = {
            "topic": example.get("topic"),
            "intent": example.get("intent_type"),
            "query": example.get("query_type"),
            "stage": example.get("stage"),
            "domain": example.get("domain_scope"),
            "advice_risk": None  # Need to bucketize
        }
        
        # Bucketize advice_risk_score
        advice_risk_score = example.get("advice_risk_score", 0.0)
        if advice_risk_score <= 0.3:
            ground_truth["advice_risk"] = "low"
        elif advice_risk_score <= 0.6:
            ground_truth["advice_risk"] = "medium"
        else:
            ground_truth["advice_risk"] = "high"
        
        # Fine-tuned prediction
        start_time = time.time()
        finetuned_pred = predict_finetuned(text, models, tokenizers, label_mappings)
        fine_tuned_time = (time.time() - start_time) * 1000  # ms
        
        # SIL prompt baseline (using Vertex AI Gemini)
        start_time = time.time()
        sil_pred = call_sil_prompt(text, gemini_model)
        sil_prompt_time = (time.time() - start_time) * 1000  # ms
        
        if sil_pred is None:
            continue
        
        fine_tuned_times.append(fine_tuned_time)
        sil_prompt_times.append(sil_prompt_time)
        
        # Log timing for each example
        logger.debug(f"Example {idx + 1}: Fine-tuned={fine_tuned_time:.2f}ms, SIL prompt={sil_prompt_time:.2f}ms")
        
        # Compare
        comparison = {
            "text": text,
            "ground_truth": ground_truth,
            "finetuned": {
                "topic": finetuned_pred.get("topic", {}).get("label"),
                "intent": finetuned_pred.get("intent", {}).get("label"),
                "query": finetuned_pred.get("query", {}).get("label"),
                "stage": finetuned_pred.get("stage", {}).get("label"),
                "domain": finetuned_pred.get("domain", {}).get("label"),
                "advice_risk": finetuned_pred.get("advice_risk", {}).get("label"),
            },
            "sil_prompt": sil_pred,
            "match": {},
            "finetuned_time_ms": fine_tuned_time,
            "sil_prompt_time_ms": sil_prompt_time
        }
        
        # Check matches
        for label_type in ["topic", "intent", "query", "stage", "domain", "advice_risk"]:
            gt = ground_truth.get(label_type)
            ft = comparison["finetuned"].get(label_type)
            sil = comparison["sil_prompt"].get(label_type)
            
            match_ft = (ft == gt) if gt else None
            match_sil = (sil == gt) if gt else None
            
            comparison["match"][label_type] = {
                "finetuned": match_ft,
                "sil_prompt": match_sil,
                "both_match": (match_ft and match_sil) if (match_ft and match_sil) else False,
                "disagree": (ft != sil) if (ft and sil) else None
            }
            
            # Track disagreements
            if ft != sil:
                results["disagreements"][label_type].append({
                    "text": text[:100],
                    "finetuned": ft,
                    "sil_prompt": sil,
                    "ground_truth": gt
                })
        
        results["comparisons"].append(comparison)
        
        if (idx + 1) % 10 == 0:
            # Show running averages
            avg_ft = np.mean(fine_tuned_times) if fine_tuned_times else 0
            avg_sil = np.mean(sil_prompt_times) if sil_prompt_times else 0
            logger.info(
                f"Processed {idx + 1}/{len(test_data)} examples | "
                f"Fine-tuned avg: {avg_ft:.2f}ms | SIL prompt avg: {avg_sil:.2f}ms"
            )
    
    # Calculate metrics
    for label_type in ["topic", "intent", "query", "stage", "domain", "advice_risk"]:
        gt_labels = []
        ft_labels = []
        sil_labels = []
        
        for comp in results["comparisons"]:
            gt = comp["ground_truth"].get(label_type)
            ft = comp["finetuned"].get(label_type)
            sil = comp["sil_prompt"].get(label_type)
            
            if gt:
                gt_labels.append(gt)
                ft_labels.append(ft)
                sil_labels.append(sil)
        
        if gt_labels:
            # Accuracy
            ft_accuracy = accuracy_score(gt_labels, ft_labels)
            sil_accuracy = accuracy_score(gt_labels, sil_labels)
            
            # F1 scores
            ft_f1_macro = f1_score(gt_labels, ft_labels, average='macro', zero_division=0)
            sil_f1_macro = f1_score(gt_labels, sil_labels, average='macro', zero_division=0)
            
            # Agreement
            agreement = sum(1 for ft, sil in zip(ft_labels, sil_labels) if ft == sil) / len(ft_labels)
            
            results["metrics"][label_type] = {
                "finetuned_accuracy": ft_accuracy,
                "sil_prompt_accuracy": sil_accuracy,
                "finetuned_f1_macro": ft_f1_macro,
                "sil_prompt_f1_macro": sil_f1_macro,
                "agreement": agreement,
                "improvement": ft_accuracy - sil_accuracy
            }
            
            results["agreement"][label_type] = agreement
    
    # Timing statistics
    if fine_tuned_times and sil_prompt_times:
        results["timing"] = {
            "finetuned": {
                "mean_ms": float(np.mean(fine_tuned_times)),
                "median_ms": float(np.median(fine_tuned_times)),
                "min_ms": float(np.min(fine_tuned_times)),
                "max_ms": float(np.max(fine_tuned_times)),
                "std_ms": float(np.std(fine_tuned_times)),
                "total_ms": float(np.sum(fine_tuned_times))
            },
            "sil_prompt": {
                "mean_ms": float(np.mean(sil_prompt_times)),
                "median_ms": float(np.median(sil_prompt_times)),
                "min_ms": float(np.min(sil_prompt_times)),
                "max_ms": float(np.max(sil_prompt_times)),
                "std_ms": float(np.std(sil_prompt_times)),
                "total_ms": float(np.sum(sil_prompt_times))
            },
            "speedup": {
                "ratio": float(np.mean(sil_prompt_times) / np.mean(fine_tuned_times)) if np.mean(fine_tuned_times) > 0 else 0,
                "faster_by_ms": float(np.mean(sil_prompt_times) - np.mean(fine_tuned_times))
            }
        }
    else:
        results["timing"] = {
            "finetuned": {},
            "sil_prompt": {}
        }
    
    return results


def save_results(results: Dict, output_dir: str, gcs_client: Optional[storage.Client] = None):
    """Save comparison results."""
    is_gcs = output_dir.startswith("gs://")
    
    if is_gcs:
        bucket_name, *blob_parts = output_dir.replace("gs://", "").split("/")
        bucket = gcs_client.bucket(bucket_name)
        prefix = "/".join(blob_parts) if blob_parts else ""
        
        # Save summary
        summary_blob = bucket.blob(f"{prefix}/comparison_summary.json")
        summary_blob.upload_from_string(
            json.dumps(results, indent=2),
            content_type="application/json"
        )
        
        # Save detailed comparisons (first 100 to avoid huge files)
        detailed = results["comparisons"][:100]
        detailed_blob = bucket.blob(f"{prefix}/detailed_comparisons.json")
        detailed_blob.upload_from_string(
            json.dumps(detailed, indent=2),
            content_type="application/json"
        )
        
        logger.info(f"Results saved to gs://{bucket_name}/{prefix}/")
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        with open(output_path / "comparison_summary.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save detailed comparisons (first 100)
        detailed = results["comparisons"][:100]
        with open(output_path / "detailed_comparisons.json", 'w') as f:
            json.dump(detailed, f, indent=2)
        
        logger.info(f"Results saved to {output_path}/")


def print_summary(results: Dict):
    """Print summary report."""
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY: Fine-tuned Models vs SIL Prompt Baseline")
    print("=" * 80)
    
    print(f"\nTotal examples tested: {results['total_examples']}")
    print(f"Valid comparisons: {len(results['comparisons'])}")
    
    print("\n" + "-" * 80)
    print("ACCURACY COMPARISON:")
    print("-" * 80)
    print(f"{'Label Type':<20} {'Fine-tuned':<15} {'SIL Prompt':<15} {'Improvement':<15} {'Agreement':<15}")
    print("-" * 80)
    
    for label_type, metrics in results["metrics"].items():
        ft_acc = metrics["finetuned_accuracy"]
        sil_acc = metrics["sil_prompt_accuracy"]
        improvement = metrics["improvement"]
        agreement = metrics["agreement"]
        
        print(f"{label_type:<20} {ft_acc:<15.3f} {sil_acc:<15.3f} {improvement:+.3f} {'':<10} {agreement:<15.3f}")
    
    print("\n" + "-" * 80)
    print("F1 SCORE COMPARISON (Macro):")
    print("-" * 80)
    print(f"{'Label Type':<20} {'Fine-tuned':<15} {'SIL Prompt':<15}")
    print("-" * 80)
    
    for label_type, metrics in results["metrics"].items():
        ft_f1 = metrics["finetuned_f1_macro"]
        sil_f1 = metrics["sil_prompt_f1_macro"]
        print(f"{label_type:<20} {ft_f1:<15.3f} {sil_f1:<15.3f}")
    
    print("\n" + "-" * 80)
    print("TIMING COMPARISON:")
    print("-" * 80)
    print(f"{'Method':<20} {'Mean (ms)':<15} {'Median (ms)':<15} {'Min (ms)':<15} {'Max (ms)':<15} {'Std (ms)':<15} {'Total (s)':<15}")
    print("-" * 80)
    
    if "finetuned" in results["timing"] and results["timing"]["finetuned"]:
        ft_timing = results["timing"]["finetuned"]
        sil_timing = results["timing"]["sil_prompt"]
        
        print(f"{'Fine-tuned':<20} {ft_timing.get('mean_ms', 0):<15.2f} {ft_timing.get('median_ms', 0):<15.2f} {ft_timing.get('min_ms', 0):<15.2f} {ft_timing.get('max_ms', 0):<15.2f} {ft_timing.get('std_ms', 0):<15.2f} {ft_timing.get('total_ms', 0)/1000:<15.2f}")
        print(f"{'SIL Prompt':<20} {sil_timing.get('mean_ms', 0):<15.2f} {sil_timing.get('median_ms', 0):<15.2f} {sil_timing.get('min_ms', 0):<15.2f} {sil_timing.get('max_ms', 0):<15.2f} {sil_timing.get('std_ms', 0):<15.2f} {sil_timing.get('total_ms', 0)/1000:<15.2f}")
        
        if "speedup" in results["timing"]:
            speedup = results["timing"]["speedup"]
            print(f"\n{'Speedup':<20} {speedup.get('ratio', 0):<15.2f}x {'Faster by':<15} {speedup.get('faster_by_ms', 0):<15.2f}ms")
            if speedup.get("faster_by_ms", 0) > 0:
                print(f"  → Fine-tuned models are {speedup.get('faster_by_ms', 0):.2f}ms faster on average")
            else:
                print(f"  → SIL prompt is {abs(speedup.get('faster_by_ms', 0)):.2f}ms faster on average")
    
    print("\n" + "-" * 80)
    print("DISAGREEMENTS (where fine-tuned and SIL prompt differ):")
    print("-" * 80)
    
    for label_type, disagreements in results["disagreements"].items():
        print(f"\n{label_type}: {len(disagreements)} disagreements")
        if disagreements:
            print("  Sample disagreements:")
            for d in disagreements[:3]:
                print(f"    Text: {d['text']}")
                print(f"      Fine-tuned: {d['finetuned']}")
                print(f"      SIL Prompt: {d['sil_prompt']}")
                print(f"      Ground Truth: {d['ground_truth']}")
                print()
    
    print("\n" + "=" * 80)


def load_test_data(test_data_path: str, gcs_client: Optional[storage.Client] = None) -> List[Dict]:
    """Load test data from file or GCS."""
    if test_data_path.startswith("gs://"):
        local_path = "/tmp/test_data.json"
        load_from_gcs(test_data_path, local_path, gcs_client)
        test_data_path = local_path
    
    test_path = Path(test_data_path)
    if not test_path.exists():
        raise FileNotFoundError(f"Test data file not found: {test_data_path}")
    
    with open(test_path, 'r') as f:
        data = json.load(f)
    
    # Handle different formats
    if isinstance(data, list):
        return data
    elif "training_data" in data:
        return data["training_data"]
    elif "test_data" in data:
        return data["test_data"]
    else:
        raise ValueError(f"Unknown test data format in {test_data_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare fine-tuned models vs SIL prompt baseline")
    parser.add_argument("--model-dir", required=True, help="Directory with fine-tuned models (local or gs://)")
    parser.add_argument("--test-data", required=True, help="Test data JSON file (local or gs://)")
    parser.add_argument("--output-dir", default="./comparison_results", help="Output directory (local or gs://)")
    parser.add_argument("--project", default="playpen-c84caa", help="GCP project ID (for Vertex AI and GCS)")
    parser.add_argument("--location", default="us-central1", help="Vertex AI location")
    parser.add_argument("--gemini-model", default="gemini-2.5-flash", help="Gemini model name")
    parser.add_argument("--max-examples", type=int, help="Limit number of test examples")
    
    args = parser.parse_args()
    
    # Initialize Vertex AI
    vertex_init(project=args.project, location=args.location)
    logger.info(f"Vertex AI initialized: project={args.project}, location={args.location}")
    
    # Initialize Gemini model
    gemini_model = GenerativeModel(args.gemini_model)
    logger.info(f"Gemini model initialized: {args.gemini_model}")
    
    # Initialize GCS client if needed
    gcs_client = None
    if args.model_dir.startswith("gs://") or args.test_data.startswith("gs://") or args.output_dir.startswith("gs://"):
        gcs_client = storage.Client(project=args.project)
        logger.info(f"GCS client initialized for project: {args.project}")
    
    # Load models
    logger.info(f"Loading fine-tuned models from {args.model_dir}")
    models, tokenizers = load_models(args.model_dir, gcs_client)
    
    if not models:
        raise ValueError("No models loaded. Check model-dir path.")
    
    logger.info(f"Loaded {len(models)} models: {list(models.keys())}")
    
    # Load test data
    logger.info(f"Loading test data from {args.test_data}")
    test_data = load_test_data(args.test_data, gcs_client)
    
    if args.max_examples:
        test_data = test_data[:args.max_examples]
        logger.info(f"Limited to {args.max_examples} examples")
    
    logger.info(f"Loaded {len(test_data)} test examples")
    
    # Run comparison
    logger.info("Starting comparison...")
    logger.info(f"Will process {len(test_data)} examples")
    logger.info("Timing will be tracked for both fine-tuned models and SIL prompt baseline")
    
    start_total = time.time()
    results = compare_predictions(
        test_data,
        models,
        tokenizers,
        gemini_model,
        LABEL_MAPPINGS
    )
    total_time = time.time() - start_total
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Total comparison time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    if "timing" in results and results["timing"].get("finetuned"):
        ft_total = results["timing"]["finetuned"].get("total_ms", 0) / 1000
        sil_total = results["timing"]["sil_prompt"].get("total_ms", 0) / 1000
        logger.info(f"  Fine-tuned inference: {ft_total:.2f}s")
        logger.info(f"  SIL prompt inference: {sil_total:.2f}s")
        logger.info(f"  Overhead (loading, comparison, etc.): {total_time - ft_total - sil_total:.2f}s")
    logger.info(f"{'='*60}\n")
    
    # Print summary
    print_summary(results)
    
    # Save results
    save_results(results, args.output_dir, gcs_client)
    
    logger.info("Comparison complete!")


if __name__ == "__main__":
    main()

