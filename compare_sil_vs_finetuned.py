#!/usr/bin/env python3
"""
compare_sil_vs_finetuned.py

Compare SIL prompt output vs fine-tuned model output side-by-side.
No ground truth required - just shows what each method predicts.

Usage:
  python compare_sil_vs_finetuned.py \
      --model-dir ./models \
      --questions ./test_questions_50.json \
      --output-dir ./comparison_results

  # Project, location, and gemini model are hardcoded (default: playpen-c84caa, us-central1, gemini-2.5-flash)
  # Override with --project, --location, --gemini-model if needed
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from google.cloud import storage
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vertexai import init as vertex_init
from vertexai.generative_models import GenerativeModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('sil_vs_finetuned_comparison.log')
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


def load_models(model_dir: str, gcs_client: Optional[storage.Client] = None) -> Dict:
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
    """Call Vertex AI Gemini with SIL prompt."""
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


def compare_questions(
    questions: List[Dict],
    models: Dict,
    tokenizers: Dict,
    gemini_model: GenerativeModel,
    label_mappings: Dict
) -> List[Dict]:
    """Compare SIL vs fine-tuned for each question."""
    
    results = []
    fine_tuned_times = []
    sil_prompt_times = []
    
    for idx, question_data in enumerate(questions):
        text = question_data.get("text", "").strip()
        if not text:
            continue
        
        logger.info(f"Processing question {idx + 1}/{len(questions)}: {text[:60]}...")
        
        # Fine-tuned prediction
        start_time = time.time()
        finetuned_pred = predict_finetuned(text, models, tokenizers, label_mappings)
        fine_tuned_time = (time.time() - start_time) * 1000  # ms
        
        # SIL prompt baseline
        start_time = time.time()
        sil_pred = call_sil_prompt(text, gemini_model)
        sil_prompt_time = (time.time() - start_time) * 1000  # ms
        
        if sil_pred is None:
            logger.warning(f"Skipping question {idx + 1} due to SIL prompt error")
            continue
        
        fine_tuned_times.append(fine_tuned_time)
        sil_prompt_times.append(sil_prompt_time)
        
        # Extract predictions
        comparison = {
            "question": text,
            "expected_topic": question_data.get("topic"),  # Optional hint from input
            "finetuned": {
                "topic": finetuned_pred.get("topic", {}).get("label"),
                "intent": finetuned_pred.get("intent", {}).get("label"),
                "query": finetuned_pred.get("query", {}).get("label"),
                "stage": finetuned_pred.get("stage", {}).get("label"),
                "domain": finetuned_pred.get("domain", {}).get("label"),
                "advice_risk": finetuned_pred.get("advice_risk", {}).get("label"),
            },
            "sil_prompt": {
                "topic": sil_pred.get("topic"),
                "intent": sil_pred.get("intent"),
                "query": sil_pred.get("query"),
                "stage": sil_pred.get("stage"),
                "domain": sil_pred.get("domain"),
                "advice_risk": sil_pred.get("advice_risk"),
            },
            "match": {},
            "finetuned_time_ms": fine_tuned_time,
            "sil_prompt_time_ms": sil_prompt_time
        }
        
        # Check if predictions match
        for label_type in ["topic", "intent", "query", "stage", "domain", "advice_risk"]:
            ft = comparison["finetuned"].get(label_type)
            sil = comparison["sil_prompt"].get(label_type)
            comparison["match"][label_type] = (ft == sil) if (ft and sil) else None
        
        results.append(comparison)
        
        # Log match status
        matches = sum(1 for m in comparison["match"].values() if m is True)
        total_labels = len([m for m in comparison["match"].values() if m is not None])
        logger.info(f"  → Matches: {matches}/{total_labels} labels | "
                   f"Fine-tuned: {fine_tuned_time:.2f}ms | SIL: {sil_prompt_time:.2f}ms")
    
    # Add summary statistics
    summary = {
        "total_questions": len(results),
        "timing": {
            "finetuned": {
                "mean_ms": float(np.mean(fine_tuned_times)) if fine_tuned_times else 0,
                "median_ms": float(np.median(fine_tuned_times)) if fine_tuned_times else 0,
                "total_ms": float(np.sum(fine_tuned_times)) if fine_tuned_times else 0
            },
            "sil_prompt": {
                "mean_ms": float(np.mean(sil_prompt_times)) if sil_prompt_times else 0,
                "median_ms": float(np.median(sil_prompt_times)) if sil_prompt_times else 0,
                "total_ms": float(np.sum(sil_prompt_times)) if sil_prompt_times else 0
            }
        }
    }
    
    return results, summary


def save_results(results: List[Dict], summary: Dict, output_dir: str, gcs_client: Optional[storage.Client] = None):
    """Save comparison results."""
    is_gcs = output_dir.startswith("gs://")
    
    output_data = {
        "summary": summary,
        "comparisons": results
    }
    
    if is_gcs:
        bucket_name, *blob_parts = output_dir.replace("gs://", "").split("/")
        bucket = gcs_client.bucket(bucket_name)
        prefix = "/".join(blob_parts) if blob_parts else ""
        
        blob = bucket.blob(f"{prefix}/sil_vs_finetuned_comparison.json")
        blob.upload_from_string(
            json.dumps(output_data, indent=2),
            content_type="application/json"
        )
        logger.info(f"Results saved to gs://{bucket_name}/{prefix}/sil_vs_finetuned_comparison.json")
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / "sil_vs_finetuned_comparison.json", 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to {output_path}/sil_vs_finetuned_comparison.json")


def print_summary(results: List[Dict], summary: Dict):
    """Print summary report."""
    print("\n" + "=" * 80)
    print("SIL PROMPT vs FINE-TUNED MODELS COMPARISON")
    print("=" * 80)
    
    print(f"\nTotal questions: {summary['total_questions']}")
    
    # Count matches per label type
    label_matches = {}
    for label_type in ["topic", "intent", "query", "stage", "domain", "advice_risk"]:
        matches = sum(1 for r in results if r["match"].get(label_type) is True)
        total = sum(1 for r in results if r["match"].get(label_type) is not None)
        label_matches[label_type] = {"matches": matches, "total": total, "agreement": matches/total if total > 0 else 0}
    
    print("\n" + "-" * 80)
    print("AGREEMENT (how often both methods predict the same):")
    print("-" * 80)
    print(f"{'Label Type':<20} {'Agreement':<15} {'Matches/Total':<20}")
    print("-" * 80)
    for label_type, stats in label_matches.items():
        print(f"{label_type:<20} {stats['agreement']:<15.2%} {stats['matches']}/{stats['total']}")
    
    print("\n" + "-" * 80)
    print("TIMING COMPARISON:")
    print("-" * 80)
    ft_timing = summary["timing"]["finetuned"]
    sil_timing = summary["timing"]["sil_prompt"]
    
    print(f"Fine-tuned: {ft_timing['mean_ms']:.2f}ms avg (total: {ft_timing['total_ms']/1000:.2f}s)")
    print(f"SIL Prompt: {sil_timing['mean_ms']:.2f}ms avg (total: {sil_timing['total_ms']/1000:.2f}s)")
    
    speedup = sil_timing['mean_ms'] / ft_timing['mean_ms'] if ft_timing['mean_ms'] > 0 else 0
    print(f"\nSpeedup: {speedup:.2f}x faster (fine-tuned vs SIL prompt)")
    
    print("\n" + "=" * 80)
    
    # Show disagreements
    print("\nQUESTIONS WHERE PREDICTIONS DIFFER:")
    print("-" * 80)
    disagreements = [r for r in results if not all(r["match"].values())]
    
    if disagreements:
        for idx, r in enumerate(disagreements[:10], 1):  # Show first 10
            print(f"\n{idx}. {r['question']}")
            print(f"   Topic:    Fine-tuned={r['finetuned']['topic']:<15} SIL={r['sil_prompt']['topic']}")
            print(f"   Intent:   Fine-tuned={r['finetuned']['intent']:<15} SIL={r['sil_prompt']['intent']}")
            print(f"   Stage:    Fine-tuned={r['finetuned']['stage']:<15} SIL={r['sil_prompt']['stage']}")
    else:
        print("No disagreements - both methods predicted the same for all questions!")


def load_questions(questions_path: str, gcs_client: Optional[storage.Client] = None) -> List[Dict]:
    """Load questions from JSON file."""
    if questions_path.startswith("gs://"):
        local_path = "/tmp/questions.json"
        load_from_gcs(questions_path, local_path, gcs_client)
        questions_path = local_path
    
    questions_file = Path(questions_path)
    if not questions_file.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")
    
    with open(questions_file, 'r') as f:
        data = json.load(f)
    
    # Handle different formats
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "questions" in data:
        return data["questions"]
    else:
        raise ValueError(f"Unknown questions format in {questions_path}")


def main():
    # Hardcoded defaults (same as generate_sil_stage_datasets.py)
    DEFAULT_PROJECT = "playpen-c84caa"
    DEFAULT_LOCATION = "us-central1"
    DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
    
    parser = argparse.ArgumentParser(description="Compare SIL prompt vs fine-tuned models")
    parser.add_argument("--model-dir", required=True, help="Directory with fine-tuned models (local or gs://)")
    parser.add_argument("--questions", required=True, help="JSON file with questions (local or gs://)")
    parser.add_argument("--output-dir", default="./comparison_results", help="Output directory (local or gs://)")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help=f"GCP project ID (default: {DEFAULT_PROJECT})")
    parser.add_argument("--location", default=DEFAULT_LOCATION, help=f"Vertex AI location (default: {DEFAULT_LOCATION})")
    parser.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL, help=f"Gemini model name (default: {DEFAULT_GEMINI_MODEL})")
    
    args = parser.parse_args()
    
    # Initialize Vertex AI
    vertex_init(project=args.project, location=args.location)
    logger.info(f"Vertex AI initialized: project={args.project}, location={args.location}")
    
    # Initialize Gemini model
    gemini_model = GenerativeModel(args.gemini_model)
    logger.info(f"Gemini model initialized: {args.gemini_model}")
    
    # Initialize GCS client if needed
    gcs_client = None
    if args.model_dir.startswith("gs://") or args.questions.startswith("gs://") or args.output_dir.startswith("gs://"):
        gcs_client = storage.Client(project=args.project)
        logger.info(f"GCS client initialized for project: {args.project}")
    
    # Load models
    logger.info(f"Loading fine-tuned models from {args.model_dir}")
    models, tokenizers = load_models(args.model_dir, gcs_client)
    
    if not models:
        raise ValueError("No models loaded. Check model-dir path.")
    
    logger.info(f"Loaded {len(models)} models: {list(models.keys())}")
    
    # Load questions
    logger.info(f"Loading questions from {args.questions}")
    questions = load_questions(args.questions, gcs_client)
    logger.info(f"Loaded {len(questions)} questions")
    
    # Run comparison
    logger.info("Starting comparison...")
    start_total = time.time()
    results, summary = compare_questions(
        questions,
        models,
        tokenizers,
        gemini_model,
        LABEL_MAPPINGS
    )
    total_time = time.time() - start_total
    
    logger.info(f"\nTotal comparison time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    
    # Print summary
    print_summary(results, summary)
    
    # Save results
    save_results(results, summary, args.output_dir, gcs_client)
    
    logger.info("\nComparison complete!")


if __name__ == "__main__":
    main()

