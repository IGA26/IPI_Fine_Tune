#!/usr/bin/env python3
"""
generate_account_specific_datasets.py

Generate account-specific training data (bank_specific domain_scope) for SIL models.
Focuses on queries that require account access: "my balance", "my pension", "my account", etc.

These examples are saved in the same directory structure as generate_sil_stage_datasets.py
so they can be picked up during training.

Usage:
  python generate_account_specific_datasets.py \
      --output ./training_data \
      --examples 200

  # Generate for specific topics
  python generate_account_specific_datasets.py \
      --output ./training_data \
      --topics banking pensions savings \
      --examples 100
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from google.cloud import storage
from vertexai import init as vertex_init
from vertexai.generative_models import GenerativeModel

# Hardcoded defaults (matches generate_sil_stage_datasets.py)
DEFAULT_PROJECT = "playpen-c84caa"
DEFAULT_LOCATION = "us-central1"
DEFAULT_MODEL = "gemini-2.5-flash"

# Topics that support account-specific queries
ACCOUNT_SPECIFIC_TOPICS = {
    "banking": {
        "intent_types": ["account_action"],
        "query_types": ["account_action"],
        "stages": ["action"],
        "brand_hint": "Lloyds, Halifax, Bank of Scotland",
        "examples": [
            "What is my account balance?",
            "Check my balance",
            "How much is in my account?",
            "View my transactions",
            "Show my recent transactions",
            "Check my statements",
            "View my statements",
            "Transfer money from my account",
            "Pay bills from my account",
            "What is my current account balance?",
            "Show me my account details",
            "What transactions have I made?",
            "How much money do I have?",
        ]
    },
    "pensions": {
        "intent_types": ["account_action"],
        "query_types": ["account_action"],
        "stages": ["accumulation", "decumulation"],
        "brand_hint": "Lloyds pension accounts",
        "examples": [
            "What is my pension balance?",
            "How much is in my pension?",
            "What is my pension pot worth?",
            "Check my pension balance",
            "How much do I have in my pension?",
            "View my pension statement",
            "What is my pension value?",
        ]
    },
    "savings": {
        "intent_types": ["account_action"],
        "query_types": ["account_action"],
        "stages": ["accumulation"],
        "brand_hint": "Lloyds, Halifax, Bank of Scotland savings accounts",
        "examples": [
            "What is my ISA balance?",
            "How much is in my ISA?",
            "Check my savings balance",
            "How much do I have in savings?",
            "What is my savings account balance?",
            "View my ISA statement",
            "How much is in my Cash ISA?",
        ]
    },
    "investments": {
        "intent_types": ["account_action"],
        "query_types": ["account_action"],
        "stages": ["execution"],
        "brand_hint": "Lloyds investment accounts",
        "examples": [
            "What is my investment balance?",
            "How much is in my investment account?",
            "What is my portfolio worth?",
            "Check my investment value",
            "View my investment statement",
            "How much have I invested?",
        ]
    },
}

PROMPT_TEMPLATE = """You are generating account-specific training data for a financial Semantic Interface Layer (SIL).

CRITICAL: All queries MUST be account-specific (require access to user's actual account data) and MUST have domain_scope="bank_specific".

Return NEWLINE-SEPARATED JSON objects (no outer array, no markdown code blocks, plain JSON only) with this schema:

{{
  "text": "<user utterance>",
  "topic": "{topic}",
  "intent_type": "account_action",
  "query_type": "account_action",
  "stage": "<stage>",
  "domain_scope": "bank_specific",
  "advice_risk_score": <number between 0.0 and 1.0>
}}

Constraints:
- UK customer voice; FCA-compliant tone.
- CRITICAL: ALL queries must be account-specific and require access to user's actual account data
- Account-specific indicators: "my account", "my balance", "my pension", "my savings", "my investments", "my ISA", "my statements", "my transactions", "check my", "view my", "show my", "what is my", "how much is in my", "transfer from my", "pay from my"
- Topic: {topic}
- intent_type: MUST be "account_action"
- query_type: MUST be "account_action"
- stage: MUST be one of {stages}
- domain_scope: MUST be "bank_specific" (all queries require account access)
- advice_risk_score: LOW RISK (0.0-0.3) for account queries - these are factual requests for account information, not advice
- Include variations:
  * With brand: "What is my Lloyds account balance?", "Check my Halifax balance"
  * Without brand: "What is my account balance?", "Check my balance"
  * Different phrasings: "How much is in my account?", "Show me my balance", "What's my balance?"
- Cover different account types: current account, savings, ISA, pension, investment account
- Include action queries: "transfer money", "pay bills", "view statements", "check transactions"
- Avoid duplicates
- Output exactly {count} lines; each line is a COMPLETE, SINGLE-LINE JSON object
- DO NOT wrap output in markdown code blocks (```json or ```). Output plain JSON only.
- CRITICAL: All quotes in the "text" field must be properly escaped as \\" (backslash-quote)
- CRITICAL: Each JSON object must be complete on a single line. Do not split JSON objects across multiple lines.

Example format (do not reuse text):
{{"text": "What is my account balance?", "topic": "banking", "intent_type": "account_action", "query_type": "account_action", "stage": "action", "domain_scope": "bank_specific", "advice_risk_score": 0.1}}
{{"text": "What is my pension balance?", "topic": "pensions", "intent_type": "account_action", "query_type": "account_action", "stage": "accumulation", "domain_scope": "bank_specific", "advice_risk_score": 0.1}}
{{"text": "How much is in my ISA?", "topic": "savings", "intent_type": "account_action", "query_type": "account_action", "stage": "accumulation", "domain_scope": "bank_specific", "advice_risk_score": 0.1}}
{{"text": "Check my Lloyds balance", "topic": "banking", "intent_type": "account_action", "query_type": "account_action", "stage": "action", "domain_scope": "bank_specific", "advice_risk_score": 0.1}}
{{"text": "View my transactions", "topic": "banking", "intent_type": "account_action", "query_type": "account_action", "stage": "action", "domain_scope": "bank_specific", "advice_risk_score": 0.1}}
{{"text": "What is my investment portfolio worth?", "topic": "investments", "intent_type": "account_action", "query_type": "account_action", "stage": "execution", "domain_scope": "bank_specific", "advice_risk_score": 0.1}}

Generate {count} new examples now.
"""


def call_gemini(model: GenerativeModel, prompt: str, expected_count: int = 100, max_retries: int = 3) -> str:
    """Call Gemini API with retry logic (same as generate_sil_stage_datasets.py)."""
    max_tokens = max(8192, expected_count * 150)
    max_tokens = min(max_tokens, 32768)
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "top_k": 40,
                    "max_output_tokens": max_tokens,
                },
                safety_settings={},
            )
            
            text = None
            if hasattr(response, 'text') and response.text:
                text = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        text_parts = [p.text for p in candidate.content.parts if hasattr(p, 'text') and p.text]
                        if text_parts:
                            text = "".join(text_parts)
                
                if hasattr(candidate, 'finish_reason'):
                    finish_reason = candidate.finish_reason
                    if finish_reason == "MAX_TOKENS" and not text:
                        print(f"⚠️  Warning: Model hit MAX_TOKENS with no output text (attempt {attempt + 1}/{max_retries})", file=sys.stderr)
                        if attempt < max_retries - 1:
                            max_tokens = int(max_tokens * 0.7)
                            print(f"  Retrying with reduced max_output_tokens: {max_tokens}", file=sys.stderr)
                            time.sleep(2)
                            continue
                        else:
                            raise Exception(f"Model hit MAX_TOKENS without producing output text. Finish reason: {finish_reason}")
            
            if not text:
                if attempt < max_retries - 1:
                    print(f"  Retrying with smaller batch size (attempt {attempt + 1}/{max_retries})...", file=sys.stderr)
                    time.sleep(2)
                    continue
                else:
                    raise Exception("Cannot get the response text.")
            
            return text
            
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str or "quota" in error_str:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 5
                    print(f"Rate limit hit. Retrying in {wait_time} seconds... (attempt {attempt + 1}/{max_retries})", file=sys.stderr)
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Rate limit error after {max_retries} attempts: {e}", file=sys.stderr)
                    raise
            else:
                print(f"Error calling Gemini: {e}", file=sys.stderr)
                raise
    
    raise Exception("Failed to get response after retries")


def parse_examples(payload: str) -> List[Dict]:
    """Parse JSON examples from Gemini response (same logic as generate_sil_stage_datasets.py)."""
    items = []
    if "```json" in payload:
        payload = payload.split("```json")[1].split("```")[0]
    elif "```" in payload:
        payload = payload.split("```")[1].split("```")[0]
    
    current_json = ""
    brace_count = 0
    in_string = False
    escape_next = False
    
    for line_num, line in enumerate(payload.strip().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        
        current_json += line
        
        for char in line:
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
        
        if brace_count == 0 and current_json:
            try:
                parsed = json.loads(current_json)
                items.append(parsed)
                current_json = ""
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON at line {line_num}: {e}", file=sys.stderr)
                current_json = ""
    
    if current_json.strip():
        missing_braces = brace_count
        if missing_braces > 0:
            if current_json.strip().startswith("{") and '"text"' in current_json:
                try:
                    fixed_json = current_json.strip()
                    if fixed_json.count('"') % 2 != 0:
                        last_quote_idx = fixed_json.rfind('"')
                        if last_quote_idx > 0:
                            before_quote = fixed_json[:last_quote_idx]
                            if before_quote.count('"') % 2 == 0:
                                fixed_json += '"'
                    fixed_json += "}" * missing_braces
                    parsed = json.loads(fixed_json)
                    items.append(parsed)
                    print(f"Note: Recovered incomplete JSON by adding {missing_braces} closing brace(s)", file=sys.stderr)
                except json.JSONDecodeError:
                    print(f"Note: Could not recover incomplete JSON (1 example lost - this is normal)", file=sys.stderr)
    
    return items


def validate_example(example: Dict, topic: str, config: Dict):
    """Validate example matches expected schema."""
    required = {"text", "topic", "intent_type", "query_type", "stage", "domain_scope", "advice_risk_score"}
    missing = required - example.keys()
    if missing:
        raise ValueError(f"Missing fields {missing} in {example}")
    if example["topic"] != topic:
        raise ValueError(f"Topic mismatch: {example}")
    if example["intent_type"] not in config["intent_types"]:
        raise ValueError(f"Invalid intent_type: {example}")
    if example["query_type"] not in config["query_types"]:
        raise ValueError(f"Invalid query_type: {example}")
    if example["stage"] not in config["stages"]:
        raise ValueError(f"Invalid stage: {example}")
    if example["domain_scope"] != "bank_specific":
        raise ValueError(f"domain_scope must be 'bank_specific' for account queries: {example}")
    if not (0.0 <= example["advice_risk_score"] <= 1.0):
        raise ValueError(f"advice_risk_score must be between 0.0 and 1.0: {example}")
    if not example["text"] or len(example["text"]) < 8:
        raise ValueError(f"Utterance too short: {example}")
    
    # Check that it's actually account-specific
    text_lower = example["text"].lower()
    account_indicators = ["my account", "my balance", "my pension", "my savings", "my investments", "my isa", 
                          "my statements", "my transactions", "check my", "view my", "show my", 
                          "what is my", "how much is in my", "transfer from my", "pay from my"]
    if not any(indicator in text_lower for indicator in account_indicators):
        raise ValueError(f"Query doesn't appear to be account-specific: {example['text']}")


def save_json(data: Dict, target: Path, gcs_client: Optional[storage.Client]):
    """Save JSON data to file or GCS."""
    if str(target).startswith("gs://"):
        bucket_name, *blob_parts = str(target).replace("gs://", "").split("/")
        blob_name = "/".join(blob_parts)
        bucket = gcs_client.bucket(bucket_name)
        bucket.blob(blob_name).upload_from_string(json.dumps(data, indent=2), content_type="application/json")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[saved] {target}")


def generate_batch(
    topic: str,
    config: Dict,
    model: GenerativeModel,
    model_name: str,
    count: int,
) -> Dict:
    """Generate account-specific examples for a topic."""
    prompt = PROMPT_TEMPLATE.format(
        topic=topic,
        stages=config["stages"],
        count=count,
    )
    
    raw = call_gemini(model, prompt, expected_count=count)
    examples = parse_examples(raw)
    
    unique = []
    seen = set()
    for ex in examples:
        try:
            validate_example(ex, topic, config)
            key = ex["text"].strip().lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(ex)
        except ValueError as e:
            print(f"Warning: Skipping invalid example: {e}", file=sys.stderr)
            continue
    
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "topic": topic,
        "schema": "sil_intent_v1",
        "fields": ["text", "topic", "intent_type", "query_type", "stage", "domain_scope", "advice_risk_score"],
        "requested_examples": count,
        "total_examples": len(unique),
        "model_name": model_name,
        "focus": "account_specific_queries",
        "domain_scope": "bank_specific",
    }
    return {"metadata": metadata, "training_data": unique}


def main():
    parser = argparse.ArgumentParser(description="Generate account-specific training data")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help=f"GCP project ID (default: {DEFAULT_PROJECT})")
    parser.add_argument("--location", default=DEFAULT_LOCATION, help=f"Vertex AI location (default: {DEFAULT_LOCATION})")
    parser.add_argument("--output", required=True, help="gs://bucket/path or local directory")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--topics", nargs="+", choices=list(ACCOUNT_SPECIFIC_TOPICS.keys()), 
                        default=list(ACCOUNT_SPECIFIC_TOPICS.keys()),
                        help="Topics to generate (default: all)")
    parser.add_argument("--examples", type=int, default=200, help="Examples per topic")
    
    args = parser.parse_args()
    
    # Ensure output directory exists (for local paths)
    if not args.output.startswith("gs://"):
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir.absolute()}", file=sys.stderr)
    
    vertex_init(project=args.project, location=args.location)
    model = GenerativeModel(args.model)
    
    gcs_client = storage.Client(project=args.project) if args.output.startswith("gs://") else None
    
    print(f"Generating account-specific examples for topics: {', '.join(args.topics)}", file=sys.stderr)
    print(f"Examples per topic: {args.examples}", file=sys.stderr)
    print(f"All examples will have domain_scope='bank_specific'", file=sys.stderr)
    
    for topic in args.topics:
        config = ACCOUNT_SPECIFIC_TOPICS[topic]
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Generating {args.examples} examples for topic '{topic}'...", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        
        dataset = generate_batch(
            topic=topic,
            config=config,
            model=model,
            model_name=args.model,
            count=args.examples,
        )
        
        if not dataset or "training_data" not in dataset:
            print(f"⚠️  Warning: Failed to generate dataset for {topic}", file=sys.stderr)
            continue
        
        if len(dataset["training_data"]) == 0:
            print(f"⚠️  Warning: Generated 0 examples for {topic}. Skipping.", file=sys.stderr)
            continue
        
        # Save in same structure as generate_sil_stage_datasets.py
        # Format: {output}/{topic}/all_stages/{topic}_all_stages_training_data_{timestamp}.json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{topic}_all_stages_account_specific_{timestamp}.json"
        output_path = Path(args.output) / topic / "all_stages" / filename
        
        print(f"Saving {len(dataset['training_data'])} examples to {output_path}...", file=sys.stderr)
        try:
            save_json(dataset, output_path, gcs_client)
            print(f"✅ Successfully saved {len(dataset['training_data'])} account-specific examples for {topic}", file=sys.stderr)
        except Exception as e:
            print(f"❌ Error saving file: {e}", file=sys.stderr)
            continue
        
        # Small delay between topics
        if topic != args.topics[-1]:
            time.sleep(2)
    
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"✅ Account-specific data generation complete!", file=sys.stderr)
    print(f"All files saved in: {args.output}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

