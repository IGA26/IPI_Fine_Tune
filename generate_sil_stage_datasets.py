#!/usr/bin/env python3
"""
generate_sil_stage_datasets.py

Generate SIL-labelled training datasets using Vertex AI Gemini.
Stages match taxonomy_financial_enhanced.yaml exactly for fine-tuning.

Usage:
  python generate_sil_stage_datasets.py \
      --output ./training_data \
      --topic savings --examples 40

  # Stage-specific
  python generate_sil_stage_datasets.py \
      --output ./training_data \
      --topic savings --stage optimisation --examples 20
  
  # Project and location are hardcoded (default: playpen-c84ca, us-central1)
  # Override with --project and --location if needed
  v2.0
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


@dataclass
class TopicConfig:
    intent_types: List[str]
    query_types: List[str]
    stages: List[str]
    domain_scopes: List[str]
    brand_hint: str
    products: List[str] = None  # Optional: specific products for bank_specific queries


# Stages extracted from taxonomy_financial_enhanced.yaml
TOPIC_MATRIX: Dict[str, TopicConfig] = {
    "savings": TopicConfig(
        intent_types=["fact_seeking", "advice_seeking", "account_action", "goal_expression", "guidance"],
        query_types=["what_is", "eligibility", "recommendation", "account_action", "goal_expression"],
        stages=["goal_setup", "accumulation", "understanding", "optimisation", "withdrawal"],
        domain_scopes=["general", "bank_specific"],
        brand_hint="Lloyds Banking Group (Lloyds, Halifax, Bank of Scotland)",
        products=["Cash ISA", "Stocks & Shares ISA", "Lifetime ISA", "Junior ISA", "Regular Savings Account", "Easy Access Savings", "Fixed Rate Savings", "Notice Savings Account"]
    ),
    "investments": TopicConfig(
        intent_types=["fact_seeking", "advice_seeking", "account_action", "guidance"],
        query_types=["what_is", "comparison", "recommendation", "account_action"],
        stages=["goal_definition", "execution"],
        domain_scopes=["general", "bank_specific"],
        brand_hint="Lloyds Banking Group investment services",
        products=["Investment ISA", "Stocks & Shares ISA", "Investment Account", "SIPP", "Investment Funds", "ETFs", "Unit Trusts", "OEICs", "Investment Platform", "Shares", "Stocks", "Bonds"]
    ),
    "pensions": TopicConfig(
        intent_types=["fact_seeking", "advice_seeking", "account_action"],
        query_types=["what_is", "eligibility", "recommendation", "account_action"],
        stages=["enrolment", "accumulation", "decumulation"],
        domain_scopes=["general", "bank_specific"],
        brand_hint="Lloyds pension advice and account servicing",
        products=["Workplace Pension", "Personal Pension", "SIPP", "Self-Invested Personal Pension", "State Pension", "Defined Benefit Pension", "Defined Contribution Pension", "Pension Transfer", "Pension Drawdown", "Annuity", "Pension Pot"]
    ),
    "mortgages": TopicConfig(
        intent_types=["fact_seeking", "advice_seeking", "account_action"],
        query_types=["what_is", "eligibility", "recommendation", "account_action"],
        stages=["application", "repayment", "remortgage"],
        domain_scopes=["general", "bank_specific"],
        brand_hint="Lloyds, Halifax, Bank of Scotland mortgage services",
        products=["Fixed Rate Mortgage", "Tracker Mortgage", "Offset Mortgage", "Variable Rate Mortgage", "Buy-to-Let Mortgage", "First-Time Buyer Mortgage", "Help to Buy Mortgage", "Shared Ownership Mortgage", "Remortgage", "Mortgage Overpayment", "Mortgage Payment Holiday", "Mortgage Porting"]
    ),
    "banking": TopicConfig(
        intent_types=["fact_seeking", "account_action"],
        query_types=["what_is", "account_action"],
        stages=["awareness", "action"],
        domain_scopes=["general", "bank_specific"],
        brand_hint="Retail and digital banking with Lloyds Banking Group",
        products=["Current Account", "Debit Card", "Credit Card", "Mobile Banking", "Online Banking", "Mobile App", "Banking App", "Standing Order", "Direct Debit", "Bank Statement", "Account Balance", "Money Transfer", "Bill Payment", "ATM", "Overdraft", "Branch Services"]
    ),
    "loans": TopicConfig(
        intent_types=["fact_seeking", "advice_seeking", "account_action"],
        query_types=["what_is", "eligibility", "recommendation", "account_action"],
        stages=["planning", "execution"],
        domain_scopes=["general", "bank_specific"],
        brand_hint="Personal loans from Lloyds Banking Group brands",
        products=["Personal Loan", "Secured Loan", "Unsecured Loan", "Debt Consolidation Loan", "Car Loan", "Home Improvement Loan", "Wedding Loan", "Holiday Loan", "Student Loan", "Loan Calculator", "Loan Repayment", "Early Repayment", "Loan Application", "Loan Approval", "Loan Interest Rate", "Fixed Rate Loan", "Variable Rate Loan"]
    ),
    "debt": TopicConfig(
        intent_types=["fact_seeking", "advice_seeking"],
        query_types=["what_is", "recommendation"],
        stages=["management"],
        domain_scopes=["general"],
        brand_hint="Debt management support (LBG and UK advice services)"
    ),
    "insurance": TopicConfig(
        intent_types=["fact_seeking", "advice_seeking", "account_action"],
        query_types=["what_is", "recommendation", "eligibility", "account_action"],
        stages=["planning", "claim"],
        domain_scopes=["general", "bank_specific"],
        brand_hint="Lloyds Banking Group insurance products",
        products=["Life Insurance", "Home Insurance", "Buildings Insurance", "Contents Insurance", "Travel Insurance", "Car Insurance", "Motor Insurance", "Pet Insurance", "Critical Illness Cover", "Income Protection Insurance", "Mobile Phone Insurance", "Landlord Insurance", "Buy-to-Let Insurance", "Insurance Claim", "Insurance Policy", "Insurance Premium", "Insurance Renewal", "Insurance Quote", "Insurance Coverage"]
    ),
    "taxation": TopicConfig(
        intent_types=["fact_seeking", "advice_seeking"],
        query_types=["what_is", "recommendation"],
        stages=["planning"],
        domain_scopes=["general"],
        brand_hint="Tax efficiency and ISA allowance questions"
    ),
    "general": TopicConfig(
        intent_types=["fact_seeking", "advice_seeking", "guidance"],
        query_types=["what_is", "recommendation"],
        stages=["planning", "understanding"],
        domain_scopes=["general"],
        brand_hint="Overall financial wellbeing guidance"
    ),
    "off_topic": TopicConfig(
        intent_types=["off_topic"],
        query_types=["what_is"],
        stages=["awareness"],
        domain_scopes=["general"],
        brand_hint="Non-financial or irrelevant queries"
    ),
}

PROMPT_TEMPLATE = """You are generating labelled training data for a financial Semantic Interface Layer (SIL).

CRITICAL: Generate output directly without excessive internal reasoning. Focus on producing the JSON examples immediately.

Return NEWLINE-SEPARATED JSON objects (no outer array, no markdown code blocks, plain JSON only) with this schema:

{{
  "text": "<user utterance>",
  "topic": "{topic}",
  "intent_type": "<intent>",
  "query_type": "<query>",
  "stage": "<stage>",
  "domain_scope": "<general or bank_specific>",
  "advice_risk_score": <number between 0.0 and 1.0>
}}

Constraints:
- UK customer voice; FCA-compliant tone.
- CRITICAL: Generate 70-80% general queries and 20-30% bank_specific queries. Most users start with general questions ("what is an ISA?") before asking about specific banks.
- Include both general questions and {brand_hint} scenarios when domain_scope is "bank_specific".
- When domain_scope is "bank_specific", include realistic queries like:
  * Product questions: "Does [brand] offer [product]?", "What is [brand] [product] rate?", "[brand] [product] features"
  * Eligibility: "Am I eligible for [brand] [product]?", "[brand] account requirements"
  * Account actions: "Check my [brand] balance", "Transfer from my [brand] account", "View my [brand] statements"
  * Comparisons: "[brand] vs [competitor] [product]", "Compare [brand] products"
  * Service queries: "[brand] branch near me", "[brand] opening hours", "[brand] customer service"
{products_hint}
- CRITICAL: Ensure good distribution across products when domain_scope is "bank_specific". Don't repeat the same product too many times - vary the products used across bank_specific examples.
- Allowable values:
  * intent_type ∈ {intent_types}
  * query_type ∈ {query_types}
  * stage ∈ {stages}
  * domain_scope ∈ {domain_scopes}
  * advice_risk_score: number (0.0 to 1.0)
    - LOW RISK (0.0-0.3): Factual questions ("what is", "how does", "explain", "tell me about"), general information ("types of", "benefits of", "eligibility"), educational content
    - MEDIUM RISK (0.4-0.6): Comparative questions ("which is better", "pros and cons", "compare"), suitability queries ("right for me", "suitable for my situation"), time-based questions ("when should", "is now good")
    - HIGH RISK (0.7-1.0): Explicit advice ("should I", "what should I", "recommend", "advise me"), decision help ("help me decide", "help me choose"), personal recommendations ("which to buy", "where to invest", "how much"), opinion seeking ("do you think", "is it smart to")
- Every utterance must logically align with the labels.
- advice_risk_score must match the intent_type: fact_seeking = low (0.0-0.3), advice_seeking = high (0.7-1.0), guidance = medium (0.4-0.6).
- IMPORTANT: Balance the distribution across all label types:
  * CRITICAL: Distribute stage values evenly across {stages}. If there are 2 stages, aim for roughly 50/50 split (e.g., 200 examples per stage for 400 total). If there are 3+ stages, distribute as evenly as possible.
  * Distribute intent_type values roughly evenly across {intent_types}
  * Distribute query_type values roughly evenly across {query_types}
  * CRITICAL: domain_scope MUST be 70-80% general and 20-30% bank_specific. Most users ask general questions first. Only use bank_specific when the query explicitly mentions a brand name (Lloyds, Halifax, Bank of Scotland) or asks about specific bank products. Generate approximately 3-4 general queries for every 1 bank_specific query.
  * Distribute advice_risk_score roughly evenly across low (0.0-0.3), medium (0.4-0.6), and high (0.7-1.0) ranges
- Cover informational asks, advice requests, goal statements, and account actions as permitted.
- For banking topic with account_action intent/query_type, include realistic account queries like: "check my balance", "what is my account balance", "view my transactions", "transfer money", "pay bills", "check my statements", etc.
- Avoid duplicates.
- Output exactly {count} lines; each line is a COMPLETE, SINGLE-LINE JSON object (no line breaks within a JSON object).
- DO NOT wrap output in markdown code blocks (```json or ```). Output plain JSON only.
- CRITICAL: All quotes in the "text" field must be properly escaped as \\" (backslash-quote). Avoid using quotes in text when possible. Ensure all JSON strings are properly closed.
- CRITICAL: Each JSON object must be complete on a single line. Do not split JSON objects across multiple lines.

Example format (do not reuse text):
{{"text": "How much can I contribute to a Cash ISA this tax year?", "topic": "savings", "intent_type": "fact_seeking", "query_type": "what_is", "stage": "understanding", "domain_scope": "general", "advice_risk_score": 0.2}}

Generate {count} new examples now.
"""


def call_gemini(model: GenerativeModel, prompt: str, expected_count: int = 40, max_retries: int = 3) -> str:
    # Calculate max_output_tokens based on expected count (roughly 100 chars per example)
    # Add buffer for safety
    max_tokens = max(8192, expected_count * 150)  # At least 8K, or 150 tokens per example
    
    # Cap at reasonable limit (Gemini has limits)
    max_tokens = min(max_tokens, 32768)  # Gemini 2.5 Flash max
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,  # Lower for structured output compliance
                    "top_p": 0.9,
                    "top_k": 40,
                    "max_output_tokens": max_tokens,
                },
                safety_settings={},
            )
            
            # Extract text from response
            text = None
            if hasattr(response, 'text') and response.text:
                text = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        # Check if parts have text
                        text_parts = [p.text for p in candidate.content.parts if hasattr(p, 'text') and p.text]
                        if text_parts:
                            text = "".join(text_parts)
                
                # Check finish reason
                if hasattr(candidate, 'finish_reason'):
                    finish_reason = candidate.finish_reason
                    if finish_reason == "MAX_TOKENS" and not text:
                        # Model hit token limit without producing output
                        # This happens when model uses all tokens on "thoughts"
                        print(f"⚠️  Warning: Model hit MAX_TOKENS with no output text (attempt {attempt + 1}/{max_retries})", file=sys.stderr)
                        if attempt < max_retries - 1:
                            # Reduce max_output_tokens and retry
                            max_tokens = int(max_tokens * 0.7)  # Reduce by 30%
                            print(f"  Retrying with reduced max_output_tokens: {max_tokens}", file=sys.stderr)
                            time.sleep(2)
                            continue
                        else:
                            raise Exception(f"Model hit MAX_TOKENS without producing output text. Finish reason: {finish_reason}")
            
            if not text:
                # Check if we have any response at all
                if hasattr(response, 'usage_metadata'):
                    usage = response.usage_metadata
                    print(f"⚠️  Warning: Empty response. Usage: {usage}", file=sys.stderr)
                
                if attempt < max_retries - 1:
                    print(f"  Retrying with smaller batch size (attempt {attempt + 1}/{max_retries})...", file=sys.stderr)
                    time.sleep(2)
                    continue
                else:
                    raise Exception("Cannot get the response text. Response candidate content has no parts (and thus no text).")
            
            return text
            
        except Exception as e:
            error_str = str(e).lower()
            # Check for rate limit errors
            if "rate limit" in error_str or "429" in error_str or "quota" in error_str:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 5  # Exponential backoff: 5s, 10s, 20s
                    print(f"Rate limit hit. Retrying in {wait_time} seconds... (attempt {attempt + 1}/{max_retries})", file=sys.stderr)
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Rate limit error after {max_retries} attempts: {e}", file=sys.stderr)
                    raise
            else:
                # Check if it's the empty response error - retry with smaller tokens
                if "cannot get" in error_str or "no parts" in error_str or "max_tokens" in error_str:
                    if attempt < max_retries - 1:
                        max_tokens = int(max_tokens * 0.7)  # Reduce tokens
                        print(f"Empty response error. Retrying with max_output_tokens={max_tokens} (attempt {attempt + 1}/{max_retries})...", file=sys.stderr)
                        time.sleep(2)
                        continue
                # Non-rate-limit error, raise immediately
                print(f"Error calling Gemini: {e}", file=sys.stderr)
                raise
    
    raise Exception("Failed to get response after retries")


def parse_examples(payload: str) -> List[Dict]:
    items = []
    # Remove markdown code blocks if present
    if "```json" in payload:
        payload = payload.split("```json")[1].split("```")[0]
    elif "```" in payload:
        payload = payload.split("```")[1].split("```")[0]
    
    # Try to handle multi-line JSON objects by accumulating lines
    current_json = ""
    brace_count = 0
    in_string = False
    escape_next = False
    
    for line_num, line in enumerate(payload.strip().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        # Skip markdown code block markers
        if line.startswith("```"):
            continue
        
        # Accumulate lines for multi-line JSON
        current_json += line
        
        # Check if we have a complete JSON object
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
        
        # If braces are balanced, try to parse
        if brace_count == 0 and current_json:
            try:
                parsed = json.loads(current_json)
                items.append(parsed)
                current_json = ""
            except json.JSONDecodeError as e:
                # Show more context for debugging
                error_pos = getattr(e, 'pos', None)
                if error_pos:
                    start = max(0, error_pos - 50)
                    end = min(len(current_json), error_pos + 50)
                    context = current_json[start:end]
                    print(f"Warning: Invalid JSON ending at line {line_num}, position {error_pos}:", file=sys.stderr)
                    print(f"  Context: ...{context}...", file=sys.stderr)
                    print(f"  Full JSON: {current_json[:300]}...", file=sys.stderr)
                else:
                    print(f"Warning: Invalid JSON ending at line {line_num}: {current_json[:200]}...", file=sys.stderr)
                print(f"  Error: {e}", file=sys.stderr)
                current_json = ""  # Reset and continue
    
    # Handle any remaining incomplete JSON - try to salvage it
    if current_json.strip():
        # Try to fix incomplete JSON by adding missing closing braces
        missing_braces = brace_count
        if missing_braces > 0:
            # Check if we have a valid structure that just needs closing
            if current_json.strip().startswith("{") and '"text"' in current_json:
                # Try to complete it
                try:
                    # Add missing closing braces and quotes if needed
                    fixed_json = current_json.strip()
                    # Close any unclosed strings (rough heuristic)
                    if fixed_json.count('"') % 2 != 0:
                        # Find last quote and add closing if needed
                        last_quote_idx = fixed_json.rfind('"')
                        if last_quote_idx > 0:
                            # Check if we're inside a string
                            before_quote = fixed_json[:last_quote_idx]
                            if before_quote.count('"') % 2 == 0:  # We're in a string
                                fixed_json += '"'
                    
                    # Add missing closing braces
                    fixed_json += "}" * missing_braces
                    
                    # Try to parse the fixed JSON
                    parsed = json.loads(fixed_json)
                    items.append(parsed)
                    print(f"Note: Recovered incomplete JSON by adding {missing_braces} closing brace(s)", file=sys.stderr)
                except json.JSONDecodeError:
                    print(f"Note: Could not recover incomplete JSON (1 example lost - this is normal): {current_json[:200]}...", file=sys.stderr)
            else:
                print(f"Note: Incomplete JSON at end of response (1 example lost - this is normal): {current_json[:200]}...", file=sys.stderr)
        else:
            print(f"Note: Incomplete JSON at end of response (1 example lost - this is normal): {current_json[:200]}...", file=sys.stderr)
    
    return items


def validate_example(example: Dict, topic: str, config: TopicConfig, stage_override: Optional[str]):
    required = {"text", "topic", "intent_type", "query_type", "stage", "domain_scope", "advice_risk_score"}
    missing = required - example.keys()
    if missing:
        raise ValueError(f"Missing fields {missing} in {example}")
    if example["topic"] != topic:
        raise ValueError(f"Topic mismatch: {example}")
    if example["intent_type"] not in config.intent_types:
        raise ValueError(f"Invalid intent_type: {example}")
    if example["query_type"] not in config.query_types:
        raise ValueError(f"Invalid query_type: {example}")
    if stage_override:
        if example["stage"] != stage_override:
            raise ValueError(f"Stage mismatch (expected {stage_override}): {example}")
    elif example["stage"] not in config.stages:
        raise ValueError(f"Invalid stage: {example}")
    if example["domain_scope"] not in config.domain_scopes:
        raise ValueError(f"Invalid domain_scope: {example}")
    # Validate advice_risk_score
    advice_risk = example.get("advice_risk_score")
    if advice_risk is None:
        raise ValueError(f"Missing advice_risk_score: {example}")
    if not isinstance(advice_risk, (int, float)):
        raise ValueError(f"advice_risk_score must be a number: {example}")
    if not (0.0 <= advice_risk <= 1.0):
        raise ValueError(f"advice_risk_score must be between 0.0 and 1.0: {example}")
    if not example["text"] or len(example["text"]) < 8:
        raise ValueError(f"Utterance too short: {example}")


def save_json(data: Dict, target: Path, gcs_client: Optional[storage.Client]):
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
    config: TopicConfig,
    model: GenerativeModel,
    model_name: str,
    count: int,
    stage_override: Optional[str],
) -> Dict:
    stages = [stage_override] if stage_override else config.stages
    products_hint = ""
    if config.products:
        products_list = ', '.join(config.products)
        # Calculate how many products to use: at least 8, or 30% of available products if more than 8
        min_products = min(max(8, int(len(config.products) * 0.3)), len(config.products))
        products_hint = f"""- PRODUCT DISTRIBUTION: When generating bank_specific queries (20-30% of examples), distribute examples across these LBG products and services: {products_list}
  * For bank_specific domain_scope, ensure queries mention or reference at least {min_products} different products across the {count} examples. Vary the products used - don't focus on just one or two products.
  * Include product-specific queries like: "What is [product]?", "Does Lloyds offer [product]?", "How do I use [product]?", "What are [product] features?"
  * For general domain_scope, products can be mentioned generically (e.g., "what is life insurance" rather than "Lloyds Life Insurance") but still cover different product types
  * CRITICAL: Ensure good variety across different product categories/types (e.g., for insurance: life, home, travel, car, pet, etc.)"""
    
    # Split into batches if count is large to avoid rate limits and ensure we get all examples
    batch_size = 100  # Generate 100 examples per batch
    if count > batch_size:
        print(f"Generating {count} examples in batches of {batch_size} to avoid rate limits...", file=sys.stderr)
        all_unique = []
        seen = set()
        max_batches = 10  # Safety limit to prevent infinite loops
        batch_num = 0
        invalid_count = 0
        duplicate_count = 0
        
        while len(all_unique) < count and batch_num < max_batches:
            remaining = count - len(all_unique)
            current_batch_size = min(batch_size, remaining)
            
            batch_num += 1
            print(f"Batch {batch_num}: Generating {current_batch_size} examples (need {remaining} more)...", file=sys.stderr)
            
            prompt = PROMPT_TEMPLATE.format(
                topic=topic,
                brand_hint=config.brand_hint,
                intent_types=config.intent_types,
                query_types=config.query_types,
                stages=stages,
                domain_scopes=config.domain_scopes,
                products_hint=products_hint,
                count=current_batch_size,
            )
            
            raw = call_gemini(model, prompt, expected_count=current_batch_size)
            examples = parse_examples(raw)
            
            print(f"  Parsed {len(examples)} JSON objects from response", file=sys.stderr)
            
            batch_unique = []
            batch_invalid = 0
            batch_duplicate = 0
            
            for ex in examples:
                try:
                    validate_example(ex, topic, config, stage_override)
                    key = ex["text"].strip().lower()
                    if key in seen:
                        batch_duplicate += 1
                        continue
                    seen.add(key)
                    batch_unique.append(ex)
                    all_unique.append(ex)
                except ValueError as e:
                    batch_invalid += 1
                    invalid_count += 1
                    continue
            
            duplicate_count += batch_duplicate
            
            print(f"  Batch result: {len(batch_unique)} valid, {batch_invalid} invalid, {batch_duplicate} duplicates", file=sys.stderr)
            print(f"  Total so far: {len(all_unique)}/{count} (invalid: {invalid_count}, duplicates: {duplicate_count})", file=sys.stderr)
            
            # If we got very few examples, warn and continue
            if len(batch_unique) < current_batch_size * 0.5:  # Got less than 50% of requested
                print(f"  ⚠️  Warning: Got only {len(batch_unique)}/{current_batch_size} examples. Response may have been truncated.", file=sys.stderr)
            
            # Warn if duplicate rate is high (more than 30% of valid examples are duplicates)
            if len(examples) > 0 and batch_duplicate > len(examples) * 0.3:
                print(f"  ⚠️  Warning: High duplicate rate ({batch_duplicate}/{len(examples)} = {batch_duplicate/len(examples)*100:.1f}%). This may slow down generation.", file=sys.stderr)
            
            # Add delay between batches to avoid rate limits
            if len(all_unique) < count:
                delay = 2  # 2 second delay between batches
                print(f"  Waiting {delay} seconds before next batch...", file=sys.stderr)
                time.sleep(delay)
        
        unique = all_unique
        if len(unique) < count:
            print(f"⚠️  Warning: Generated {len(unique)}/{count} examples ({len(unique)/count*100:.1f}%). Consider running again or checking for issues.", file=sys.stderr)
        else:
            print(f"✅ Generated {len(unique)} total unique examples (requested {count})", file=sys.stderr)
        
        if duplicate_count > 0:
            print(f"   Note: {duplicate_count} duplicates were filtered out (not counted toward total)", file=sys.stderr)
    else:
        # Single batch for smaller requests
        prompt = PROMPT_TEMPLATE.format(
            topic=topic,
            brand_hint=config.brand_hint,
            intent_types=config.intent_types,
            query_types=config.query_types,
            stages=stages,
            domain_scopes=config.domain_scopes,
            products_hint=products_hint,
            count=count,
        )
        raw = call_gemini(model, prompt, expected_count=count)
        examples = parse_examples(raw)

        unique = []
        seen = set()
        for ex in examples:
            try:
                validate_example(ex, topic, config, stage_override)
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
        "stage_filter": stage_override,
        "schema": "sil_intent_v1",
        "fields": ["text", "topic", "intent_type", "query_type", "stage", "domain_scope", "advice_risk_score"],
        "requested_examples": count,
        "total_examples": len(unique),
        "model_name": model_name,
    }
    return {"metadata": metadata, "training_data": unique}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate SIL datasets by topic (optional stage).")
    parser.add_argument("--project", default="playpen-c84caa", help="GCP project ID")
    parser.add_argument("--location", default="us-central1", help="Vertex AI location")
    parser.add_argument("--output", required=True, help="gs://bucket/path or local directory")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name")
    parser.add_argument("--topic", choices=TOPIC_MATRIX.keys(), required=True, help="Topic to generate")
    parser.add_argument("--stage", help="Optional stage filter; must belong to the topic's stages")
    parser.add_argument("--examples", type=int, default=40, help="Examples to request")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    config = TOPIC_MATRIX[args.topic]
    if args.stage and args.stage not in config.stages:
        raise ValueError(f"Stage '{args.stage}' not valid for topic '{args.topic}'. Valid: {config.stages}")

    # Ensure output directory exists (for local paths)
    if not args.output.startswith("gs://"):
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir.absolute()}", file=sys.stderr)

    vertex_init(project=args.project, location=args.location)
    model = GenerativeModel(args.model)

    gcs_client = storage.Client(project=args.project) if args.output.startswith("gs://") else None

    print(f"Generating {args.examples} examples for topic '{args.topic}'...", file=sys.stderr)
    dataset = generate_batch(
        topic=args.topic,
        config=config,
        model=model,
        model_name=args.model,
        count=args.examples,
        stage_override=args.stage,
    )

    if not dataset or "training_data" not in dataset:
        raise Exception("Failed to generate dataset - empty or malformed response")
    
    if len(dataset["training_data"]) == 0:
        print("⚠️  Warning: Generated 0 examples. Nothing to save.", file=sys.stderr)
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Organize output by topic/stage in filesystem
    if args.stage:
        stage_dir = args.stage
    else:
        stage_dir = "all_stages"
    
    filename = f"{args.topic}_{stage_dir}_training_data_{timestamp}.json"
    output_path = Path(args.output) / args.topic / stage_dir / filename
    
    print(f"Saving {len(dataset['training_data'])} examples to {output_path}...", file=sys.stderr)
    try:
        save_json(dataset, output_path, gcs_client)
        print(f"✅ Successfully saved to {output_path}", file=sys.stderr)
    except Exception as e:
        print(f"❌ Error saving file: {e}", file=sys.stderr)
        raise

    manifest = {
        "topic": args.topic,
        "stage": args.stage,
        "example_count": dataset["metadata"]["total_examples"],
        "output_path": str(output_path),
    }
    manifest_path = Path(args.output) / f"manifest_{timestamp}.json"
    try:
        save_json(manifest, manifest_path, gcs_client)
        print(f"Manifest saved: {manifest_path}", file=sys.stderr)
    except Exception as e:
        print(f"⚠️  Warning: Could not save manifest: {e}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

