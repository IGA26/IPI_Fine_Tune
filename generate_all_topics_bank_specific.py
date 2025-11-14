#!/usr/bin/env python3
"""
generate_all_topics_bank_specific.py

Generate SIL-labelled training datasets with 80% bank_specific queries across all topics.
Focuses on product offering queries like "what pensions do you offer" which must be bank_specific.

Usage:
  python generate_all_topics_bank_specific.py --output ./training_data
  
  # Custom examples per topic
  python generate_all_topics_bank_specific.py --output ./training_data --savings 100 --pensions 80
  
  # Specific topics only
  python generate_all_topics_bank_specific.py --output ./training_data --topics savings,pensions,banking
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

# Import TOPIC_MATRIX from the existing script
from generate_sil_stage_datasets import TOPIC_MATRIX, TopicConfig, call_gemini, parse_examples, save_json, validate_example


# Default examples per topic (can be overridden via command line)
DEFAULT_EXAMPLES_PER_TOPIC: Dict[str, int] = {
    "savings": 100,
    "investments": 80,
    "pensions": 80,
    "mortgages": 80,
    "banking": 100,
    "loans": 60,
    "debt": 40,  # Lower because domain_scope is only "general"
    "insurance": 60,
    "taxation": 40,  # Lower because domain_scope is only "general"
    "general": 40,  # Lower because domain_scope is only "general"
    "off_topic": 30,  # Lower because domain_scope is only "general"
}


PROMPT_TEMPLATE = """You are generating labelled training data for a financial Semantic Interface Layer (SIL).

Output NEWLINE-SEPARATED JSON objects ONLY (no array, no markdown).

========================================================
SCHEMA
========================================================

{{
  "text": "<user utterance>",
  "topic": "{topic}",
  "intent_type": "<intent_type>",
  "query_type": "<query_type>",
  "stage": "<stage>",
  "domain_scope": "<general|bank_specific>",
  "advice_risk_score": <float 0.0-1.0>
}}

========================================================
STRICT FORMAT RULES
========================================================

- Output EXACTLY {count} JSON LINES.
- ONE JSON OBJECT per line.
- NO markdown code blocks (```json or ```).
- NO multi-line text within JSON objects.
- NO extra commentary between lines.
- All JSON fields REQUIRED.
- topic MUST be "{topic}".
- Text MUST be natural UK customer voice.
- Escape quotes as \\" if needed.
- Each JSON object must be complete on a single line.

========================================================
DOMAIN SCOPE RULES (CRITICAL)
========================================================

TARGET DISTRIBUTION:
- bank_specific: 80%
- general: 20%

A query MUST be bank_specific if ANY of these apply:

--------------------------------------------------------
1. PRODUCT OFFERING & AVAILABILITY PATTERNS
--------------------------------------------------------

Examples include (use ALL these variations):
- Direct: "what [products] do you offer", "what accounts do you offer", "what [products] are available", "what [products] do you have", "what [products] can I get", "what [products] are on offer", "what [products] do you provide", "what [products] can you help me with", "what's on offer"
- Yes/No: "do you offer X", "do you have X", "can I get X", "are there any X", "do you provide X", "do you do X"
- Imperative: "show me your [products]", "list your [products]", "tell me about your [products]", "show me what [products] you have", "give me information about your [products]", "show me available [products]"
- Availability: "what's available", "what options do you have", "what choices do I have", "what types of X do you offer", "what kinds of X are available", "which X do you offer", "what options are there for X"
- Service: "what services do you provide", "what can you help me with", "what do you provide", "what services are available"
- Possessive: "your [products]", "your [services]", "your accounts", "your mortgages"

--------------------------------------------------------
2. BRAND-SPECIFIC QUERIES
--------------------------------------------------------

If the text contains **ANY** of these brand forms:

**Lloyds Banking Group brands:**
- "Lloyds Bank", "Lloyds"
- "Halifax"
- "Bank of Scotland", "BoS"
- "MBNA"

**Scottish Widows (pensions, life insurance, investments context):**
- "Scottish Widows" (full name)
- "Scottish Wids" (common abbreviation)
- "SW" (ONLY when context clearly implies Scottish Widows brand: pensions, life insurance, investments)

(NOTE: Any mention of these brands makes the query bank_specific.)

--------------------------------------------------------
3. ACCOUNT-SPECIFIC ACTIONS
--------------------------------------------------------

Any query involving the customer's personal accounts MUST be bank_specific:
- "my balance", "my account balance", "check my balance", "what is my account balance"
- "my transactions", "view my transactions", "show my recent transactions"
- "my statements", "check my statements", "view my statements"
- "transfer money from my account", "pay bills from my account"
- "what is my ISA balance", "how much is in my ISA"
- "my pension balance", "my Scottish Widows pension", "how much is in my pension"
- "my savings", "my ISA", "my mortgage", "my investments"

--------------------------------------------------------
4. PRODUCT-SPECIFIC BANK QUERIES
--------------------------------------------------------

Examples:
- "does Lloyds offer X", "does Halifax offer X", "does Bank of Scotland offer X"
- "features of Lloyds X", "Bank of Scotland mortgage rates", "Scottish Widows pension options"
- "SW life insurance features" (when SW clearly refers to Scottish Widows)

--------------------------------------------------------
GENERAL SCOPE ONLY IF:
--------------------------------------------------------

- Definition questions: "what is an ISA", "what is a pension"
- Consumer finance education (generic, not bank-specific)
- Generic comparisons not referencing bank products
- Non-brand, non-account, non-offering questions

========================================================
LABEL RULES (STRICT)
========================================================

intent_type ∈ {intent_types}
query_type ∈ {query_types}
stage ∈ {stages}
domain_scope ∈ {domain_scopes}

ADVICE RISK ALIGNMENT:
- fact_seeking → 0.0–0.3
- guidance → 0.4–0.6
- advice_seeking → 0.7–1.0
- account_action → 0.0–0.3

STAGE DISTRIBUTION:
- CRITICAL: Distribute stage values EVENLY across {stages}
- If 2 stages: aim for 50/50 split
- If 3+ stages: distribute as evenly as possible (e.g., 3 stages = ~33% each, 5 stages = ~20% each)

INTENT & QUERY TYPE DISTRIBUTION:
- Distribute intent_type values roughly evenly across {intent_types}
- Distribute query_type values roughly evenly across {query_types}

ADVICE RISK SCORE DISTRIBUTION:
- Distribute roughly evenly across low (0.0-0.3), medium (0.4-0.6), and high (0.7-1.0) ranges

========================================================
PRODUCT VARIETY RULES
========================================================

{products_hint}

Requirements:
- Use at least 8 different product types OR 30% of available products (whichever is higher)
- Do NOT concentrate too many examples on one product
- Mix product categories (savings, pensions, investments, mortgages, insurance, loans, accounts)

========================================================
QUALITY RULES
========================================================

- UK customer voice ONLY
- FCA-compliant tone
- No personalised financial advice
- No product inventions—use real product categories (savings, mortgages, ISAs, loans, pensions, insurance, cards)
- Use natural variations in phrasing
- Avoid contradictions or impossible asks
- Every utterance must logically align with the labels

========================================================
EXAMPLES (DO NOT COPY)
========================================================

{{"text": "What pensions do you offer", "topic": "{topic}", "intent_type": "fact_seeking", "query_type": "what_is", "stage": "enrolment", "domain_scope": "bank_specific", "advice_risk_score": 0.2}}
{{"text": "Does Scottish Widows offer a workplace pension", "topic": "{topic}", "intent_type": "fact_seeking", "query_type": "what_is", "stage": "understanding", "domain_scope": "bank_specific", "advice_risk_score": 0.2}}
{{"text": "What current accounts are available", "topic": "{topic}", "intent_type": "fact_seeking", "query_type": "what_is", "stage": "awareness", "domain_scope": "bank_specific", "advice_risk_score": 0.2}}
{{"text": "Can I get a personal loan", "topic": "{topic}", "intent_type": "fact_seeking", "query_type": "what_is", "stage": "planning", "domain_scope": "bank_specific", "advice_risk_score": 0.2}}
{{"text": "Show me your mortgage options", "topic": "{topic}", "intent_type": "fact_seeking", "query_type": "what_is", "stage": "application", "domain_scope": "bank_specific", "advice_risk_score": 0.2}}
{{"text": "What is my account balance", "topic": "{topic}", "intent_type": "account_action", "query_type": "account_action", "stage": "action", "domain_scope": "bank_specific", "advice_risk_score": 0.1}}
{{"text": "What is an ISA", "topic": "{topic}", "intent_type": "fact_seeking", "query_type": "what_is", "stage": "understanding", "domain_scope": "general", "advice_risk_score": 0.2}}

========================================================
GENERATE EXACTLY {count} NEW EXAMPLES NOW.
========================================================
"""


def generate_batch(
    topic: str,
    config: TopicConfig,
    model: GenerativeModel,
    model_name: str,
    count: int,
) -> Dict:
    """Generate training data for a single topic with 80% bank_specific focus."""
    products_hint = ""
    if config.products:
        products_list = ', '.join(config.products)
        # Calculate how many products to use: at least 8, or 30% of available products
        min_products = min(max(8, int(len(config.products) * 0.3)), len(config.products))
        products_hint = f"""- PRODUCT DISTRIBUTION: When generating bank_specific queries (80% of examples), distribute examples across these LBG products and services: {products_list}
  * For bank_specific domain_scope, ensure queries mention or reference at least {min_products} different products across the {count} examples. Vary the products used - don't focus on just one or two products.
  * CRITICAL: Focus on product offering queries using ALL these variations:
    - Direct: "What [products] do you offer?", "What [products] are available?", "What [products] do you have?", "What [products] can I get?", "What [products] do you provide?"
    - Yes/No: "Do you offer [product]?", "Do you have [product]?", "Can I get [product]?", "Do you provide [product]?", "Are there any [products]?"
    - Imperative: "Show me your [products]", "List your [products]", "Tell me about your [products]", "Show me what [products] you have"
    - Availability: "What types of [product] do you offer?", "What kinds of [product] are available?", "Which [products] do you offer?", "What options are there for [product]?"
    - With brand: "What [products] does Lloyds offer?", "Does Halifax offer [product]?", "What products does Bank of Scotland have?"
  * Include product-specific queries like: "What is [product]?", "Does Lloyds offer [product]?", "How do I use [product]?", "What are [product] features?"
  * For general domain_scope (20% of examples), products can be mentioned generically (e.g., "what is life insurance" rather than "Lloyds Life Insurance") but still cover different product types
  * CRITICAL: Ensure good variety across different product categories/types (e.g., for insurance: life, home, travel, car, pet, etc.)"""
    
    # Split into batches if count is large to avoid rate limits
    batch_size = 100  # Generate 100 examples per batch
    if count > batch_size:
        print(f"Generating {count} examples in batches of {batch_size} to avoid rate limits...", file=sys.stderr)
        all_unique = []
        seen = set()
        max_batches = 10  # Safety limit
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
                stages=config.stages,
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
                    validate_example(ex, topic, config, None)  # No stage override
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
            
            if len(batch_unique) < current_batch_size * 0.5:
                print(f"  ⚠️  Warning: Got only {len(batch_unique)}/{current_batch_size} examples. Response may have been truncated.", file=sys.stderr)
            
            if len(examples) > 0 and batch_duplicate > len(examples) * 0.3:
                print(f"  ⚠️  Warning: High duplicate rate ({batch_duplicate}/{len(examples)} = {batch_duplicate/len(examples)*100:.1f}%).", file=sys.stderr)
            
            if len(all_unique) < count:
                delay = 2
                print(f"  Waiting {delay} seconds before next batch...", file=sys.stderr)
                time.sleep(delay)
        
        unique = all_unique
        if len(unique) < count:
            print(f"⚠️  Warning: Generated {len(unique)}/{count} examples ({len(unique)/count*100:.1f}%).", file=sys.stderr)
        else:
            print(f"✅ Generated {len(unique)} total unique examples (requested {count})", file=sys.stderr)
        
        if duplicate_count > 0:
            print(f"   Note: {duplicate_count} duplicates were filtered out", file=sys.stderr)
    else:
        # Single batch for smaller requests
        prompt = PROMPT_TEMPLATE.format(
            topic=topic,
            brand_hint=config.brand_hint,
            intent_types=config.intent_types,
            query_types=config.query_types,
            stages=config.stages,
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
                validate_example(ex, topic, config, None)
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
        "stage_filter": None,
        "schema": "sil_intent_v1",
        "fields": ["text", "topic", "intent_type", "query_type", "stage", "domain_scope", "advice_risk_score"],
        "requested_examples": count,
        "total_examples": len(unique),
        "model_name": model_name,
        "bank_specific_ratio": "80%",
    }
    return {"metadata": metadata, "training_data": unique}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate SIL datasets with 80% bank_specific queries across all topics."
    )
    parser.add_argument("--project", default="playpen-c84caa", help="GCP project ID")
    parser.add_argument("--location", default="us-central1", help="Vertex AI location")
    parser.add_argument("--output", required=True, help="Output directory (local path)")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name")
    
    # Topic-specific example counts (optional, overrides defaults)
    parser.add_argument("--savings", type=int, help="Examples for savings topic")
    parser.add_argument("--investments", type=int, help="Examples for investments topic")
    parser.add_argument("--pensions", type=int, help="Examples for pensions topic")
    parser.add_argument("--mortgages", type=int, help="Examples for mortgages topic")
    parser.add_argument("--banking", type=int, help="Examples for banking topic")
    parser.add_argument("--loans", type=int, help="Examples for loans topic")
    parser.add_argument("--debt", type=int, help="Examples for debt topic")
    parser.add_argument("--insurance", type=int, help="Examples for insurance topic")
    parser.add_argument("--taxation", type=int, help="Examples for taxation topic")
    parser.add_argument("--general", type=int, help="Examples for general topic")
    parser.add_argument("--off_topic", type=int, help="Examples for off_topic topic")
    
    # Optional: specify which topics to generate (default: all)
    parser.add_argument(
        "--topics",
        help="Comma-separated list of topics to generate (default: all). Example: savings,pensions,banking"
    )
    
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    
    # Determine which topics to generate
    if args.topics:
        topics_to_generate = [t.strip() for t in args.topics.split(",")]
        invalid_topics = [t for t in topics_to_generate if t not in TOPIC_MATRIX]
        if invalid_topics:
            raise ValueError(f"Invalid topics: {invalid_topics}. Valid: {list(TOPIC_MATRIX.keys())}")
    else:
        topics_to_generate = list(TOPIC_MATRIX.keys())
    
    # Build examples_per_topic dict from defaults and overrides
    examples_per_topic = DEFAULT_EXAMPLES_PER_TOPIC.copy()
    
    # Apply command-line overrides
    topic_arg_map = {
        "savings": args.savings,
        "investments": args.investments,
        "pensions": args.pensions,
        "mortgages": args.mortgages,
        "banking": args.banking,
        "loans": args.loans,
        "debt": args.debt,
        "insurance": args.insurance,
        "taxation": args.taxation,
        "general": args.general,
        "off_topic": args.off_topic,
    }
    
    for topic, count in topic_arg_map.items():
        if count is not None:
            examples_per_topic[topic] = count
    
    # Filter to only topics we're generating
    examples_per_topic = {k: v for k, v in examples_per_topic.items() if k in topics_to_generate}
    
    # Ensure output directory exists
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}", file=sys.stderr)
    print(f"Topics to generate: {topics_to_generate}", file=sys.stderr)
    print(f"Examples per topic: {examples_per_topic}", file=sys.stderr)
    
    # Initialize Vertex AI
    vertex_init(project=args.project, location=args.location)
    model = GenerativeModel(args.model)
    
    gcs_client = None  # Only local paths supported for now
    
    # Generate for each topic
    total_examples = 0
    successful_topics = []
    failed_topics = []
    
    for topic in topics_to_generate:
        if topic not in examples_per_topic:
            print(f"⚠️  Skipping {topic} (no example count specified)", file=sys.stderr)
            continue
        
        count = examples_per_topic[topic]
        config = TOPIC_MATRIX[topic]
        
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Generating {count} examples for topic '{topic}'...", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        
        try:
            dataset = generate_batch(
                topic=topic,
                config=config,
                model=model,
                model_name=args.model,
                count=count,
            )
            
            if not dataset or "training_data" not in dataset:
                print(f"❌ Failed to generate dataset for {topic} - empty response", file=sys.stderr)
                failed_topics.append(topic)
                continue
            
            if len(dataset["training_data"]) == 0:
                print(f"⚠️  Warning: Generated 0 examples for {topic}. Skipping.", file=sys.stderr)
                failed_topics.append(topic)
                continue
            
            # Save to training_data/{topic}/all_stages/{filename}.json
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{topic}_all_stages_bank_specific_{timestamp}.json"
            output_path = output_dir / topic / "all_stages" / filename
            
            print(f"Saving {len(dataset['training_data'])} examples to {output_path}...", file=sys.stderr)
            try:
                save_json(dataset, output_path, gcs_client)
                print(f"✅ Successfully saved to {output_path}", file=sys.stderr)
                total_examples += len(dataset["training_data"])
                successful_topics.append(topic)
            except Exception as e:
                print(f"❌ Error saving file for {topic}: {e}", file=sys.stderr)
                failed_topics.append(topic)
            
            # Add delay between topics to avoid rate limits
            if topic != topics_to_generate[-1]:  # Don't delay after last topic
                delay = 3
                print(f"Waiting {delay} seconds before next topic...", file=sys.stderr)
                time.sleep(delay)
                
        except Exception as e:
            print(f"❌ Error generating dataset for {topic}: {e}", file=sys.stderr)
            import traceback
            print(traceback.format_exc(), file=sys.stderr)
            failed_topics.append(topic)
            continue
    
    # Summary
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"SUMMARY", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"Total examples generated: {total_examples}", file=sys.stderr)
    print(f"Successful topics ({len(successful_topics)}): {successful_topics}", file=sys.stderr)
    if failed_topics:
        print(f"Failed topics ({len(failed_topics)}): {failed_topics}", file=sys.stderr)
    print(f"Output directory: {output_dir.absolute()}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        import traceback
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)

