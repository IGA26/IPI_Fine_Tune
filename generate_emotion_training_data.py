#!/usr/bin/env python3
"""
generate_emotion_training_data.py

Generate training data for fine-tuning ProsusAI/finbert to predict:
- emotion (positive/neutral/negative/confused)
- distress_flag (boolean)
- vulnerability_flag (boolean)

Uses Vertex AI Gemini to generate labeled examples.

Usage:
  python generate_emotion_training_data.py \
      --output ./training_data/emotion \
      --examples 500
"""

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from google.cloud import storage
from vertexai import init as vertex_init
from vertexai.generative_models import GenerativeModel

PROMPT_TEMPLATE = """You are generating labelled training data for fine-tuning ProsusAI/finbert to detect emotion, distress, and vulnerability in financial queries.

FCA Compliance References:
- FG21/1: Vulnerability Guidance - Distress and vulnerability detection per FCA guidelines
- This training data must align with FCA FG21/1 vulnerability indicators for regulatory compliance

Return NEWLINE-SEPARATED JSON objects (no outer array, no markdown code blocks, plain JSON only) with this schema:

{{
  "text": "<user utterance>",
  "emotion": "<positive|neutral|negative|confused>",
  "sentiment_score": <number between -1.0 and 1.0>,
  "distress_flag": <boolean>,
  "vulnerability_flag": <boolean>
}}

Constraints:
- UK customer voice; financial domain queries
- Emotion labels:
  * positive: excited, optimistic, confident, happy, pleased
  * neutral: calm, matter-of-fact, informational, standard
  * negative: worried, frustrated, stressed, anxious, concerned
  * confused: uncertain, overwhelmed, seeking clarity, unclear
  
- Distress flag (true when) - Per FCA FG21/1 vulnerability indicators:
  * Life events: job loss, divorce, bereavement, illness, redundancy (as per FG21/1)
  * Financial stress: debt problems, payment difficulties, can't afford, struggling, unable to pay
  * Emotional distress: anxiety, depression, panic, desperation, overwhelmed (FG21/1 emotional indicators)
  * Urgent language: "desperate", "emergency", "crisis", "urgent", "help me now", "I need help"
  * Self-harm or suicidal ideation indicators: "I want to hurt myself", "I can't go on", "I don't see a way out", "life isn't worth it", "I wish I wasn't here", "I feel like ending it all" - CRITICAL: Any mention of self-harm or suicidal thoughts MUST set distress_flag=true
  * FCA FG21/1 guidance: Flags customers who may need immediate support or escalation to human agent
  * CRITICAL: Only set distress_flag=true when clear indicators of FCA-defined vulnerability distress are present
  * CRITICAL: Self-harm indicators (even subtle) MUST always set distress_flag=true - this is the highest priority distress signal
  
- Vulnerability flag (true when) - Per FCA vulnerability characteristics (FG21/1):
  * Age-related: very young (<18 mentioned or implied), elderly (80+ mentioned or implied) - FCA age vulnerability
  * Health conditions: mental health issues, cognitive impairment, disability, long-term illness - FCA health vulnerability
  * Life events: recent major changes (bereavement, relationship breakdown, job loss) - FCA life event vulnerability
  * Financial capability: low financial literacy indicators ("I don't understand", "confused about", "not sure how", "never done this before") - FCA capability vulnerability
  * FCA FG21/1 guidance: Identifies customers who may need additional support, clearer explanations, or different communication approaches
  * CRITICAL: Only set vulnerability_flag=true when clear FCA-defined vulnerability characteristics are present
  
- sentiment_score mapping:
  * positive emotion → 0.5 to 1.0
  * neutral emotion → -0.2 to 0.2
  * negative emotion → -1.0 to -0.3
  * confused emotion → -0.3 to 0.0
  
- IMPORTANT: Balance the distribution:
  * CRITICAL: Distribute emotion values evenly: ~25% positive, ~25% neutral, ~25% negative, ~25% confused
  * Distress flag: 70-80% false, 20-30% true (most queries are not distressed)
  * Vulnerability flag: 80-90% false, 10-20% true (vulnerability is less common)
  * Ensure distress/vulnerability flags align with emotion (distressed users usually negative/confused)
  
- Cover various financial topics: savings, investments, pensions, mortgages, banking, loans, debt, insurance
- Include both general and bank-specific queries
- Avoid duplicates
- Output exactly {count} lines; each line is a COMPLETE, SINGLE-LINE JSON object
- DO NOT wrap output in markdown code blocks (```json or ```). Output plain JSON only.
- CRITICAL: All quotes in the "text" field must be properly escaped as \\" (backslash-quote)
- CRITICAL: Each JSON object must be complete on a single line. Do not split JSON objects across multiple lines.

⚠️ CRITICAL SAFETY REQUIREMENTS:
- DO NOT generate or simulate explicit self-harm content, suicidal ideation, or graphic descriptions of harm
- DO NOT include phrases that explicitly describe self-harm methods or intentions
- Focus on financial distress and vulnerability indicators without explicit harm language
- If generating examples about severe distress, use appropriate language that indicates need for help without explicit harm references
- This training data is for FCA compliance and must be safe for fine-tuning and review

⚠️ IMPORTANT LABELING RULE FOR DISTRESS DETECTION:
- While we avoid generating explicit self-harm content, IF any generated text contains indicators of self-harm or suicidal ideation (even subtle phrases like "I can't go on", "I don't see a way out", "life isn't worth it"), you MUST set distress_flag=true
- This ensures the fine-tuned model learns to detect self-harm indicators in production, even if they appear in training data
- Self-harm indicators are the highest priority distress signals and should always trigger distress_flag=true

Example format (do not reuse text):
{{"text": "I'm really worried about my debt and don't know what to do", "emotion": "negative", "sentiment_score": -0.7, "distress_flag": true, "vulnerability_flag": false}}
{{"text": "What is an ISA?", "emotion": "neutral", "sentiment_score": 0.0, "distress_flag": false, "vulnerability_flag": false}}
{{"text": "I lost my job and can't pay my mortgage", "emotion": "negative", "sentiment_score": -0.9, "distress_flag": true, "vulnerability_flag": true}}
{{"text": "I feel like I can't go on with all this debt", "emotion": "negative", "sentiment_score": -0.95, "distress_flag": true, "vulnerability_flag": false}}

Generate {count} new examples now.
"""

# Curated list of potentially triggering words/phrases to filter out
# These are words that should not appear in training data even in financial context
TRIGGERING_PATTERNS = [
    # Explicit self-harm references (case-insensitive regex patterns)
    r'\bkill\s+myself\b',
    r'\bend\s+it\s+all\b',
    r'\bend\s+my\s+life\b',
    r'\bnot\s+worth\s+living\b',
    r'\bno\s+point\s+in\s+living\b',
    # Note: We don't filter general words like "kill" or "end" as they appear in legitimate financial contexts
    # ("kill the debt", "end of month", etc.)
]

# Subtle self-harm indicators that should trigger distress_flag=true
# These are less explicit but still indicate distress that requires attention
SELF_HARM_INDICATORS = [
    r'\bcan\'?t\s+go\s+on\b',
    r'\bdon\'?t\s+see\s+a\s+way\s+out\b',
    r'\blife\s+isn\'?t\s+worth\b',
    r'\bwish\s+I\s+wasn\'?t\s+here\b',
    r'\bfeel\s+like\s+ending\s+it\s+all\b',
    r'\bno\s+way\s+out\b',
    r'\bhopeless\b',
    r'\bno\s+point\s+anymore\b',
]


def filter_triggering_content(text: str) -> bool:
    """
    Check if text contains potentially triggering content that should be filtered.
    
    Args:
        text: Text to check
        
    Returns:
        True if text should be filtered (contains triggering content), False otherwise
    """
    text_lower = text.lower()
    
    for pattern in TRIGGERING_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    
    return False


def call_gemini(model: GenerativeModel, prompt: str, expected_count: int = 100, max_retries: int = 3) -> str:
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


def validate_example(example: Dict) -> bool:
    required = {"text", "emotion", "sentiment_score", "distress_flag", "vulnerability_flag"}
    if not all(field in example for field in required):
        return False
    
    if example["emotion"] not in ["positive", "neutral", "negative", "confused"]:
        return False
    
    if not isinstance(example["sentiment_score"], (int, float)):
        return False
    if not (-1.0 <= example["sentiment_score"] <= 1.0):
        return False
    
    if not isinstance(example["distress_flag"], bool):
        return False
    
    if not isinstance(example["vulnerability_flag"], bool):
        return False
    
    if not example["text"] or len(example["text"]) < 8:
        return False
    
    # Safety check: If self-harm indicators are present, distress_flag MUST be true
    text_lower = example["text"].lower()
    has_self_harm_indicator = any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in SELF_HARM_INDICATORS)
    if has_self_harm_indicator and not example["distress_flag"]:
        print(f"⚠️  Warning: Self-harm indicator detected but distress_flag=false. Text: '{example['text'][:60]}...'", file=sys.stderr)
        # Auto-correct: Set distress_flag=true for self-harm indicators
        example["distress_flag"] = True
    
    return True


def generate_batch(model: GenerativeModel, count: int) -> List[Dict]:
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
        filtered_count_total = 0
        
        while len(all_unique) < count and batch_num < max_batches:
            remaining = count - len(all_unique)
            current_batch_size = min(batch_size, remaining)
            
            batch_num += 1
            print(f"Batch {batch_num}: Generating {current_batch_size} examples (need {remaining} more)...", file=sys.stderr)
            
            prompt = PROMPT_TEMPLATE.format(count=current_batch_size)
            raw = call_gemini(model, prompt, expected_count=current_batch_size)
            examples = parse_examples(raw)
            
            print(f"  Parsed {len(examples)} JSON objects from response", file=sys.stderr)
            
            batch_unique = []
            batch_invalid = 0
            batch_duplicate = 0
            batch_filtered = 0
            
            for ex in examples:
                try:
                    # Ethical safeguard: Filter triggering content
                    if filter_triggering_content(ex.get("text", "")):
                        batch_filtered += 1
                        continue
                    
                    if validate_example(ex):
                        key = ex["text"].strip().lower()
                        if key in seen:
                            batch_duplicate += 1
                            continue
                        seen.add(key)
                        batch_unique.append(ex)
                        all_unique.append(ex)
                except Exception as e:
                    batch_invalid += 1
                    invalid_count += 1
                    continue
            
            duplicate_count += batch_duplicate
            filtered_count_total += batch_filtered
            
            print(f"  Batch result: {len(batch_unique)} valid, {batch_invalid} invalid, {batch_duplicate} duplicates, {batch_filtered} filtered (triggering content)", file=sys.stderr)
            print(f"  Total so far: {len(all_unique)}/{count} (invalid: {invalid_count}, duplicates: {duplicate_count}, filtered: {filtered_count_total})", file=sys.stderr)
            
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
        if filtered_count_total > 0:
            print(f"   Note: {filtered_count_total} examples with triggering content were filtered out (not counted toward total)", file=sys.stderr)
    else:
        # Single batch for smaller requests
        prompt = PROMPT_TEMPLATE.format(count=count)
        raw = call_gemini(model, prompt, expected_count=count)
        examples = parse_examples(raw)

        unique = []
        seen = set()
        filtered_count = 0
        for ex in examples:
            try:
                # Ethical safeguard: Filter triggering content
                if filter_triggering_content(ex.get("text", "")):
                    filtered_count += 1
                    continue
                
                if validate_example(ex):
                    key = ex["text"].strip().lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    unique.append(ex)
            except Exception as e:
                print(f"Warning: Skipping invalid example: {e}", file=sys.stderr)
                continue
        
        if filtered_count > 0:
            print(f"Note: Filtered {filtered_count} examples with triggering content", file=sys.stderr)
    
    return unique


def main():
    parser = argparse.ArgumentParser(description="Generate emotion training data")
    parser.add_argument("--project", default="playpen-c84caa", help="GCP project ID")
    parser.add_argument("--location", default="us-central1", help="Vertex AI location")
    parser.add_argument("--output", required=True, help="Output directory (local or gs://)")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name")
    parser.add_argument("--examples", type=int, default=500, help="Number of examples to generate")
    
    args = parser.parse_args()
    
    if not args.output.startswith("gs://"):
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir.absolute()}", file=sys.stderr)
    
    vertex_init(project=args.project, location=args.location)
    model = GenerativeModel(args.model)
    
    gcs_client = storage.Client(project=args.project) if args.output.startswith("gs://") else None
    
    print(f"Generating {args.examples} emotion training examples...", file=sys.stderr)
    examples = generate_batch(model, args.examples)
    
    if not examples:
        print("⚠️  Warning: Generated 0 examples. Nothing to save.", file=sys.stderr)
        return
    
    # Calculate statistics for governance metadata
    total_chars = sum(len(ex.get("text", "")) for ex in examples)
    avg_text_length = total_chars / len(examples) if examples else 0
    
    # Label distribution
    emotion_dist = {}
    distress_true = 0
    vulnerability_true = 0
    
    for ex in examples:
        emotion = ex.get("emotion", "neutral")
        emotion_dist[emotion] = emotion_dist.get(emotion, 0) + 1
        if ex.get("distress_flag", False):
            distress_true += 1
        if ex.get("vulnerability_flag", False):
            vulnerability_true += 1
    
    # Generate prompt hash for tracking
    prompt_hash = hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest()[:16]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"emotion_training_data_{timestamp}.json"
    
    # Governance metadata
    governance_metadata = {
        "taxonomy_version": "8.0",
        "sil_version": "2.0",
        "generator_version": "2.1",
        "fca_category": "vulnerability",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": args.model,
        "prompt_hash": prompt_hash,
        "total_examples": len(examples),
        "statistics": {
            "average_text_length": round(avg_text_length, 2),
            "label_distribution": {
                "emotion": {k: v for k, v in emotion_dist.items()},
                "distress_flag": {
                    "true": distress_true,
                    "false": len(examples) - distress_true,
                    "true_percentage": round(distress_true / len(examples) * 100, 2) if examples else 0
                },
                "vulnerability_flag": {
                    "true": vulnerability_true,
                    "false": len(examples) - vulnerability_true,
                    "true_percentage": round(vulnerability_true / len(examples) * 100, 2) if examples else 0
                }
            }
        }
    }
    
    # Prepare output with governance metadata
    output_data = {
        "metadata": governance_metadata,
        "training_data": examples
    }
    
    if args.output.startswith("gs://"):
        bucket_name, *blob_parts = args.output.replace("gs://", "").split("/")
        blob_name = "/".join(blob_parts + [filename])
        bucket = gcs_client.bucket(bucket_name)
        bucket.blob(blob_name).upload_from_string(
            json.dumps(output_data, indent=2),
            content_type="application/json"
        )
        print(f"✅ Saved {len(examples)} examples to gs://{bucket_name}/{blob_name}", file=sys.stderr)
    else:
        output_path = Path(args.output) / filename
        output_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
        print(f"✅ Saved {len(examples)} examples to {output_path}", file=sys.stderr)
    
    # Print statistics
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"GOVERNANCE METADATA:", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"  Taxonomy Version: {governance_metadata['taxonomy_version']}", file=sys.stderr)
    print(f"  SIL Version: {governance_metadata['sil_version']}", file=sys.stderr)
    print(f"  Generator Version: {governance_metadata['generator_version']}", file=sys.stderr)
    print(f"  FCA Category: {governance_metadata['fca_category']}", file=sys.stderr)
    print(f"  Prompt Hash: {governance_metadata['prompt_hash']}", file=sys.stderr)
    print(f"\nSTATISTICS:", file=sys.stderr)
    print(f"  Average text length: {governance_metadata['statistics']['average_text_length']} chars", file=sys.stderr)
    print(f"  Emotion distribution: {governance_metadata['statistics']['label_distribution']['emotion']}", file=sys.stderr)
    print(f"  Distress flag: {governance_metadata['statistics']['label_distribution']['distress_flag']['true']}/{len(examples)} ({governance_metadata['statistics']['label_distribution']['distress_flag']['true_percentage']}%)", file=sys.stderr)
    print(f"  Vulnerability flag: {governance_metadata['statistics']['label_distribution']['vulnerability_flag']['true']}/{len(examples)} ({governance_metadata['statistics']['label_distribution']['vulnerability_flag']['true_percentage']}%)", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)


if __name__ == "__main__":
    main()

