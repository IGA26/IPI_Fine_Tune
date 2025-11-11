#!/usr/bin/env python3
"""
generate_emotion_training_data.py

Generate training data for fine-tuning ProsusAI/finbert to predict:
- emotion (positive/neutral/negative/confused)
- distress_flag (boolean)
- distress_severity (0.0-1.0) - FCA FG21/1 severity scoring
- handover_required (boolean) - Immediate human handover needed
- vulnerability_flag (boolean)

Uses Vertex AI Gemini to generate labeled examples.

Usage:
  python generate_emotion_training_data.py \
      --output ./training_data/emotion \
      --examples 500
  
  # For large datasets (4000+ examples), the script automatically uses:
  # - Smaller batch sizes (60 examples per batch)
  # - Longer delays (4 seconds base delay)
  # - Adaptive rate limiting with exponential backoff
  
  # Custom batch size and delay (optional):
  python generate_emotion_training_data.py \
      --output ./training_data/emotion \
      --examples 4000 \
      --batch-size 50 \
      --delay 5.0
  
  Project and location are hardcoded (default: playpen-c84caa, us-central1)
  Override with --project and --location if needed
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

# Hardcoded defaults (matches generate_sil_stage_datasets.py)
DEFAULT_PROJECT = "playpen-c84caa"
DEFAULT_LOCATION = "us-central1"
DEFAULT_MODEL = "gemini-2.5-flash"

PROMPT_TEMPLATE = """You are generating labelled training data for fine-tuning ProsusAI/finbert to detect emotion, distress, and vulnerability in financial queries.

FCA Compliance References:
- FG21/1: Vulnerability Guidance - Distress and vulnerability detection per FCA guidelines
- This training data must align with FCA FG21/1 vulnerability indicators for regulatory compliance

Return NEWLINE-SEPARATED JSON objects (no outer array, no markdown code blocks, plain JSON only) with this schema:

{{
  "text": "<user utterance>",
  "emotion": "<positive|neutral|negative>",
  "sentiment_score": <number between -1.0 and 1.0>,
  "distress_flag": <boolean>,
  "distress_severity": <number between 0.0 and 1.0>,
  "handover_required": <boolean>,
  "vulnerability_flag": <boolean>
}}

Constraints:
- UK customer voice; financial domain queries
- Emotion labels:
  * positive: excited, optimistic, confident, happy, pleased, reassured
  * neutral: calm, matter-of-fact, informational, standard, balanced
  * negative: worried, frustrated, stressed, anxious, concerned, confused, uncertain, overwhelmed
  
- Distress flag (true when) - Per FCA FG21/1 vulnerability indicators:
  * Life events: job loss, divorce, bereavement, illness, redundancy (as per FG21/1)
  * Financial stress: debt problems, payment difficulties, can't afford, struggling, unable to pay
  * Emotional distress: anxiety, depression, panic, desperation, overwhelmed (FG21/1 emotional indicators)
  * Urgent language: "desperate", "emergency", "crisis", "urgent", "help me now", "I need help"
  * Security/fraud incidents: account hacked, unauthorized transactions, money stolen, fraud detected, suspicious activity, account compromised, all my money is gone, someone took my money, I didn't make these transactions - CRITICAL: Security incidents are HIGH priority distress and require immediate handover
  * Self-harm or suicidal ideation indicators: "I want to hurt myself", "I can't go on", "I don't see a way out", "life isn't worth it", "I wish I wasn't here", "I feel like ending it all" - CRITICAL: Any mention of self-harm or suicidal thoughts MUST set distress_flag=true
  * FCA FG21/1 guidance: Flags customers who may need immediate support or escalation to human agent
  * CRITICAL: Only set distress_flag=true when clear indicators of FCA-defined vulnerability distress are present
  * CRITICAL: Self-harm indicators (even subtle) MUST always set distress_flag=true - this is the highest priority distress signal
  * CRITICAL: Security/fraud incidents MUST always set distress_flag=true and distress_severity ≥0.8 (HIGH/CRITICAL) - these require immediate human intervention

- Distress severity (0.0-1.0) - FCA FG21/1 severity scoring for routing decisions:
  * CRITICAL (0.95-1.0): Self-harm indicators, suicidal ideation ("I can't go on", "no way out", "hopeless", "life isn't worth it") → handover_required=true
  * HIGH (0.8-0.95): Crisis language ("desperate", "emergency", "urgent help", "crisis"), severe financial crisis ("losing my home", "bankruptcy", "can't pay anything"), severe panic, security/fraud incidents ("my account is hacked", "all my money is gone", "unauthorized transactions", "fraud detected") → handover_required=true
  * MEDIUM (0.6-0.8): Life events + financial stress (job loss + money worries, bereavement + debt), vulnerability + moderate distress → handover_required=false (specialist support, 3 min timeout)
  * LOW (0.3-0.6): Mild concern ("I'm worried about...", "a bit concerned", "slightly stressed") → handover_required=false (continue with empathy)
  * NONE (0.0-0.3): Normal queries, no distress indicators → handover_required=false (normal conversation)
  * CRITICAL: distress_severity must align with distress_flag (if distress_flag=true, severity must be ≥0.3; if false, severity should be <0.3)
  * CRITICAL: distress_severity ≥0.8 MUST set handover_required=true (immediate handover per FCA FG21/1)
  * CRITICAL: Security/fraud incidents MUST have distress_severity ≥0.8 (HIGH/CRITICAL) and handover_required=true - these require immediate fraud team intervention

- Handover required (boolean) - Immediate human handover needed per FCA FG21/1:
  * true when distress_severity ≥0.8 (CRITICAL or HIGH severity)
  * true when self-harm indicators present (even subtle: "can't go on", "no way out", "hopeless")
  * true when user explicitly requests human/operator/advisor: "I want to speak to a human", "can I talk to someone", "transfer me to an operator", "I need to speak to a person", "connect me to an advisor", "let me speak to a real person", "I want human help", "talk to a representative"
  * false when distress_severity <0.8 AND no explicit human request (agent can continue with empathy or specialist support)
  * CRITICAL: Explicit human requests ALWAYS set handover_required=true regardless of distress_severity (user preference must be respected)
  * CRITICAL: This field determines routing - true = immediate handover (<1 min), false = continue with appropriate support
  
- Vulnerability flag (true when) - Per FCA vulnerability characteristics (FG21/1):
  * Age-related: very young (<18 mentioned or implied), elderly (80+ mentioned or implied) - FCA age vulnerability
  * Health conditions: mental health issues, cognitive impairment, disability, long-term illness - FCA health vulnerability
  * Life events: recent major changes (bereavement, relationship breakdown, job loss) - FCA life event vulnerability
  * Financial capability: low financial literacy indicators ("I don't understand", "confused about", "not sure how", "never done this before") - FCA capability vulnerability
  * FCA FG21/1 guidance: Identifies customers who may need additional support, clearer explanations, or different communication approaches
  * CRITICAL: Only set vulnerability_flag=true when clear FCA-defined vulnerability characteristics are present
  
- sentiment_score mapping:
  * positive emotion → 0.4 to 1.0
  * neutral emotion → -0.2 to 0.2
  * negative emotion → -1.0 to -0.2 (includes confused, worried, frustrated, stressed)
  
- IMPORTANT: Balance the distribution:
  * CRITICAL: Distribute emotion values roughly evenly: ~33% positive, ~33% neutral, ~33% negative
  * Distress flag: 70-80% false, 20-30% true (most queries are not distressed)
  * Distress severity: Distribute across ranges - ~60% 0.0-0.3 (none/low), ~20% 0.3-0.6 (low), ~15% 0.6-0.8 (medium), ~5% 0.8-1.0 (high/critical)
  * Handover required: 85-95% false, 5-15% true (most queries don't need immediate handover)
  * Include ~2-5% examples with explicit human/operator requests (e.g., "I want to speak to a human", "can I talk to someone") - these MUST have handover_required=true
  * Vulnerability flag: 80-90% false, 10-20% true (vulnerability is less common)
  * Ensure distress/vulnerability flags align with emotion (distressed users usually negative)
  * CRITICAL: Ensure distress_severity and handover_required align correctly (severity ≥0.8 → handover=true)
  
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
{{"text": "I'm really worried about my debt and don't know what to do", "emotion": "negative", "sentiment_score": -0.7, "distress_flag": true, "distress_severity": 0.5, "handover_required": false, "vulnerability_flag": false}}
{{"text": "What is an ISA?", "emotion": "neutral", "sentiment_score": 0.0, "distress_flag": false, "distress_severity": 0.1, "handover_required": false, "vulnerability_flag": false}}
{{"text": "I lost my job and can't pay my mortgage", "emotion": "negative", "sentiment_score": -0.9, "distress_flag": true, "distress_severity": 0.75, "handover_required": false, "vulnerability_flag": true}}
{{"text": "I feel like I can't go on with all this debt", "emotion": "negative", "sentiment_score": -0.95, "distress_flag": true, "distress_severity": 0.95, "handover_required": true, "vulnerability_flag": false}}
{{"text": "I'm desperate and need urgent help with my finances", "emotion": "negative", "sentiment_score": -0.9, "distress_flag": true, "distress_severity": 0.85, "handover_required": true, "vulnerability_flag": false}}
{{"text": "I'm a bit concerned about my savings", "emotion": "negative", "sentiment_score": -0.4, "distress_flag": false, "distress_severity": 0.25, "handover_required": false, "vulnerability_flag": false}}
{{"text": "I want to speak to a human about my mortgage", "emotion": "neutral", "sentiment_score": 0.0, "distress_flag": false, "distress_severity": 0.2, "handover_required": true, "vulnerability_flag": false}}
{{"text": "Can I talk to someone about my pension options?", "emotion": "neutral", "sentiment_score": 0.1, "distress_flag": false, "distress_severity": 0.15, "handover_required": true, "vulnerability_flag": false}}
{{"text": "I'm confused about ISAs, can I speak to an advisor?", "emotion": "negative", "sentiment_score": -0.3, "distress_flag": false, "distress_severity": 0.3, "handover_required": true, "vulnerability_flag": true}}
{{"text": "Transfer me to an operator please", "emotion": "neutral", "sentiment_score": 0.0, "distress_flag": false, "distress_severity": 0.1, "handover_required": true, "vulnerability_flag": false}}
{{"text": "My account has been hacked and all my money is gone", "emotion": "negative", "sentiment_score": -0.95, "distress_flag": true, "distress_severity": 0.9, "handover_required": true, "vulnerability_flag": false}}
{{"text": "I see unauthorized transactions on my account, someone stole my money", "emotion": "negative", "sentiment_score": -0.9, "distress_flag": true, "distress_severity": 0.85, "handover_required": true, "vulnerability_flag": false}}
{{"text": "There's fraud on my account, I need help immediately", "emotion": "negative", "sentiment_score": -0.9, "distress_flag": true, "distress_severity": 0.88, "handover_required": true, "vulnerability_flag": false}}

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

# Security/fraud incident indicators that should trigger HIGH severity distress and immediate handover
SECURITY_FRAUD_INDICATORS = [
    r'\baccount\s+(has\s+)?been\s+hacked\b',
    r'\baccount\s+compromised\b',
    r'\bunauthorized\s+transactions?\b',
    r'\bfraud\s+(detected|on|in)\b',
    r'\bsuspicious\s+activity\b',
    r'\bmoney\s+(stolen|gone|missing|taken)\b',
    r'\ball\s+my\s+money\s+is\s+gone\b',
    r'\bsomeone\s+(stole|took)\s+my\s+money\b',
    r'\bI\s+didn\'?t\s+make\s+these\s+transactions\b',
    r'\bsomeone\s+accessed\s+my\s+account\b',
    r'\bmy\s+account\s+was\s+breached\b',
]

# Explicit human/operator request indicators that should trigger handover_required=true
HUMAN_REQUEST_INDICATORS = [
    r'\bwant\s+to\s+speak\s+to\s+(a\s+)?(human|person|operator|advisor|representative|agent)\b',
    r'\bcan\s+I\s+talk\s+to\s+(someone|a\s+(human|person|operator|advisor|representative|agent))\b',
    r'\btransfer\s+me\s+to\s+(an\s+)?(operator|human|advisor|representative)\b',
    r'\bconnect\s+me\s+to\s+(an\s+)?(operator|human|advisor|representative)\b',
    r'\bneed\s+to\s+speak\s+to\s+(a\s+)?(human|person|operator|advisor|representative)\b',
    r'\blet\s+me\s+speak\s+to\s+(a\s+)?(real\s+)?(human|person|operator|advisor|representative)\b',
    r'\bI\s+want\s+(human|person|operator|advisor|representative)\s+help\b',
    r'\btalk\s+to\s+(a\s+)?(human|person|operator|advisor|representative)\b',
    r'\bspeak\s+with\s+(a\s+)?(human|person|operator|advisor|representative)\b',
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


# Target distributions for class balance
TARGET_DISTRIBUTIONS = {
    "emotion": {
        "positive": 0.33,  # ~33%
        "neutral": 0.33,   # ~33%
        "negative": 0.34,  # ~33% (slightly higher to account for rounding)
    },
    "distress_flag": {
        "true": 0.25,   # 20-30% target, use 25% as midpoint
        "false": 0.75,  # 70-80% target, use 75% as midpoint
    },
    "handover_required": {
        "true": 0.10,   # 5-15% target, use 10% as midpoint
        "false": 0.90,  # 85-95% target, use 90% as midpoint
    },
    "vulnerability_flag": {
        "true": 0.15,   # 10-20% target, use 15% as midpoint
        "false": 0.85,  # 80-90% target, use 85% as midpoint
    },
}

# Tolerance thresholds (how much deviation is acceptable before rebalancing)
BALANCE_TOLERANCE = 0.10  # 10% deviation is acceptable (e.g., 30% instead of 33% is OK)


def check_distribution_balance(examples: List[Dict]) -> Dict:
    """Check if the distribution matches target distributions."""
    if not examples:
        return {"balanced": True, "issues": []}
    
    total = len(examples)
    issues = []
    
    # Check emotion distribution
    emotion_counts = {"positive": 0, "neutral": 0, "negative": 0}
    for ex in examples:
        emotion = ex.get("emotion", "neutral")
        if emotion in emotion_counts:
            emotion_counts[emotion] += 1
    
    for emotion, target_pct in TARGET_DISTRIBUTIONS["emotion"].items():
        actual_pct = emotion_counts[emotion] / total
        deviation = abs(actual_pct - target_pct)
        if deviation > BALANCE_TOLERANCE:
            issues.append({
                "type": "emotion",
                "class": emotion,
                "target": target_pct,
                "actual": actual_pct,
                "deviation": deviation,
                "count": emotion_counts[emotion],
                "needed": max(0, int((target_pct * total) - emotion_counts[emotion]))
            })
    
    # Check distress_flag distribution
    distress_true = sum(1 for ex in examples if ex.get("distress_flag", False))
    distress_pct = distress_true / total
    target_distress = TARGET_DISTRIBUTIONS["distress_flag"]["true"]
    if abs(distress_pct - target_distress) > BALANCE_TOLERANCE:
        issues.append({
            "type": "distress_flag",
            "class": "true",
            "target": target_distress,
            "actual": distress_pct,
            "deviation": abs(distress_pct - target_distress),
            "count": distress_true,
            "needed": max(0, int((target_distress * total) - distress_true))
        })
    
    # Check handover_required distribution
    handover_true = sum(1 for ex in examples if ex.get("handover_required", False))
    handover_pct = handover_true / total
    target_handover = TARGET_DISTRIBUTIONS["handover_required"]["true"]
    if abs(handover_pct - target_handover) > BALANCE_TOLERANCE:
        issues.append({
            "type": "handover_required",
            "class": "true",
            "target": target_handover,
            "actual": handover_pct,
            "deviation": abs(handover_pct - target_handover),
            "count": handover_true,
            "needed": max(0, int((target_handover * total) - handover_true))
        })
    
    # Check vulnerability_flag distribution
    vulnerability_true = sum(1 for ex in examples if ex.get("vulnerability_flag", False))
    vulnerability_pct = vulnerability_true / total
    target_vulnerability = TARGET_DISTRIBUTIONS["vulnerability_flag"]["true"]
    if abs(vulnerability_pct - target_vulnerability) > BALANCE_TOLERANCE:
        issues.append({
            "type": "vulnerability_flag",
            "class": "true",
            "target": target_vulnerability,
            "actual": vulnerability_pct,
            "deviation": abs(vulnerability_pct - target_vulnerability),
            "count": vulnerability_true,
            "needed": max(0, int((target_vulnerability * total) - vulnerability_true))
        })
    
    return {
        "balanced": len(issues) == 0,
        "issues": issues,
        "emotion_counts": emotion_counts,
        "distress_true": distress_true,
        "handover_true": handover_true,
        "vulnerability_true": vulnerability_true,
    }


def generate_rebalancing_examples(
    model: GenerativeModel,
    issues: List[Dict],
    existing_examples: List[Dict],
    seen: set,
    max_additional: int = 200
) -> List[Dict]:
    """Generate additional examples to rebalance underrepresented classes."""
    if not issues:
        return []
    
    # Group issues by type and class to generate efficiently
    rebalance_needs = {}
    for issue in issues:
        if issue["needed"] > 0:
            key = (issue["type"], issue["class"])
            if key not in rebalance_needs:
                rebalance_needs[key] = 0
            rebalance_needs[key] += issue["needed"]
    
    if not rebalance_needs:
        return []
    
    # Limit total additional examples to avoid excessive generation
    total_needed = sum(rebalance_needs.values())
    if total_needed > max_additional:
        # Scale down proportionally
        scale = max_additional / total_needed
        rebalance_needs = {k: int(v * scale) for k, v in rebalance_needs.items()}
        total_needed = sum(rebalance_needs.values())
    
    print(f"\n🔄 Rebalancing: Generating {total_needed} additional examples for underrepresented classes...", file=sys.stderr)
    
    new_examples = []
    
    for (issue_type, issue_class), needed_count in rebalance_needs.items():
        if needed_count <= 0:
            continue
        
        print(f"  Generating {needed_count} examples for {issue_type}={issue_class}...", file=sys.stderr)
        
        # Create a focused prompt for this specific class
        if issue_type == "emotion":
            emotion_constraint = f"CRITICAL: ALL examples must have emotion=\"{issue_class}\". Do not generate any examples with other emotion values."
        elif issue_type == "distress_flag":
            emotion_constraint = f"CRITICAL: ALL examples must have distress_flag={issue_class.lower()}. Generate examples that clearly indicate {'distress' if issue_class == 'true' else 'no distress'}."
        elif issue_type == "handover_required":
            emotion_constraint = f"CRITICAL: ALL examples must have handover_required={issue_class.lower()}. Generate examples that {'require immediate human handover' if issue_class == 'true' else 'do not require immediate handover'}."
        elif issue_type == "vulnerability_flag":
            emotion_constraint = f"CRITICAL: ALL examples must have vulnerability_flag={issue_class.lower()}. Generate examples that {'show vulnerability indicators' if issue_class == 'true' else 'show no vulnerability indicators'}."
        else:
            emotion_constraint = ""
        
        rebalance_prompt = f"""{PROMPT_TEMPLATE}

{emotion_constraint}

Generate exactly {needed_count} examples that match the above constraint. These examples are for rebalancing the dataset, so ensure they are diverse and not duplicates of existing examples.
"""
        
        try:
            raw = call_gemini(model, rebalance_prompt.format(count=needed_count), expected_count=needed_count)
            examples = parse_examples(raw)
            
            batch_new = []
            for ex in examples:
                try:
                    # Filter triggering content
                    if filter_triggering_content(ex.get("text", "")):
                        continue
                    
                    if validate_example(ex):
                        key = ex["text"].strip().lower()
                        if key in seen:
                            continue
                        seen.add(key)
                        batch_new.append(ex)
                        new_examples.append(ex)
                except Exception:
                    continue
            
            print(f"    Generated {len(batch_new)} valid examples (requested {needed_count})", file=sys.stderr)
            
            # Small delay between rebalancing batches
            if len(rebalance_needs) > 1:
                time.sleep(1)
        
        except Exception as e:
            print(f"    ⚠️  Warning: Failed to generate rebalancing examples for {issue_type}={issue_class}: {e}", file=sys.stderr)
            continue
    
    return new_examples


def validate_example(example: Dict) -> bool:
    required = {"text", "emotion", "sentiment_score", "distress_flag", "distress_severity", "handover_required", "vulnerability_flag"}
    if not all(field in example for field in required):
        return False
    
    if example["emotion"] not in ["positive", "neutral", "negative"]:
        return False
    
    if not isinstance(example["sentiment_score"], (int, float)):
        return False
    if not (-1.0 <= example["sentiment_score"] <= 1.0):
        return False
    
    if not isinstance(example["distress_flag"], bool):
        return False
    
    # Validate distress_severity
    if not isinstance(example["distress_severity"], (int, float)):
        return False
    if not (0.0 <= example["distress_severity"] <= 1.0):
        return False
    
    # Validate handover_required
    if not isinstance(example["handover_required"], bool):
        return False
    
    if not isinstance(example["vulnerability_flag"], bool):
        return False
    
    if not example["text"] or len(example["text"]) < 8:
        return False
    
    # Safety check: If self-harm indicators are present, distress_flag MUST be true and severity high
    text_lower = example["text"].lower()
    has_self_harm_indicator = any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in SELF_HARM_INDICATORS)
    if has_self_harm_indicator:
        if not example["distress_flag"]:
            print(f"⚠️  Warning: Self-harm indicator detected but distress_flag=false. Auto-correcting. Text: '{example['text'][:60]}...'", file=sys.stderr)
            example["distress_flag"] = True
        if example["distress_severity"] < 0.95:
            print(f"⚠️  Warning: Self-harm indicator detected but distress_severity too low ({example['distress_severity']}). Auto-correcting to 0.95.", file=sys.stderr)
            example["distress_severity"] = 0.95
        if not example["handover_required"]:
            print(f"⚠️  Warning: Self-harm indicator detected but handover_required=false. Auto-correcting.", file=sys.stderr)
            example["handover_required"] = True
    
    # Safety check: If security/fraud indicators are present, distress_flag MUST be true, severity HIGH, and handover required
    has_security_fraud_indicator = any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in SECURITY_FRAUD_INDICATORS)
    if has_security_fraud_indicator:
        if not example["distress_flag"]:
            print(f"⚠️  Warning: Security/fraud indicator detected but distress_flag=false. Auto-correcting. Text: '{example['text'][:60]}...'", file=sys.stderr)
            example["distress_flag"] = True
        if example["distress_severity"] < 0.8:
            print(f"⚠️  Warning: Security/fraud indicator detected but distress_severity too low ({example['distress_severity']}). Auto-correcting to 0.85 (HIGH severity).", file=sys.stderr)
            example["distress_severity"] = 0.85
        if not example["handover_required"]:
            print(f"⚠️  Warning: Security/fraud indicator detected but handover_required=false. Auto-correcting (requires immediate fraud team intervention).", file=sys.stderr)
            example["handover_required"] = True
    
    # Validation: distress_severity and distress_flag alignment
    if example["distress_flag"] and example["distress_severity"] < 0.3:
        print(f"⚠️  Warning: distress_flag=true but distress_severity too low ({example['distress_severity']}). Auto-correcting to 0.3.", file=sys.stderr)
        example["distress_severity"] = 0.3
    if not example["distress_flag"] and example["distress_severity"] >= 0.3:
        print(f"⚠️  Warning: distress_flag=false but distress_severity too high ({example['distress_severity']}). Auto-correcting flag.", file=sys.stderr)
        example["distress_flag"] = True
    
    # Validation: handover_required and distress_severity alignment (FCA FG21/1 rule)
    if example["distress_severity"] >= 0.8 and not example["handover_required"]:
        print(f"⚠️  Warning: distress_severity ≥0.8 ({example['distress_severity']}) but handover_required=false. Auto-correcting per FCA FG21/1.", file=sys.stderr)
        example["handover_required"] = True
    
    # Safety check: If explicit human request is present, handover_required MUST be true
    has_human_request = any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in HUMAN_REQUEST_INDICATORS)
    if has_human_request and not example["handover_required"]:
        print(f"⚠️  Warning: Explicit human request detected but handover_required=false. Auto-correcting. Text: '{example['text'][:60]}...'", file=sys.stderr)
        example["handover_required"] = True
    
    return True


def generate_batch(model: GenerativeModel, count: int, batch_size: int = None, base_delay: float = None) -> List[Dict]:
    """
    Generate examples in batches with adaptive rate limiting.
    
    Args:
        model: Gemini model instance
        count: Total number of examples to generate
        batch_size: Examples per batch (auto-calculated if None)
        base_delay: Base delay between batches in seconds (auto-calculated if None)
    """
    # Adaptive batch sizing: smaller batches for very large requests to reduce rate limit risk
    if batch_size is None:
        if count <= 200:
            batch_size = 100
        elif count <= 1000:
            batch_size = 100
        elif count <= 3000:
            batch_size = 80  # Smaller batches for large requests
        else:
            batch_size = 60  # Even smaller for very large requests (4000+)
    
    # Adaptive base delay: longer delays for larger requests
    if base_delay is None:
        if count <= 500:
            base_delay = 2.0
        elif count <= 2000:
            base_delay = 3.0
        else:
            base_delay = 4.0  # Longer delay for 4000+ examples
    
    if count > batch_size:
        estimated_batches = (count + batch_size - 1) // batch_size
        estimated_time_minutes = (estimated_batches * (base_delay + 5)) / 60  # Rough estimate
        print(f"Generating {count} examples in batches of {batch_size} to avoid rate limits...", file=sys.stderr)
        print(f"Estimated: ~{estimated_batches} batches, ~{estimated_time_minutes:.1f} minutes", file=sys.stderr)
        all_unique = []
        seen = set()
        # Dynamic max_batches: allow enough batches for large requests
        max_batches = max(50, (count // batch_size) + 10)  # At least enough for full request + buffer
        batch_num = 0
        invalid_count = 0
        duplicate_count = 0
        filtered_count_total = 0
        consecutive_rate_limits = 0  # Track consecutive rate limit hits
        current_delay = base_delay  # Start with base delay, increase if rate limited
        
        while len(all_unique) < count and batch_num < max_batches:
            remaining = count - len(all_unique)
            current_batch_size = min(batch_size, remaining)
            
            batch_num += 1
            progress_pct = (len(all_unique) / count) * 100 if count > 0 else 0
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"Batch {batch_num}/{estimated_batches} ({progress_pct:.1f}% complete)", file=sys.stderr)
            print(f"Generating {current_batch_size} examples (need {remaining} more)...", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)
            
            try:
                prompt = PROMPT_TEMPLATE.format(count=current_batch_size)
                raw = call_gemini(model, prompt, expected_count=current_batch_size)
                examples = parse_examples(raw)
                
                # Reset consecutive rate limits on success
                consecutive_rate_limits = 0
                current_delay = base_delay  # Reset to base delay after success
                
            except Exception as e:
                error_str = str(e).lower()
                if "rate limit" in error_str or "429" in error_str or "quota" in error_str:
                    consecutive_rate_limits += 1
                    # Exponential backoff: increase delay after rate limit
                    current_delay = base_delay * (2 ** consecutive_rate_limits)
                    current_delay = min(current_delay, 30.0)  # Cap at 30 seconds
                    print(f"  ⚠️  Rate limit hit (consecutive: {consecutive_rate_limits}). Increasing delay to {current_delay:.1f}s...", file=sys.stderr)
                    if consecutive_rate_limits >= 3:
                        print(f"  ⚠️  Multiple rate limits. Waiting {current_delay:.1f} seconds before retry...", file=sys.stderr)
                        time.sleep(current_delay)
                        continue
                    else:
                        time.sleep(current_delay)
                        continue
                else:
                    print(f"  ⚠️  Error in batch {batch_num}: {e}", file=sys.stderr)
                    # For non-rate-limit errors, wait a bit and continue
                    time.sleep(base_delay)
                    continue
            
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
            print(f"  Total so far: {len(all_unique)}/{count} ({progress_pct:.1f}%)", file=sys.stderr)
            print(f"  Cumulative: invalid={invalid_count}, duplicates={duplicate_count}, filtered={filtered_count_total}", file=sys.stderr)
            
            # If we got very few examples, warn and continue
            if len(batch_unique) < current_batch_size * 0.5:  # Got less than 50% of requested
                print(f"  ⚠️  Warning: Got only {len(batch_unique)}/{current_batch_size} examples. Response may have been truncated.", file=sys.stderr)
            
            # Warn if duplicate rate is high (more than 30% of valid examples are duplicates)
            if len(examples) > 0 and batch_duplicate > len(examples) * 0.3:
                print(f"  ⚠️  Warning: High duplicate rate ({batch_duplicate}/{len(examples)} = {batch_duplicate/len(examples)*100:.1f}%). This may slow down generation.", file=sys.stderr)
            
            # Adaptive delay between batches
            if len(all_unique) < count:
                # Increase delay slightly as we progress (to avoid hitting cumulative rate limits)
                progress_factor = len(all_unique) / count
                adaptive_delay = current_delay * (1 + progress_factor * 0.5)  # Up to 50% longer delay near end
                adaptive_delay = min(adaptive_delay, 10.0)  # Cap at 10 seconds for normal delays
                
                print(f"  Waiting {adaptive_delay:.1f} seconds before next batch...", file=sys.stderr)
                time.sleep(adaptive_delay)
        
        unique = all_unique
        if len(unique) < count:
            print(f"⚠️  Warning: Generated {len(unique)}/{count} examples ({len(unique)/count*100:.1f}%). Consider running again or checking for issues.", file=sys.stderr)
        else:
            print(f"✅ Generated {len(unique)} total unique examples (requested {count})", file=sys.stderr)
        
        if duplicate_count > 0:
            print(f"   Note: {duplicate_count} duplicates were filtered out (not counted toward total)", file=sys.stderr)
        if filtered_count_total > 0:
            print(f"   Note: {filtered_count_total} examples with triggering content were filtered out (not counted toward total)", file=sys.stderr)
        
        # Check distribution balance and rebalance if needed
        print(f"\n📊 Checking distribution balance...", file=sys.stderr)
        balance_check = check_distribution_balance(unique)
        
        if not balance_check["balanced"]:
            print(f"⚠️  Distribution imbalance detected:", file=sys.stderr)
            for issue in balance_check["issues"]:
                print(f"   {issue['type']}={issue['class']}: {issue['actual']:.1%} actual vs {issue['target']:.1%} target (deviation: {issue['deviation']:.1%}, need {issue['needed']} more)", file=sys.stderr)
            
            # Generate rebalancing examples
            rebalance_examples = generate_rebalancing_examples(
                model, balance_check["issues"], unique, seen, max_additional=min(200, count // 2)
            )
            
            if rebalance_examples:
                unique.extend(rebalance_examples)
                print(f"✅ Added {len(rebalance_examples)} rebalancing examples", file=sys.stderr)
                
                # Re-check balance after rebalancing
                balance_check_after = check_distribution_balance(unique)
                if balance_check_after["balanced"]:
                    print(f"✅ Distribution is now balanced!", file=sys.stderr)
                else:
                    remaining_issues = len(balance_check_after["issues"])
                    print(f"⚠️  Distribution improved but {remaining_issues} imbalance(s) remain. Consider generating more examples.", file=sys.stderr)
            else:
                print(f"⚠️  Could not generate rebalancing examples. Distribution may be imbalanced.", file=sys.stderr)
        else:
            print(f"✅ Distribution is well-balanced!", file=sys.stderr)
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
        
        # Check distribution balance and rebalance if needed (for single batch too)
        print(f"\n📊 Checking distribution balance...", file=sys.stderr)
        balance_check = check_distribution_balance(unique)
        
        if not balance_check["balanced"]:
            print(f"⚠️  Distribution imbalance detected:", file=sys.stderr)
            for issue in balance_check["issues"]:
                print(f"   {issue['type']}={issue['class']}: {issue['actual']:.1%} actual vs {issue['target']:.1%} target (deviation: {issue['deviation']:.1%}, need {issue['needed']} more)", file=sys.stderr)
            
            # Generate rebalancing examples
            rebalance_examples = generate_rebalancing_examples(
                model, balance_check["issues"], unique, seen, max_additional=min(200, count // 2)
            )
            
            if rebalance_examples:
                unique.extend(rebalance_examples)
                print(f"✅ Added {len(rebalance_examples)} rebalancing examples", file=sys.stderr)
                
                # Re-check balance after rebalancing
                balance_check_after = check_distribution_balance(unique)
                if balance_check_after["balanced"]:
                    print(f"✅ Distribution is now balanced!", file=sys.stderr)
                else:
                    remaining_issues = len(balance_check_after["issues"])
                    print(f"⚠️  Distribution improved but {remaining_issues} imbalance(s) remain. Consider generating more examples.", file=sys.stderr)
            else:
                print(f"⚠️  Could not generate rebalancing examples. Distribution may be imbalanced.", file=sys.stderr)
        else:
            print(f"✅ Distribution is well-balanced!", file=sys.stderr)
    
    return unique


def main():
    parser = argparse.ArgumentParser(description="Generate emotion training data")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help=f"GCP project ID (default: {DEFAULT_PROJECT})")
    parser.add_argument("--location", default=DEFAULT_LOCATION, help=f"Vertex AI location (default: {DEFAULT_LOCATION})")
    parser.add_argument("--output", required=True, help="Output directory (local or gs://)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--examples", type=int, default=500, help="Number of examples to generate")
    parser.add_argument("--batch-size", type=int, default=None, help="Examples per batch (auto-calculated based on total count if not specified)")
    parser.add_argument("--delay", type=float, default=None, help="Base delay between batches in seconds (auto-calculated based on total count if not specified)")
    
    args = parser.parse_args()
    
    if not args.output.startswith("gs://"):
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir.absolute()}", file=sys.stderr)
    
    vertex_init(project=args.project, location=args.location)
    model = GenerativeModel(args.model)
    
    gcs_client = storage.Client(project=args.project) if args.output.startswith("gs://") else None
    
    print(f"Generating {args.examples} emotion training examples...", file=sys.stderr)
    if args.batch_size:
        print(f"Using custom batch size: {args.batch_size}", file=sys.stderr)
    if args.delay:
        print(f"Using custom delay: {args.delay}s", file=sys.stderr)
    examples = generate_batch(model, args.examples, batch_size=args.batch_size, base_delay=args.delay)
    
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
    handover_true = 0
    severity_ranges = {"none_low": 0, "medium": 0, "high_critical": 0}  # 0.0-0.6, 0.6-0.8, 0.8-1.0
    
    for ex in examples:
        emotion = ex.get("emotion", "neutral")
        emotion_dist[emotion] = emotion_dist.get(emotion, 0) + 1
        if ex.get("distress_flag", False):
            distress_true += 1
        if ex.get("vulnerability_flag", False):
            vulnerability_true += 1
        if ex.get("handover_required", False):
            handover_true += 1
        
        # Categorize severity
        severity = ex.get("distress_severity", 0.0)
        if severity < 0.6:
            severity_ranges["none_low"] += 1
        elif severity < 0.8:
            severity_ranges["medium"] += 1
        else:
            severity_ranges["high_critical"] += 1
    
    # Generate prompt hash for tracking
    prompt_hash = hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest()[:16]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"emotion_training_data_{timestamp}.json"
    
    # Governance metadata
    governance_metadata = {
        "taxonomy_version": "8.0",
        "sil_version": "2.0",
        "generator_version": "2.2",
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
                },
                "handover_required": {
                    "true": handover_true,
                    "false": len(examples) - handover_true,
                    "true_percentage": round(handover_true / len(examples) * 100, 2) if examples else 0
                },
                "distress_severity_ranges": {
                    "none_low_0.0_0.6": severity_ranges["none_low"],
                    "medium_0.6_0.8": severity_ranges["medium"],
                    "high_critical_0.8_1.0": severity_ranges["high_critical"]
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
    print(f"  Handover required: {governance_metadata['statistics']['label_distribution']['handover_required']['true']}/{len(examples)} ({governance_metadata['statistics']['label_distribution']['handover_required']['true_percentage']}%)", file=sys.stderr)
    print(f"  Vulnerability flag: {governance_metadata['statistics']['label_distribution']['vulnerability_flag']['true']}/{len(examples)} ({governance_metadata['statistics']['label_distribution']['vulnerability_flag']['true_percentage']}%)", file=sys.stderr)
    print(f"  Distress severity ranges: {governance_metadata['statistics']['label_distribution']['distress_severity_ranges']}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)


if __name__ == "__main__":
    main()

