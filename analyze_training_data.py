#!/usr/bin/env python3
"""
analyze_training_data.py

Analyze training data quality and distribution for SIL fine-tuning.
Checks balance across labels, data quality, and provides recommendations.

Usage:
  python analyze_training_data.py --data-dir ./training_data
  python analyze_training_data.py --data-dir ./training_data --topic savings
  python analyze_training_data.py --data-dir ./training_data --topic savings --stage goal_setup
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def load_training_data(data_dir: Path, topic: Optional[str] = None, stage: Optional[str] = None) -> List[Dict]:
    """Load all training data from JSON files."""
    examples = []
    
    for topic_dir in data_dir.iterdir():
        if not topic_dir.is_dir():
            continue
        
        topic_name = topic_dir.name
        if topic and topic_name != topic:
            continue
        
        for stage_dir in topic_dir.iterdir():
            if not stage_dir.is_dir():
                continue
            
            stage_name = stage_dir.name
            if stage and stage_name != stage:
                continue
            
            for json_file in stage_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if "training_data" in data:
                            examples.extend(data["training_data"])
                        elif isinstance(data, list):
                            examples.extend(data)
                except Exception as e:
                    print(f"Warning: Failed to load {json_file}: {e}", file=sys.stderr)
    
    return examples


def bucketize_advice_risk(score: float) -> str:
    """Bucketize advice_risk_score into low/medium/high."""
    if score <= 0.3:
        return "low"
    elif score <= 0.6:
        return "medium"
    else:
        return "high"


def analyze_distribution(examples: List[Dict]) -> Dict:
    """Analyze label distribution across all dimensions."""
    stats = {
        "total_examples": len(examples),
        "topics": Counter(),
        "intent_types": Counter(),
        "query_types": Counter(),
        "stages": Counter(),
        "domain_scopes": Counter(),
        "advice_risk_buckets": Counter(),
        "topic_stage": Counter(),
        "topic_intent": Counter(),
        "intent_advice_risk": Counter(),
        "text_lengths": [],
        "missing_fields": [],
    }
    
    for ex in examples:
        # Basic labels
        stats["topics"][ex.get("topic", "missing")] += 1
        stats["intent_types"][ex.get("intent_type", "missing")] += 1
        stats["query_types"][ex.get("query_type", "missing")] += 1
        stats["stages"][ex.get("stage", "missing")] += 1
        stats["domain_scopes"][ex.get("domain_scope", "missing")] += 1
        
        # Advice risk bucket
        advice_risk = ex.get("advice_risk_score")
        if advice_risk is not None:
            stats["advice_risk_buckets"][bucketize_advice_risk(advice_risk)] += 1
        
        # Combinations
        topic = ex.get("topic", "unknown")
        stage = ex.get("stage", "unknown")
        intent = ex.get("intent_type", "unknown")
        stats["topic_stage"][f"{topic}.{stage}"] += 1
        stats["topic_intent"][f"{topic}.{intent}"] += 1
        
        if advice_risk is not None:
            bucket = bucketize_advice_risk(advice_risk)
            stats["intent_advice_risk"][f"{intent}.{bucket}"] += 1
        
        # Text quality
        text = ex.get("text", "")
        stats["text_lengths"].append(len(text))
        
        # Check for missing required fields
        required = {"text", "topic", "intent_type", "query_type", "stage", "domain_scope", "advice_risk_score"}
        missing = required - set(ex.keys())
        if missing:
            stats["missing_fields"].append((ex.get("text", "")[:50], missing))
    
    return stats


def calculate_balance_metrics(stats: Dict, total: int) -> Dict:
    """Calculate balance metrics and recommendations."""
    metrics = {}
    
    # Intent type balance
    intent_counts = stats["intent_types"]
    if intent_counts:
        intent_std = np.std(list(intent_counts.values()))
        intent_mean = np.mean(list(intent_counts.values()))
        intent_cv = intent_std / intent_mean if intent_mean > 0 else 0
        metrics["intent_balance"] = {
            "coefficient_of_variation": float(intent_cv),
            "status": "balanced" if intent_cv < 0.3 else "imbalanced" if intent_cv > 0.5 else "moderate",
            "distribution": dict(intent_counts),
        }
    
    # Query type balance
    query_counts = stats["query_types"]
    if query_counts:
        query_std = np.std(list(query_counts.values()))
        query_mean = np.mean(list(query_counts.values()))
        query_cv = query_std / query_mean if query_mean > 0 else 0
        metrics["query_balance"] = {
            "coefficient_of_variation": float(query_cv),
            "status": "balanced" if query_cv < 0.3 else "imbalanced" if query_cv > 0.5 else "moderate",
            "distribution": dict(query_counts),
        }
    
    # Domain scope balance
    domain_counts = stats["domain_scopes"]
    if domain_counts:
        general = domain_counts.get("general", 0)
        bank_specific = domain_counts.get("bank_specific", 0)
        total_domain = general + bank_specific
        if total_domain > 0:
            general_pct = general / total_domain
            bank_pct = bank_specific / total_domain
            metrics["domain_balance"] = {
                "general_percentage": float(general_pct),
                "bank_specific_percentage": float(bank_pct),
                "status": "balanced" if 0.7 <= general_pct <= 0.8 else "imbalanced",
                "distribution": dict(domain_counts),
            }
    
    # Advice risk balance
    risk_counts = stats["advice_risk_buckets"]
    if risk_counts:
        risk_std = np.std(list(risk_counts.values()))
        risk_mean = np.mean(list(risk_counts.values()))
        risk_cv = risk_std / risk_mean if risk_mean > 0 else 0
        metrics["advice_risk_balance"] = {
            "coefficient_of_variation": float(risk_cv),
            "status": "balanced" if risk_cv < 0.3 else "imbalanced" if risk_cv > 0.5 else "moderate",
            "distribution": dict(risk_counts),
        }
    
    return metrics


def print_report(stats: Dict, metrics: Dict, topic: Optional[str] = None, stage: Optional[str] = None):
    """Print comprehensive analysis report."""
    total = stats["total_examples"]
    
    print("=" * 80)
    print("TRAINING DATA ANALYSIS REPORT")
    print("=" * 80)
    if topic:
        print(f"Topic: {topic}")
    if stage:
        print(f"Stage: {stage}")
    print(f"Total Examples: {total}")
    print()
    
    if total == 0:
        print("⚠️  No training data found!")
        return
    
    # Basic distribution
    print("-" * 80)
    print("BASIC DISTRIBUTION")
    print("-" * 80)
    
    print(f"\nTopics ({len(stats['topics'])}):")
    for topic_name, count in stats["topics"].most_common():
        pct = (count / total) * 100
        print(f"  {topic_name:20s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\nStages ({len(stats['stages'])}):")
    for stage_name, count in stats["stages"].most_common():
        pct = (count / total) * 100
        print(f"  {stage_name:20s}: {count:4d} ({pct:5.1f}%)")
    
    # Balance analysis
    print("\n" + "-" * 80)
    print("BALANCE ANALYSIS")
    print("-" * 80)
    
    if "intent_balance" in metrics:
        ib = metrics["intent_balance"]
        print(f"\nIntent Type Balance (CV: {ib['coefficient_of_variation']:.2f}) - {ib['status'].upper()}:")
        for intent, count in sorted(ib["distribution"].items()):
            pct = (count / total) * 100
            print(f"  {intent:20s}: {count:4d} ({pct:5.1f}%)")
    
    if "query_balance" in metrics:
        qb = metrics["query_balance"]
        print(f"\nQuery Type Balance (CV: {qb['coefficient_of_variation']:.2f}) - {qb['status'].upper()}:")
        for query, count in sorted(qb["distribution"].items()):
            pct = (count / total) * 100
            print(f"  {query:20s}: {count:4d} ({pct:5.1f}%)")
    
    if "domain_balance" in metrics:
        db = metrics["domain_balance"]
        print(f"\nDomain Scope Balance - {db['status'].upper()}:")
        print(f"  General:        {db['general_percentage']*100:.1f}%")
        print(f"  Bank Specific: {db['bank_specific_percentage']*100:.1f}%")
        print(f"  (Recommended: 70-80% general, 20-30% bank_specific)")
    
    if "advice_risk_balance" in metrics:
        arb = metrics["advice_risk_balance"]
        print(f"\nAdvice Risk Balance (CV: {arb['coefficient_of_variation']:.2f}) - {arb['status'].upper()}:")
        for risk, count in sorted(arb["distribution"].items()):
            pct = (count / total) * 100
            print(f"  {risk:20s}: {count:4d} ({pct:5.1f}%)")
    
    # Data quality
    print("\n" + "-" * 80)
    print("DATA QUALITY")
    print("-" * 80)
    
    text_lengths = stats["text_lengths"]
    if text_lengths:
        print(f"\nText Length Statistics:")
        print(f"  Min:    {min(text_lengths)} chars")
        print(f"  Max:    {max(text_lengths)} chars")
        print(f"  Mean:   {np.mean(text_lengths):.1f} chars")
        print(f"  Median: {np.median(text_lengths):.1f} chars")
        print(f"  Std:    {np.std(text_lengths):.1f} chars")
        
        # Flag very short or very long texts
        short = [l for l in text_lengths if l < 10]
        long = [l for l in text_lengths if l > 200]
        if short:
            print(f"  ⚠️  {len(short)} examples with <10 characters")
        if long:
            print(f"  ⚠️  {len(long)} examples with >200 characters")
    
    if stats["missing_fields"]:
        print(f"\n⚠️  {len(stats['missing_fields'])} examples with missing fields:")
        for text, missing in stats["missing_fields"][:10]:
            print(f"    '{text}...' - missing: {missing}")
        if len(stats["missing_fields"]) > 10:
            print(f"    ... and {len(stats['missing_fields']) - 10} more")
    
    # Recommendations
    print("\n" + "-" * 80)
    print("RECOMMENDATIONS")
    print("-" * 80)
    
    recommendations = []
    
    if total < 100:
        recommendations.append(f"⚠️  Low sample size ({total} examples). Recommend at least 200-300 per stage.")
    elif total < 200:
        recommendations.append(f"⚠️  Moderate sample size ({total} examples). Consider 250-300 for better quality.")
    else:
        recommendations.append(f"✅ Good sample size ({total} examples)")
    
    if "intent_balance" in metrics:
        ib = metrics["intent_balance"]
        if ib["status"] == "imbalanced":
            recommendations.append("⚠️  Intent types are imbalanced. Generate more examples for underrepresented intents.")
        elif ib["status"] == "moderate":
            recommendations.append("⚠️  Intent types are moderately balanced. Consider generating more examples for underrepresented intents.")
    
    if "domain_balance" in metrics:
        db = metrics["domain_balance"]
        if db["status"] == "imbalanced":
            rec_pct = db["general_percentage"] * 100
            if rec_pct < 70:
                recommendations.append(f"⚠️  Too many bank_specific queries ({100-rec_pct:.1f}%). Aim for 70-80% general.")
            else:
                recommendations.append(f"⚠️  Too few bank_specific queries ({100-rec_pct:.1f}%). Aim for 20-30% bank_specific.")
    
    if "advice_risk_balance" in metrics:
        arb = metrics["advice_risk_balance"]
        if arb["status"] == "imbalanced":
            recommendations.append("⚠️  Advice risk buckets are imbalanced. Ensure good distribution across low/medium/high.")
    
    if not recommendations:
        recommendations.append("✅ Data looks good for fine-tuning!")
    
    for rec in recommendations:
        print(f"  {rec}")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Analyze training data quality and distribution")
    parser.add_argument("--data-dir", required=True, help="Directory containing training data")
    parser.add_argument("--topic", help="Filter by topic")
    parser.add_argument("--stage", help="Filter by stage")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Load data
    examples = load_training_data(data_dir, topic=args.topic, stage=args.stage)
    
    if not examples:
        print(f"Error: No training data found in {data_dir}", file=sys.stderr)
        if args.topic:
            print(f"  (filtered by topic: {args.topic})", file=sys.stderr)
        if args.stage:
            print(f"  (filtered by stage: {args.stage})", file=sys.stderr)
        sys.exit(1)
    
    # Analyze
    stats = analyze_distribution(examples)
    metrics = calculate_balance_metrics(stats, len(examples))
    
    # Output
    if args.json:
        output = {
            "summary": {
                "total_examples": stats["total_examples"],
                "topic": args.topic,
                "stage": args.stage,
            },
            "distribution": {
                "topics": dict(stats["topics"]),
                "stages": dict(stats["stages"]),
                "intent_types": dict(stats["intent_types"]),
                "query_types": dict(stats["query_types"]),
                "domain_scopes": dict(stats["domain_scopes"]),
                "advice_risk_buckets": dict(stats["advice_risk_buckets"]),
            },
            "balance_metrics": metrics,
            "quality": {
                "text_length_stats": {
                    "min": int(min(stats["text_lengths"])) if stats["text_lengths"] else 0,
                    "max": int(max(stats["text_lengths"])) if stats["text_lengths"] else 0,
                    "mean": float(np.mean(stats["text_lengths"])) if stats["text_lengths"] else 0,
                    "median": float(np.median(stats["text_lengths"])) if stats["text_lengths"] else 0,
                    "std": float(np.std(stats["text_lengths"])) if stats["text_lengths"] else 0,
                },
                "missing_fields_count": len(stats["missing_fields"]),
            },
        }
        print(json.dumps(output, indent=2))
    else:
        print_report(stats, metrics, topic=args.topic, stage=args.stage)


if __name__ == "__main__":
    main()

