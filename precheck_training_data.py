#!/usr/bin/env python3
"""
precheck_training_data.py

Pre-check training data before fine-tuning to catch issues early.
Validates format, distribution, and quality.

Usage:
  python precheck_training_data.py --training-data ./training_data
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set

# Label definitions from finetune_sil_model.py
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

ADVICE_RISK_RANGES = {
    "low": (0.0, 0.3),
    "medium": (0.4, 0.6),
    "high": (0.7, 1.0)
}

REQUIRED_FIELDS = ["text", "topic", "intent_type", "query_type", "stage", "domain_scope", "advice_risk_score"]


def load_training_data(data_dir: Path) -> tuple[List[Dict], Dict[str, int]]:
    """Load all training data and track file sources."""
    examples = []
    file_stats = {}
    
    for topic_dir in sorted(data_dir.iterdir()):
        if not topic_dir.is_dir():
            continue
        
        for stage_dir in sorted(topic_dir.iterdir()):
            if not stage_dir.is_dir():
                continue
            
            for json_file in sorted(stage_dir.glob("*.json")):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if "training_data" in data:
                            file_examples = data["training_data"]
                        elif isinstance(data, list):
                            file_examples = data
                        else:
                            print(f"⚠️  Warning: Unknown format in {json_file}", file=sys.stderr)
                            continue
                        
                        file_stats[str(json_file)] = len(file_examples)
                        examples.extend(file_examples)
                except Exception as e:
                    print(f"❌ Error loading {json_file}: {e}", file=sys.stderr)
    
    return examples, file_stats


def validate_example(example: Dict, idx: int) -> tuple[bool, List[str]]:
    """Validate a single example and return (is_valid, errors)."""
    errors = []
    
    # Check required fields
    missing = [f for f in REQUIRED_FIELDS if f not in example]
    if missing:
        errors.append(f"Missing fields: {missing}")
    
    # Validate text
    text = example.get("text", "")
    if not text or not isinstance(text, str):
        errors.append(f"Invalid text: {type(text)}")
    elif len(text.strip()) < 5:
        errors.append(f"Text too short: {len(text)} chars")
    elif len(text) > 1000:
        errors.append(f"Text too long: {len(text)} chars")
    
    # Validate topic
    topic = example.get("topic", "")
    if topic not in TOPIC_LABELS:
        errors.append(f"Invalid topic: {topic}")
    
    # Validate intent_type
    intent_type = example.get("intent_type", "")
    if intent_type not in INTENT_TYPE_LABELS:
        errors.append(f"Invalid intent_type: {intent_type}")
    
    # Validate query_type
    query_type = example.get("query_type", "")
    if query_type not in QUERY_TYPE_LABELS:
        errors.append(f"Invalid query_type: {query_type}")
    
    # Validate stage
    stage = example.get("stage", "")
    if stage not in STAGE_LABELS:
        errors.append(f"Invalid stage: {stage}")
    
    # Validate domain_scope
    domain_scope = example.get("domain_scope", "")
    if domain_scope not in DOMAIN_SCOPE_LABELS:
        errors.append(f"Invalid domain_scope: {domain_scope}")
    
    # Validate advice_risk_score
    advice_risk = example.get("advice_risk_score")
    if advice_risk is None:
        errors.append("Missing advice_risk_score")
    elif not isinstance(advice_risk, (int, float)):
        errors.append(f"advice_risk_score must be number, got {type(advice_risk)}")
    elif not (0.0 <= advice_risk <= 1.0):
        errors.append(f"advice_risk_score out of range: {advice_risk}")
    
    return len(errors) == 0, errors


def check_data_quality(examples: List[Dict]) -> Dict:
    """Check data quality metrics."""
    stats = {
        "total_examples": len(examples),
        "valid_examples": 0,
        "invalid_examples": 0,
        "validation_errors": defaultdict(int),
        "duplicates": 0,
        "text_lengths": [],
        "label_distributions": {
            "topic": Counter(),
            "intent_type": Counter(),
            "query_type": Counter(),
            "stage": Counter(),
            "domain_scope": Counter(),
            "advice_risk_buckets": Counter(),
        },
        "topic_stage_matrix": defaultdict(lambda: Counter()),
        "domain_distribution": Counter(),
        "advice_risk_scores": [],
    }
    
    seen_texts = set()
    
    for idx, example in enumerate(examples):
        is_valid, errors = validate_example(example, idx)
        
        if is_valid:
            stats["valid_examples"] += 1
            
            # Check duplicates
            text_lower = example["text"].strip().lower()
            if text_lower in seen_texts:
                stats["duplicates"] += 1
            else:
                seen_texts.add(text_lower)
            
            # Text length
            stats["text_lengths"].append(len(example["text"]))
            
            # Label distributions
            stats["label_distributions"]["topic"][example["topic"]] += 1
            stats["label_distributions"]["intent_type"][example["intent_type"]] += 1
            stats["label_distributions"]["query_type"][example["query_type"]] += 1
            stats["label_distributions"]["stage"][example["stage"]] += 1
            stats["label_distributions"]["domain_scope"][example["domain_scope"]] += 1
            
            # Topic-stage matrix
            stats["topic_stage_matrix"][example["topic"]][example["stage"]] += 1
            
            # Advice risk bucket
            advice_risk = example["advice_risk_score"]
            stats["advice_risk_scores"].append(advice_risk)
            if advice_risk <= 0.3:
                bucket = "low"
            elif advice_risk <= 0.6:
                bucket = "medium"
            else:
                bucket = "high"
            stats["label_distributions"]["advice_risk_buckets"][bucket] += 1
        else:
            stats["invalid_examples"] += 1
            for error in errors:
                stats["validation_errors"][error] += 1
    
    return stats


def check_coverage(stats: Dict) -> Dict:
    """Check label coverage."""
    coverage = {
        "topics_missing": [],
        "intent_types_missing": [],
        "query_types_missing": [],
        "stages_missing": [],
        "domain_scopes_missing": [],
        "advice_risk_buckets_missing": [],
    }
    
    # Check topic coverage
    present_topics = set(stats["label_distributions"]["topic"].keys())
    coverage["topics_missing"] = [t for t in TOPIC_LABELS if t not in present_topics]
    
    # Check intent_type coverage
    present_intents = set(stats["label_distributions"]["intent_type"].keys())
    coverage["intent_types_missing"] = [i for i in INTENT_TYPE_LABELS if i not in present_intents]
    
    # Check query_type coverage
    present_queries = set(stats["label_distributions"]["query_type"].keys())
    coverage["query_types_missing"] = [q for q in QUERY_TYPE_LABELS if q not in present_queries]
    
    # Check stage coverage
    present_stages = set(stats["label_distributions"]["stage"].keys())
    coverage["stages_missing"] = [s for s in STAGE_LABELS if s not in present_stages]
    
    # Check domain_scope coverage
    present_domains = set(stats["label_distributions"]["domain_scope"].keys())
    coverage["domain_scopes_missing"] = [d for d in DOMAIN_SCOPE_LABELS if d not in present_domains]
    
    # Check advice_risk buckets
    present_buckets = set(stats["label_distributions"]["advice_risk_buckets"].keys())
    coverage["advice_risk_buckets_missing"] = [b for b in ["low", "medium", "high"] if b not in present_buckets]
    
    return coverage


def calculate_balance_metrics(stats: Dict) -> Dict:
    """Calculate balance metrics (coefficient of variation)."""
    balance = {}
    
    for label_type, distribution in stats["label_distributions"].items():
        if not distribution:
            continue
        
        counts = list(distribution.values())
        if len(counts) > 1:
            mean_count = sum(counts) / len(counts)
            if mean_count > 0:
                variance = sum((c - mean_count) ** 2 for c in counts) / len(counts)
                std_dev = variance ** 0.5
                cv = std_dev / mean_count if mean_count > 0 else float('inf')
                balance[label_type] = {
                    "cv": cv,
                    "min": min(counts),
                    "max": max(counts),
                    "mean": mean_count,
                }
    
    return balance


def print_report(stats: Dict, coverage: Dict, balance: Dict, file_stats: Dict):
    """Print comprehensive report."""
    print("=" * 80)
    print("TRAINING DATA PRE-CHECK REPORT")
    print("=" * 80)
    print()
    
    # File statistics
    print("📁 FILE STATISTICS:")
    print(f"   Total files loaded: {len(file_stats)}")
    total_in_files = sum(file_stats.values())
    print(f"   Total examples in files: {total_in_files}")
    print()
    
    # Data quality
    print("✅ DATA QUALITY:")
    print(f"   Total examples: {stats['total_examples']}")
    print(f"   Valid examples: {stats['valid_examples']} ({stats['valid_examples']/stats['total_examples']*100:.1f}%)")
    print(f"   Invalid examples: {stats['invalid_examples']} ({stats['invalid_examples']/stats['total_examples']*100:.1f}%)")
    print(f"   Duplicates: {stats['duplicates']}")
    
    if stats['text_lengths']:
        print(f"   Text length: min={min(stats['text_lengths'])}, max={max(stats['text_lengths'])}, avg={sum(stats['text_lengths'])/len(stats['text_lengths']):.1f}")
    print()
    
    # Validation errors
    if stats['validation_errors']:
        print("❌ VALIDATION ERRORS:")
        for error, count in sorted(stats['validation_errors'].items(), key=lambda x: -x[1]):
            print(f"   {error}: {count}")
        print()
    
    # Label distributions
    print("📊 LABEL DISTRIBUTIONS:")
    for label_type, distribution in stats['label_distributions'].items():
        if distribution:
            print(f"\n   {label_type.upper()}:")
            for label, count in sorted(distribution.items(), key=lambda x: -x[1]):
                pct = count / stats['valid_examples'] * 100
                print(f"      {label}: {count} ({pct:.1f}%)")
    print()
    
    # Topic-stage matrix
    print("📋 TOPIC-STAGE MATRIX (sample):")
    for topic, stages in list(stats['topic_stage_matrix'].items())[:5]:  # Show first 5 topics
        print(f"   {topic}:")
        for stage, count in sorted(stages.items(), key=lambda x: -x[1])[:3]:  # Top 3 stages
            print(f"      {stage}: {count}")
    if len(stats['topic_stage_matrix']) > 5:
        print(f"   ... and {len(stats['topic_stage_matrix']) - 5} more topics")
    print()
    
    # Coverage
    print("🎯 LABEL COVERAGE:")
    any_missing = False
    if coverage['topics_missing']:
        print(f"   ⚠️  Missing topics: {', '.join(coverage['topics_missing'])}")
        any_missing = True
    if coverage['intent_types_missing']:
        print(f"   ⚠️  Missing intent_types: {', '.join(coverage['intent_types_missing'])}")
        any_missing = True
    if coverage['query_types_missing']:
        print(f"   ⚠️  Missing query_types: {', '.join(coverage['query_types_missing'])}")
        any_missing = True
    if coverage['stages_missing']:
        print(f"   ⚠️  Missing stages: {', '.join(coverage['stages_missing'])}")
        any_missing = True
    if coverage['domain_scopes_missing']:
        print(f"   ⚠️  Missing domain_scopes: {', '.join(coverage['domain_scopes_missing'])}")
        any_missing = True
    if coverage['advice_risk_buckets_missing']:
        print(f"   ⚠️  Missing advice_risk buckets: {', '.join(coverage['advice_risk_buckets_missing'])}")
        any_missing = True
    if not any_missing:
        print("   ✅ All expected labels present")
    print()
    
    # Balance metrics
    print("⚖️  DATA BALANCE (Coefficient of Variation):")
    print("   CV < 0.5 = well balanced, CV > 1.0 = imbalanced")
    for label_type, metrics in balance.items():
        cv_status = "✅" if metrics['cv'] < 0.5 else "⚠️" if metrics['cv'] < 1.0 else "❌"
        print(f"   {cv_status} {label_type}: CV={metrics['cv']:.2f}, min={metrics['min']}, max={metrics['max']}, mean={metrics['mean']:.1f}")
    print()
    
    # Domain scope distribution
    domain_dist = stats['label_distributions']['domain_scope']
    if domain_dist:
        total_domain = sum(domain_dist.values())
        general_pct = domain_dist.get('general', 0) / total_domain * 100
        bank_pct = domain_dist.get('bank_specific', 0) / total_domain * 100
        print("🏦 DOMAIN SCOPE DISTRIBUTION:")
        print(f"   general: {domain_dist.get('general', 0)} ({general_pct:.1f}%)")
        print(f"   bank_specific: {domain_dist.get('bank_specific', 0)} ({bank_pct:.1f}%)")
        if general_pct < 70 or general_pct > 80:
            print(f"   ⚠️  Warning: Expected 70-80% general, got {general_pct:.1f}%")
        print()
    
    # Advice risk distribution
    if stats['advice_risk_scores']:
        avg_risk = sum(stats['advice_risk_scores']) / len(stats['advice_risk_scores'])
        print("⚠️  ADVICE RISK SCORES:")
        print(f"   Average: {avg_risk:.3f}")
        print(f"   Range: {min(stats['advice_risk_scores']):.3f} - {max(stats['advice_risk_scores']):.3f}")
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY:")
    issues = []
    if stats['invalid_examples'] > 0:
        issues.append(f"{stats['invalid_examples']} invalid examples")
    if stats['duplicates'] > 0:
        issues.append(f"{stats['duplicates']} duplicates")
    if any_missing:
        issues.append("Missing labels")
    if stats['valid_examples'] == 0:
        issues.append("No valid examples")
    
    if issues:
        print(f"❌ Issues found: {', '.join(issues)}")
        print("   Fix these before fine-tuning!")
        return False
    else:
        print("✅ All checks passed! Data is ready for fine-tuning.")
        return True


def main():
    parser = argparse.ArgumentParser(description="Pre-check training data before fine-tuning")
    parser.add_argument("--training-data", required=True, help="Directory containing training JSON files")
    parser.add_argument("--output-report", help="Optional: Save JSON report to file")
    
    args = parser.parse_args()
    
    data_dir = Path(args.training_data)
    if not data_dir.exists():
        print(f"❌ Error: Training data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading training data from: {data_dir}")
    examples, file_stats = load_training_data(data_dir)
    
    if not examples:
        print(f"❌ Error: No training data found in {data_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loaded {len(examples)} examples from {len(file_stats)} files\n")
    
    # Run checks
    stats = check_data_quality(examples)
    coverage = check_coverage(stats)
    balance = calculate_balance_metrics(stats)
    
    # Print report
    is_valid = print_report(stats, coverage, balance, file_stats)
    
    # Save report if requested
    if args.output_report:
        report = {
            "stats": {k: dict(v) if isinstance(v, (Counter, defaultdict)) else v for k, v in stats.items()},
            "coverage": coverage,
            "balance": balance,
            "file_stats": file_stats,
        }
        with open(args.output_report, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n📄 Report saved to: {args.output_report}")
    
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()

