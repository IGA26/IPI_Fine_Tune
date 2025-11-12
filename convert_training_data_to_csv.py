#!/usr/bin/env python3
"""
convert_training_data_to_csv.py

Convert training data JSON files to CSV format.
For a given topic (e.g., "savings"), finds all JSON files in that topic's directory,
extracts the training_data array from each, and combines them into a single CSV file.

Usage:
  # Convert all JSON files for a specific topic
  python convert_training_data_to_csv.py \
      --input-dir ./training_data \
      --topic savings \
      --output ./training_data/savings/savings_all.csv

  # Convert all topics
  python convert_training_data_to_csv.py \
      --input-dir ./training_data \
      --all-topics \
      --output ./training_data

  # Convert specific JSON files
  python convert_training_data_to_csv.py \
      --input-dir ./training_data \
      --topic savings \
      --files savings_training_data_20251021_130708.json \
      --output ./training_data/savings/savings_specific.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Set


def load_json_file(file_path: Path) -> List[Dict]:
    """Load training data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both formats: direct array or object with "training_data" key
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "training_data" in data:
            return data["training_data"]
        else:
            print(f"⚠️  Warning: Unexpected JSON structure in {file_path}", file=sys.stderr)
            return []
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in {file_path}: {e}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}", file=sys.stderr)
        return []


def get_all_fields(examples: List[Dict]) -> Set[str]:
    """Extract all unique field names from examples."""
    fields = set()
    for example in examples:
        fields.update(example.keys())
    return fields


def convert_to_csv(
    input_dir: Path,
    topic: str = None,
    all_topics: bool = False,
    specific_files: List[str] = None,
    output_path: Path = None
):
    """
    Convert training data JSON files to CSV.
    
    Args:
        input_dir: Base directory containing topic subdirectories
        topic: Specific topic to convert (e.g., "savings")
        all_topics: If True, convert all topics found in input_dir
        specific_files: List of specific JSON filenames to convert (optional)
        output_path: Output CSV file path (or directory if all_topics)
    """
    if all_topics:
        # Find all topic directories
        topic_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
        if not topic_dirs:
            print(f"❌ No topic directories found in {input_dir}", file=sys.stderr)
            sys.exit(1)
        
        print(f"Found {len(topic_dirs)} topic directories")
        for topic_dir in sorted(topic_dirs):
            topic_name = topic_dir.name
            output_file = output_path / f"{topic_name}_all.csv" if output_path.is_dir() else output_path
            convert_topic_to_csv(topic_dir, topic_name, output_file)
    else:
        if not topic:
            print("❌ Error: Must specify either --topic or --all-topics", file=sys.stderr)
            sys.exit(1)
        
        topic_dir = input_dir / topic
        if not topic_dir.exists():
            print(f"❌ Error: Topic directory not found: {topic_dir}", file=sys.stderr)
            sys.exit(1)
        
        if output_path is None:
            output_path = topic_dir / f"{topic}_all.csv"
        
        convert_topic_to_csv(topic_dir, topic, output_path, specific_files)


def convert_topic_to_csv(
    topic_dir: Path,
    topic_name: str,
    output_path: Path,
    specific_files: List[str] = None
):
    """Convert all JSON files for a specific topic to CSV."""
    # Find all JSON files in the topic directory
    if specific_files:
        json_files = [topic_dir / f for f in specific_files if (topic_dir / f).exists()]
        if not json_files:
            print(f"❌ Error: None of the specified files found in {topic_dir}", file=sys.stderr)
            sys.exit(1)
    else:
        json_files = sorted(topic_dir.glob("*.json"))
        if not json_files:
            print(f"⚠️  Warning: No JSON files found in {topic_dir}", file=sys.stderr)
            return
    
    print(f"\n📁 Processing topic: {topic_name}")
    print(f"   Found {len(json_files)} JSON file(s)")
    
    # Load all examples from all JSON files
    all_examples = []
    total_examples = 0
    
    for json_file in json_files:
        print(f"   Loading: {json_file.name}...", end=" ", flush=True)
        examples = load_json_file(json_file)
        all_examples.extend(examples)
        total_examples += len(examples)
        print(f"✅ {len(examples)} examples")
    
    if not all_examples:
        print(f"⚠️  Warning: No training examples found for topic '{topic_name}'", file=sys.stderr)
        return
    
    # Get all unique field names
    all_fields = sorted(get_all_fields(all_examples))
    
    print(f"\n   Total examples: {total_examples}")
    print(f"   Fields: {', '.join(all_fields)}")
    
    # Write to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"   Writing CSV to: {output_path}...", end=" ", flush=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=all_fields, extrasaction='ignore')
        writer.writeheader()
        
        for example in all_examples:
            # Ensure all fields are present (fill missing with empty string)
            row = {field: example.get(field, '') for field in all_fields}
            writer.writerow(row)
    
    print(f"✅ Done!")
    print(f"   Output: {output_path}")
    print(f"   Rows: {len(all_examples)}")
    print(f"   Columns: {len(all_fields)}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert training data JSON files to CSV format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all JSON files for 'savings' topic
  python convert_training_data_to_csv.py \\
      --input-dir ./training_data \\
      --topic savings \\
      --output ./training_data/savings/savings_all.csv

  # Convert all topics (creates one CSV per topic)
  python convert_training_data_to_csv.py \\
      --input-dir ./training_data \\
      --all-topics \\
      --output ./training_data

  # Convert specific files for a topic
  python convert_training_data_to_csv.py \\
      --input-dir ./training_data \\
      --topic savings \\
      --files savings_training_data_20251021_130708.json \\
      --output ./training_data/savings/savings_specific.csv
        """
    )
    
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Base directory containing topic subdirectories (e.g., ./training_data)"
    )
    
    parser.add_argument(
        "--topic",
        type=str,
        help="Specific topic to convert (e.g., 'savings', 'investments')"
    )
    
    parser.add_argument(
        "--all-topics",
        action="store_true",
        help="Convert all topics found in input-dir"
    )
    
    parser.add_argument(
        "--files",
        nargs="+",
        help="Specific JSON filenames to convert (optional, only used with --topic)"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV file path (or directory if --all-topics). Default: {topic_dir}/{topic}_all.csv"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not args.input_dir.exists():
        print(f"❌ Error: Input directory not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)
    
    if not args.input_dir.is_dir():
        print(f"❌ Error: Input path is not a directory: {args.input_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Convert
    try:
        convert_to_csv(
            input_dir=args.input_dir,
            topic=args.topic,
            all_topics=args.all_topics,
            specific_files=args.files,
            output_path=args.output
        )
        print("\n✅ Conversion complete!")
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

