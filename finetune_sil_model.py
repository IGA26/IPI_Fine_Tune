#!/usr/bin/env python3
"""
finetune_sil_model.py

Fine-tune a transformer model for SIL classification using Vertex AI Custom Training.
Multi-label classification: topic, intent_type, query_type, stage, domain_scope

Usage:
  python finetune_sil_model.py \
      --project playpen-c84caa \
      --location us-central1 \
      --training-data ./training_data \
      --output-dir gs://your-bucket/sil-models \
      --epochs 5 \
      --batch-size 16
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer


# Model configuration
BASE_MODEL = "distilbert-base-uncased"  # Fast and efficient
# Alternative: "ProsusAI/finbert" (financial domain, but larger)
# Alternative: "roberta-base" (better accuracy, slower)

# Label mappings
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


def load_training_data(data_dir: Path) -> List[Dict]:
    """Load all JSON training files from topic/stage directories."""
    examples = []
    
    for topic_dir in data_dir.iterdir():
        if not topic_dir.is_dir():
            continue
        
        for stage_dir in topic_dir.iterdir():
            if not stage_dir.is_dir():
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


def encode_labels(example: Dict, label_mappings: Dict[str, List[str]]) -> Dict:
    """Encode labels as integers for multi-label classification."""
    encoded = {}
    
    # Topic (single label)
    topic_idx = label_mappings["topic"].get(example.get("topic", ""), 0)
    encoded["labels_topic"] = topic_idx
    
    # Intent type (single label)
    intent_idx = label_mappings["intent_type"].get(example.get("intent_type", ""), 0)
    encoded["labels_intent"] = intent_idx
    
    # Query type (single label)
    query_idx = label_mappings["query_type"].get(example.get("query_type", ""), 0)
    encoded["labels_query"] = query_idx
    
    # Stage (single label)
    stage_idx = label_mappings["stage"].get(example.get("stage", ""), 0)
    encoded["labels_stage"] = stage_idx
    
    # Domain scope (single label)
    domain_idx = label_mappings["domain_scope"].get(example.get("domain_scope", ""), 0)
    encoded["labels_domain"] = domain_idx
    
    return encoded


def create_label_mappings() -> Dict[str, Dict[str, int]]:
    """Create label to index mappings."""
    mappings = {}
    
    # Topic mapping
    mappings["topic"] = {label: idx for idx, label in enumerate(TOPIC_LABELS)}
    mappings["topic"]["unknown"] = 0  # Default fallback
    
    # Intent type mapping
    mappings["intent_type"] = {label: idx for idx, label in enumerate(INTENT_TYPE_LABELS)}
    mappings["intent_type"]["unknown"] = 0
    
    # Query type mapping
    mappings["query_type"] = {label: idx for idx, label in enumerate(QUERY_TYPE_LABELS)}
    mappings["query_type"]["unknown"] = 0
    
    # Stage mapping
    mappings["stage"] = {label: idx for idx, label in enumerate(STAGE_LABELS)}
    mappings["stage"]["unknown"] = 0
    
    # Domain scope mapping
    mappings["domain_scope"] = {label: idx for idx, label in enumerate(DOMAIN_SCOPE_LABELS)}
    mappings["domain_scope"]["unknown"] = 0
    
    return mappings


def compute_metrics(eval_pred):
    """Compute metrics for a specific label type."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}


def train_model(
    training_data: List[Dict],
    output_dir: str,
    project: str,
    location: str,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
):
    """Fine-tune the model on SIL classification task."""
    
    print(f"Loaded {len(training_data)} training examples")
    
    # Create label mappings
    label_mappings = create_label_mappings()
    
    # Encode labels
    processed_data = []
    for ex in training_data:
        processed = {
            "text": ex.get("text", ""),
            **encode_labels(ex, label_mappings)
        }
        if processed["text"]:
            processed_data.append(processed)
    
    print(f"Processed {len(processed_data)} valid examples")
    
    # Create dataset
    dataset = Dataset.from_list(processed_data)
    
    # Split train/val (80/20)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]
    
    # Load tokenizer and model
    print(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    # Create separate models for each label type (multi-head approach)
    # Or use a single model with multiple outputs
    num_labels = {
        "topic": len(TOPIC_LABELS),
        "intent": len(INTENT_TYPE_LABELS),
        "query": len(QUERY_TYPE_LABELS),
        "stage": len(STAGE_LABELS),
        "domain": len(DOMAIN_SCOPE_LABELS),
    }
    
    models = {}
    trainers = {}
    
    # Train separate models for each label type
    for label_type, num_label in num_labels.items():
        print(f"\n{'='*60}")
        print(f"Training model for: {label_type}")
        print(f"{'='*60}")
        
        model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL,
            num_labels=num_label
        )
        
        # Tokenize datasets
        def tokenize(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=128,
            )
        
        # Select the appropriate label column
        label_col = f"labels_{label_type}"
        
        train_tokenized = train_dataset.map(
            tokenize,
            batched=True,
            remove_columns=[col for col in train_dataset.column_names if col != label_col]
        )
        train_tokenized = train_tokenized.rename_column(label_col, "labels")
        
        val_tokenized = val_dataset.map(
            tokenize,
            batched=True,
            remove_columns=[col for col in val_dataset.column_names if col != label_col]
        )
        val_tokenized = val_tokenized.rename_column(label_col, "labels")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"{output_dir}/{label_type}",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/{label_type}/logs",
            logging_steps=50,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            push_to_hub=False,
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=val_tokenized,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        # Train
        trainer.train()
        
        # Evaluate
        results = trainer.evaluate()
        print(f"\n{label_type} Results:")
        print(json.dumps(results, indent=2))
        
        # Save model
        trainer.save_model()
        tokenizer.save_pretrained(f"{output_dir}/{label_type}")
        
        models[label_type] = model
        trainers[label_type] = trainer
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Models saved to: {output_dir}")
    print(f"{'='*60}")
    
    return models, trainers


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SIL classification model")
    parser.add_argument("--project", default="playpen-c84caa", help="GCP project ID")
    parser.add_argument("--location", default="us-central1", help="Vertex AI location")
    parser.add_argument("--training-data", required=True, help="Directory containing training JSON files")
    parser.add_argument("--output-dir", required=True, help="Output directory (local or gs://)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--model", default=BASE_MODEL, help="Base model to fine-tune")
    
    args = parser.parse_args()
    
    # Load training data
    data_dir = Path(args.training_data)
    if not data_dir.exists():
        print(f"Error: Training data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)
    
    training_data = load_training_data(data_dir)
    
    if not training_data:
        print(f"Error: No training data found in {data_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Train models
    train_model(
        training_data=training_data,
        output_dir=args.output_dir,
        project=args.project,
        location=args.location,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()

