#!/usr/bin/env python3
"""
finetune_sil_model.py

Fine-tune a transformer model for SIL classification using Vertex AI Custom Training.
Multi-label classification: topic, intent_type, query_type, stage, domain_scope, advice_risk

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
import logging
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
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('finetune.log')
    ]
)
logger = logging.getLogger(__name__)


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

ADVICE_RISK_LABELS = ["low", "medium", "high"]  # 0.0-0.3, 0.4-0.6, 0.7-1.0


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
                # Skip manifest files
                if "manifest" in json_file.name.lower():
                    continue
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
    
    # Advice risk (bucketize continuous score into low/medium/high)
    advice_risk_score = example.get("advice_risk_score", 0.0)
    if isinstance(advice_risk_score, (int, float)):
        if advice_risk_score <= 0.3:
            advice_risk_bucket = "low"
        elif advice_risk_score <= 0.6:
            advice_risk_bucket = "medium"
        else:
            advice_risk_bucket = "high"
    else:
        advice_risk_bucket = "low"  # Default fallback
    
    advice_risk_idx = label_mappings["advice_risk"].get(advice_risk_bucket, 0)
    encoded["labels_advice_risk"] = advice_risk_idx
    
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
    
    # Advice risk mapping
    mappings["advice_risk"] = {label: idx for idx, label in enumerate(ADVICE_RISK_LABELS)}
    mappings["advice_risk"]["unknown"] = 0
    
    return mappings


def compute_metrics(eval_pred):
    """Compute metrics for a specific label type (accuracy, F1 macro, F1 micro)."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    f1_micro = f1_score(labels, predictions, average='micro', zero_division=0)
    
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
    }


def train_model(
    training_data: List[Dict],
    output_dir: str,
    project: str,
    location: str,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    seed: int = 42,
):
    """Fine-tune the model on SIL classification task."""
    
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Set random seed to {seed} for reproducibility")
    
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        logger.info(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        logger.info(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("⚠️  No GPU detected. Training will use CPU (will be slower).")
    logger.info(f"Using device: {device}\n")
    
    logger.info(f"Loaded {len(training_data)} training examples")
    
    # Create label mappings
    label_mappings = create_label_mappings()
    
    # Save label mappings to JSON for deployment
    label_map_file = Path(output_dir) / "label_mappings.json"
    label_map_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to serializable format
    label_map_serializable = {}
    for label_type, mapping in label_mappings.items():
        label_map_serializable[label_type] = {
            "labels": list(mapping.keys()),
            "label_to_idx": mapping
        }
    
    with open(label_map_file, 'w') as f:
        json.dump(label_map_serializable, f, indent=2)
    logger.info(f"Saved label mappings to {label_map_file}")
    
    # Encode labels and validate
    processed_data = []
    invalid_count = 0
    for ex in training_data:
        text = ex.get("text", "").strip()
        if not text or len(text) < 5:
            invalid_count += 1
            continue
        
        try:
            encoded = encode_labels(ex, label_mappings)
            processed = {
                "text": text,
                **encoded
            }
            processed_data.append(processed)
        except Exception as e:
            invalid_count += 1
            if invalid_count <= 10:  # Only show first 10 errors
                print(f"Warning: Skipping invalid example: {e}", file=sys.stderr)
    
    if invalid_count > 0:
        logger.warning(f"⚠️  Skipped {invalid_count} invalid examples")
    logger.info(f"Processed {len(processed_data)} valid examples")
    
    if not processed_data:
        logger.error("❌ Error: No valid training examples after processing")
        sys.exit(1)
    
    # Deduplicate based on text (case-insensitive, stripped)
    seen_texts = set()
    deduplicated_data = []
    duplicate_count = 0
    
    for ex in processed_data:
        text_key = ex["text"].strip().lower()
        if text_key not in seen_texts:
            seen_texts.add(text_key)
            deduplicated_data.append(ex)
        else:
            duplicate_count += 1
    
    if duplicate_count > 0:
        logger.warning(f"⚠️  Removed {duplicate_count} duplicate examples (based on text content)")
        logger.info(f"After deduplication: {len(deduplicated_data)} unique examples")
    
    if not deduplicated_data:
        logger.error("❌ Error: No unique examples after deduplication")
        sys.exit(1)
    
    # Create dataset
    dataset = Dataset.from_list(deduplicated_data)
    
    # Split train/val (80/20)
    # Note: For stratified split by topic, we'd need to use sklearn's train_test_split
    # but HuggingFace datasets doesn't support stratification directly
    if len(dataset) < 10:
        logger.warning("⚠️  Very small dataset. Using all data for training (no validation split).")
        train_dataset = dataset
        val_dataset = dataset  # Use same for validation (will be ignored)
    else:
        dataset = dataset.train_test_split(test_size=0.2, seed=seed)
        train_dataset = dataset["train"]
        val_dataset = dataset["test"]
        logger.info(f"Split dataset: {len(train_dataset)} train, {len(val_dataset)} validation")
    
    # Load tokenizer and model
    logger.info(f"Loading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    # Create separate models for each label type (multi-head approach)
    # Or use a single model with multiple outputs
    num_labels = {
        "topic": len(TOPIC_LABELS),
        "intent": len(INTENT_TYPE_LABELS),
        "query": len(QUERY_TYPE_LABELS),
        "stage": len(STAGE_LABELS),
        "domain": len(DOMAIN_SCOPE_LABELS),
        "advice_risk": len(ADVICE_RISK_LABELS),
    }
    
    models = {}
    trainers = {}
    
    # Train separate models for each label type (multi-head approach)
    # This is correct: each label type (topic, intent, etc.) is independent
    # and gets its own classification head, avoiding interference
    for label_type, num_label in num_labels.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Training model for: {label_type}")
        logger.info(f"{'='*60}")
        
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
        
        # Verify label column exists
        if label_col not in train_dataset.column_names:
            logger.warning(f"⚠️  Label column '{label_col}' not found in dataset. Skipping {label_type}.")
            continue
        
        train_tokenized = train_dataset.map(
            tokenize,
            batched=True,
            remove_columns=[col for col in train_dataset.column_names if col not in ["text", label_col]]
        )
        train_tokenized = train_tokenized.rename_column(label_col, "labels")
        
        val_tokenized = val_dataset.map(
            tokenize,
            batched=True,
            remove_columns=[col for col in val_dataset.column_names if col not in ["text", label_col]]
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
            eval_strategy="epoch",
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
            eval_dataset=val_tokenized if len(val_tokenized) > 0 else None,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        # Train
        logger.info(f"Training {label_type} model...")
        trainer.train()
        
        # Evaluate
        if len(val_tokenized) > 0:
            results = trainer.evaluate()
            logger.info(f"\n{label_type} Results:")
            logger.info(json.dumps(results, indent=2))
            
            # Generate confusion matrix for detailed analysis
            predictions = trainer.predict(val_tokenized)
            pred_labels = np.argmax(predictions.predictions, axis=1)
            true_labels = predictions.label_ids
            
            cm = confusion_matrix(true_labels, pred_labels)
            logger.info(f"\n{label_type} Confusion Matrix (first 10x10):")
            logger.info(f"\n{cm[:10, :10]}")
        else:
            logger.info(f"\n{label_type}: Training complete (no validation set)")
        
        # Save model, tokenizer, and config
        trainer.save_model()
        tokenizer.save_pretrained(f"{output_dir}/{label_type}")
        logger.info(f"Saved {label_type} model to {output_dir}/{label_type}")
        
        models[label_type] = model
        trainers[label_type] = trainer
    
    logger.info(f"\n{'='*60}")
    logger.info("Training complete!")
    logger.info(f"Models saved to: {output_dir}")
    logger.info(f"Label mappings saved to: {output_dir}/label_mappings.json")
    logger.info(f"{'='*60}")
    
    return models, trainers


def main():
    import shutil
    
    parser = argparse.ArgumentParser(description="Fine-tune SIL classification model")
    parser.add_argument("--project", default="playpen-c84caa", help="GCP project ID")
    parser.add_argument("--location", default="us-central1", help="Vertex AI location")
    parser.add_argument("--training-data", required=True, help="Directory containing training JSON files")
    parser.add_argument("--output-dir", required=True, help="Output directory (local or gs://)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--model", default=BASE_MODEL, help="Base model to fine-tune")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--clear-output", action="store_true", help="Clear output directory if it exists (prevents shape mismatch errors)")
    
    args = parser.parse_args()
    
    # Handle output directory clearing (only for local paths, not GCS)
    output_path = Path(args.output_dir)
    if not str(args.output_dir).startswith("gs://"):
        if output_path.exists() and args.clear_output:
            # Only clear SIL-specific subdirectories, not emotion or other directories
            sil_subdirs = ["topic", "intent", "query", "stage", "domain", "advice_risk"]
            cleared_dirs = []
            for subdir in sil_subdirs:
                subdir_path = output_path / subdir
                if subdir_path.exists():
                    logger.warning(f"Clearing SIL model directory: {subdir_path}")
                    shutil.rmtree(subdir_path)
                    cleared_dirs.append(subdir)
            
            # Also clear label_mappings.json if it exists
            label_map_file = output_path / "label_mappings.json"
            if label_map_file.exists():
                logger.warning(f"Clearing label mappings: {label_map_file}")
                label_map_file.unlink()
            
            if cleared_dirs:
                logger.info(f"Cleared SIL model directories: {', '.join(cleared_dirs)}")
                logger.info(f"NOTE: Emotion models in '{output_path / 'emotion'}' were NOT touched.")
            else:
                logger.info(f"No existing SIL model directories found to clear.")
            
            # Ensure output directory exists
            output_path.mkdir(parents=True, exist_ok=True)
        elif output_path.exists():
            # Check if it contains SIL model files that might cause conflicts
            sil_subdirs = ["topic", "intent", "query", "stage", "domain", "advice_risk"]
            existing_sil_dirs = [d for d in sil_subdirs if (output_path / d).exists()]
            if existing_sil_dirs:
                logger.warning(f"Output directory exists and contains SIL model directories: {', '.join(existing_sil_dirs)}")
                logger.warning(f"This may cause shape mismatch errors.")
                logger.warning(f"Use --clear-output to remove existing SIL models and start fresh.")
                logger.warning(f"NOTE: This will only clear SIL model directories - emotion models are safe.")
    
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
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

