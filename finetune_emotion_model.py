#!/usr/bin/env python3
"""
finetune_emotion_model.py

Fine-tune ProsusAI/finbert for emotion, distress, and vulnerability detection.
Multi-task learning: emotion classification + distress flag + vulnerability flag.

Usage:
  python finetune_emotion_model.py \
      --training-data ./training_data/emotion/emotion_training_data_*.json \
      --output-dir ./models/emotion \
      --epochs 3 \
      --batch-size 16
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
import random

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('finetune_emotion.log')
    ]
)
logger = logging.getLogger(__name__)

# Base model
BASE_MODEL = "ProsusAI/finbert"

# Emotion labels (matches generate_emotion_training_data.py)
EMOTION_LABELS = ["positive", "neutral", "negative"]


def load_training_data(data_path: str) -> List[Dict]:
    """Load training data from JSON file(s)."""
    data_path = Path(data_path)
    all_data = []
    
    if data_path.is_file():
        files = [data_path]
    elif data_path.is_dir():
        # Load all JSON files in directory
        files = list(data_path.glob("*.json"))
    else:
        # Try glob pattern
        import glob
        files = [Path(f) for f in glob.glob(str(data_path))]
    
    if not files:
        raise ValueError(f"No training data files found: {data_path}")
    
    for file_path in files:
        if "manifest" in file_path.name:
            continue
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Handle different formats
            if isinstance(data, list):
                all_data.extend(data)
            elif isinstance(data, dict) and "training_data" in data:
                all_data.extend(data["training_data"])
            elif isinstance(data, dict) and "data" in data:
                all_data.extend(data["data"])
            else:
                logger.warning(f"Unknown format in {file_path}, skipping")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            continue
    
    logger.info(f"Loaded {len(all_data)} examples from {len(files)} file(s)")
    return all_data


def encode_labels(example: Dict, emotion_mapping: Dict[str, int]) -> Dict:
    """Encode labels for multi-task learning."""
    # Emotion classification
    emotion = example.get("emotion", "neutral")
    emotion_idx = emotion_mapping.get(emotion, 0)
    
    # Distress flag (binary: 0 or 1)
    distress_flag = 1 if example.get("distress_flag", False) else 0
    
    # Vulnerability flag (binary: 0 or 1)
    vulnerability_flag = 1 if example.get("vulnerability_flag", False) else 0
    
    # Handover required (binary: 0 or 1)
    handover_required = 1 if example.get("handover_required", False) else 0
    
    # Distress severity (regression: float 0.0-1.0)
    distress_severity = float(example.get("distress_severity", 0.0))
    # Clamp to valid range
    distress_severity = max(0.0, min(1.0, distress_severity))
    
    return {
        "labels_emotion": emotion_idx,
        "labels_distress": distress_flag,
        "labels_vulnerability": vulnerability_flag,
        "labels_handover": handover_required,
        "labels_severity": distress_severity,
    }


def create_label_mappings() -> Dict[str, Dict[str, int]]:
    """Create label mappings."""
    return {
        "emotion": {label: idx for idx, label in enumerate(EMOTION_LABELS)}
    }


def tokenize(examples: Dict, tokenizer) -> Dict:
    """Tokenize text examples."""
    # Don't use return_tensors="pt" here - let DataCollator handle padding
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,  # DataCollator will handle padding
        max_length=128,
    )


def compute_metrics(eval_pred, task: str = "emotion"):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    
    if task == "emotion":
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='macro', zero_division=0)
        return {
            "accuracy": accuracy,
            "f1_macro": f1
        }
    elif task == "regression":
        # Regression task for distress_severity
        # predictions is shape (batch_size, 1) for regression
        if predictions.ndim > 1:
            predictions = predictions.squeeze()
        if labels.ndim > 1:
            labels = labels.squeeze()
        
        # Mean Absolute Error (MAE)
        mae = np.mean(np.abs(predictions - labels))
        # Mean Squared Error (MSE)
        mse = np.mean((predictions - labels) ** 2)
        # Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mse)
        # R-squared
        ss_res = np.sum((labels - predictions) ** 2)
        ss_tot = np.sum((labels - np.mean(labels)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "r2": float(r2)
        }
    else:
        # Binary classification for distress/vulnerability/handover
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='binary', zero_division=0)
        return {
            "accuracy": accuracy,
            "f1": f1
        }


def train_model(
    training_data: List[Dict],
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    seed: int = 42
):
    """Train emotion detection model with multi-task learning."""
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create label mappings
    label_mappings = create_label_mappings()
    
    # Process and encode data
    processed_data = []
    for ex in training_data:
        if not ex.get("text") or len(ex.get("text", "")) < 8:
            continue
        
        encoded = encode_labels(ex, label_mappings["emotion"])
        processed_data.append({
            "text": ex["text"],
            **encoded
        })
    
    if not processed_data:
        raise ValueError("No valid training data after processing")
    
    logger.info(f"Processed {len(processed_data)} examples")
    
    # Create datasets
    dataset = Dataset.from_list(processed_data)
    
    # Split train/val (80/20)
    dataset = dataset.train_test_split(test_size=0.2, seed=seed)
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Initialize tokenizer and model
    # Always load from BASE_MODEL, not from output_dir to avoid shape mismatches
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(EMOTION_LABELS),
        ignore_mismatched_sizes=True  # Safety: ignore if there's a mismatch
    )
    
    # Tokenize (preserve label columns)
    train_tokenized = train_dataset.map(
        lambda x: tokenize(x, tokenizer),
        batched=True,
        remove_columns=["text"]  # Keep labels_emotion, labels_distress, labels_vulnerability
    )
    val_tokenized = val_dataset.map(
        lambda x: tokenize(x, tokenizer),
        batched=True,
        remove_columns=["text"]  # Keep labels_emotion, labels_distress, labels_vulnerability
    )
    
    # Rename label column to "labels" (expected by Trainer)
    train_tokenized = train_tokenized.rename_column("labels_emotion", "labels")
    val_tokenized = val_tokenized.rename_column("labels_emotion", "labels")
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        seed=seed,
    )
    
    # Trainer for emotion classification
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, "emotion"),
    )
    
    # Train
    logger.info("Training emotion classification model...")
    trainer.train()
    
    # Evaluate
    results = trainer.evaluate()
    logger.info(f"\nEmotion Classification Results:")
    logger.info(json.dumps(results, indent=2))
    
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Now train separate binary classifiers for distress, vulnerability, and handover
    # We'll use the same base model but with 2 labels (binary classification)
    
    for task_name, task_label in [("distress", "labels_distress"), ("vulnerability", "labels_vulnerability"), ("handover", "labels_handover")]:
        logger.info(f"\nTraining {task_name} flag classifier...")
        
        # Create binary classification dataset
        task_dataset = Dataset.from_list([
            {
                "text": ex["text"],
                "labels": ex[task_label]
            }
            for ex in processed_data
        ])
        
        task_dataset = task_dataset.train_test_split(test_size=0.2, seed=seed)
        task_train = task_dataset["train"]
        task_val = task_dataset["test"]
        
        # Load fresh model for binary classification
        # Always load from BASE_MODEL to avoid shape mismatches
        task_model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL, 
            num_labels=2,
            ignore_mismatched_sizes=True  # Safety: ignore if there's a mismatch
        )
        
        # Tokenize (preserve label column)
        task_train_tokenized = task_train.map(
            lambda x: tokenize(x, tokenizer),
            batched=True,
            remove_columns=["text"]  # Keep "labels" column
        )
        task_val_tokenized = task_val.map(
            lambda x: tokenize(x, tokenizer),
            batched=True,
            remove_columns=["text"]  # Keep "labels" column
        )
        
        # Training arguments for binary task
        task_training_args = TrainingArguments(
            output_dir=f"{output_dir}/{task_name}",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/{task_name}/logs",
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            seed=seed,
        )
        
        # Trainer for binary task
        task_trainer = Trainer(
            model=task_model,
            args=task_training_args,
            train_dataset=task_train_tokenized,
            eval_dataset=task_val_tokenized,
            data_collator=data_collator,
            compute_metrics=lambda eval_pred: compute_metrics(eval_pred, "binary"),
        )
        
        # Train
        task_trainer.train()
        
        # Evaluate
        task_results = task_trainer.evaluate()
        logger.info(f"\n{task_name.capitalize()} Flag Results:")
        logger.info(json.dumps(task_results, indent=2))
        
        # Save
        task_trainer.save_model()
        tokenizer.save_pretrained(f"{output_dir}/{task_name}")
    
    # Train distress_severity regression model
    logger.info(f"\nTraining distress_severity regression model...")
    
    # Create regression dataset
    severity_dataset = Dataset.from_list([
        {
            "text": ex["text"],
            "labels": ex["labels_severity"]
        }
        for ex in processed_data
    ])
    
    severity_dataset = severity_dataset.train_test_split(test_size=0.2, seed=seed)
    severity_train = severity_dataset["train"]
    severity_val = severity_dataset["test"]
    
    # Load fresh model for regression (num_labels=1 for regression)
    severity_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=1,  # Regression: single output
        problem_type="regression",  # Specify regression problem type
        ignore_mismatched_sizes=True
    )
    
    # Tokenize (preserve label column)
    severity_train_tokenized = severity_train.map(
        lambda x: tokenize(x, tokenizer),
        batched=True,
        remove_columns=["text"]  # Keep "labels" column
    )
    severity_val_tokenized = severity_val.map(
        lambda x: tokenize(x, tokenizer),
        batched=True,
        remove_columns=["text"]  # Keep "labels" column
    )
    
    # Training arguments for regression
    severity_training_args = TrainingArguments(
        output_dir=f"{output_dir}/severity",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/severity/logs",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rmse",  # Lower is better for RMSE
        greater_is_better=False,
        seed=seed,
    )
    
    # Trainer for regression
    severity_trainer = Trainer(
        model=severity_model,
        args=severity_training_args,
        train_dataset=severity_train_tokenized,
        eval_dataset=severity_val_tokenized,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, "regression"),
    )
    
    # Train
    severity_trainer.train()
    
    # Evaluate
    severity_results = severity_trainer.evaluate()
    logger.info(f"\nDistress Severity Regression Results:")
    logger.info(json.dumps(severity_results, indent=2))
    
    # Save
    severity_trainer.save_model()
    tokenizer.save_pretrained(f"{output_dir}/severity")
    
    logger.info(f"\n{'='*60}")
    logger.info("Training complete!")
    logger.info(f"Models saved to: {output_dir}")
    logger.info(f"  - emotion/ (emotion classification: positive/neutral/negative)")
    logger.info(f"  - distress/ (distress flag binary classifier)")
    logger.info(f"  - vulnerability/ (vulnerability flag binary classifier)")
    logger.info(f"  - handover/ (handover_required binary classifier)")
    logger.info(f"  - severity/ (distress_severity regression: 0.0-1.0)")
    logger.info(f"{'='*60}")


def main():
    import shutil
    
    parser = argparse.ArgumentParser(description="Fine-tune ProsusAI/finbert for emotion detection")
    parser.add_argument("--training-data", required=True, help="Path to training JSON file(s)")
    parser.add_argument("--output-dir", required=True, help="Output directory for models")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--clear-output", action="store_true", help="Clear output directory if it exists (prevents shape mismatch errors)")
    
    args = parser.parse_args()
    
    # Clear output directory if requested or if it exists and contains conflicting models
    output_path = Path(args.output_dir)
    if output_path.exists() and args.clear_output:
        logger.warning(f"Clearing output directory: {output_path}")
        shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
    elif output_path.exists():
        # Check if it contains model files that might cause conflicts
        model_files = list(output_path.glob("**/pytorch_model.bin")) + list(output_path.glob("**/model.safetensors"))
        if model_files:
            logger.warning(f"Output directory exists and contains model files. This may cause shape mismatch errors.")
            logger.warning(f"Use --clear-output to remove existing models and start fresh.")
            logger.warning(f"Or delete the directory manually: {output_path}")
    
    # Load training data
    training_data = load_training_data(args.training_data)
    
    if not training_data:
        print(f"Error: No training data found", file=sys.stderr)
        sys.exit(1)
    
    # Train models
    train_model(
        training_data=training_data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

