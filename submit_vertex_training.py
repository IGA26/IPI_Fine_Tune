#!/usr/bin/env python3
"""
submit_vertex_training.py

Submit a Vertex AI Custom Training job for SIL model fine-tuning.

Usage:
  python submit_vertex_training.py \
      --project playpen-c84caa \
      --location us-central1 \
      --training-data gs://your-bucket/training_data \
      --output-dir gs://your-bucket/sil-models \
      --epochs 5
"""

import argparse
from datetime import datetime
from google.cloud import aiplatform


def submit_training_job(
    project: str,
    location: str,
    training_data_gcs: str,
    output_dir_gcs: str,
    epochs: int = 5,
    batch_size: int = 16,
    machine_type: str = "n1-standard-4",
    accelerator_type: str = "NVIDIA_TESLA_T4",
    accelerator_count: int = 1,
):
    """Submit a Vertex AI Custom Training job."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = f"sil-finetune-{timestamp}"
    
    # Initialize Vertex AI
    aiplatform.init(project=project, location=location)
    
    # Define the training script
    script_path = "finetune_sil_model.py"
    
    # Create a CustomJob
    job = aiplatform.CustomTrainingJob(
        display_name=job_name,
        script_path=script_path,
        container_uri="gcr.io/cloud-aiplatform/training/pytorch-gpu.2-0:latest",
        requirements=["finetuning_requirements.txt"],
        model_serving_container_image_uri=None,  # We'll save models to GCS
    )
    
    # Submit the job
    print(f"Submitting training job: {job_name}")
    print(f"Training data: {training_data_gcs}")
    print(f"Output directory: {output_dir_gcs}")
    
    model = job.run(
        args=[
            "--project", project,
            "--location", location,
            "--training-data", training_data_gcs,
            "--output-dir", output_dir_gcs,
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
        ],
        replica_count=1,
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        sync=False,  # Run asynchronously
    )
    
    print(f"\nJob submitted successfully!")
    print(f"Job name: {job_name}")
    print(f"Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={project}")
    
    return job, model


def main():
    parser = argparse.ArgumentParser(description="Submit Vertex AI training job")
    parser.add_argument("--project", default="playpen-c84caa", help="GCP project ID")
    parser.add_argument("--location", default="us-central1", help="Vertex AI location")
    parser.add_argument("--training-data", required=True, help="GCS path to training data")
    parser.add_argument("--output-dir", required=True, help="GCS output directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--machine-type", default="n1-standard-4", help="Machine type")
    parser.add_argument("--accelerator-type", default="NVIDIA_TESLA_T4", help="GPU type")
    parser.add_argument("--accelerator-count", type=int, default=1, help="Number of GPUs")
    
    args = parser.parse_args()
    
    submit_training_job(
        project=args.project,
        location=args.location,
        training_data_gcs=args.training_data,
        output_dir_gcs=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        machine_type=args.machine_type,
        accelerator_type=args.accelerator_type,
        accelerator_count=args.accelerator_count,
    )


if __name__ == "__main__":
    main()

