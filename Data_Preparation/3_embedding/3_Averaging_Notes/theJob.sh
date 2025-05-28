#!/bin/bash
#SBATCH --partition=gpu                  # Use GPU partition
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --cpus-per-task=2                # Number of CPU cores per task
#SBATCH --mem=1000G                        # Memory per node
#SBATCH --time=48:00:00                  # Time limit
#SBATCH --output=log/output_%j.log           # Output log file
#SBATCH --error=log/error_%j.log             # Error log file

#SBATCH --job-name=text_to_embedding

mkdir -p log

# export emb_model="Bio_ClinicalBERT"
# export emb_model="ClinicalBERT"
export emb_model="gatortron-base"

export thePath="../../../Data/unstructured"
export batch_size=32

python emb_extraction.py
