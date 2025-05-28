#!/bin/bash
#SBATCH --partition=gpu                  # Use GPU partition
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --cpus-per-task=2                # Number of CPU cores per task
#SBATCH --mem=32G                        # Memory per node
#SBATCH --time=48:00:00                  # Time limit
#SBATCH --output=log/output_%j.log           # Output log file
#SBATCH --error=log/error_%j.log             # Error log file

# SBATCH --job-name=text_to_embedding_t5_small2
# SBATCH --job-name=text_to_embedding_bart_large_cnn
# SBATCH --job-name=text_to_embedding_medical_summarization

#SBATCH --job-name=text_to_embedding

mkdir -p log

# export thePath="../../../Data/unstructured"

# # export smrz_model="1_t5_small2"
# # export smrz_model="3_bart_large_cnn"
# export smrz_model="4_medical_summarization"

# python BioClinicalBert.py
# python clinicalBert.py
python GatorTron.py
