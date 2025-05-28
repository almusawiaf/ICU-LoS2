#!/bin/bash
#SBATCH --job-name=unifying_models
#SBATCH --output=logs/unifying_%A_%a.out
#SBATCH --error=logs/unifying_%A_%a.err
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00

# Set paths
STRUCTURED_PATH="../../../Data/structured"
UNSTRUCTURED_PATH="../../../Data/unstructured/emb"
SAVING_PATH="../../../Data/XY3"

# List of models
MODELS=("bioclinicalbert" "clinicalbert" "gatortron")

# Select model based on SLURM array task ID
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

# Run the Python script for the selected model
python unifying.py --model $MODEL --structured_path $STRUCTURED_PATH --unstructured_path $UNSTRUCTURED_PATH --saving_path $SAVING_PATH

# Ensure the logs directory exists
mkdir -p logs