#!/bin/bash
#SBATCH --partition=cpu                  # Use CPU partition
#SBATCH --cpus-per-task=32                # Increase CPU cores for better performance
#SBATCH --mem=260G                        # Adjust memory as needed
#SBATCH --time=3-00:00:00                 # Time limit (3 days, adjust if needed)
#SBATCH --output=log/output_parallel_%j.log        # Output log file
#SBATCH --error=log/error_parallel_%j.log          # Error log file

#SBATCH --job-name=SM

mkdir -p log

# python 2_2_generating_BoW_cpu.py
# python BoW_parallel.py
# python BoW_parallel_medspacy.py

python BoW_parallel_sci_sm.py

# jupyter nbconvert --to notebook --execute sci_sm.ipynb --output sci_sm2.ipynb

