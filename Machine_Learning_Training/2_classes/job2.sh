#!/bin/bash

# models=("LSTM.py" "ANN.py") 
models=("ml_LR.py" "ml_RF.py" "ml_XGBoost.py") 

data_types=("Z" "F" "E" "ZS" "ZSF" "ZSE" "ZSEF")

# Loop through models and data types
for model in "${models[@]}"; do
    model_name=$(basename "$model" .py)  # Extract model name without .py extension

    for data_type in "${data_types[@]}"; do
        sbatch <<EOF
#!/bin/bash
#SBATCH --partition=gpu                  # Use GPU partition
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --cpus-per-task=1                # Number of CPU cores per task
#SBATCH --mem=64G                        # Memory per node
#SBATCH --time=01:00:00                  # Time limit

#SBATCH --job-name=${model_name}_${data_type}_SM  # Job name
#SBATCH --output=logs/normalized/log_sci_sm_2000_F_7_days/Lung_Cancer/200_features_KBoW_200_SMOTE/output_${model_name}_${data_type}_%j.log  # Output log file
#SBATCH --error=logs/normalized/log_sci_sm_2000_F_7_days/Lung_Cancer/200_features_KBoW_200_SMOTE/error_${model_name}_${data_type}_%j.log    # Error log file

mkdir -p logs/normalized/log_sci_sm_2000_F_7_days/Lung_Cancer/200_features_KBoW_200_SMOTE

# Set environment variables
export base_path="../../../Data/XY_BoW/2_classes"
export training_model="BoW_sci_sm_2000_F_7_days_Lung_Cancer"
export data_type=${data_type}
export exp="results_BoW"

export num_features="200"

export smote_ing="yes"
export pca_ing="no"

export epochs="15"
learning_rate="0.001"

python ${model}
EOF
    done
done
