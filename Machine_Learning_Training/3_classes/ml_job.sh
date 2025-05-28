#!/bin/bash

# models=("ml_LR.py" "ml_RF.py" "ml_XGBoost.py" "ANN.py") 
models=("ANN.py") 

data_types=("Z" "F" "E" "ZS" "ZSF" "ZSE" "ZSEF")

epochs="15"
learning_rate="0.001"

# normalization="not_normalized"
normalization="normalized"

# num_days="_3_7_days"
num_days=""

num_classes="3_classes"

feature_selection="yes"
num_features="200"
k_bow="200"

file_name="BoW_sci_sm_2000_F_Lung_Cancer"

saving_path="${normalization}/${num_classes}/${file_name}"

smote_ing="yes"
pca_ing="no"

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
#SBATCH --time=00:15:00                  # Time limit


#SBATCH --job-name=${model_name}_${data_type}_SM  # Job name
#SBATCH --output=logs/${saving_path}/output_${model_name}_${data_type}_%j.log  # Output log file
#SBATCH --error=logs/${saving_path}/error_${model_name}_${data_type}_%j.log    # Error log file

mkdir -p logs/${saving_path}


export base_path="../../../Data/XY_BoW/${normalization}/3_classes"
export training_model=${file_name}
export data_type=${data_type}
export exp="results_BoW"

export feature_selection=${feature_selection}
export num_features=${num_features}
export k_bow=${k_bow}

export smote_ing=${smote_ing}
export pca_ing=${pca_ing}

export epochs=${epochs}
export learning_rate=${learning_rate}

python ${model}
EOF
    done
done
