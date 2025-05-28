#!/bin/bash

# models=("ml_LR.py" "ml_RF.py" "ml_XGBoost.py" "ANN.py" "ml_SVM.py") 
models=("ml_SVM.py") 

data_types=("Z" "F" "E" "ZS" "ZSF" "ZSE" "ZSEF")

epochs="15"
learning_rate="0.001"

# normalization="not_normalized"
normalization="normalized"

# num_classes="3_classes"
# num_days="_3_5_days"

num_classes="4_classes"
num_days="_3_7_30_days"

feature_selection="no"
num_features="5000"
k_bow="5000"

# saving_path="${normalization}/log_sci_sm_2000_F/${k_bow}_features_KBoW_${num_features}${num_days}"
saving_path="${normalization}/log_sci_sm_2000_F"

smote_ing="no"
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
#SBATCH --time=00:35:00                  # Time limit


#SBATCH --job-name=${model_name}_${data_type}_SM  # Job name
#SBATCH --output=logs/${saving_path}/${num_days}/output_${model_name}_${data_type}_%j.log  # Output log file
#SBATCH --error=logs/${saving_path}/${num_days}/error_${model_name}_${data_type}_%j.log    # Error log file

mkdir -p logs/${saving_path}


export base_path="../../../Data/XY_BoW/${normalization}/${num_classes}"
export training_model="BoW_sci_sm_2000_F${num_days}"
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
