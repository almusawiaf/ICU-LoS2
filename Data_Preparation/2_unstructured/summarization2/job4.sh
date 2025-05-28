#!/bin/bash

#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=10-00:00:00

mkdir -p log

# Define files and models
# files=("NursingOther" "Nursing" "Radiology")
files=("ALL_first_last")

# models=("1_t5_small2.py" "2_longchat.py")
models=("1_t5_small2.py") # "3_bart_large_cnn.py" "4_medical_summarization.py")

# Loop over each file and model, submitting jobs in parallel
for theFile in "${files[@]}"; do
    for model in "${models[@]}"; do
        export  input_file="../../../Data/unstructured/text/${theFile}.csv"
        export output_file="../../../Data/unstructured/summarized/${theFile}_${model%.py}.csv"  # Ensure unique filenames

        sbatch --job-name="${theFile}_${model%.py}" --output="log/${theFile}_${model%.py}_%j.txt" --error="log/${theFile}_${model%.py}_%j.err" <<EOF
#!/bin/bash
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=10-00:00:00

python ${model}
EOF
    done
done
