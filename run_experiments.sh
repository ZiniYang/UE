#!/bin/bash

# run_comprehensive.sh - Complete script for testing all combinations
# - Tests classwise and samplewise poisoning
# - Tests min-min and min-max attacks
# - Tests different poison percentages
# - Tests different detection methods
# - Generates comparison graphs

# Setup experiment parameters
EXP_NAME="comprehensive_experiments"
CLASS_IDX=3  # Cat class in CIFAR-10
POISON_TYPES=("classwise" "samplewise")
ATTACK_TYPES=("min-min" "min-max")
POISON_PERCENTAGES=(0 0.2 0.4 0.6 0.8 1.0)
DETECTION_METHODS=("simple-2NN" "simple-linear" "bias-0.5" "bias+0.5")
NUM_EPOCHS=50

# Create experiment directory
mkdir -p "$EXP_NAME"

# Function to generate perturbation
generate_perturbation() {
    local attack_type=$1
    local perturb_type=$2
    local perturb_dir="${EXP_NAME}/${perturb_type}_${attack_type}"
    local perturb_path="${perturb_dir}/perturbation.pt"
    
    mkdir -p "$perturb_dir"
    
    if [ ! -f "$perturb_path" ]; then
        echo "Generating ${attack_type} perturbation for ${perturb_type}..."
        python3 perturbation.py --exp_name ${perturb_dir} --attack_type ${attack_type} --perturb_type ${perturb_type}
        if [ ! -f "$perturb_path" ]; then
            echo "Error: Perturbation generation failed for ${attack_type} ${perturb_type}"
            return 1
        fi
        echo "Perturbation saved to $perturb_path"
    else
        echo "Using existing perturbation at $perturb_path"
    fi
    
    return 0
}

# Function to train model
train_model() {
    local attack_type=$1
    local perturb_type=$2
    local percentage=$3
    local perturb_dir="${EXP_NAME}/${perturb_type}_${attack_type}"
    local perturb_path="${perturb_dir}/perturbation.pt"
    local percentage_int=$(echo "$percentage * 100" | bc | cut -d'.' -f1)
    local model_dir="${perturb_dir}/class_${CLASS_IDX}_p${percentage_int}"
    local model_path="${model_dir}/resnet18/checkpoints/resnet18.pth"
    
    mkdir -p "$model_dir"
    
    if [ ! -f "$model_path" ]; then
        echo "Training model with ${perturb_type} ${attack_type} at ${percentage_int}% poison rate..."
        
        # Add --poison_classwise args only for classwise poisoning
        if [ "$perturb_type" == "classwise" ]; then
            python3 main.py --train \
                --train_data_type PoisonCIFAR10 \
                --poison_classwise \
                --poison_classwise_idx ${CLASS_IDX} \
                --poison_class_percentage ${percentage} \
                --perturb_tensor_filepath ${perturb_path} \
                --exp_name ${model_dir} \
                --version resnet18 \
                --plot
        else
            python3 main.py --train \
                --train_data_type PoisonCIFAR10 \
                --perturb_type ${perturb_type} \
                --poison_rate ${percentage} \
                --perturb_tensor_filepath ${perturb_path} \
                --exp_name ${model_dir} \
                --version resnet18 \
                --plot
        fi
        
        if [ ! -f "$model_path" ]; then
            echo "Error: Model training failed for ${perturb_type} ${attack_type} at ${percentage_int}%"
            return 1
        fi
        echo "Model saved to $model_path"
    else
        echo "Using existing model at $model_path"
    fi
    
    return 0
}

# Function to run detection
run_detection() {
    local attack_type=$1
    local perturb_type=$2
    local percentage=$3
    local detection_method=$4
    local perturb_dir="${EXP_NAME}/${perturb_type}_${attack_type}"
    local perturb_path="${perturb_dir}/perturbation.pt"
    local percentage_int=$(echo "$percentage * 100" | bc | cut -d'.' -f1)
    local model_dir="${perturb_dir}/class_${CLASS_IDX}_p${percentage_int}"
    local detect_dir="${model_dir}/detection_${detection_method}"
    local results_file="${detect_dir}/detection_results_${detection_method}.npz"
    
    mkdir -p "$detect_dir"
    
    if [ ! -f "$results_file" ]; then
        echo "Running ${detection_method} detection on ${perturb_type} ${attack_type} at ${percentage_int}%..."
        
        # Add --poison_classwise args only for classwise poisoning
        if [ "$perturb_type" == "classwise" ]; then
            python3 detect.py \
                --detect_dir ${detect_dir} \
                --detection_method ${detection_method} \
                --poison_method ${perturb_type} \
                --perturb_tensor_filepath ${perturb_path} \
                --poison_classwise \
                --poison_classwise_idx ${CLASS_IDX} \
                --poison_class_percentage ${percentage} \
                --epochs ${NUM_EPOCHS}
        else
            python3 detect.py \
                --detect_dir ${detect_dir} \
                --detection_method ${detection_method} \
                --poison_method ${perturb_type} \
                --perturb_tensor_filepath ${perturb_path} \
                --perturb_type ${perturb_type} \
                --poison_rate ${percentage} \
                --epochs ${NUM_EPOCHS}
        fi
        
        if [ ! -f "$results_file" ]; then
            echo "Error: Detection failed for ${perturb_type} ${attack_type} at ${percentage_int}% with ${detection_method}"
            return 1
        fi
    else
        echo "Using existing detection results at $results_file"
    fi
    
    # Extract ROC AUC score for summary
    local roc_auc=$(python3 -c "
import numpy as np
results = np.load('${results_file}')
if 'detection_roc_auc' in results:
    print(results['detection_roc_auc'])
else:
    print('N/A')
")
    
    echo "${perturb_type},${attack_type},${percentage_int},${detection_method},${roc_auc}" >> ${EXP_NAME}/results_summary.csv
    
    return 0
}

# Check if main.py has poison_class_percentage parameter
echo "=== Checking for poison_class_percentage parameter ==="
if ! grep -q "poison_class_percentage" main.py; then
    echo "Adding poison_class_percentage parameter to main.py..."
    LINE_NUMBER=$(grep -n "poison_classwise_idx" main.py | cut -d ':' -f 1)
    if [ -n "$LINE_NUMBER" ]; then
        NEW_LINE_NUMBER=$((LINE_NUMBER + 1))
        sed -i "${NEW_LINE_NUMBER}i parser.add_argument('--poison_class_percentage', default=1.0, type=float, help='Percentage of target class to poison')" main.py
        echo "Parameter added to main.py"
    else
        echo "Could not find poison_classwise_idx in main.py. Please add the parameter manually."
        exit 1
    fi
else
    echo "Parameter poison_class_percentage already exists in main.py"
fi

# Initialize results summary file
echo "perturb_type,attack_type,percentage,detection_method,roc_auc" > ${EXP_NAME}/results_summary.csv

# Main execution loop
for perturb_type in "${POISON_TYPES[@]}"; do
    for attack_type in "${ATTACK_TYPES[@]}"; do
        # Generate perturbation
        generate_perturbation "$attack_type" "$perturb_type" || continue
        
        for percentage in "${POISON_PERCENTAGES[@]}"; do
            # Train model
            train_model "$attack_type" "$perturb_type" "$percentage" || continue
            
            for detection_method in "${DETECTION_METHODS[@]}"; do
                # Run detection
                run_detection "$attack_type" "$perturb_type" "$percentage" "$detection_method" || continue
            done
        done
    done
done

# Generate comparison graphs using Python
echo "=== Generating comparison graphs ==="
python3 - <<EOF
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Load results summary
results_df = pd.read_csv('${EXP_NAME}/results_summary.csv')

# Create output directory for plots
os.makedirs('${EXP_NAME}/plots', exist_ok=True)

# 1. Plot ROC AUC by poison percentage for each detection method and attack type
plt.figure(figsize=(12, 10))

for perturb_type in results_df['perturb_type'].unique():
    plt.subplot(1, 2, 1 if perturb_type == 'classwise' else 2)
    plt.title(f'{perturb_type.capitalize()} Poisoning Detection Performance')
    
    for attack_type in results_df['attack_type'].unique():
        for detection_method in results_df['detection_method'].unique():
            data = results_df[(results_df['perturb_type'] == perturb_type) & 
                             (results_df['attack_type'] == attack_type) &
                             (results_df['detection_method'] == detection_method)]
            
            if not data.empty and 'N/A' not in data['roc_auc'].values:
                plt.plot(data['percentage'], data['roc_auc'].astype(float), 
                         marker='o', linestyle='-', 
                         label=f'{attack_type}, {detection_method}')
    
    plt.xlabel('Poison Percentage')
    plt.ylabel('Detection ROC AUC')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.ylim(0.5, 1.0)

plt.tight_layout()
plt.savefig('${EXP_NAME}/plots/detection_performance_by_percentage.png')

# 2. Compare detection methods
plt.figure(figsize=(15, 10))
grouped = results_df.groupby(['detection_method', 'perturb_type', 'attack_type'])['roc_auc'].mean().reset_index()

for i, method in enumerate(grouped['detection_method'].unique()):
    plt.subplot(2, 2, i+1)
    plt.title(f'{method} Performance')
    
    for perturb_type in grouped['perturb_type'].unique():
        for attack_type in grouped['attack_type'].unique():
            data = grouped[(grouped['detection_method'] == method) & 
                         (grouped['perturb_type'] == perturb_type) &
                         (grouped['attack_type'] == attack_type)]
            
            if not data.empty and 'N/A' not in data['roc_auc'].values:
                plt.bar(f'{perturb_type}\n{attack_type}', 
                        data['roc_auc'].astype(float).values[0],
                        label=f'{perturb_type}, {attack_type}')
    
    plt.ylabel('Average ROC AUC')
    plt.ylim(0.5, 1.0)
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('${EXP_NAME}/plots/detection_methods_comparison.png')

# 3. Compare attack types
plt.figure(figsize=(12, 6))
grouped = results_df.groupby(['attack_type', 'perturb_type'])['roc_auc'].mean().reset_index()

for i, attack in enumerate(grouped['attack_type'].unique()):
    plt.subplot(1, 2, i+1)
    plt.title(f'{attack} Attack Detectability')
    
    for perturb_type in grouped['perturb_type'].unique():
        data = grouped[(grouped['attack_type'] == attack) & 
                     (grouped['perturb_type'] == perturb_type)]
        
        if not data.empty and 'N/A' not in data['roc_auc'].values:
            plt.bar(perturb_type, data['roc_auc'].astype(float).values[0],
                    label=perturb_type)
    
    plt.ylabel('Average ROC AUC')
    plt.ylim(0.5, 1.0)
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')
    plt.legend()

plt.tight_layout()
plt.savefig('${EXP_NAME}/plots/attack_types_comparison.png')

print("Graphs generated successfully in ${EXP_NAME}/plots/")
EOF

echo "=== Experiment Completed ==="
echo "Summary:"
echo "- Results saved to ${EXP_NAME}/"
echo "- Comparison graphs saved to ${EXP_NAME}/plots/"
echo "- Full results available in ${EXP_NAME}/results_summary.csv"