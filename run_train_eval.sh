#!/bin/bash
set -e

# Check if any config files are provided as arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <config_file1> [config_file2] [config_file3] ..."
    echo "Example: $0 config_1.yaml config_2.yaml"
    exit 1
fi

# Use command line arguments as config files
CONFIG_FILES=("$@")
N_DEVICES=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)

# Loop through each config file
for CONFIG_PATH in "${CONFIG_FILES[@]}"
do
    echo "Processing config file: $CONFIG_PATH"
    
    # Check if config file exists
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "Config file not found: $CONFIG_PATH. Skipping..."
        continue
    fi

    # Train the model
    torchrun --nproc_per_node=${N_DEVICES} train_ir.py $CONFIG_PATH

    # Evaluate the model
    torchrun --nproc_per_node=${N_DEVICES} evaluate_beir.py $CONFIG_PATH

    echo "Completed processing $CONFIG_PATH"
    echo "----------------------------------------"
done
