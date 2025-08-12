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

# Loop through each config file
for CONFIG_PATH in "${CONFIG_FILES[@]}"
do
    echo "Processing config file: $CONFIG_PATH"
    
    # Check if config file exists
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "Config file not found: $CONFIG_PATH. Skipping..."
        continue
    fi
    
    # Extract parameters from config file
    OUTPUT_DIR=$(grep "^output_dir:" "$CONFIG_PATH" | sed "s/^output_dir: //")
    MAX_STEPS=$(grep "^max_steps:" "$CONFIG_PATH" | sed "s/^max_steps: //")
    N_DEVICES=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
    echo $N_DEVICES $OUTPUT_DIR $MAX_STEPS

    # Handle IDF settings
    IDF_REQUIRES_GRAD=$(grep "^idf_requires_grad:" "$CONFIG_PATH" | sed "s/^idf_requires_grad: //")
    IDF_PATH=$(grep "^idf_path:" "$CONFIG_PATH" | sed "s/^idf_path: //")
    if [ "$IDF_REQUIRES_GRAD" = "true" ]; then
        IDF_PATH="${OUTPUT_DIR}/checkpoint-${MAX_STEPS}/idf.json"
    elif [ -n "$IDF_PATH" ]; then
        IDF_PATH=$IDF_PATH
    else
        IDF_PATH="null"
    fi
    echo "IDF_PATH: $IDF_PATH"

    # Handle use_l0 parameter
    USE_L0=$(grep "^use_l0:" "$CONFIG_PATH" | sed "s/^use_l0: //")
    if [ -n "$USE_L0" ]; then
        USE_L0=$USE_L0
    else
        USE_L0="true"
    fi
    echo "USE_L0: $USE_L0"

    # Handle preprocess function
    PREPROCESS_FUNC=$(grep "^preprocess_func:" "$CONFIG_PATH" | sed "s/^preprocess_func: //")
    if [ -n "$PREPROCESS_FUNC" ]; then
        PREPROCESS_FUNC=$PREPROCESS_FUNC
    else
        PREPROCESS_FUNC="null"
    fi
    echo "PREPROCESS_FUNC: $PREPROCESS_FUNC"

    # Train the model
    torchrun --nproc_per_node=${N_DEVICES} train_ir.py $CONFIG_PATH

    # Evaluate the model
    for step in {$MAX_STEPS,}
    do
        torchrun --nproc_per_node=${N_DEVICES} evaluate_beir.py \
          --model_name_or_path ${OUTPUT_DIR}/checkpoint-${step} \
          --inf_free true \
          --output_dir ${OUTPUT_DIR} \
          --log_level info \
          --per_device_eval_batch_size 50 \
          --idf_path $IDF_PATH \
          --preprocess_func $PREPROCESS_FUNC \
          --use_l0 $USE_L0
    done

    echo "Completed processing $CONFIG_PATH"
    echo "----------------------------------------"
done
