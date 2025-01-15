set -e


# l0 mask loss vary threshold 
for threshold in 50 100 150 200 250 300 350 400 450 500 1000
do
    CONFIG_PATH="./config/mask_vary_threshold/${threshold}_0.04.yaml"
    OUTPUT_DIR=$(grep "^output_dir:" "$CONFIG_PATH" | sed "s/^output_dir: //")
    MAX_STEPS=$(grep "^max_steps:" "$CONFIG_PATH" | sed "s/^max_steps: //")
    N_DEVICES=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
    echo $N_DEVICES $OUTPUT_DIR $MAX_STEPS

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

    torchrun --nproc_per_node=${N_DEVICES} train_ir.py $CONFIG_PATH

    for step in {$MAX_STEPS,}
    do
    torchrun --nproc_per_node=${N_DEVICES} evaluate_beir.py \
    --model_name_or_path ${OUTPUT_DIR}/checkpoint-${step} \
    --inf_free true \
    --output_dir ${OUTPUT_DIR} \
    --log_level info \
    --per_device_eval_batch_size 50 \
    --idf_path $IDF_PATH
    done
done

# l0 mask loss+l0 activation vary threshold 
for threshold in 50 100 150 200 250 300 350 400 450 500 1000
do
    CONFIG_PATH="./config/mask+activation_vary_threshold/${threshold}_0.04.yaml"
    OUTPUT_DIR=$(grep "^output_dir:" "$CONFIG_PATH" | sed "s/^output_dir: //")
    MAX_STEPS=$(grep "^max_steps:" "$CONFIG_PATH" | sed "s/^max_steps: //")
    N_DEVICES=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
    echo $N_DEVICES $OUTPUT_DIR $MAX_STEPS

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

    torchrun --nproc_per_node=${N_DEVICES} train_ir.py $CONFIG_PATH

    for step in {$MAX_STEPS,}
    do
    torchrun --nproc_per_node=${N_DEVICES} evaluate_beir.py \
    --model_name_or_path ${OUTPUT_DIR}/checkpoint-${step} \
    --inf_free true \
    --output_dir ${OUTPUT_DIR} \
    --log_level info \
    --per_device_eval_batch_size 50 \
    --idf_path $IDF_PATH
    done
done

# l0 mask loss+l0 activation vary lambda
for lambda in 0.1 0.01 0.001 0.2 0.4 0.05 0.8 0.025 0.035
do
    CONFIG_PATH="./config/mask+activation_vary_lambda/200_${lambda}.yaml"
    OUTPUT_DIR=$(grep "^output_dir:" "$CONFIG_PATH" | sed "s/^output_dir: //")
    MAX_STEPS=$(grep "^max_steps:" "$CONFIG_PATH" | sed "s/^max_steps: //")
    N_DEVICES=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
    echo $N_DEVICES $OUTPUT_DIR $MAX_STEPS

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

    torchrun --nproc_per_node=${N_DEVICES} train_ir.py $CONFIG_PATH

    for step in {$MAX_STEPS,}
    do
    torchrun --nproc_per_node=${N_DEVICES} evaluate_beir.py \
    --model_name_or_path ${OUTPUT_DIR}/checkpoint-${step} \
    --inf_free true \
    --output_dir ${OUTPUT_DIR} \
    --log_level info \
    --per_device_eval_batch_size 50 \
    --idf_path $IDF_PATH
    done
done

# l0 mask loss vary lambda
for lambda in 0.1 0.01 0.001 0.2 0.4 0.05 0.8 0.025 0.035
do
    CONFIG_PATH="./config/mask_vary_lambda/200_${lambda}.yaml"
    OUTPUT_DIR=$(grep "^output_dir:" "$CONFIG_PATH" | sed "s/^output_dir: //")
    MAX_STEPS=$(grep "^max_steps:" "$CONFIG_PATH" | sed "s/^max_steps: //")
    N_DEVICES=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
    echo $N_DEVICES $OUTPUT_DIR $MAX_STEPS

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

    torchrun --nproc_per_node=${N_DEVICES} train_ir.py $CONFIG_PATH

    for step in {$MAX_STEPS,}
    do
    torchrun --nproc_per_node=${N_DEVICES} evaluate_beir.py \
    --model_name_or_path ${OUTPUT_DIR}/checkpoint-${step} \
    --inf_free true \
    --output_dir ${OUTPUT_DIR} \
    --log_level info \
    --per_device_eval_batch_size 50 \
    --idf_path $IDF_PATH
    done
done

# l0 activation vary lambda
for lambda in 0.1 0.01 0.4 0.04 0.004 0.035
do
    CONFIG_PATH="./config/just_activation/${lambda}.yaml"
    OUTPUT_DIR=$(grep "^output_dir:" "$CONFIG_PATH" | sed "s/^output_dir: //")
    MAX_STEPS=$(grep "^max_steps:" "$CONFIG_PATH" | sed "s/^max_steps: //")
    N_DEVICES=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
    echo $N_DEVICES $OUTPUT_DIR $MAX_STEPS

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

    torchrun --nproc_per_node=${N_DEVICES} train_ir.py $CONFIG_PATH

    for step in {$MAX_STEPS,}
    do
    torchrun --nproc_per_node=${N_DEVICES} evaluate_beir.py \
    --model_name_or_path ${OUTPUT_DIR}/checkpoint-${step} \
    --inf_free true \
    --output_dir ${OUTPUT_DIR} \
    --log_level info \
    --per_device_eval_batch_size 50 \
    --idf_path $IDF_PATH
    done
done

# vary l0 activation 
for type in decouple l2 l3 mask_decouple
do
    CONFIG_PATH="./config/vary_activation/${type}.yaml"
    OUTPUT_DIR=$(grep "^output_dir:" "$CONFIG_PATH" | sed "s/^output_dir: //")
    MAX_STEPS=$(grep "^max_steps:" "$CONFIG_PATH" | sed "s/^max_steps: //")
    N_DEVICES=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
    echo $N_DEVICES $OUTPUT_DIR $MAX_STEPS

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

    torchrun --nproc_per_node=${N_DEVICES} train_ir.py $CONFIG_PATH

    for step in {$MAX_STEPS,}
    do
    torchrun --nproc_per_node=${N_DEVICES} evaluate_beir.py \
    --model_name_or_path ${OUTPUT_DIR}/checkpoint-${step} \
    --inf_free true \
    --output_dir ${OUTPUT_DIR} \
    --log_level info \
    --per_device_eval_batch_size 50 \
    --idf_path $IDF_PATH
    done
done

# baseline
for lambda in 0.01 0.001 0.0001 0.4 0.004 0.15
do
    CONFIG_PATH="./config/baseline/${lambda}.yaml"
    OUTPUT_DIR=$(grep "^output_dir:" "$CONFIG_PATH" | sed "s/^output_dir: //")
    MAX_STEPS=$(grep "^max_steps:" "$CONFIG_PATH" | sed "s/^max_steps: //")
    N_DEVICES=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
    echo $N_DEVICES $OUTPUT_DIR $MAX_STEPS

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

    torchrun --nproc_per_node=${N_DEVICES} train_ir.py $CONFIG_PATH

    for step in {$MAX_STEPS,}
    do
    torchrun --nproc_per_node=${N_DEVICES} evaluate_beir.py \
    --model_name_or_path ${OUTPUT_DIR}/checkpoint-${step} \
    --inf_free true \
    --output_dir ${OUTPUT_DIR} \
    --log_level info \
    --per_device_eval_batch_size 50 \
    --idf_path $IDF_PATH
    done
done