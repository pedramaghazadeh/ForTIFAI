#!/bin/bash
# GPUs to be used
# read -p "Enter number of devices: " NUM_DEVICES
# read -p "Enter the devices: " DEVICES
# read -p "Enter the seed: " SEED
# read -p "Enter the number of fractions: " DATASET_FRACTIONS
# read -p "Enter the Focal Gamma: " FOCAL_GAMMA
# read -p "Enter the max epochs: " MAX_EPOCHS
# read -p "Enter the AWS Access Key: " AWS_ACCESS_KEY
# read -p "Enter the AWS Secret Key: " AWS_SECRET_KEY

# Default values
NUM_DEVICES=1
DEVICES=0
LOSS_TYPE="focal"
SEED=1
DATASET_FRACTIONS=6
FOCAL_GAMMA=2.0
MAX_EPOCHS=10
AWS_ACCESS_KEY="default_key"
AWS_SECRET_KEY="default_secret"
UNIQUE_ID=$(date +"%Y%m%d%H%M%S")
TASK="Imagination"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --num-devices) OTHER_OPTION="$2"; shift ;;
        --devices) DEVICES="$2"; shift ;;
        --loss-type) LOSS_TYPE="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --dataset-fractions) DATASET_FRACTIONS="$2"; shift ;;
        --focal-gamma) FOCAL_GAMMA="$2"; shift ;;
        --max-epochs) MAX_EPOCHS="$2"; shift ;;
        --aws-access-key) AWS_ACCESS_KEY="$2"; shift ;;
        --aws-secret-key) AWS_SECRET_KEY="$2"; shift ;;
        --unique-id) UNIQUE_ID="$2"; shift ;;
        --task) TASK="$2"; shift ;;
        --help)
            echo "Usage: $0 [--aws-access-key <key>] [--other-option <value>]"
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done


MODEL_TAG="meta-llama/Llama-3.2-1B"
# MODEL_TAG="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# MODEL_TAG="google/gemma-2-2b-it"
MODEL_NAME="llama1b"
# MODEL_NAME="gemma2b"
# MODEL_NAME="deepseek1.5b"
# DATASET_FRACTIONS=1

# FOCAL_GAMMA="2.0"
FOCAL_ALPHA=1.0
USE_CONSISTENCY_BLOCK="False"

# Loss type
LOSS_TYPE="${LOSS_TYPE}_${FOCAL_GAMMA}"

# LOSS_TYPE="dynamic_clipped_${FOCAL_GAMMA}"
# LOSS_TYPE="clipped_${FOCAL_GAMMA}"
# LOSS_TYPE="focal_${FOCAL_GAMMA}"
# LOSS_TYPE="clpfocal_${FOCAL_GAMMA}"
# LOSS_TYPE="entropy_${FOCAL_GAMMA}"

TAGS="${LOSS_TYPE}"

# Training settings
BATCH_SIZE=64
ACCUMULATE_GRAD_BATCHES=32
NUM_WORKERS=127
BLOCK_SIZE=128
LEARNING_RATE=2e-5
OPTIMIZER="adam"

LOAD_GENERATE="/opt/dlami/nvme/echollm/generated_datasets/${MODEL_NAME}/${LOSS_TYPE}/frac_${DATASET_FRACTIONS}/batch${BATCH_SIZE}-lr${LEARNING_RATE}-seed${SEED}/"
SERVER="aws"

# Evaluations
BLIMP_EVAL="adjunct_island,causative"
EVALUATE_KT="kt_dataset_gpt_mini.json"

# Not to be changed frequently
STRATEGY="ddp"
GENERATED_LENGTH=32
GENERATION_CONTEXT_LENGTH=512
GENERATE_BATCH_SIZE=16
GENERATE_NUM_WORKERS=64

# Loop through dataset fractions
for ((stage=0; stage<DATASET_FRACTIONS; stage++)); do
    CUDA_VISIBLE_DEVICES=$DEVICES python pipeline.py \
        --stage $stage \
        --loss_type $LOSS_TYPE \
        --model_tag $MODEL_TAG \
        --focal_gamma $FOCAL_GAMMA \
        --focal_alpha $FOCAL_ALPHA \
        --batch_size $BATCH_SIZE \
        --num_devices $NUM_DEVICES \
        --accumulate_grad_batches $ACCUMULATE_GRAD_BATCHES \
        --num_workers $NUM_WORKERS \
        --block_size $BLOCK_SIZE \
        --max_epochs $MAX_EPOCHS \
        --strategy $STRATEGY \
        --seed $SEED \
        --learning_rate $LEARNING_RATE \
        --optimizer $OPTIMIZER \
        --blimp_eval $BLIMP_EVAL \
        --evaluate_kt $EVALUATE_KT \
        --dataset_fractions $DATASET_FRACTIONS \
        --load_generate $LOAD_GENERATE \
        --generated_length $GENERATED_LENGTH \
        --generation_context_length $GENERATION_CONTEXT_LENGTH \
        --use_consistency_block "False" \
        --generate_batch_size $GENERATE_BATCH_SIZE \
        --generate_num_workers $GENERATE_NUM_WORKERS \
        --tags $TAGS \
        --server $SERVER \
        --AWS_ACCESS_KEY $AWS_ACCESS_KEY\
        --AWS_SECRET_KEY $AWS_SECRET_KEY\
        --UNIQUE_ID $UNIQUE_ID \
        --task $TASK \
        # --test_mode $TEST_MODE # Uncomment to run in test mode
    # Check if the last command was successful
    if [ $? -ne 0 ]; then
        echo "Error: pipeline failed at stage $stage. Stopping execution."
        exit 1
    fi
done