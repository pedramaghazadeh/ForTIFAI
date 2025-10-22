#!/bin/bash
# Rading unique ID for the run from the command-line
UNIQUE_ID=$(date +"%Y%m%d%H%M%S")
INIT_STAGE=0
# GPUs to be used
NUM_DEVICES=1
DEVICES=0
SEED=1

# MODEL_TAG="meta-llama/Llama-3.2-1B"
# MODEL_TAG="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_TAG="google/gemma-3-1b-pt"
DATASET_FRACTIONS=6

FOCAL_GAMMA="0.99"
FOCAL_ALPHA=1.0

AWS_ACCESS_KEY="NONE"
AWS_SECRET_KEY="NONE"

TASK="wikitext"
# TASK="imagination"

# Training settings
BATCH_SIZE=64
ACCUMULATE_GRAD_BATCHES=32
NUM_WORKERS=127
BLOCK_SIZE=128
MAX_EPOCHS=10
LEARNING_RATE=2e-5 # For Gemma3 1B
# LEARNING_RATE=2e-4 # For DeepSeek 1.5B
# LEARNING_RATE=2e-5 # For LLaMA 1.5B
OPTIMIZER="adam"

SCALE=0
END_STAGE=2

if [ $SCALE -eq 0 ]; then
    END_STAGE=$DATASET_FRACTIONS
fi

# Evaluations
BLIMP_EVAL="adjunct_island,causative"
EVALUATE_KT="kt_dataset_gpt_mini_imagination.json"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --restart-stage) INIT_STAGE="$2"; shift ;;
        --unique-id) UNIQUE_ID="$2"; shift ;;
        --aws-access-key) AWS_ACCESS_KEY="$2"; shift ;;
        --aws-secret-key) AWS_SECRET_KEY="$2"; shift ;;
        --task) TASK="$2"; shift ;;
        --loss-type) LOSS_TYPE="$2"; shift ;;
        --model-tag) MODEL_TAG="$2"; shift ;;
        --dataset-fractions) DATASET_FRACTIONS="$2"; shift ;;
        --focal-gamma) FOCAL_GAMMA="$2"; shift ;;
        --focal-alpha) FOCAL_ALPHA="$2"; shift ;;
        --batch-size) BATCH_SIZE="$2"; shift ;;
        --num-devices) NUM_DEVICES="$2"; shift ;;
        --accumulate-grad-batches) ACCUMULATE_GRAD_BATCHES="$2"; shift ;;
        --num-workers) NUM_WORKERS="$2"; shift ;;
        --block-size) BLOCK_SIZE="$2"; shift ;;
        --max-epochs) MAX_EPOCHS="$2"; shift ;;
        --learning-rate) LEARNING_RATE="$2"; shift ;;
        --optimizer) OPTIMIZER="$2"; shift ;;
        --blimp-eval) BLIMP_EVAL="$2"; shift ;;
        --evaluate-kt) EVALUATE_KT="$2"; shift ;;
        --server) SERVER="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --devices) DEVICES="$2"; shift ;;
        --scale) SCALE="$2"; shift ;;
        --help) 
            echo "Usage: $0 [--unique-id <unique time id>]"
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Loss types
# LOSS_TYPE="lovasz"
# LOSS_TYPE="dynamic_clipped_alpha_${FOCAL_ALPHA}"
LOSS_TYPE="clipped_${FOCAL_GAMMA}"
# LOSS_TYPE="focal_${FOCAL_GAMMA}"
# LOSS_TYPE="clpfocal_${FOCAL_GAMMA}_clip_0.99"
# LOSS_TYPE="entropy_${FOCAL_GAMMA"
# LOSS_TYPE="baseline"
TAGS="${LOSS_TYPE}"
LOAD_GENERATE="/scr/echollm/generated_datasets/llama-1b/${LOSS_TYPE}/frac_${DATASET_FRACTIONS}/batch${BATCH_SIZE}-lr${LEARNING_RATE}-seed${SEED}-${UNIQUE_ID}/"
SERVER="local"

# Not to be changed frequently
STRATEGY="ddp"
GENERATED_LENGTH=32
GENERATION_CONTEXT_LENGTH=512
GENERATE_BATCH_SIZE=16
GENERATE_NUM_WORKERS=64

USE_CONSISTENCY_BLOCK="False"
TEST_MODE="False"

# Loop through dataset fractions
for ((stage=$INIT_STAGE; stage<$END_STAGE; stage++)); do
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
        --use_consistency_block $USE_CONSISTENCY_BLOCK \
        --generate_batch_size $GENERATE_BATCH_SIZE \
        --generate_num_workers $GENERATE_NUM_WORKERS \
        --tags $TAGS \
        --AWS_ACCESS_KEY $AWS_ACCESS_KEY \
        --AWS_SECRET_KEY $AWS_SECRET_KEY \
        --UNIQUE_ID $UNIQUE_ID \
        --task $TASK \
        --scale $SCALE \
        # --test_mode $TEST_MODE # Uncomment to run in test mode
    # Check if the last command was successful
    if [ $? -ne 0 ]; then
        echo "Error: pipeline failed at stage $stage. Stopping execution."
        exit 1
    fi
done
