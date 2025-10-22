#!/bin/bash

# Default values
ACCESS_KEY=""
SECRET_KEY=""
HF_TOKEN=""
SERVER="aws"
MODEL_PATH="/scr/echollm/checkpoints/AWS/"
NAME=""
DEVICE="0"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --access-key) ACCESS_KEY="$2"; shift ;;  # Shift to skip value
        --secret-key) SECRET_KEY="$2"; shift ;;  # Shift to skip value
        --hf-token) HF_TOKEN="$2"; shift ;;  # Shift to skip value
        --server) SERVER="$2"; shift ;;  # Shift to skip value
        --path) MODEL_PATH="$2"; shift ;;  # Shift to skip value
        --name) NAME="$2"; shift ;;  # Shift to skip value
        --device) DEVICE="$2"; shift ;;  # Shift to skip value
        --help) 
            echo "Usage: $0 [--access-key <AWS access-key>] [--secret-key <AWS secret-key>] [--hf-token <Hugging Face token>]"
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [ "$SERVER" == "aws" ]; then
    echo "Grabbing models from AWS"
    for LOSS in "clipped_0.99" "focal_2" "focal_0"; do
        for STAGE in "0" "1" "2" "3" "4" "5"; do
            echo "$LOSS M$STAGE"
            python convert_model.py --access-key $ACCESS_KEY \
            --secret-key $SECRET_KEY \
            --bucket-name aceslab \
            --aws-path checkpoints/$LOSS/frac6/meta-llama/Llama-3.2-1B/M$STAGE-llama1b-batch64-lr2e-05-seed1/last.ckpt \
            --local-path /scr/echollm/checkpoints/AWS/$LOSS/frac6/meta-llama/Llama-3.2-1B/M$STAGE-llama1b-batch64-lr2e-05-seed1/last.ckpt \
            --token $HF_TOKEN

            echo "Evaluating the model"
            lm_eval --model hf --model_args pretrained=/scr/echollm/checkpoints/AWS/$LOSS/frac6/meta-llama/Llama-3.2-1B/M$STAGE-llama1b-batch64-lr2e-05-seed1 \
            --tasks gsm8k,hellaswag,mathqa --device cuda:$DEVICE --batch_size 64 --trust_remote_code \
            --output_path /scr/echollm/checkpoints/AWS/$LOSS/frac6/meta-llama/Llama-3.2-1B/M$STAGE-llama1b-batch64-lr2e-05-seed1/results_new_128

        done
    done
else
    for STAGE in "0" "1" "2" "3" "4" "5"; do
        echo "Converting the model to HF"
        python convert_model.py --local-path "${MODEL_PATH}M${STAGE}${NAME}" --token $HF_TOKEN --server $SERVER
        echo "Evaluating the model"
        lm_eval --model hf --model_args pretrained="${MODEL_PATH}M${STAGE}${NAME}" --tasks gsm8k,hellaswag --device cuda:$DEVICE --batch_size 64 --trust_remote_code \
        --output_path "${MODEL_PATH}M${STAGE}${NAME}/results_gsm8k_hellaswag_wikitext" --wandb_args project=lm-eval-echollm,name=eval_M${STAGE}${NAME}
    done
fi