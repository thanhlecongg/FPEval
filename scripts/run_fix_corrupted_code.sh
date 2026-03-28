#! /bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HUB_OFFLINE=1
# export MODEL_NAME=./local_models/Qwen3-Coder-30B-A3B-Instruct
export MODEL_NAME=/working/local_models/Qwen2.5-Coder-3B-Instruct
export OUTPUT_PATH=/working/output_fixed/
export FIXED_OUTPUT_PATH=/working/output_fixed_v2/

# Change to scripts directory so Python can find the script
cd "$(dirname "$0")" || exit 1

python3 ./fix_corrupted_code.py \
    --model        $MODEL_NAME \
    --output_path  $OUTPUT_PATH \
    --languages    ocaml haskell scala java \
    --fixed_output_path $FIXED_OUTPUT_PATH \
    --temperature  1.0 \
    --max_tokens   2048 \
    --debug     
     # uncomment to scan without fixing
    # uncomment to print folder tree + before/after code
