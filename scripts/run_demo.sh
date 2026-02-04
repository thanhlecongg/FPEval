#! /bin/bash

export HUGGINGFACE_TOKEN=hxxxx
export HUGGINFACE_REPO_ID=FPEvalRepoPublic/LeetCodeProblem
export OPENAI_API_KEY=xxxxx
export BASE_URL=xxxx
export HF_TOKEN=hxxxx
export LANGUAGE=scala # has
export WORKFLOW=basic
# export MODEL_NAME=openai/gpt-4o-mini #openai/gpt-4o
export MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
export OUTPUT_PATH=/workspace/output/

# Change to core directory so Python can find the modules
cd "$(dirname "$0")/../core" || exit 1
# --token $HUGGINGFACE_TOKEN \
python3 ./run_dataset.py \
                --repo_id $HUGGINFACE_REPO_ID \
                --language $LANGUAGE \
                --workflow $WORKFLOW \
                --model_name $MODEL_NAME \
                --output_path $OUTPUT_PATH \
                --no-download
