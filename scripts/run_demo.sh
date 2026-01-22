#! /bin/bash

export HUGGINGFACE_TOKEN=xxxxxxxxxxx
export HUGGINFACE_REPO_ID=FPEvalRepoPublic/LeetCodeProblem
export LANGUAGE=scala
export WORKFLOW=basic
export MODEL_NAME=openai/gpt-4o-mini
export OUTPUT_PATH=/workspace/output/demo

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
