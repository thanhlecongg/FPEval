#! /bin/bash


export HUGGINGFACE_TOKEN=hxxxx
export HUGGINFACE_REPO_ID=FPEvalRepoPublic/LeetCodeProblem

export BASE_URL=http://localhost:8088/v1   
# export BASE_URL=http://localhost:8888/v1    
export OPENAI_API_KEY=hf_____
export HF_TOKEN=hf_____
# export WORKFLOW=basic
export WORKFLOW=basic
# export MODEL_NAME=Qwen/Qwen2.5-Coder-14B-Instruct
# export MODEL_NAME=deepseek-ai/deepseek-coder-1.3b-instruct
export MODEL_NAME=Qwen/Qwen3-Coder-30B-A3B-Instruct
# export MODEL_NAME=Qwen/Qwen2.5-Coder-3B-Instruct-moe-finetuned-ver-1-all-peft-500k-merged
export OUTPUT_PATH=/workspace/output/

# Change to core directory so Python can find the modules
cd "$(dirname "$0")/../core" || exit 1

# haskell
# scala
# ocaml
# java
# Loop over languages
# for LANGUAGE in haskell; do
#     export LANGUAGE=$LANGUAGE
#     echo "Processing language: $LANGUAGE"
#     # --token $HUGGINGFACE_TOKEN \
#     python3 ./run_dataset.py \
#                     --repo_id $HUGGINFACE_REPO_ID \
#                     --language $LANGUAGE \
#                     --workflow $WORKFLOW \
#                     --model_name $MODEL_NAME \
#                     --output_path $OUTPUT_PATH \
#                     --no-download
# done

# for LANGUAGE in scala; do
#     echo "Processing language: $LANGUAGE"
#     # --token $HUGGINGFACE_TOKEN \
#     python3 ./run_dataset.py \
#                     --repo_id $HUGGINFACE_REPO_ID \
#                     --language $LANGUAGE \
#                     --workflow $WORKFLOW \
#                     --model_name $MODEL_NAME \
#                     --output_path $OUTPUT_PATH \
#                     --no-download
# done

for LANGUAGE in ocaml; do
    echo "Processing language: $LANGUAGE"
    # --token $HUGGINGFACE_TOKEN \
    python3 ./run_dataset.py \
                    --repo_id $HUGGINFACE_REPO_ID \
                    --language $LANGUAGE \
                    --workflow $WORKFLOW \
                    --model_name $MODEL_NAME \
                    --output_path $OUTPUT_PATH \
                    --no-download
done

# for LANGUAGE in java; do
#     echo "Processing language: $LANGUAGE"
#     # --token $HUGGINGFACE_TOKEN \
#     python3 ./run_dataset.py \
#                     --repo_id $HUGGINFACE_REPO_ID \
#                     --language $LANGUAGE \
#                     --workflow $WORKFLOW \
#                     --model_name $MODEL_NAME \
#                     --output_path $OUTPUT_PATH \
#                     --no-download
# done
