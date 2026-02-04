#! /bin/bash


export HUGGINGFACE_TOKEN=hxxxx
export HUGGINFACE_REPO_ID=FPEvalRepoPublic/LeetCodeProblem
export OPENAI_API_KEY=xxxxx
export BASE_URL=xxxx
export HF_TOKEN=hxxxx
export WORKFLOW=basic
# export MODEL_NAME=openai/gpt-4o-mini #openai/gpt-4o
export MODEL_NAME= 
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

# for LANGUAGE in ocaml; do
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

for LANGUAGE in java; do
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
