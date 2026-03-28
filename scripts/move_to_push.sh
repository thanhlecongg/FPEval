#! /bin/bash

# set -euo pipefail

# Copy generated code for selected models from FPEval/output
# to local-models/FPEval-Generated-Code/code/basics/{lang}/Qwen/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUTPUT_DIR="${PROJECT_ROOT}/output"
DEST_BASE="/home/locpb4/local-models/FPEval-Generated-Code/code/basics"

LANGS=(haskell ocaml scala java)

# Ensure destination directories exist
for lang in "${LANGS[@]}"; do
  mkdir -p "${DEST_BASE}/${lang}/Qwen"
done

for MODEL_NAME in \
    "Qwen/Qwen2.5-Coder-1.5B-Instruct-all-peft-500k-merged" \
    "Qwen/Qwen2.5-Coder-1.5B-Instruct-ocaml-peft-500k-merged" \
    "Qwen/Qwen2.5-Coder-1.5B-Instruct-scala-peft-500k-merged" \
    "Qwen/Qwen2.5-Coder-1.5B-Instruct-moe-finetuned-ver-1-all-peft-500k-merged-100k-finetuned" \
    "Qwen/Qwen2.5-Coder-3B-Instruct-moe-finetuned-ver-1-all-peft-500k-merged-100k-finetuned"; do
    echo "Copying model: $MODEL_NAME"

    cp -r "${OUTPUT_DIR}/haskell/${MODEL_NAME}" "${DEST_BASE}/haskell/Qwen/"
    cp -r "${OUTPUT_DIR}/ocaml/${MODEL_NAME}"   "${DEST_BASE}/ocaml/Qwen/"
    cp -r "${OUTPUT_DIR}/scala/${MODEL_NAME}"   "${DEST_BASE}/scala/Qwen/"
    cp -r "${OUTPUT_DIR}/java/${MODEL_NAME}"    "${DEST_BASE}/java/Qwen/"
done

echo "Done copying all models."
