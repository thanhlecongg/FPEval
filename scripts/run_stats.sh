#! /bin/bash

# Run executors for multiple languages in a loop

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configuration (can be overridden by env vars)
# MODEL_NAME="${MODEL_NAME:-gpt-3.5-turbo}"
# MODEL_NAME="${MODEL_NAME:-gpt-4o}"
# MODEL_NAME="${MODEL_NAME:-gpt-5}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
# haskell ocaml scala java
LANGUAGES=(${LANGUAGES:-haskell ocaml scala})

PRIVATE_TESTCASE_PATH="${PRIVATE_TESTCASE_PATH:-${PROJECT_ROOT}/dataset/PrivateTestCase}"
META_PATH="${META_PATH:-${PROJECT_ROOT}/dataset/LeetCodeMetaData}"

echo "Project root: $PROJECT_ROOT"
echo "Model name:   $MODEL_NAME"
echo "Languages:    ${LANGUAGES[*]}"
echo "Private TC:   $PRIVATE_TESTCASE_PATH"
echo "Meta path:    $META_PATH"
echo

# Change to core directory so Python can find the modules
cd "$SCRIPT_DIR/../core" || exit 1

for lang in "${LANGUAGES[@]}"; do
    executor="executor_${lang}.py"

    # LLM_OUTPUT_DIR="${LLM_OUTPUT_DIR_BASE:-${PROJECT_ROOT}/dataset/LLMsGeneratedCode/CodeGenerated}/${lang}/${MODEL_NAME}"
    LLM_OUTPUT_DIR="${LLM_OUTPUT_DIR_BASE:-${PROJECT_ROOT}/output/${lang}/${MODEL_NAME}}"
    OUTPUT_DIR="${OUTPUT_DIR_BASE:-${PROJECT_ROOT}/debug/results_llm_reasoning}/CodeGenerated/${lang}/${MODEL_NAME}"

    echo "========================================"
    echo "Running executor for language: $lang"
    echo "Executor:        $executor"
    echo "LLM_OUTPUT_DIR:  $LLM_OUTPUT_DIR"
    echo "OUTPUT_DIR:      $OUTPUT_DIR"
    echo "========================================"

    if [[ ! -f "$executor" ]]; then
        echo "Warning: executor script '$executor' not found, skipping $lang"
        echo
        continue
    fi

    python3 "./$executor" \
        --llm-output-dir "$LLM_OUTPUT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --private-testcase-path "$PRIVATE_TESTCASE_PATH" \
        --meta-path "$META_PATH"

    # echo

    # executor="executor_${lang}.py"

    # LLM_OUTPUT_DIR="${LLM_OUTPUT_DIR_BASE:-${PROJECT_ROOT}/dataset/LLMsGeneratedCode/InstructionRepair}/${lang}/${MODEL_NAME}"
    # OUTPUT_DIR="${OUTPUT_DIR_BASE:-${PROJECT_ROOT}/debug/results_llm_reasoning}/InstructionRepair/${lang}/${MODEL_NAME}"

    # echo "========================================"
    # echo "Running executor for language: $lang"
    # echo "Executor:        $executor"
    # echo "LLM_OUTPUT_DIR:  $LLM_OUTPUT_DIR"
    # echo "OUTPUT_DIR:      $OUTPUT_DIR"
    # echo "========================================"

    # if [[ ! -f "$executor" ]]; then
    #     echo "Warning: executor script '$executor' not found, skipping $lang"
    #     echo
    #     continue
    # fi

    # python3 "./$executor" \
    #     --llm-output-dir "$LLM_OUTPUT_DIR" \
    #     --output-dir "$OUTPUT_DIR" \
    #     --private-testcase-path "$PRIVATE_TESTCASE_PATH" \
    #     --meta-path "$META_PATH"

    # echo
done
