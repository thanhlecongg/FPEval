#! /bin/bash

# Collect stats for multiple languages in a loop

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Configuration (can be overridden by env vars)
# MODEL_NAME="${MODEL_NAME:-gpt-3.5-turbo}"
# MODEL_NAME="${MODEL_NAME:-gpt-4o}"
MODEL_NAME="${MODEL_NAME:-gpt-5}"
# java
# ocaml scala
LANGUAGES=(${LANGUAGES:-haskell ocaml scala})
# BaselineRepair CodeGenerated InstructionRepair
PIPELINE_NAME="${PIPELINE_NAME:-CodeGenerated}"

PRIVATE_TESTCASE_PATH="${PRIVATE_TESTCASE_PATH:-${PROJECT_ROOT}/dataset/PrivateTestCase}"
META_PATH="${META_PATH:-${PROJECT_ROOT}/dataset/LeetCodeMetaData}"
PROBLEM_PATH="${PROBLEM_PATH:-${PROJECT_ROOT}/dataset/LeetCodeProblem}"
OUTPUT_STATS_FOLDER="${OUTPUT_STATS_FOLDER:-${PROJECT_ROOT}/stats_output/$PIPELINE_NAME}"

echo "Project root: $PROJECT_ROOT"
echo "Model name:   $MODEL_NAME"
echo "Languages:    ${LANGUAGES[*]}"
echo "Private TC:   $PRIVATE_TESTCASE_PATH"
echo "Meta path:    $META_PATH"
echo "Problem path: $PROBLEM_PATH"
echo "Output stats: $OUTPUT_STATS_FOLDER"
echo "Pipeline name: $PIPELINE_NAME"
echo

# Change to core directory so Python can find the modules
cd "$SCRIPT_DIR/../core" || exit 1

# Build base paths (without language/model)
LLM_OUTPUT_DIR_BASE="${LLM_OUTPUT_DIR_BASE:-${PROJECT_ROOT}/dataset/LLMsGeneratedCode/$PIPELINE_NAME}"
EXECUTER_OUTPUT_DIR_BASE="${EXECUTER_OUTPUT_DIR_BASE:-${PROJECT_ROOT}/results_llm_reasoning}/$PIPELINE_NAME"

echo "========================================"
echo "Collecting stats for all languages and models"
echo "LLM_OUTPUT_DIR_BASE:  $LLM_OUTPUT_DIR_BASE"
echo "EXECUTER_OUTPUT_DIR_BASE:  $EXECUTER_OUTPUT_DIR_BASE"
echo "========================================"

python3 "./collect_result.py" \
    --problem-path "$PROBLEM_PATH" \
    --executer-results-folder "$EXECUTER_OUTPUT_DIR_BASE" \
    --input-result-folder "$LLM_OUTPUT_DIR_BASE" \
    --output-stats-folder "$OUTPUT_STATS_FOLDER" \
    --langs "${LANGUAGES[@]}" \
    --models "$MODEL_NAME"

echo ""
echo "========================================"
echo "Summarizing metadata across all languages"
echo "========================================"

python3 "./summarize_metadata.py" \
    --output-stats-folder "$OUTPUT_STATS_FOLDER" \
    --langs "${LANGUAGES[@]}" \
    --models "$MODEL_NAME"

echo "Summary complete!"