#! /bin/bash

# Count JSON files in each subfolder of output directory
# Print subfolders that have fewer than 721 files

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output}"
MIN_FILES="${MIN_FILES:-721}"

echo "Counting JSON files in model subfolders of: $OUTPUT_DIR"
echo "Checking directories at depth 3 (language/vendor/model)"
echo "Minimum expected files: $MIN_FILES"
echo

# Check if output directory exists
if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "Error: Output directory '$OUTPUT_DIR' does not exist"
    exit 1
fi

# Flag to track if any subfolder has fewer files
found_low_count=false

# Find all model subfolders (at depth 3: language/vendor/model)
# These are the actual directories containing JSON files
while IFS= read -r -d '' model_dir; do
    # Get relative path from OUTPUT_DIR for cleaner output
    relative_path="${model_dir#$OUTPUT_DIR/}"
    
    # Count JSON files in this model directory (not recursive, just in this dir)
    file_count=$(find "$model_dir" -maxdepth 1 -type f -name "*.json" | wc -l)
    
    # Check if count is less than minimum
    if [[ $file_count -lt $MIN_FILES ]]; then
        echo "WARNING: $relative_path has $file_count JSON files (less than $MIN_FILES)"
        found_low_count=true
    else
        echo "OK: $relative_path has $file_count JSON files"
    fi
done < <(find "$OUTPUT_DIR" -mindepth 3 -maxdepth 3 -type d -print0)

echo

# Exit with appropriate code
if [[ "$found_low_count" == true ]]; then
    echo "Some model subfolders have fewer than $MIN_FILES JSON files"
    exit 1
else
    echo "All model subfolders have at least $MIN_FILES JSON files"
    exit 0
fi
