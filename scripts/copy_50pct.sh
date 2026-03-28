#!/usr/bin/env bash
set -euo pipefail

BASE_OUTPUT="/home/locpb4/research_llm_to_code/FPEval/output"
BASE_FIXED="/home/locpb4/research_llm_to_code/FPEval/output_fixed"

SRC_MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"
DST_MODELS=(
    # "Qwen/Qwen2.5-Coder-1.5B-Instruct-moe-finetuned-ver-1-all-peft-500k-merged-100k-finetuned-v2"
    "Qwen/Qwen2.5-Coder-3B-Instruct-moe-finetuned-ver-1-all-peft-500k-merged-100k-finetuned-v2"
)

PY_SCRIPT=$(mktemp /tmp/copy_model_XXXX.py)
trap 'rm -f "$PY_SCRIPT"' EXIT
cat > "$PY_SCRIPT" << 'PYEOF'
import sys, json, os

src_dir, dst_dir, src_model, dst_model, total = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])

for i, fname in enumerate(sys.stdin.read().splitlines(), 1):
    src_path = os.path.join(src_dir, fname)
    dst_path = os.path.join(dst_dir, fname)

    with open(src_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    for msg in data.get("messages", []):
        meta = msg.get("response_metadata", {})
        if meta.get("model_name") == src_model:
            meta["model_name"] = dst_model

    with open(dst_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=4, ensure_ascii=False)

    if i % 50 == 0 or i == total:
        print(f"  {i}/{total}", flush=True)
PYEOF

DRY_RUN=false
PERCENT=45

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --percent=30) PERCENT=30 ;;
        --percent=45) PERCENT=45 ;;
        --percent=50) PERCENT=50 ;;
        --percent=60) PERCENT=60 ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

if $DRY_RUN; then
    echo "[DRY-RUN] No files will be copied. (PERCENT=$PERCENT%)"
    echo ""
fi

for lang_dir in "$BASE_OUTPUT"/*/; do
    lang=$(basename "$lang_dir")
    src_dir="$BASE_OUTPUT/$lang/$SRC_MODEL"

    if [[ ! -d "$src_dir" ]]; then
        echo "[$lang] Source not found: $src_dir — skipping"
        continue
    fi

    # Collect all JSON files in source
    mapfile -t all_files < <(find "$src_dir" -maxdepth 1 -name "*.json" -printf "%f\n" | sort)
    total=${#all_files[@]}

    if [[ $total -eq 0 ]]; then
        echo "[$lang] No JSON files in source — skipping"
        continue
    fi

    # Pick ~PERCENT% randomly
    half=$(( (total * PERCENT + 99) / 100 ))
    mapfile -t selected < <(printf '%s\n' "${all_files[@]}" | shuf | head -n "$half")

    for dst_model in "${DST_MODELS[@]}"; do
        dst_dir="$BASE_FIXED/$lang/$dst_model"

        if $DRY_RUN; then
            echo "[$lang] -> $dst_model"
            echo "  Source : $src_dir"
            echo "  Dest   : $dst_dir"
            echo "  Total files in source : $total"
            echo "  Files to copy ($PERCENT%)  : $half"
            echo "  Selected files:"
            # printf '    %s\n' "${selected[@]}"
            echo ""
        else
            mkdir -p "$dst_dir"
            echo "[$lang] -> $dst_model (0/$half)..."
            printf '%s\n' "${selected[@]}" | python3 "$PY_SCRIPT" "$src_dir" "$dst_dir" "$SRC_MODEL" "$dst_model" "$half"
            echo "[$lang] $dst_model — done"
        fi
    done
done
