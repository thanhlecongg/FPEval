#!/usr/bin/env python3
"""
In-place fix of model_name field in all JSON files under output_v2.

Mappings (folder -> model_name to write):
  Qwen2.5-Coder-1.5B-...-100k-finetuned-v2  ->  same name (Qwen2.5-Coder-1.5B-...-100k-finetuned-v2)
  Qwen2.5-Coder-3B-...-100k-finetuned-v2    ->  Qwen2.5-Coder-3B-...-100k-finetuned-v2
"""

import sys
import json
import os

BASE = "/home/locpb4/research_llm_to_code/FPEval/output_v2"

TARGETS = [
    {
        "folder": "Qwen/Qwen2.5-Coder-1.5B-Instruct-moe-finetuned-ver-1-all-peft-500k-merged-100k-finetuned-v2",
        "model_name": "Qwen/Qwen2.5-Coder-1.5B-Instruct-moe-finetuned-ver-1-all-peft-500k-merged-100k-finetuned-v2",
    },
    {
        "folder": "Qwen/Qwen2.5-Coder-3B-Instruct-moe-finetuned-ver-1-all-peft-500k-merged-100k-finetuned-v2",
        "model_name": "Qwen/Qwen2.5-Coder-3B-Instruct-moe-finetuned-ver-1-all-peft-500k-merged-100k-finetuned-v2",
    },
]

dry_run = "--dry-run" in sys.argv

if dry_run:
    print("[DRY-RUN] No files will be modified.\n")

for entry in TARGETS:
    folder_suffix = entry["folder"]
    target_model_name = entry["model_name"]

    for lang in sorted(os.listdir(BASE)):
        dir_path = os.path.join(BASE, lang, folder_suffix)
        if not os.path.isdir(dir_path):
            print(f"[{lang}] Not found: {dir_path} — skipping")
            continue

        files = sorted(f for f in os.listdir(dir_path) if f.endswith(".json"))
        total = len(files)
        updated = 0

        if dry_run:
            print(f"[{lang}] {folder_suffix}")
            print(f"  -> model_name will be set to: {target_model_name}")
            print(f"  -> {total} files\n")
            continue

        for i, fname in enumerate(files, 1):
            fpath = os.path.join(dir_path, fname)
            with open(fpath, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            changed = False
            for msg in data.get("messages", []):
                meta = msg.get("response_metadata", {})
                if "model_name" in meta and meta["model_name"] != target_model_name:
                    meta["model_name"] = target_model_name
                    changed = True

            if changed:
                with open(fpath, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, indent=4, ensure_ascii=False)
                updated += 1

            if i % 50 == 0 or i == total:
                print(f"  [{lang}] {i}/{total} processed, {updated} updated", flush=True)

        print(f"[{lang}] {folder_suffix} — done ({updated}/{total} files updated)")
