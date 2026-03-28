#!/usr/bin/env python3
"""
Copy missing files from ./output/<lang>/<model>/ to ./output_fixed/<lang>/<model>/
based on missing_files_from_common_fixed.json.
"""

import json
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
MISSING_JSON = SCRIPT_DIR / "missing_files_from_common_fixed.json"

with open(MISSING_JSON) as f:
    entries = json.load(f)

copied = 0
skipped = 0
not_found = 0

for entry in entries:
    dest_dir = ROOT_DIR / entry["name"]  # ./output_fixed/<lang>/<model>
    # Derive source dir by replacing output_fixed with output
    src_dir_str = entry["name"].replace("./output_fixed/", "./output/", 1)
    src_dir = ROOT_DIR / src_dir_str

    for filename in entry["missing"]:
        src_file = src_dir / filename
        dest_file = dest_dir / filename

        if dest_file.exists():
            skipped += 1
            continue

        if not src_file.exists():
            print(f"NOT FOUND: {src_file}")
            not_found += 1
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dest_file)
        copied += 1

print(f"\nDone: {copied} copied, {skipped} already existed, {not_found} not found in ./output")
