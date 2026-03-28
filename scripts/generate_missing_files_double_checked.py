#!/usr/bin/env python3
"""
Double-check `scripts/missing_files.json` against the filesystem and write a
cleaned version to `scripts/missing_files_double_checked.json`.

For each entry:
  {
      "name": "./output/haskell/Qwen/Qwen2.5-1.5B-Instruct",
      "missing": ["foo.json", "bar.json", ...]
  }

we verify each file in "missing":
  - If `<name>/<file>` does **not** exist, we keep it in the new "missing".
  - If it **does** exist, we drop it (it is not actually missing).

The structure of the output file is identical to the input, but with
"missing" filtered to only truly absent files.

Usage (from project root or any directory):
  python scripts/generate_missing_files_double_checked.py
"""

import json
import os
import sys
from typing import Dict, List


def get_project_root() -> str:
    """Return the absolute path to the project root (parent of scripts/)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)


def load_existing_missing(project_root: str) -> List[Dict]:
    """Load scripts/missing_files.json."""
    path = os.path.join(project_root, "scripts", "missing_files_from_common_fixed.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"missing_files.json not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("missing_files.json must contain a top-level list")

    return data


def filter_missing_for_entry(project_root: str, entry: Dict) -> Dict:
    """
    Given a single entry from missing_files.json, filter its "missing" list to
    include only files that are truly absent on disk.
    """
    name = entry.get("name")
    missing_list = entry.get("missing", [])

    if not isinstance(name, str):
        raise ValueError(f"Invalid entry (missing or non-string 'name'): {entry!r}")
    if not isinstance(missing_list, list):
        raise ValueError(
            f"Invalid entry (missing is not a list) for {name!r}: {missing_list!r}"
        )

    model_dir = os.path.normpath(os.path.join(project_root, name))
    if not os.path.isdir(model_dir):
        # If the directory itself is missing, we cannot verify; preserve the list.
        print(
            f"[WARN] Model directory does not exist, preserving original 'missing': {name}",
            file=sys.stderr,
        )
        return {"name": name, "missing": list(missing_list)}

    still_missing: List[str] = []
    now_present: List[str] = []

    for filename in missing_list:
        if not isinstance(filename, str):
            print(
                f"[WARN] Non-string filename in 'missing' for {name}: {filename!r}",
                file=sys.stderr,
            )
            continue
        file_path = os.path.join(model_dir, filename)
        if os.path.exists(file_path):
            now_present.append(filename)
        else:
            still_missing.append(filename)

    print(
        f"{name}: {len(still_missing)} still missing, "
        f"{len(now_present)} found on disk and removed"
    )

    return {"name": name, "missing": still_missing}


def main() -> int:
    project_root = get_project_root()

    try:
        existing = load_existing_missing(project_root)
    except Exception as e:  # noqa: BLE001
        print(f"Failed to load missing_files.json: {e}", file=sys.stderr)
        return 1

    print(f"Project root: {project_root}")
    print(f"Entries in missing_files.json: {len(existing)}")
    print()

    new_entries: List[Dict] = []
    for entry in existing:
        new_entry = filter_missing_for_entry(project_root, entry)
        new_entries.append(new_entry)

    output_path = os.path.join(
        project_root, "scripts", "missing_files_double_checked_fixed.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_entries, f, indent=4, ensure_ascii=False)

    print()
    print(f"Wrote double-checked missing list to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


