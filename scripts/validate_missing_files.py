#!/usr/bin/env python3
"""
Validate that the entries in scripts/missing_files.json are actually missing
from the filesystem.

For each object in missing_files.json:
  {
      "name": "./output/haskell/Qwen/Qwen2.5-1.5B-Instruct",
      "missing": ["foo.json", "bar.json", ...]
  }

this script checks whether each listed file really does *not* exist under
the corresponding directory.

Usage (from project root or any directory):
  python scripts/validate_missing_files.py
"""

import json
import os
import sys
from typing import Dict, List, Tuple


def get_project_root() -> str:
    """Return the absolute path to the project root (parent of scripts/)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)


def load_missing_file_config(project_root: str, path: str = "scripts/missing_files.json"):
    """Load the missing_files.json configuration."""
    config_path = os.path.join(project_root, path)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"missing_files.json not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("missing_files.json must contain a top-level list")

    return data


def validate_entry(project_root: str, entry: Dict) -> Tuple[str, List[str]]:
    """
    Validate a single entry from missing_files.json.

    Returns (relative_model_path, incorrect_files) where incorrect_files are
    those that are marked as missing but actually exist on disk.
    """
    name = entry.get("name")
    missing_list = entry.get("missing")

    if not isinstance(name, str) or not isinstance(missing_list, list):
        raise ValueError(f"Invalid entry format: {entry!r}")

    # The paths in JSON are relative like "./output/...", resolve from project root.
    model_dir = os.path.normpath(os.path.join(project_root, name))
    relative_model_path = os.path.relpath(model_dir, project_root)

    if not os.path.isdir(model_dir):
        print(f"[ERROR] Model directory does not exist: {relative_model_path}", file=sys.stderr)
        return relative_model_path, []

    incorrect: List[str] = []

    for filename in missing_list:
        if not isinstance(filename, str):
            print(f"[WARN] Non-string filename in entry for {relative_model_path}: {filename!r}", file=sys.stderr)
            continue

        file_path = os.path.join(model_dir, filename)
        if os.path.exists(file_path):
            incorrect.append(filename)

    return relative_model_path, incorrect


def main() -> int:
    project_root = get_project_root()

    try:
        config = load_missing_file_config(project_root)
    except Exception as e:  # noqa: BLE001
        print(f"Failed to load missing_files.json: {e}", file=sys.stderr)
        return 1

    any_incorrect = False

    print("Validating entries in scripts/missing_files.json\n")
    print(f"Project root: {project_root}")
    print()

    for entry in config:
        model_path, incorrect_files = validate_entry(project_root, entry)

        if incorrect_files:
            any_incorrect = True
            print(f"[INCORRECT] {model_path}")
            print("  Files marked as missing but present on disk:")
            for fname in sorted(incorrect_files):
                print(f"    - {fname}")
            print()
        else:
            # All listed missing files are indeed absent (or directory missing was already reported)
            print(f"[OK] {model_path} - all listed 'missing' files are absent")

    print()
    if any_incorrect:
        print("Some entries in missing_files.json are incorrect (files exist but are marked as missing).")
        return 2

    print("All entries in missing_files.json are consistent with the filesystem.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

