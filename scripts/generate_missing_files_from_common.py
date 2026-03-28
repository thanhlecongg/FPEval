#!/usr/bin/env python3
"""
Generate a JSON file listing, for each model subfolder under ./output,
which files from core/common_files.pkl are missing.

The output format matches scripts/missing_files.json:
[
    {
        "name": "./output/haskell/Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "missing": ["foo.json", "bar.json", ...]
    },
    ...
]

Usage (from project root or any directory):
  python scripts/generate_missing_files_from_common.py
"""

import json
import os
import pickle
from typing import List, Dict


def get_project_root() -> str:
    """Return the absolute path to the project root (parent of scripts/)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)


def load_common_files(project_root: str) -> List[str]:
    """
    Load the list of common files from core/common_files.pkl.

    The pickle is expected to contain an iterable of file names like
    'add-edges-to-make-degrees-of-all-nodes-even.json'.
    We will convert these to the underscore style used in output folders,
    e.g. 'add_edges_to_make_degrees_of_all_nodes_even.json'.
    """
    pkl_path = os.path.join(project_root, "core", "common_files.pkl")
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"common_files.pkl not found at: {pkl_path}")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, (list, tuple, set)):
        raise ValueError(
            f"Expected common_files.pkl to contain a list/tuple/set, got {type(data)!r}"
        )

    converted: List[str] = []
    for name in data:
        if not isinstance(name, str):
            continue
        # Convert kebab-case to snake_case to match filenames under ./output
        converted.append(name.replace("-", "_"))

    return converted


def find_model_dirs(project_root: str, output_rel: str = "output_fixed") -> List[str]:
    """
    Discover all model directories under the output directory.

    We follow the same convention as scripts/count_json_files.sh and look for
    directories at depth 3 under ./output (language/vendor/model).

    Returns a list of paths relative to project_root, e.g.
      "./output/haskell/Qwen/Qwen2.5-Coder-1.5B-Instruct"
    """
    output_dir = os.path.join(project_root, output_rel)
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    model_dirs: List[str] = []

    # Walk output_dir and collect directories at depth 3:
    # output/<lang>/<vendor>/<model>
    for lang in os.listdir(output_dir):
        lang_path = os.path.join(output_dir, lang)
        if not os.path.isdir(lang_path):
            continue

        for vendor in os.listdir(lang_path):
            vendor_path = os.path.join(lang_path, vendor)
            if not os.path.isdir(vendor_path):
                continue

            for model in os.listdir(vendor_path):
                model_path = os.path.join(vendor_path, model)
                if not os.path.isdir(model_path):
                    continue

                rel_path = os.path.relpath(model_path, project_root)
                # Match the "./output/..." style used in existing JSON
                model_dirs.append(f"./{rel_path}")

    return sorted(model_dirs)


def compute_missing_for_model(project_root: str, model_dir_rel: str, common_files: List[str]) -> Dict:
    """
    For a single model directory (relative path like "./output/..."), compute
    which common files are missing.
    """
    model_dir_abs = os.path.normpath(os.path.join(project_root, model_dir_rel))
    missing: List[str] = []

    if not os.path.isdir(model_dir_abs):
        # If the directory itself does not exist, we consider all common files missing.
        # This mirrors the style of other scripts which preserve missing lists when dirs
        # cannot be checked.
        return {"name": model_dir_rel, "missing": list(common_files)}

    for filename in common_files:
        if not filename.endswith(".json"):
            # Only care about JSON files in the output dirs
            continue
        file_path = os.path.join(model_dir_abs, filename)
        if not os.path.exists(file_path):
            missing.append(filename)

    return {"name": model_dir_rel, "missing": missing}


def main() -> int:
    project_root = get_project_root()

    try:
        common_files = load_common_files(project_root)
    except Exception as e:  # noqa: BLE001
        print(f"Failed to load common_files.pkl: {e}")
        return 1

    print(f"Project root: {project_root}")
    print(f"Total common files: {len(common_files)}")

    try:
        model_dirs = find_model_dirs(project_root)
    except Exception as e:  # noqa: BLE001
        print(f"Failed to discover model directories: {e}")
        return 1

    print(f"Discovered {len(model_dirs)} model directories under ./output")

    entries: List[Dict] = []
    for model_dir_rel in model_dirs:
        entry = compute_missing_for_model(project_root, model_dir_rel, common_files)
        print(f"{model_dir_rel}: {len(entry['missing'])} missing from common set")
        entries.append(entry)

    output_path = os.path.join(project_root, "scripts", "missing_files_from_common_fixed.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=4, ensure_ascii=False)

    print()
    print(f"Wrote missing-from-common list to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

