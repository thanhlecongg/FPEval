#!/usr/bin/env bash

set -euo pipefail

echo "=== Hugging Face folder upload (Python) ==="

if ! command -v python3 >/dev/null 2>&1; then
  echo "'python3' not found on PATH."
  exit 1
fi
export HUGGINGFACE_TOKEN=hf_____
export HUGGINFACE_REPO_ID=BaoLocTown/FPEval-Generated-Code-Zipped
export OUTPUT_PATH=/home/locpb4/local-models/FPEval-Generated-Code-Zipped

python3 - << 'EOF'
import os
import sys
import getpass

try:
    from huggingface_hub import HfApi
except ImportError:
    print("The 'huggingface_hub' package is required.")
    print("Install it with: pip install huggingface_hub")
    sys.exit(1)


def main() -> None:
    print("=== Hugging Face folder upload using huggingface_hub ===")

    # token = getpass.getpass("Enter your Hugging Face token: ").strip()
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        print("Token cannot be empty.")
        sys.exit(1)

    # repo_id = input("Enter dataset repo id (e.g. username/dataset-name): ").strip()
    repo_id = os.getenv("HUGGINFACE_REPO_ID")
    if not repo_id:
        print("Dataset repo id cannot be empty.")
        sys.exit(1)

    folder_path = os.getenv("OUTPUT_PATH")
    if not folder_path:
        print("Folder path cannot be empty.")
        sys.exit(1)

    if not os.path.isdir(folder_path):
        print(f"Folder '{folder_path}' does not exist or is not a directory.")
        sys.exit(1)

    api = HfApi(token=token)

    # Ensure the dataset repo exists (will not fail if it already exists)
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
    )

    print(f"Uploading folder '{folder_path}' to dataset '{repo_id}' on Hugging Face...")

    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=folder_path,
    )

    print("Upload complete.")


if __name__ == "__main__":
    main()
EOF

# python3 - << 'EOF'
# import os
# import sys
# from pathlib import Path

# try:
#     from huggingface_hub import HfApi
# except ImportError:
#     print("The 'huggingface_hub' package is required.")
#     print("Install it with: pip install huggingface_hub")
#     sys.exit(1)


# def iter_local_files(folder_path: Path):
#     """Yield all files under folder_path recursively."""
#     for path in folder_path.rglob("*"):
#         if path.is_file():
#             yield path


# def main() -> None:
#     print("=== Hugging Face folder upload: new files only ===")

#     token = os.getenv("HUGGINGFACE_TOKEN")
#     if not token:
#         print("Token cannot be empty.")
#         sys.exit(1)

#     # Note: your original code used HUGGINFACE_REPO_ID (missing 'G' after HUG).
#     # Keeping the same env var here for compatibility.
#     repo_id = os.getenv("HUGGINFACE_REPO_ID")
#     if not repo_id:
#         print("Dataset repo id cannot be empty.")
#         sys.exit(1)

#     folder_path_str = os.getenv("OUTPUT_PATH")
#     if not folder_path_str:
#         print("Folder path cannot be empty.")
#         sys.exit(1)

#     folder_path = Path(folder_path_str)
#     if not folder_path.is_dir():
#         print(f"Folder '{folder_path}' does not exist or is not a directory.")
#         sys.exit(1)

#     api = HfApi(token=token)

#     # Ensure the dataset repo exists
#     api.create_repo(
#         repo_id=repo_id,
#         repo_type="dataset",
#         exist_ok=True,
#     )

#     print(f"Reading existing files from dataset '{repo_id}'...")
#     existing_files = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))

#     to_upload = []
#     for local_file in iter_local_files(folder_path):
#         rel_path = local_file.relative_to(folder_path).as_posix()
#         if rel_path not in existing_files:
#             to_upload.append((local_file, rel_path))

#     if not to_upload:
#         print("No new files to upload.")
#         return

#     print(f"Found {len(to_upload)} new file(s) to upload.")

#     for local_file, rel_path in to_upload:
#         print(f"Uploading: {rel_path}")
#         api.upload_file(
#             path_or_fileobj=str(local_file),
#             path_in_repo=rel_path,
#             repo_id=repo_id,
#             repo_type="dataset",
#         )

#     print("Upload complete.")
#     print(f"Uploaded {len(to_upload)} new file(s).")


# if __name__ == "__main__":
#     main()
# EOF

