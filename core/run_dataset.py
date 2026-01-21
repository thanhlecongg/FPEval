import argparse
import uuid
import os
import json
from generator import CodeGenerator
import urllib3
from huggingface_hub import HfApi, login, snapshot_download
from basic_self_repair import main as start_repairing

# Increase timeout for potential large downloads/uploads
http = urllib3.PoolManager(timeout=30.0)  

# Keep the base directory consistent
TARGET_ROOT_DIR = "/workspace/dataset"

def process_repo(repo_id, target_dir):
    print(f"Downloading dataset from {repo_id} to {target_dir}...")
    
    # This downloads the 'dataset' folder from the repo into {target_dir}/dataset
    downloaded_path = snapshot_download(
        repo_id=repo_id, 
        allow_patterns=["dataset/*"], 
        local_dir=target_dir,
        repo_type="dataset"
    )

    dataset_path = os.path.join(target_dir, "dataset")
    return dataset_path

def main():
    parser = argparse.ArgumentParser(description="Generate specifications from a program.")
    parser.add_argument("--no-download", action="store_true", help="Skip downloading if already exists")
    # Updated default repo_id
    parser.add_argument("--repo_id", default="LLMFP/LeetCodeProblem", help="Repository ID to download")
    parser.add_argument("--language", default="scala", help="Programming language")
    parser.add_argument("--workflow", default="gen_code_workflow", help="Workflow to use for the generator")
    parser.add_argument("--model_name", default="gpt-3.5-turbo", help="Model name for the coding assistant")
    parser.add_argument("--output_path", default="output", help="Path to save the generated code")
    
    args = parser.parse_args()

    # Create the specific output subdirectory early to simplify processing
    output_dir = os.path.join(args.output_path, args.language, args.model_name)
    os.makedirs(output_dir, exist_ok=True)

    if not args.no_download:
        print(f"Downloading dataset from {args.repo_id} to {TARGET_ROOT_DIR}...")
        dataset_path = process_repo(args.repo_id, TARGET_ROOT_DIR)
    else:
        dataset_path = os.path.join(TARGET_ROOT_DIR, "dataset")
        
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path does not exist: {dataset_path}")
        return

    # Check for existing output files to support resuming
    already_processed = set()
    if os.path.exists(output_dir):
        for existing_file in os.listdir(output_dir):
            if existing_file.endswith('.json'):
                # Assumes file_name.json structure
                base_name = existing_file.replace('.json', '')
                already_processed.add(base_name)
    
    print(f"Already processed files: {len(already_processed)}")
    
    # Process files
    files_to_process = [f for f in os.listdir(dataset_path) if f not in already_processed]
    if args.workflow == "self-repair":
        print(f"Starting repair for {args.language} using {args.model_name}...")
        
        start_repairing(
            langs=[args.language], 
            models=[args.model_name]
        )
    else:
        print(f"Generating code for {len(files_to_process)} files in {args.language} using {args.model_name}...")
        for i, file_name in enumerate(files_to_process):
            print(f"[{i+1}/{len(files_to_process)}] Processing: {file_name}")
            
            try:
                file_path = os.path.join(dataset_path, file_name)
                
                # Re-init config for each run to keep thread_ids unique if desired
                config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            
                generator = CodeGenerator(
                    workflow=args.workflow, 
                    language=args.language, 
                    model_name=args.model_name, 
                    timeout_limit=90
                )
                
                results = generator.generate(file_path, config=config)
                
                output_file = os.path.join(output_dir, f"{file_name}.json")
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)
                
                print(f"Successfully saved results to {output_file}")
                    
            except Exception as e:
                print(f"Exception with file {file_name}: {str(e)}")
                with open("error_log.txt", "a") as f:
                    f.write(f"{file_name}: {str(e)}\n")

if __name__ == "__main__":
    main()
    start_repairing()