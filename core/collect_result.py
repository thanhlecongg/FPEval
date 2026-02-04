import os
import json
import re
import pickle as pkl
import argparse
from pathlib import Path
from config import logger

def classify_test_results(results):
    has_compile_error = any(r[:3] == [-1, -1, -1] for r in results)
    has_timeout = any(r[:3] == [-2, -1, -1] for r in results)
    pass_count = sum(1 for r in results if r[0] == 0 and r[2] == 0)
    has_failed = any(r[0] == -3 for r in results)

    if pass_count > 8:
        return "pass"
    elif has_timeout:
        return "timeout"
    elif has_compile_error and pass_count <1:
        return "compile_error"
    elif has_failed or pass_count > 0:
        return "fail"

def main(problem_path, executer_results_folder, input_result_folder, output_stats_folder, langs, models):
    executer_results_folder = Path(executer_results_folder)
    input_result_folder = Path(input_result_folder)
    output_stats_folder = Path(output_stats_folder)
    # logger.info(f"Problem path: {problem_path}")
    # logger.info(f"Input result folder: {input_result_folder}")
    # logger.info(f"Output stats folder: {output_stats_folder}")
    logger.info(f"Languages: {langs}")
    logger.info(f"Models: {models}")
    for lang in langs:
        for model_name in models:
            output = Path(output_stats_folder) / lang / model_name
            output.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Processing {lang} {model_name}")
            
            # Initialize metadata tracking
            metadata = {
                "language": lang,
                "model": model_name,
                "total_files": 0,
                "processed_files": 0,
                "missing_files": [],
                "missing_executer_files": [],
                "classification_counts": {
                    "pass": 0,
                    "fail": 0,
                    "timeout": 0,
                    "compile_error": 0
                },
                "files_by_classification": {
                    "pass": [],
                    "fail": [],
                    "timeout": [],
                    "compile_error": []
                }
            }
            
            # logger.debug(f"Executer results folder: {executer_results_folder}")
            for file_name in sorted(os.listdir(problem_path)):
                # logger.debug(f"Processing {file_name}")
                # clean_file_name = file_name.replace("-", "_")
                clean_file_name = file_name
                metadata["total_files"] += 1
                
                if os.path.exists(f"{output}/{clean_file_name}.json"):
                    # logger.debug(f"Continue exist {file_name}")
                    # Still count it in metadata if we want to track already processed files
                    continue
                    
                input_path = input_result_folder / lang / model_name / (clean_file_name + ".json")
                file_name_for_executer = file_name.replace("_", "-")
                executer_result_path = executer_results_folder / lang / model_name / (file_name_for_executer + ".json")  
                
                # Check for missing files
                if not executer_result_path.exists():
                    metadata["missing_executer_files"].append(clean_file_name)
                    logger.warning(f"Executer result file not found: {executer_result_path}")
                    continue
                    
                if not input_path.exists():
                    metadata["missing_files"].append(clean_file_name)
                    logger.warning(f"File not found: {input_path}")
                    continue
                
                # logger.debug(f"Input path: {input_path}")
                # logger.debug(f"Executer result path: {executer_result_path}")
                # for item in all_warnings:
                #     if item["file"] == clean_file_name:
                #         file_warnings = item["hlint_issues"]
                #         warnings_map = [issue["hint"] for issue in file_warnings if "hint" in issue]
                #         break
                try:
                    with open(executer_result_path) as f:
                        result = json.load(f)
                except Exception as e:
                    logger.warning(f"Error reading test data for {file_name}: {e}")
                    metadata["missing_executer_files"].append(clean_file_name)
                    continue        
                    
                results_test = [d["Result"] for d in result if isinstance(d.get("Result", []), list)]
                classification = classify_test_results(results_test)
                logger.debug(f"Classification: {classification}")
                
                # Update metadata counts
                if classification in metadata["classification_counts"]:
                    metadata["classification_counts"][classification] += 1
                    metadata["files_by_classification"][classification].append(clean_file_name)
                else:
                    logger.warning(f"Unknown classification: {classification}")
                
                if classification in ['compile_error']:
                    error = [d["Result"] for d in result if isinstance(d.get("Result", []), list)][:1]
                else:
                    # error = classification
                    error = None

                with open(input_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Collect and save classification result
                result_data = {
                    "classification": classification,
                    "file_name": clean_file_name,
                    "language": lang,
                    "model": model_name,
                    "test_results": results_test,
                    "error": error,
                    "input_data": data
                }
                
                output_file = output / (clean_file_name + ".json")
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(result_data, f, indent=2, ensure_ascii=False)
                
                metadata["processed_files"] += 1
                logger.debug(f"Saved classification result to {output_file}")
            
            # Compute percentage statistics for this language/model
            total_processed = metadata["processed_files"]
            if total_processed > 0:
                metadata["percentages"] = {
                    "pass": round(
                        100
                        * metadata["classification_counts"]["pass"]
                        / total_processed,
                        2,
                    ),
                    "fail": round(
                        100
                        * metadata["classification_counts"]["fail"]
                        / total_processed,
                        2,
                    ),
                    "timeout": round(
                        100
                        * metadata["classification_counts"]["timeout"]
                        / total_processed,
                        2,
                    ),
                    "compile_error": round(
                        100
                        * metadata["classification_counts"]["compile_error"]
                        / total_processed,
                        2,
                    ),
                }
            else:
                metadata["percentages"] = {
                    "pass": 0.0,
                    "fail": 0.0,
                    "timeout": 0.0,
                    "compile_error": 0.0,
                }

            # Save metadata summary
            metadata_file = output / "metadata.json"
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.debug(f"Metadata file: {metadata_file}")
            logger.info(f"Metadata for {lang}/{model_name}: "
                       f"Total: {metadata['total_files']}, "
                       f"Processed: {metadata['processed_files']}, "
                       f"Missing: {len(metadata['missing_files'])}, "
                       f"Missing Executer: {len(metadata['missing_executer_files'])}, "
                       f"Pass: {metadata['classification_counts']['pass']}, "
                       f"Fail: {metadata['classification_counts']['fail']}, "
                       f"Timeout: {metadata['classification_counts']['timeout']}, "
                       f"Compile Error: {metadata['classification_counts']['compile_error']}")

                


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect and process execution results")
    parser.add_argument(
        "--problem-path",
        type=str,
        required=True,
        help="Path to the problem directory"
    )
    parser.add_argument(
        "--executer-results-folder",
        type=str,
        required=True,
        help="Path to the executor results folder"
    )
    parser.add_argument(
        "--input-result-folder",
        type=str,
        required=True,
        help="Path to the input result folder (generated code results)"
    )
    parser.add_argument(
        "--output-stats-folder",
        type=str,
        required=True,
        help="Path to the output stats folder"
    )
    parser.add_argument(
        "--langs",
        type=str,
        nargs='+',
        required=True,
        help="List of languages to process (e.g., haskell ocaml scala java)"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs='+',
        required=True,
        help="List of model names to process (e.g., gpt-3.5-turbo gpt-4o)"
    )
    
    args = parser.parse_args()
    # logger.info(f"Problem path: {args.problem_path}")
    # logger.info(f"Executer results folder: {args.executer_results_folder}")
    # logger.info(f"Input result folder: {args.input_result_folder}")
    # logger.info(f"Output stats folder: {args.output_stats_folder}")
    # logger.info(f"Languages: {args.langs}")
    # logger.info(f"Models: {args.models}")
    main(
        problem_path=args.problem_path,
        executer_results_folder=args.executer_results_folder,
        input_result_folder=args.input_result_folder,
        output_stats_folder=args.output_stats_folder,
        langs=args.langs,
        models=args.models
    )
