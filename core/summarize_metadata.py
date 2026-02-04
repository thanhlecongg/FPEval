import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from config import logger

def summarize_metadata(output_stats_folder, langs, model_name):
    """
    Summarize metadata.json files from all languages for a given model.
    
    Args:
        output_stats_folder: Base folder containing language/model subdirectories
        langs: List of languages to process
        model_name: Model name to summarize
    """
    output_stats_folder = Path(output_stats_folder)
    
    # Initialize summary structure
    summary = {
        "model": model_name,
        "languages": langs,
        "total_across_languages": {
            "total_files": 0,
            "processed_files": 0,
            "missing_files": 0,
            "missing_executer_files": 0,
            "classification_counts": {
                "pass": 0,
                "fail": 0,
                "timeout": 0,
                "compile_error": 0
            }
        },
        "by_language": {},
        "all_missing_files": [],
        "all_missing_executer_files": [],
        "files_by_classification": {
            "pass": [],
            "fail": [],
            "timeout": [],
            "compile_error": []
        }
    }
    
    # Process each language
    for lang in langs:
        metadata_path = output_stats_folder / lang / model_name / "metadata.json"
        
        if not metadata_path.exists():
            logger.warning(f"Metadata file not found: {metadata_path}")
            summary["by_language"][lang] = {
                "status": "missing",
                "error": "metadata.json not found"
            }
            continue
        
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            # Add language-specific data
            processed_files = metadata.get("processed_files", 0)
            # Use percentages from metadata if present; otherwise compute per-language percentages
            if "percentages" in metadata and isinstance(
                metadata.get("percentages"), dict
            ):
                lang_percentages = metadata["percentages"]
            elif processed_files > 0:
                counts = metadata.get("classification_counts", {})
                lang_percentages = {
                    "pass": round(100 * counts.get("pass", 0) / processed_files, 2),
                    "fail": round(100 * counts.get("fail", 0) / processed_files, 2),
                    "timeout": round(
                        100 * counts.get("timeout", 0) / processed_files, 2
                    ),
                    "compile_error": round(
                        100 * counts.get("compile_error", 0) / processed_files, 2
                    ),
                }
            else:
                lang_percentages = {
                    "pass": 0.0,
                    "fail": 0.0,
                    "timeout": 0.0,
                    "compile_error": 0.0,
                }

            summary["by_language"][lang] = {
                "status": "processed",
                "total_files": metadata.get("total_files", 0),
                "processed_files": processed_files,
                "missing_files": len(metadata.get("missing_files", [])),
                "missing_executer_files": len(metadata.get("missing_executer_files", [])),
                "classification_counts": metadata.get("classification_counts", {}),
                "missing_files_list": metadata.get("missing_files", []),
                "missing_executer_files_list": metadata.get("missing_executer_files", []),
                "percentages": lang_percentages,
            }
            
            # Aggregate totals
            summary["total_across_languages"]["total_files"] += metadata.get("total_files", 0)
            summary["total_across_languages"]["processed_files"] += metadata.get("processed_files", 0)
            summary["total_across_languages"]["missing_files"] += len(metadata.get("missing_files", []))
            summary["total_across_languages"]["missing_executer_files"] += len(metadata.get("missing_executer_files", []))
            
            # Aggregate classification counts
            for classification in ["pass", "fail", "timeout", "compile_error"]:
                count = metadata.get("classification_counts", {}).get(classification, 0)
                summary["total_across_languages"]["classification_counts"][classification] += count
                
                # Collect files by classification (with language prefix)
                files = metadata.get("files_by_classification", {}).get(classification, [])
                for file_name in files:
                    summary["files_by_classification"][classification].append(f"{lang}/{file_name}")
            
            # Collect all missing files (with language prefix)
            for file_name in metadata.get("missing_files", []):
                summary["all_missing_files"].append(f"{lang}/{file_name}")
            
            for file_name in metadata.get("missing_executer_files", []):
                summary["all_missing_executer_files"].append(f"{lang}/{file_name}")
            
            logger.info(f"Processed metadata for {lang}/{model_name}")
            
        except Exception as e:
            logger.error(f"Error reading metadata from {metadata_path}: {e}")
            summary["by_language"][lang] = {
                "status": "error",
                "error": str(e)
            }
    
    # Calculate percentages
    total_processed = summary["total_across_languages"]["processed_files"]
    if total_processed > 0:
        summary["percentages"] = {
            "pass": round(100 * summary["total_across_languages"]["classification_counts"]["pass"] / total_processed, 2),
            "fail": round(100 * summary["total_across_languages"]["classification_counts"]["fail"] / total_processed, 2),
            "timeout": round(100 * summary["total_across_languages"]["classification_counts"]["timeout"] / total_processed, 2),
            "compile_error": round(100 * summary["total_across_languages"]["classification_counts"]["compile_error"] / total_processed, 2)
        }
    else:
        summary["percentages"] = {
            "pass": 0.0,
            "fail": 0.0,
            "timeout": 0.0,
            "compile_error": 0.0
        }
    
    # Save summary
    summary_file = output_stats_folder / f"summary_{model_name}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Summary saved to: {summary_file}")
    
    # Print summary to console
    print("\n" + "="*60)
    print(f"SUMMARY FOR MODEL: {model_name}")
    print("="*60)
    print(f"Languages processed: {len([l for l in summary['by_language'].values() if l.get('status') == 'processed'])}/{len(langs)}")
    print(f"\nTotal across all languages:")
    print(f"  Total files:        {summary['total_across_languages']['total_files']}")
    print(f"  Processed files:    {summary['total_across_languages']['processed_files']}")
    print(f"  Missing files:      {summary['total_across_languages']['missing_files']}")
    print(f"  Missing executer:   {summary['total_across_languages']['missing_executer_files']}")
    print(f"\nClassification counts:")
    print(f"  Pass:              {summary['total_across_languages']['classification_counts']['pass']} ({summary['percentages']['pass']}%)")
    print(f"  Fail:              {summary['total_across_languages']['classification_counts']['fail']} ({summary['percentages']['fail']}%)")
    print(f"  Timeout:           {summary['total_across_languages']['classification_counts']['timeout']} ({summary['percentages']['timeout']}%)")
    print(f"  Compile Error:     {summary['total_across_languages']['classification_counts']['compile_error']} ({summary['percentages']['compile_error']}%)")
    
    print(f"\nBy language:")
    for lang in langs:
        lang_data = summary["by_language"].get(lang, {})
        if lang_data.get("status") == "processed":
            lang_pct = lang_data.get("percentages", {})
            print(f"  {lang}:")
            print(f"    Total: {lang_data['total_files']}, Processed: {lang_data['processed_files']}")
            print(
                f"    Pass: {lang_data['classification_counts'].get('pass', 0)} "
                f"({lang_pct.get('pass', 0.0)}%), "
                f"Fail: {lang_data['classification_counts'].get('fail', 0)} "
                f"({lang_pct.get('fail', 0.0)}%), "
                f"Timeout: {lang_data['classification_counts'].get('timeout', 0)} "
                f"({lang_pct.get('timeout', 0.0)}%), "
                f"Compile Error: {lang_data['classification_counts'].get('compile_error', 0)} "
                f"({lang_pct.get('compile_error', 0.0)}%)"
            )
        else:
            print(f"  {lang}: {lang_data.get('status', 'unknown')} - {lang_data.get('error', 'N/A')}")
    
    print("="*60 + "\n")
    
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize metadata.json files across all languages for a model")
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
    
    for model_name in args.models:
        summarize_metadata(
            output_stats_folder=args.output_stats_folder,
            langs=args.langs,
            model_name=model_name
        )
