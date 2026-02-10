import json
from pathlib import Path
import re
import pickle
from collections import defaultdict

with open("FPEval/core/common_files.pkl", "rb") as f:
    results_path = set(pickle.load(f))

# folder JSON-code
folder = Path("LLMFP/CodeGenerated/scala/gpt-5")


fp_violations = {
    "mutable_var": re.compile(r"\bvar\b"),
    "imperative_while": re.compile(r"\bwhile\b"),
    "imperative_for": re.compile(r"\bfor\s*\("),
    "null_usage": re.compile(r"\bnull\b"),
    "println": re.compile(r"\bprintln\s*\("),
    "mutable_keyword": re.compile(r"\bmutable\b"),


}


def check_violations(code: str) -> dict:
   
    violations_found = {}
    for name, pattern in fp_violations.items():
        matches = pattern.findall(code)
        if matches:
            violations_found[name] = len(matches)
    return violations_found

def scan_json_files(folder: Path):
    results = []

    for json_file in results_path:
        json_file = json_file.replace("-","_")
        json_file = folder / json_file
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                
                    data = json.load(f)
        except:
                
                print(f"JSON error: {json_file}")
                continue

       
        if not isinstance(data, dict) or "code_traces" not in data:
            continue

        code_snippets = data["code_traces"]
        if not isinstance(code_snippets, list):
            continue

        for idx, code in enumerate(code_snippets):
            if not isinstance(code, str):
                continue
            violations = check_violations(code)
            if violations:
                results.append({
                    "file": json_file.name,
                    "index": idx,
                    "violations": violations,
                    "code_snippet": code
                })

    return results

all_violations = scan_json_files(folder)

print(f"\nThe number of violation files: {len(all_violations)}")
for entry in all_violations[5:8]:  
    print(f"\n File: {entry['file']} [Code index: {entry['index']}]")
    print(f" Violations: {entry['violations']}")
    print(f" Code snippet:\n{entry['code_snippet']}")


violation_file_map = defaultdict(set)

for entry in all_violations:
    file_name = entry["file"]
    for violation_type in entry["violations"].keys():
        violation_file_map[violation_type].add(file_name)

violation_file_counts = {k: len(v) for k, v in violation_file_map.items()}

sorted_violation_counts = sorted(violation_file_counts.items(), key=lambda x: x[1], reverse=True)

print("Violation Type: Number of Files")
for violation, file_count in sorted_violation_counts:
    print(f"- {violation}: {file_count} files")

files_with_violations = set(entry["file"] for entry in all_violations)

num_files_with_violations = len(files_with_violations)
print(f"\n The number of files with at least one code violation: {num_files_with_violations}")
