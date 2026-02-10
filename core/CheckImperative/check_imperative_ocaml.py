import json
from pathlib import Path
import re
from collections import defaultdict

with open("FPEval/core/common_files.pkl", "rb") as f:
    results_path = set(pickle.load(f))

# folder JSON-code
folder = Path("LLMFP/CodeGenerated/ocaml/gpt-4o")

fp_violations = {
    "mutable_variable": re.compile(r"\bmutable\b"),
    "imperative_ref": re.compile(r"\bref\b"),
    "imperative_assignment": re.compile(r":=\s*"),  # assignment operator in OCaml
    "imperative_while": re.compile(r"\bwhile\b"),
    "imperative_for": re.compile(r"\bfor\b"),
    "print_endline": re.compile(r"\bprint_endline\b"),
    "print_string": re.compile(r"\bprint_string\b"),
    "null_emulation": re.compile(r"\bNone\b|\bnull\b"),  # OCaml không có null nhưng vẫn có None
}
fp_violations = {
    # 1. Mutable Variables
    "mutable_keyword": re.compile(r"\bmutable\b"),
    "ref_usage": re.compile(r"\bref\b"),
    "assignment_operator": re.compile(r":="), # OCaml assignment
    
    # 2. Imperative Loops
    "imperative_while": re.compile(r"\bwhile\b"),
    "imperative_for": re.compile(r"\bfor\b"),
    
    # 3. In-place Mutation of Data Structures  
    "array_mutation": re.compile(r"\w+\.\(\d+\)\s*<-"), # arr.(0) <- value
    "string_mutation": re.compile(r"\w+\.\[\d+\]\s*<-"), # str.[0] <- 'c'
    "mutable_record": re.compile(r"\{.*mutable.*\}"), # mutable record fields
    
    # 4. Unsafe operations instead of Option types
    "list_hd_tl": re.compile(r"\b(List\.hd|List\.tl)\b"), # can raise Failure
    "array_get": re.compile(r"\w+\.\(\d+\)"), # direct array access without bounds check
    "failwith_usage": re.compile(r"\bfailwith\b"), # should use Result/Option instead
}

def check_violations(code: str) -> dict:
    violations_found = {}
    for name, pattern in fp_violations.items():
        matches = pattern.findall(code)
        if matches:
            violations_found[name] = len(matches)
    return violations_found

import pickle

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
