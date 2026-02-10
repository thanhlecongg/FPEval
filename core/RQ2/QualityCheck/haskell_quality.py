import json
import os
from pathlib import Path
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def get_project_root():
    return Path(__file__).resolve().parents[1]
    #[0] core
    #[1] FPEval
project_root = get_project_root()
code_trace_dir = project_root / "result_llms/haskell/gpt-4o"
metadata_dir = project_root / "LeetCodeMeta"
result_dir =  project_root/ "ResultExecution/RQ2/haskell/gpt4o"
output_dir = Path("haskell_outputs")
output_dir.mkdir(exist_ok=True)

results = []
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

def write_haskell_file(code_str, file_path):
    with open(file_path, "w") as f:
        f.write(code_str)

def run_hlint(file_path):
    try:
        result = subprocess.run(
            ["hlint", str(file_path), "--json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10
        )
        if result.returncode != 0 and "parse error" in result.stderr.lower():
            return None, "parse_error"
        issues = json.loads(result.stdout)
        serious = [i for i in issues if i["severity"] != "Ignore"]
        return serious, None  
    except subprocess.TimeoutExpired:
        return None, "timeout"
def run_ghc_warning_check(file_path):
    try:
        result = subprocess.run(
            ["ghc", "-Wall", "-Werror", "-fno-code", str(file_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            return False, result.stderr.strip()
        return True, None
    except subprocess.TimeoutExpired:
        return False, "timeout"

count = 0
for json_path in result_dir.glob("*.json"):

    json_file = (json_path.name).replace("-","_")
    file_path = result_dir / json_path
    try:
        with open(file_path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading test data for {json_path}: {e}")
        continue
    meta_path = json_file.replace("_","-")
    metadata_file = metadata_dir / meta_path
    difficulty = "Unknown"
    if metadata_file.exists():
        with open(metadata_file) as f:
            meta = json.load(f)
            difficulty = meta.get("difficulty", "Unknown")
            func_name = meta["metadata"]["func_name"]

    results_ghc = [d["Result"] for d in data if isinstance(d.get("Result", []), list)]
    classification = classify_test_results(results_ghc)
    if classification not in ['pass']:
        continue
    
    code_file = code_trace_dir/json_file
    if not code_file.exists():
        print(f"File does not exist. {code_file}")
        continue
    try:
        with open(code_file) as f:
            trace_data = json.load(f)
    except Exception as e:
        print(f"Error reading code trace for {json_file}: {e}")
        continue
    
    code_traces = trace_data.get("code_traces", [])
    if not code_traces:
        continue

    code = code_traces[0]  
    code = code+ f'''
main :: IO ()
main = do
    let _ = {func_name}
    return ()'''
    haskell_file = output_dir / "temp.hs"
    write_haskell_file(code, haskell_file)
    hlint_count, hlint_error = run_hlint(haskell_file)
    ghc_passed, ghc_error = run_ghc_warning_check(haskell_file)
    print(ghc_error)
    has_errors = hlint_error is not None or not ghc_passed

    loc = code.count("\n") + 1

    result = {
        "file": json_file,
        "difficulty": difficulty,
        "test_result": classification,  
        "loc": loc,
        "hlint_issues": hlint_count if hlint_count is not None else -1,
        "has_errors": has_errors,
        "ghc_pass" : ghc_passed
    }
    results.append(result)
    print(result)


with open("haskell_quality_results_4o.json", "w") as out:
    json.dump(results, out, indent=2)


threshold = 0  
grouped = defaultdict(lambda: {"Clean": 0, "Issues": 0})

for r in results:
    test_result = r["test_result"]
    if test_result not in ["pass", "fail"]:
        continue

    if isinstance(r["hlint_issues"], list) and len(r["hlint_issues"]) > threshold or not r['ghc_pass']:
        grouped[test_result]["Issues"] += 1
    else:
        grouped[test_result]["Clean"] += 1
test_categories = ["pass", "fail"]
clean_percents = []
issue_percents = []

for category in test_categories:
    total = grouped[category]["Clean"] + grouped[category]["Issues"]
    if total == 0:
        clean_percents.append(0)
        issue_percents.append(0)
    else:
        clean_percents.append(grouped[category]["Clean"] / total * 100)
        issue_percents.append(grouped[category]["Issues"] / total * 100)


print(f"\nTest results:")
for i, category in enumerate(test_categories):
    print(f"  Clean: {clean_percents[i]:.1f}%")
    print(f"  Issues: {issue_percents[i]:.1f}%")

