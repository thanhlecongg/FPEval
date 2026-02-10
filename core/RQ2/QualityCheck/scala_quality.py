import json
import subprocess
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter

def get_project_root():
    return Path(__file__).resolve().parents[1]
    #[0] core
    #[1] FPEval
project_root = get_project_root()
code_trace_dir = project_root / "result_llms/scala/gpt-4o"
metadata_dir = project_root / "LeetCodeMeta"
result_dir =  project_root/ "ResultExecution/RQ2/scala/gpt4o"
output_dir = Path("scala_outputs")
output_dir.mkdir(exist_ok=True)
envs_root = Path("FPEval/core/envs/scala")
main_file = envs_root / "src/main/scala/Main.scala"
scala_src_dir = main_file.parent


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

def write_scala_file(code_str, file_path):
    with open(file_path, "w") as f:
        f.write(code_str)

def run_scalastyle_cli():

    proc = subprocess.run(
        ["sbt", "scalastyle"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=project_root
    )
    return proc.stdout


def parse_scalastyle_output(output):
    errors = []
    warnings = []
    ignored_warning_keywords = ["whitespace", 'header does not match expected text','warnings exist', 'println']

    for line in output.splitlines():
        if "[error]" in line:
            errors.append(line)
        elif "[warn]" in line:
           
            if any(keyword in line.lower() for keyword in ignored_warning_keywords):
                continue
            print(f"This is output:{line}")
            warnings.append(line)
    return errors, warnings

json_files = sorted(code_trace_dir.glob("*.json"))
for index, json_path in enumerate(json_files):
    
    json_file = json_path.name.replace("-", "_")
    code_file = code_trace_dir / json_file
    file_path = result_dir / json_path.name.replace("_", "-")
    if not code_file.exists():
        print(f"⚠️ Missing: {code_file}")
        continue

    with open(code_file) as f:
        trace_data = json.load(f)
    code_traces = trace_data.get("code_traces", [])
    if not code_traces:
        continue

    code = code_traces[0]
    write_scala_file(code, main_file)

    output = run_scalastyle_cli()
   
    errors, warnings = parse_scalastyle_output(output)
    loc = code.count("\n") + 1

    meta_path = json_path.name
    difficulty = "Unknown"
    metadata_file = metadata_dir / meta_path
    if metadata_file.exists():
        with open(metadata_file) as f:
            meta = json.load(f)
            difficulty = meta.get("difficulty", "Unknown")

    try:
        with open(file_path) as f:
            data = json.load(f)
            results_classi = [d["Result"] for d in data if isinstance(d.get("Result", []), list)]
            classification = classify_test_results(results_classi)

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        continue

    # if classification not in ['pass', 'fail']:
    #     continue
    print(classification)
    results.append({
        "file": json_path.name,
        "difficulty": difficulty,
        "test_result": classification,  
        "loc": loc,
        "style_status": "ok" if len(errors) == 0 and len(warnings) == 0 else "style_errors",
        "num_errors": len(errors),
        "num_warnings": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "has_errors": len(errors) + len(warnings) > 0
    })
    print(f"✅ Done {json_file}: {len(errors)} errors, {len(warnings)} warnings")
    print(f" length of result: {len(results)}")

    # try:
    #     target_file.unlink()
    # except Exception as e:
    #     print(f"⚠️ Error deleting {scala_filename}: {e}")

with open("scala_quality_results_5.json", "w") as f:
    json.dump(results, f, indent=2)
all_result = results

def parse_scalastyle_output(output):
    errors = []
    warnings = []
    ignored_warning_keywords = ["whitespace", 'header does not match expected text','warnings exist', 'println','newline']

    for line in output.splitlines():
        if "[error]" in line:
            errors.append(line)
        elif "[warn]" in line:
           
            if any(keyword in line.lower() for keyword in ignored_warning_keywords):
                continue
            print(f"This is output:{line}")
            warnings.append(line)
    return errors, warnings
results = []
for ind, result in enumerate(all_result):
    
    warning = result['warnings']
    warnings = []
    for warn in warning:
        if 'complexity of 11' in warn:
            continue
        if 'newline' in warn:
            continue
        if 'whitespace' in warn:
            continue
        warnings.append(warn)
    results.append({
         "test_result": result['test_result'],
         "warnings": warnings  
    })

grouped = {
    "Clean": 0,
    "Issues": 0
}

for r in results:
    print(r)
    test_result = r["test_result"]
    if test_result not in ["pass"]:
        continue
   
    if len(r["warnings"])>0:

        grouped["Issues"] += 1
    else:
        grouped["Clean"] += 1

total = grouped["Clean"] + grouped["Issues"]

if total == 0:
    clean_percent = 0
    issue_percent = 0
else:
    clean_percent = grouped["Clean"] / total * 100
    issue_percent = grouped["Issues"] / total * 100


print(f"  Clean: {clean_percent:.1f}%")
print(f"  Issues: {issue_percent:.1f}%")

error_counter = Counter()
warning_counter = Counter()

for r in results:
    for warn in list(r["warnings"]):
        _,_,key = warn.rpartition(":")
        warning_counter[key] += 1

print("\n🔟 Top 10 Scala Style Errors:")
for err, count in error_counter.most_common(10):
    print(f"{err}: {count} times")

print("\n🔟 Top 10 Scala Style Warnings:")
for warn, count in warning_counter.most_common(10):
    print(f"{warn}: {count} times")

