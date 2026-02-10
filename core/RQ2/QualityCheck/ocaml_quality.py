import json
from pathlib import Path
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
import re
import difflib
import pickle


def get_project_root():
    return Path(__file__).resolve().parents[1]
    #[0] core
    #[1] FPEval
project_root = get_project_root()
code_trace_dir = project_root / "result_llms/ocaml/gpt-4o"
metadata_dir = project_root / "LeetCodeMeta"
result_dir =  project_root/ "ResultExecution/RQ2/CodeGenerated/ocaml/gpt4o"
output_dir = Path("ocaml_outputs")
output_dir.mkdir(exist_ok=True)
envs_root = Path("FPEval/envs/ocaml")
main_file = Path('FPEval/envs/ocaml/bin/main.ml')



results = []
warning_list = []
def show_side_by_side_diff(original_code, formatted_code):
    norm_orig = normalize_code(original_code)
    norm_formatted = normalize_code(formatted_code)
    
    orig_lines = norm_orig.split('\n')
    formatted_lines = norm_formatted.split('\n')
    
    max_lines = max(len(orig_lines), len(formatted_lines))
    
    print(f"{'ORIGINAL':<50} | {'FORMATTED'}")
    print("-" * 50 + " | " + "-" * 50)
    
    for i in range(max_lines):
        orig_line = orig_lines[i] if i < len(orig_lines) else ""
        formatted_line = formatted_lines[i] if i < len(formatted_lines) else ""
        marker = ">>>" if orig_line != formatted_line else "   "
        
        print(f"{marker} {orig_line[:45]:<45} | {formatted_line}")
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


def write_main_ml(code):
    with open(main_file, "w") as f:
        f.write(code)

def normalize_code(s):
    return "".join(s.split())
# === Tools for Statics analysis===
def run_ocamlc_check():
    try:
        result = subprocess.run(
            ["ocamlc", "-c", "-warn-error", "+a", str(main_file)],
            cwd=envs_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10
        )
        print(result)

        if result.returncode != 0:
            errors = result.stderr.strip().splitlines()
            return False, errors
        return True, []
    except subprocess.TimeoutExpired:
        return False, ["Timeout"]


def run_dune_build_check():
    try:
        result = subprocess.run(
            ["dune", "build"],
            cwd=envs_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=20
        )
        if result.returncode != 0:
            errors = result.stderr.strip().splitlines()
            return False, errors
        return True, []
    except subprocess.TimeoutExpired:
        return False, ["Timeout from dune build @check"]
def run_merlin_errors():
    try:
        with open(main_file, "r") as f:
            file_content = f.read()

        result = subprocess.run(
            ["ocamlmerlin", "single", "errors", "-filename", str(main_file)],
            input=file_content,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            return False, ["Merlin error: " + result.stderr.strip()]

        data = json.loads(result.stdout)
        if data.get("class") != "return":
            return False, ["Merlin: invalid output"]

        errors = data.get("value", [])
        if not errors:
            return True, [] 
        messages = []
        
        for e in errors:
            if isinstance(e, dict):
                msg_type = e.get("type", "unknown")
                msg = e.get("message", "")
                start_line = e.get("start", {}).get("line", "?")
                start_col = e.get("start", {}).get("col", "?")
                messages.append(f"[{msg_type}] Line {start_line} Col {start_col}: {msg}")
            else:
                messages.append(str(e))
        return False, messages  

    except Exception as e:
        return False, [str(e)]
def run_ocamlformat_on_main():
    try:
        result = subprocess.run(
            ["ocamlformat", "--enable-outside-detected-project", "bin/main.ml"],
            cwd=envs_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            return "bad_format", result.stderr.strip(), None
        formatted_code = result.stdout
        return "ok", None, formatted_code
    except subprocess.TimeoutExpired:
        return "timeout", None, None

def extract_ocaml_warning_messages(output_lines):
    clean_lines = []
    for line in output_lines:
        line = line.strip()
        if re.fullmatch(r"\^+", line):
            continue
        if "Warning" in line or "Error" in line :
            clean_lines.append(line)
            print(line)
    return clean_lines

def run_ocaml_warning_check():
    try:
        result = subprocess.run(
            ["ocaml", "-w", "+a", str(main_file)],
            cwd=envs_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5
        )
        lines = result.stderr.strip().splitlines()
        warnings = extract_ocaml_warning_messages(lines)
        return warnings
    except subprocess.TimeoutExpired:
        return ["Timeout"]

count = 0
with open("FPEval/core/common_files.pkl", "rb") as f:
    results_path = set(pickle.load(f))
for json_path in results_path:
    
    json_file = json_path.replace("-", "_")
    file_path = result_dir / json_path

     # Metadata
    meta_path = json_file.replace("_", "-")
    metadata_file = metadata_dir / meta_path
    difficulty = "Unknown"
    if metadata_file.exists():
        with open(metadata_file) as f:
            meta = json.load(f)
            difficulty = meta.get("difficulty", "Unknown")
            func_name = meta["metadata"]["func_name"]
    try:
        with open(file_path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading test data for {json_path}: {e}")
        continue

    results_test = [d["Result"] for d in data if isinstance(d.get("Result", []), list)]
    classification = classify_test_results(results_test)
    if classification not in ['pass']:
        continue
    
    code_file = code_trace_dir / json_file
    if not code_file.exists():
        print(f"⚠️ Missing: {code_file}")
        continue
    
    with open(code_file) as f:
        trace_data = json.load(f)
    
    code_traces = trace_data.get("code_traces", [])
    
    if not code_traces:
        continue
    
    code = code_traces[0]
    code = code + f'\n let _ = {func_name}'
    write_main_ml(code)
    
    
    loc = code.count("\n") + 1
     # === Check warning ===
    warnings = run_ocaml_warning_check()
    has_warning = len(warnings) > 0 and "Timeout" not in warnings
    warning_list.extend(warnings)
    

    # === Check warning ===
    print("ok")
    compile_ok, compile_errors = run_ocamlc_check()
    # merlin_ok, merlin_errors = run_merlin_errors()
    dune_ok, dune_errors = run_dune_build_check()
    has_errors = False
    format_status, error_msg, formatted_code = run_ocamlformat_on_main()
    if format_status != "ok":
        has_errors = True
    else:
        normalized_original = normalize_code(code)
        normalized_formatted = normalize_code(formatted_code)
        
        if normalized_original != normalized_formatted:
            has_errors = True
            diff = difflib.unified_diff(
                normalized_original.splitlines(keepends=True),
                normalized_formatted.splitlines(keepends=True),
                fromfile='original',
                tofile='formatted',
                lineterm=''
            )
            
            print("Differences found:")
            for line in diff:
                print(line, end='\n')
        
            
        
    
    result = {
        "file": json_file,
        "difficulty": difficulty,
        "loc": loc,
        "compile_ok": compile_ok,
        "compile_errors": compile_errors,
        "dune_ok": dune_ok,
        "dune_errors": dune_errors,
        "format_status": format_status,
        "has_errors": has_errors,
        "has_warning": has_warning,
        "warning_count": len(warnings)
    }

    count +=1
    print(result)
    results.append(result)
print(f"Total files processed: {count}")

with open("ocaml_quality_results.json", "w") as f:
    json.dump(results, f, indent=2)

grouped = {
    "Clean": 0,
    "Issues": 0
}

for r in results:

    if not r["compile_ok"] or not r["dune_ok"] :
        grouped["Issues"] += 1
        
    else:
        grouped["Clean"] += 1

def flatten_error_list(error_lists):

    flattened = []
    for sublist in error_lists:
        if isinstance(sublist, list):
            flattened.extend(sublist)
        else:
            flattened.append(sublist)
    return flattened


total = grouped["Clean"] + grouped["Issues"]

if total == 0:
    clean_percent = 0
    issue_percent = 0
else:
    clean_percent = grouped["Clean"] / total * 100
    issue_percent = grouped["Issues"] / total * 100


print(f"  Clean: {clean_percent:.1f}%")
print(f"  Issues: {issue_percent:.1f}%")

