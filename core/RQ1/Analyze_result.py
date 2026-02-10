import json
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import pickle
import numpy as np

def get_project_root():
    return Path(__file__).resolve().parents[1]
    #[0] core
    #[1] FPEval
project_root = get_project_root()
languages = ['haskell', 'ocaml', 'scala','java']
model = 'gpt3.5'
labels = ["Pass", "Fail", "Compile_error", "Timeout"]
colors = {
    "haskell": "#d7cfdb",
    "ocaml": "#b13030",
    "scala": "#9db19d",
    "java": "#5b64a4"
}
common_file_path = Path("FPEval/core/common_files.pkl")
with open(common_file_path, "rb") as f:
    common_files = set(pickle.load(f))


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

lang_results = {lang: Counter() for lang in languages}
for lang in languages:
    folder = project_root / f"ResultExecution/RQ1/{lang}/{model}"
    i = 1
    o = 0
    for file_name in common_files:
        file = folder / file_name
        if not file.exists():
            print(f"File {file} not found.")
            continue

        try:
            with open(file) as f:
                data = json.load(f)
                results = [d["Result"] for d in data if isinstance(d.get("Result", []), list)]
                classification = classify_test_results(results)
                if classification:
                    lang_results[lang][classification] += 1
        except Exception as e:
            continue

percentages = defaultdict(list)
print(f"This is the model {model}")
for label in labels:
    label = label.lower()
    for lang in languages:
        total = sum(lang_results[lang].values())
        value = (lang_results[lang][label] / total * 100) if total > 0 else 0
        percentages[label].append(value)
        print (f"The {label} for {lang}: {value}")
percentages = defaultdict(list)  # key = difficulty, value = [pass_rate_haskell, pass_rate_ocaml, ...]





