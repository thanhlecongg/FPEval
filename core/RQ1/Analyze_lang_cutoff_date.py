from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Configuration
languages = [ 'java','scala', 'haskell', 'ocaml']
model = 'gpt4o'
base_folder = Path(".../FPEval")
output_image = Path(f"images/RQ1/cutoff_date/multiline_{model}.png")
metadata_dir = Path(".../LeetCodeMeta")

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

def get_period_from_date(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    if dt < datetime(2022, 10, 1):
        return "~Sep2022"
    elif dt < datetime(2023, 4, 1):
        return "Oct2022-Mar2023"
    elif dt < datetime(2023, 10, 1):
        return "Apr2023-Sep2023"
    elif dt < datetime(2024, 4, 1):
        return "Oct2023-Mar2024"
    elif dt < datetime(2024, 10, 1):
        return "Apr2024-Sep2024"
    else:
        return "Oct2024~"

periods = ["~Sep2022","Oct2022-Mar2023","Apr2023-Sep2023", "Oct2023-Mar2024", "Apr2024-Sep2024", "Oct2024~"]
# Load common files
common_file_path = Path("FPEval/core/common_files.pkl")
with open(common_file_path, "rb") as f:
    common_files = set(pickle.load(f))

# Data structure to store results for all languages
language_results = {}

# Process each language
for lang in languages:
    print(f"Processing {lang}...")
    folder = base_folder / f"{lang}_full" / model
    
    pass_counts_by_period = Counter()
    total_counts_by_period = Counter()
    
    for file_name in common_files:
        meta_file = metadata_dir / file_name
        if not meta_file.exists():
            continue
        
        try:
            # Read metadata
            with open(meta_file) as f:
                metadata = json.load(f)
            
            date_str = metadata.get("date")
            if not date_str:
                continue
            
            period = get_period_from_date(date_str)
            if period is None:
                continue
                
        except Exception as e:
            print(f"Error reading metadata for {file_name}: {e}")
            continue
        
        try:
            # Read test results
            test_file = folder / file_name
            if not test_file.exists():
                continue
                
            with open(test_file) as f:
                data = json.load(f)
            
            results = [d["Result"] for d in data if isinstance(d.get("Result", []), list)]
            classification = classify_test_results(results)
            
        except Exception as e:
            print(f"Error reading test data for {file_name}: {e}")
            continue
        
        # Count totals and passes
        total_counts_by_period[period] += 1
        if classification == "pass":
            pass_counts_by_period[period] += 1
    
    pass_rates = []
    
    for period in periods:
        if total_counts_by_period[period] > 0:
            pass_rate = pass_counts_by_period[period] / total_counts_by_period[period]
        else:
            pass_rate = 0
        pass_rates.append(pass_rate)
        print(f"{lang} - {period}: {pass_counts_by_period[period]}/{total_counts_by_period[period]} = {pass_rate:.3f}")
    
    language_results[lang] = pass_rates

# Create the line plot
plt.rcParams["font.family"] = "Times New Roman"
plt.figure(figsize=(10, 5))

# Define colors for each language
colors = {
    'java': '#FF7F0E',
    'scala': '#D62728',
    'haskell': '#1F77B4', 

    'ocaml': '#9467BD'
}

# Define markers for each language
markers = {
    'scala': 'o',
    'haskell': 's',
    'java': '^',
    'ocaml': 'D'
}
linestyles = {
    'scala': '-',      # Solid line
    'haskell': '--',    # Solid line
    'java': '-',      # Dashed line (Nét đứt)
    'ocaml': '--'       # Solid line
}


x_positions = range(len(periods))  # = range(5)



# Plot lines for each language
for lang in languages:
    plt.plot(x_positions, language_results[lang], 
             marker=markers[lang], 
             color=colors[lang],
             linestyle=linestyles[lang], 
             linewidth=2.5, 
             markersize=8, 
             label=lang.capitalize(),
             markeredgecolor='black',
             markeredgewidth=1)

# Customize the plot

plt.ylabel("Pass@1", fontweight='bold', fontsize=15)


plt.xticks(x_positions, periods, fontsize=13, fontweight='bold')
plt.yticks(np.arange(0, 1.1, 0.1), fontsize =13, fontweight='bold')
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend(frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig(output_image, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Saved multi-language line plot to {output_image}")

# Print summary statistics
print("\n📊 Summary Statistics:")
for lang in languages:
    print(f"\n{lang.capitalize()}:")
    for i, period in enumerate(periods):
        print(f"  {period}: {language_results[lang][i]:.3f}")