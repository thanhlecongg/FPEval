import re
import json
import sys
import os
import shutil
import uuid
import subprocess
import tempfile
import sys
import pickle
import argparse
from zipp import Path
import gc

sys.set_int_max_str_digits(5000) # Increase the limit for integer string conversion
def get_project_root():
    # return Path(__file__).resolve().parents[1]
    return "/workspace"
    #[0] core
    #[1] FPEval
project_root = get_project_root()

# Parse command line arguments
parser = argparse.ArgumentParser(description='Haskell executor for LLM code evaluation')
parser.add_argument('--llm-output-dir', type=str, default=f"{project_root}/results_llm_reasoning/haskell/gpt4",
                    help='LLM output directory: reads LLM-generated code from here and saves execution results here')
parser.add_argument('--output-dir', type=str, default=f"{project_root}/results_llm_reasoning/haskell/gpt4",
                    help='Output directory: saves execution results here')
parser.add_argument('--private-testcase-path', type=str, default=f"{project_root}/PrivateTestCase",
                    help='Path to private testcase directory')
parser.add_argument('--meta-path', type=str, default=f"{project_root}/LeetCodeMeta",
                    help='Path to metadata directory')
parser.add_argument('--base-env-path', type=str, default=f"{project_root}/envs/haskell",
                    help='Path to base environment directory')
parser.add_argument('--tmp-env-path', type=str, default=f"{project_root}/tmp/haskell_env_{uuid.uuid4().hex[:8]}",
                    help='Path to temporary environment directory')
args = parser.parse_args()

llm_output_dir = args.llm_output_dir
output_dir = args.output_dir
private_testcase_path = args.private_testcase_path
meta_base_path = args.meta_path

####### HASKELL EXECUTOR ############
from executor import HaskellExecutor
from config import logger
executor = HaskellExecutor()

def create_haskell_env_copy():
    base_env_path = args.base_env_path
    temp_env_dir = args.tmp_env_path
    # temp_env_dir.parent.mkdir(parents=True, exist_ok=True)
    os.makedirs(temp_env_dir, exist_ok=True)
    shutil.copytree(base_env_path, temp_env_dir, dirs_exist_ok=True)

    #Path to the cabal file after copying
    cabal_file_path = os.path.join(temp_env_dir, "haskell.cabal")

    # Open and replace non-breaking space (U+00A0) with regular space
    with open(cabal_file_path, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace("\u00A0", " ")  # replace non-breaking space
    with open(cabal_file_path, "w", encoding="utf-8") as f:
        f.write(content)

    return temp_env_dir

def run_single_haskell_test(env_dir:str,test_code: str, haskell_code: str):
    main_file_path = os.path.join(env_dir, "app", "Main.hs")

    try:
       
        with open(main_file_path, "w") as f:
            f.write(haskell_code)
       
        executor = HaskellExecutor()
        executor.apply_code(test_code, main_file_path)
        output = executor.execute(env_dir)
        return output
    except Exception as e:
        logger.error(f"Error in run_single_haskell_test: {e}")
        raise e

###### CREAT HASKELL TEST FILE ###########
def extract_types_from_docstring(docstring):
    """Extract types from docstring, including lists."""
    type_pattern = r":type (\w+): ([\w\[\], ]+)"
    return_pattern = r":rtype: ([\w\[\], ]+)"

    param_types = re.findall(type_pattern, docstring)
    return_type_match = re.search(return_pattern, docstring)

    param_dict = {param: p_type for param, p_type in param_types}
    return_type = return_type_match.group(1) if return_type_match else "Unknown"
    return param_dict, return_type
def python_type_to_haskell(python_type):
    """Convert Python data types to Haskell, including nested lists."""
    type_mapping = {
        "int": "Int",
        "float": "Float",
        "str": "String",
        "bool": "Bool",
        "Integer":"Int",
    }
    # Test if it's nested list (List[...])
   
    list_match = re.match(r"List\[(.+)\]", python_type)
    if list_match:
        inner_type = list_match.group(1)
        return f"[{python_type_to_haskell(inner_type)}]" 

    return type_mapping.get(python_type, "Unknown")


def extract_haskell_signature(python_template):
    """Convert Python function signature to Haskell"""
    function_match = re.search(r"def (\w+)\(", python_template)
    if not function_match:
        return None, None

    function_name = function_match.group(1)


    docstring_match = re.search(r'"""(.*?)"""', python_template, re.DOTALL)
    if not docstring_match:
        return None, None

    docstring = docstring_match.group(1)


    param_types, return_type = extract_types_from_docstring(docstring)


    haskell_params = [python_type_to_haskell(param_types[param]) for param in param_types]
    haskell_return = python_type_to_haskell(return_type)


    haskell_signature = f"{function_name} :: " + " -> ".join(haskell_params + [haskell_return])
    haskell_function = f"{function_name} " + " ".join(param_types.keys())
    params = list(param_types.keys())
    return haskell_signature, haskell_function,params, haskell_params
py = '''    
class Solution(object):
    def maxContainers(self, n, w, maxWeight):
        """
        :type n: int
        :type w: int
        :type maxWeight: int
        :rtype: int
        """
'''
sig, fn,params , has_pa= extract_haskell_signature(py)
print(has_pa)
print(sig)
print(fn)
print(params)


####################################


base_type_mappings = {
    "int": "Int",
    "float": "Float",
    "str": "String",
    "bool": "Bool",
    "List[int]": "[Int]",
    "List[float]": "[Float]",
    "List[str]": "[String]",
    "List[bool]": "[Bool]",
}

def extract_value_between_params(input_str, param1, param2=None):
    """
    Trích xuất giá trị giữa param1 và param2 từ input_str,
    hỗ trợ cả kiểu danh sách (List) và chuỗi có dấu "..."
    """
    """
    Extract the value between param1 and param2 from input_str,
    supporting both list (List) and string types with quotes "..."
    """
    if param2:

        # Catch list, string or integer/negative number between param1 and param2
        pattern = rf"{param1} = (\[.*?\]|\".*?\"|\S+)(?=, {param2} = |$)"
    else:
        # Catch value for param1 to end of string if param2 is not present
        pattern = rf"{param1} = (\[.*?\]|\".*?\"|\S+)$"

    match = re.search(pattern, input_str)

    if match:
        return match.group(1).strip() 
    return None




def generate_public_haskell_test_file(question_title, func_name, case, python_template, check_temp:int):
        haskell_signature, haskell_function, params, haskell_pa = extract_haskell_signature(str(python_template))
      
        if check_temp == 0:
            temp_function = "" 
        else:
            temp_function = f"{haskell_signature}\n{haskell_function} = undefined\n"
        input_var = []
        if case['output']=='' or case['output'] == 'None':
            print("Error")
            return
        if case["output"] == "tru":
            case["output"] = 'true'
        for j in range(len(params)):
            if j == (len(params) - 1):
                inp = extract_value_between_params(case['input'], params[j])
                input_var.append(inp)
                break
            inp = extract_value_between_params(case['input'], params[j], params[j + 1])
            input_var.append(inp)

        try:
            test_input_var = " ".join(inp for inp in input_var).strip()
        except:
            return
        test_input_var = re.sub(r'(^|[\s\[,])-\s*(\d+)', r'\1(-\2)', test_input_var)

        cleaned_lines = [line.replace('\"', " \\\"") for line in test_input_var]  # Handle escaped quotes
        test_input = "".join(cleaned_lines).strip()
        test_input =  re.sub(r'(^|[\s\[,])-\s*(\d+)', r'\1(-\2)', test_input)
        test_input = test_input.replace("'",'"')
        test_output = case['output'].split("Cons", 1)[0].strip()
        test_output = test_output.replace('\"','"').replace(" ","")
        test_output = " ".join(test_output.split("\n"))
        test_output = re.sub(r'(^|[\s\[,])-\s*(\d+)', r'\1(-\2)',test_output)
       
        test_output = test_output.replace('true', "True").replace('false', "False").replace('"','\"')
        return f"""module Main where
import Test.HUnit

--Program start
{temp_function}
--Program end

-- Test case
test1 :: Test
test1 = TestCase (assertEqual "for ({func_name} {test_input})," {test_output} ({func_name} {test_input_var}))

-- Running the test
main :: IO Counts
main = runTestTT test1
"""

def is_list_of_single_item_lists(text):
    # First extract the full outer list
    match = re.search(r'\[\s*(\[[^\[\]]+\](\s*,\s*\[[^\[\]]+\])*)\s*\]', text)
    if not match:
        return False
    # Try parsing the string as a Python list
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return False

    # Check if it's a list of lists with one element each
    return all(isinstance(inner, list) and len(inner) == 1 and isinstance(inner[0], int) for inner in parsed)


def generate_private_haskell_test_file(problem_name, func_name, case, python_template, check_temp:int):
    haskell_signature, haskell_function, params, haskell_pa = extract_haskell_signature(str(python_template))  # Assuming params are returned here
    if check_temp == 0:
        temp_function = ""
    else:
        temp_function = f"{haskell_signature}\n{haskell_function} = undefined\n"
    input_var = []
    input_list = case['input']

    if len(params) > 1:
        for item in input_list:
            if is_list_of_single_item_lists(str(item)):
                item = str(item).replace(" ","").replace("[[",'[').replace('],[',',').replace(']]',']')
        input_var.extend(str(item) for item in input_list)
    else:
        if is_list_of_single_item_lists(str(input_list)):
                input_list= str(input_list).replace(" ","").replace("[[",'[').replace('],[',',').replace(']]',']')
        input_var.append(str(input_list))
    try:
        for i, types in enumerate(haskell_pa):
            if 'Int' in types:
                input_var[i] = input_var[i].replace("\"","").replace('"','').replace("\'", "").replace("'","")
            if "String" in types:
                if not input_var[i].startswith('"') and not input_var[i].startswith('\\"'):
                    input_var[i] = '"' + input_var[i] +'"' 
        test_input_var = " ".join(inp for inp in input_var).strip()

    except Exception as e:
        return
    
    test_input_var = re.sub(r'(^|[\s\[,])-\s*(\d+)', r'\1(-\2)', test_input_var)

    cleaned_lines = [line.replace('\"', " \\\"") for line in test_input_var]  # Handle escaped quotes
    test_input = "".join(cleaned_lines).strip()
    test_input =  re.sub(r'(^|[\s\[,])-\s*(\d+)', r'\1(-\2)', test_input)
    test_input = test_input.replace("'",'"')
    test_output = str(case['output'])
    test_output = test_output.replace('\"','"').replace(" ","")
    test_output = " ".join(test_output.split("\n"))
    test_output = re.sub(r'(^|[\s\[,])-\s*(\d+)', r'\1(-\2)',test_output)
    
    test_output = test_output.replace('true', "True").replace('false', "False").replace('"','\"')
    
    return f"""module Main where
import Test.HUnit

--Program start
{temp_function}
--Program end

-- Test case
test1 :: Test
test1 = TestCase (assertEqual "for ({func_name} {test_input})," {test_output} ({func_name} {test_input_var}))

-- Running the test
main :: IO Counts
main = runTestTT test1
"""

import os
import subprocess

def check_haskell_syntax_and_types(path:str,code: str) -> bool:
    
    test_file_path = os.path.join(path, "app", "Main2.hs")
    if code == None:
        return False
    

    try:
        with open(test_file_path, "w") as f:
            f.write(code)
        result = subprocess.run(
            ["cabal", "exec", "--", "ghc", "-fno-code", "app/Main2.hs"],
            cwd=path,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120
        )
       
        return True
    except subprocess.CalledProcessError as e:
        print("STDERR:\n", e.stderr.decode())
        print(e)
        return False
    

######### MAIN FUNCTION TO SAVE FILES ##############

os.makedirs(output_dir, exist_ok=True)
def read_file(filename):
    try:
        meta_path = f"{meta_base_path}/{filename}"
        private_path = f"{private_testcase_path}/{filename}"
        filename = filename.replace('-','_')
        # llms_path = f"{project_root}/output/haskell/gpt-5/{filename}"
        llms_path = f"{llm_output_dir}/{filename}"
        print(f"meta_path: {meta_path}")
        print(f"private_path: {private_path}")
        print(f"llms_path: {llms_path}")
        with open(meta_path, 'r', encoding='utf-8') as f:
            row = json.load(f)
           
        with open(private_path, 'r', encoding='utf-8') as f:
            private_row = json.load(f)
            # print(private_row)
        with open(llms_path, 'r', encoding='utf-8') as f:
            llms_mess = json.load(f)
           
        test_code = llms_mess['code_traces'][0]
        test_code = re.sub(
            r'^module\b[\s\S]*?\bwhere\s*\n',
            '',
            test_code,
            flags=re.MULTILINE
        )
        test_code = re.sub(
            r'^\s*\{\-\#\s*LANGUAGE\s+BangPatterns\s*\#\-\}\s*\n',
            '',
            test_code,
            flags=re.MULTILINE
        )
        test_code = re.sub(
            r'(?<!\S)!(?=\w)',
            '',
            test_code
        )
        return row, private_row, test_code,meta_path
    except:
        print(f"Error in reading file {filename}")
        return None, None, None, None
common_files = []
def save_haskell_files(problem_path):
    i = 0
    env_dir = create_haskell_env_copy()
    for filename in sorted(os.listdir(problem_path)):
        try:  
            i+=1
            problem_name = filename.replace(".json","")

            # Skip if result for this problem already exists
            if os.path.exists(os.path.join(output_dir, f"{problem_name}.json")):
                print(f"Skip {filename} because output already exists")
                continue

            if os.path.exists(f"{llm_output_dir}/{filename}"):
                print(f"Continue {filename}")
                continue
            # break
            # print(f"Processing the problem {i}th {filename}")
            problem_results = []
            row, private_row, test_code,meta_path = read_file(filename)
            if row == None:
                continue
            if not os.path.exists(meta_path):
                continue
            try:
                metadata = row['metadata']
            except:
                    continue
            func_name = metadata['func_name']
        
            question_title = row['name']
            public_test_cases = row['test_case']
            python_template = row['python_template']
            # Process public test_case
            if isinstance(public_test_cases, str):
                public_cases = json.loads(public_test_cases)  # Parse string to list
            elif isinstance(public_test_cases, list):
                public_cases = public_test_cases  # Directly use list
            else:
                continue
            count = 0
            for case in public_cases:
                count+=1
              
                haskell_code = generate_public_haskell_test_file(question_title, func_name, case, python_template,0)
                haskell_syntax = generate_public_haskell_test_file(question_title, func_name, case, python_template,1)
                if haskell_syntax == None:
                    break
                if check_haskell_syntax_and_types(env_dir, haskell_syntax):
                    print(" True Syntax")
                    try:
                        result = run_single_haskell_test(env_dir, test_code, haskell_code)
                        problem_results.append({
                    "Test_num": count,
                    "Result": result})
                    except:
                        continue
                else:
                        result = run_single_haskell_test(env_dir, test_code, haskell_code)
                        problem_results.append({
                    "Test_num": count,
                    "Result": result})
                        print("False Syntax")
                        break
            if result[0] == -1 and result[1] == -1 and result[2] == -1:
                    print("\n\n Error llms message ")
                    if problem_results:
                        with open(os.path.join(output_dir, f"{problem_name}.json"), "w") as f:
                            json.dump(problem_results, f, indent=2)
                    continue
            for ind in range(len(private_row)):

                count+=1
                if count >= 11:
                    break
                private = private_row[ind]
                haskell_code = generate_private_haskell_test_file(question_title,func_name,private, python_template,0)
               
                haskell_syntax = generate_private_haskell_test_file(question_title, func_name, private, python_template,1)
                if haskell_syntax == None:
                    break
                if check_haskell_syntax_and_types(env_dir, haskell_syntax):
                    print("True Syntax")
                   
                    try:
                        result = run_single_haskell_test(env_dir, test_code, haskell_code)
                        problem_results.append({
                    "Test_num": count,
                    "Result": result})
                    except:
                        continue
                else:
                        result = run_single_haskell_test(env_dir, test_code, haskell_code)
                        problem_results.append({
                    "Test_num": count,
                    "Result": result})
                        print("False Syntax")
                        break
            
                if result[0] == -2 and result[1] == -1 and result[2] == -1:
                    print("\n\nTime out need break ")
                    break

            
            if problem_results:
                print(f"Saving results to {output_dir}/{problem_name}.json")
                with open(os.path.join(output_dir, f"{problem_name}.json"), "w") as f:
                    json.dump(problem_results, f, indent=2)

            gc.collect()
        except subprocess.CalledProcessError as e:
            if len(problem_results)>2:
                print(f"Error in saving results to {output_dir}/{problem_name}.json")
                with open(os.path.join(output_dir, f"{problem_name}.json"), "w") as f:
                    json.dump(problem_results, f, indent=2)
            print(e)
            continue
        # break
        
    


save_haskell_files(private_testcase_path)

print("Done!")  