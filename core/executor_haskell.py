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
sys.set_int_max_str_digits(5000) # Increase the limit for integer string conversion

sys.path.append(os.path.abspath('/data/scratch/projects/punim1928/NA/LLM4FunctionalProgramming'))

####### HASKELL EXECUTOR ############
from core.executor import HaskellExecutor
executor = HaskellExecutor()

def create_haskell_env_copy():
    base_env_path = "/data/scratch/projects/punim1928/NA/LLM4FunctionalProgramming/envs/haskell"
    temp_env_dir = f"/data/scratch/projects/punim1928/NA/tmp/haskell_env_{uuid.uuid4().hex[:8]}"
    shutil.copytree(base_env_path, temp_env_dir)

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
        print(e)
        raise e


# def run_haskell_testcase(test_code: str, source_file: str):
#     env_dir = create_haskell_env_copy()
#     main_file = os.path.join(env_dir, "app", "Main.hs")
#     executor.apply_code(test_code, main_file)
#     return executor.execute(env_dir)

###### CREAT HASKELL TEST FILE ###########
def extract_types_from_docstring(docstring):
    """Trích xuất kiểu dữ liệu từ docstring, bao gồm cả danh sách."""
    type_pattern = r":type (\w+): ([\w\[\], ]+)"
    return_pattern = r":rtype: ([\w\[\], ]+)"

    param_types = re.findall(type_pattern, docstring)
    return_type_match = re.search(return_pattern, docstring)

    param_dict = {param: p_type for param, p_type in param_types}
    return_type = return_type_match.group(1) if return_type_match else "Unknown"
    return param_dict, return_type
def python_type_to_haskell(python_type):
    """Chuyển kiểu dữ liệu Python sang Haskell, bao gồm cả danh sách lồng nhau."""
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
    """Chuyển đổi chữ ký hàm Python sang Haskell."""
    # Lấy tên hàm
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

#############################
with open('/data/scratch/projects/punim1928/NA/LLM4FunctionalProgramming/leetcode_updated.json', 'r') as f:
    data = json.load(f) 

####################################
import os
import json
import re

import subprocess
import gc


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
           
            del_file.append(question_title)
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
        print("error:", e)  # Log the error for debugging
        del_file.append(problem_name)
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
output_dir = "/scratch/punim1928/NA/results_llm_reasoning/haskell/gpt4"
problem_path = '/data/scratch/projects/punim1928/NA/LLM4FunctionalProgramming/remain2'
os.makedirs(output_dir, exist_ok=True)
def read_file(filename):
    try:
    
        private_path = f"{problem_path}/{filename}"
        meta_path = f'/data/scratch/projects/punim1928/NA/LLM4FunctionalProgramming/split_by_name/{filename}'
        llm = filename.replace('-','_')
        llms_path = f'/data/scratch/projects/punim1928/NA/RQ3/result_llms_reasoning/haskell/gpt-4o/{llm}'
        
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
        return None, None, None, None
common_files = []
def save_haskell_files(problem_path, error_log_file="error.txt"):
    del_file = ['count_the_number_of_inversions.json','']
    err = []
    
    i = 0
    env_dir = create_haskell_env_copy()
    for filename in sorted(os.listdir(problem_path)):
        try:  
            if os.path.exists(f"{output_dir}/{filename}"):
                print(f"Continue {filename}")
                continue
            i+=1
            print(f"Processing the problem {i}th {filename}")
            problem_results = []
            problem_name = filename.replace(".json","")
            row, private_row, test_code,meta_path = read_file(filename)
            if row == None:
                print("Continue NOne " + filename)
                continue
            if not os.path.exists(meta_path):
                del_file.append(problem_name)
                print("do not have file " + problem_name)
                continue
            try:
                metadata = row['metadata']
            except:
                    print(f"skip {question_title} because metadata is None")
                    del_file.append(row['name'])
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
                del_file.append(filename)
                print(f"⚠️ Bỏ qua {question_title}: public_test_cases không hợp lệ ({type(public_test_cases)})")
                continue
            count = 0
            print("This is public")
            for case in public_cases:
                count+=1
              
                haskell_code = generate_public_haskell_test_file(question_title, func_name, case, python_template,0)
                haskell_syntax = generate_public_haskell_test_file(question_title, func_name, case, python_template,1)
                if haskell_syntax == None:
                    err.append("Syntaxx == none")
                    break
                if check_haskell_syntax_and_types(env_dir, haskell_syntax):
                    print(" True Syntax")
                    try:
                        result = run_single_haskell_test(env_dir, test_code, haskell_code)
                        problem_results.append({
                    "Test_num": count,
                    "Result": result})
                    except:
                        err.append("Output err")
                        continue
                else:
                        result = run_single_haskell_test(env_dir, test_code, haskell_code)
                        problem_results.append({
                    "Test_num": count,
                    "Result": result})
                        print("False Syntax")
                        err.append("Syntax error")
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
                    err.append("Syntaxx == none")
                    break
                if check_haskell_syntax_and_types(env_dir, haskell_syntax):
                    print("True Syntax")
                   
                    try:
                        result = run_single_haskell_test(env_dir, test_code, haskell_code)
                        problem_results.append({
                    "Test_num": count,
                    "Result": result})
                    except:
                        err.append("Output err")
                        continue
                else:
                        result = run_single_haskell_test(env_dir, test_code, haskell_code)
                        problem_results.append({
                    "Test_num": count,
                    "Result": result})
                        print("False Syntax")
                        err.append("Syntax error")
                        break
            
                if result[0] == -2 and result[1] == -1 and result[2] == -1:
                    print("\n\nTime out need break ")
                    break

            
            if problem_results:
                with open(os.path.join(output_dir, f"{problem_name}.json"), "w") as f:
                    json.dump(problem_results, f, indent=2)

            gc.collect()
        except subprocess.CalledProcessError as e:
            if len(problem_results)>2:
                with open(os.path.join(output_dir, f"{problem_name}.json"), "w") as f:
                    json.dump(problem_results, f, indent=2)
            print(e)
            continue
        
    


save_haskell_files(problem_path)

print("Done!")  
