import re
import json
import sys
import os
import shutil
import uuid
import subprocess
import tempfile
import sys
sys.set_int_max_str_digits(5000)  # Tăng lên đủ lớn hơn số chữ số bạn cần

sys.path.append(os.path.abspath('/data/scratch/projects/punim1928/NA/LLM4FunctionalProgramming'))
import subprocess

def execute_command(command, timeout=60):
    try:
        result = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            text=True
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        return "Timeout"

####### SCALA EXECUTOR ############
from core.executor import ScalaExecutor
executor = ScalaExecutor()

def create_scala_env_copy():

    base_env_path = "/data/scratch/projects/punim1928/NA/LLM4FunctionalProgramming/envs/scala"
    temp_env_dir = f"/data/scratch/projects/punim1928/NA/tmp_4o/scala_env_{uuid.uuid4().hex[:8]}"  # hoặc bạn có thể dùng /workspace/tmp_test
    shutil.copytree(base_env_path, temp_env_dir)
    return temp_env_dir


def main_code(scala_code: str):
    env_dir = create_scala_env_copy()
    main_file_path = os.path.join(env_dir, "src/main/scala/Main.scala")
    
    try:
       

        with open(main_file_path, "r") as f:
            content = f.read()

        # Nếu dòng đầu chưa có package
        if not content.startswith("package "):
            content = "package scala\n\n" + content
            with open(main_file_path, "w") as f:
                f.write(content)

        executor = ScalaExecutor()
        executor.apply_code(scala_code, main_file_path)

        # ✅ Thêm bước compile
        curr_dir = os.getcwd()
        os.chdir(env_dir)

        compile_output = execute_command("sbt compile", timeout=60)

        os.chdir(curr_dir)
        assert os.getcwd() == curr_dir, "Current directory not restored"

        if compile_output == "Timeout":
            if os.path.exists(env_dir):
                shutil.rmtree(env_dir)
            return [-2, -1, -1, "Timeout"]
        if "[error]" in compile_output:
            if os.path.exists(env_dir):
                shutil.rmtree(env_dir)
            return [-1, -1, -1, compile_output]


    except Exception as e:
        print(e)
        raise e
    
    return env_dir

def run_single_scala_test(env_dir: str, test_code: str):
    test_file_path = os.path.join(env_dir, "src/test/scala/MySuite.scala")
    
    # Ghi code chính
    
    try:
        # Ghi code test
        test_code_with_package = "package scala\n\n" + test_code

        with open(test_file_path, "w") as f:
            f.write(test_code_with_package)
        executor = ScalaExecutor()
        output = executor.execute(env_dir)
        print("Raw output: "+ str(output))
        return output
    finally:
        if os.path.exists(env_dir):
            shutil.rmtree(env_dir)
        


# def run_scala_testcase(test_code: str, source_file: str):
#     env_dir = create_scala_env_copy()
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
def python_type_to_scala(python_type):
    base_type_mappings = {
        "int": "Int",
        "float": "Float",
        "str": "String",
        "bool": "Boolean",
    }

    def parse_type(py_type):
        if py_type in base_type_mappings:
            return base_type_mappings[py_type]

        list_match = re.match(r"List\[(.+)\]", py_type)
        if list_match:
            inner_type = list_match.group(1)
            return f"List[{parse_type(inner_type)}]"

        return "unknown"

    return parse_type(python_type)
def extract_scala_signature(python_template):
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


    scala_params = [python_type_to_scala(param_types[param]) for param in param_types]
    scala_return = python_type_to_scala(return_type)
    scala_signature = f"{function_name} :: {' '.join(scala_params)} -> {scala_return}"


    params = list(param_types.keys())
    formatted_args = ",".join([f"{name}: {arg_type}" for name, arg_type in zip(params, scala_params)])
    scala_function = f"def {function_name}({formatted_args}): {scala_return} = {{ \n    \n}}"

    return scala_return, scala_signature, scala_function, params, scala_params

python_template = '''
class Solution(object):
    def maxContainers(self, n, w, maxWeight):
        """
        :type n: int
        :type w: int
        :type maxWeight: int
        :rtype: int
        """
'''

scala_return,sig, fn,params,param_types = extract_scala_signature(python_template)
print(param_types)
print(params)
print(scala_return)
print(sig)  # maxContainers :: Int -> Int -> Int -> Int
print(fn)   # maxContainers n w maxWeight
import re

#############################
with open('/data/scratch/projects/punim1928/NA/LLM4FunctionalProgramming/leetcode_updated.json', 'r') as f:
    data = json.load(f) # Không phải json.loads(f.read()) hay đọc từng dòng
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
    if param2:
        # Bắt danh sách, chuỗi hoặc số nguyên/dấu âm
        pattern = rf"{param1} = (\[.*?\]|\".*?\"|\S+)(?=, {param2} = |$)"
    else:
        # Bắt giá trị cho param1 đến cuối chuỗi nếu param2 không có
        pattern = rf"{param1} = (\[.*?\]|\".*?\"|\S+)$"

    match = re.search(pattern, input_str)

    if match:
        return match.group(1).strip()  # Loại bỏ khoảng trắng thừa
    return None


def generate_public_scala_test_file(question_title, func_name, case,  python_template,check_temp:int):
        _,scala_signature, scala_function, params,_ = extract_scala_signature(python_template)
        temp_function = f"{scala_function} ???\n"
   
    
        input_var=[]
        if case['output']=='' or case['output'] == 'None':
            return
        if case["output"] == "tru":
            case["output"] = 'true'
        if case["output"] == "fals":
            case['output'] = 'false'

        for j in range(len(params)):
            if j == (len(params)-1):
                inp = extract_value_between_params(case['input'], params[j]) #Fixed: using params instead of param
                input_var.append(inp)
                break

            inp = extract_value_between_params(case['input'], params[j], params[j+1])
            # Check if inp is None before calling replace
            if inp is not None:
                inp = inp.replace("'",'"')
            input_var.append(inp)

        try:
            test_input = ",".join(inp for inp in input_var if inp)
        except:
            print("error test_input")

            return
        test_input = change_lst(test_input)
        test_output = " ".join(re.sub(r"\bConstraints:.*", "", case['output']).strip().split("\n"))
        test_output = test_output.replace('True', "true").replace('False', "false").replace(" ","")
        test_output = change_lst(test_output)
        return f"""
class MySuite extends munit.FunSuite {{
    test("test1") {{
        assertEquals(Main.{func_name}({test_input}), {test_output})
    }}
}}
"""


def change_lst(lst):
    def convert_number(match):
        num = match.group(0)
        try:
            if abs(int(num)) > 2_147_483_647:  # Nếu vượt quá Int.MaxValue
                return f"BigInt(\"{num}\")"
            return num  # Nếu trong phạm vi Int, giữ nguyên
        except ValueError:
            return num  # Nếu không phải số, giữ nguyên
    if lst.startswith("\""):
      return lst
    lst = lst.replace('[', "List(").replace(']', ")")
    # Thêm "L" vào số lớn hơn Int.MaxValue hoặc nhỏ hơn Int.MinValue
    lst = re.sub(r'-?\d+', convert_number, lst)

    return lst
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

def generate_private_scala_test_file(problem_name, func_name, case, python_template, check_temp:int):
    type_return, scala_signature, scala_function, params,param_types = extract_scala_signature(python_template)
    temp_function = f"{scala_function} ???\n"
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
        for i, types in enumerate(param_types):
            if 'Int' in types:
                input_var[i] = input_var[i].replace("\"","").replace('"','').replace("\'", "").replace("'","")
                input_var[i] = change_lst(input_var[i])
            if "String" in types:
                if not input_var[i].startswith('"') and not input_var[i].startswith('\\"'):
                    input_var[i] = '"' + input_var[i] +'"' 
        test_input = ",".join(inp for inp in input_var if inp)    
    except:
            print("error test_input")
            return
    print(case['output'])
    test_output = " ".join(str(case['output']).strip().split("\n"))
    test_output = test_output.replace('True', "true").replace('False', "false").replace(" ", "")

    test_output = str(test_output).replace('True', "true").replace('False', "false").replace(" ","")
    test_output = change_lst(test_output)
   
    return f"""
class MySuite extends munit.FunSuite {{
    test("test1") {{
        assertEquals(Main.{func_name}({test_input}), {test_output})
    }}
}}
"""

import os
import subprocess

# def check_haskell_syntax_and_types(code: str) -> bool:
#     base_path = "/workspace/LLM4FunctionalProgramming/envs/hs"
#     test_file_path = os.path.join(base_path, "app", "Main2.hs")
#     with open(test_file_path, "w") as f:
#         f.write(code)

#     try:
#         result = subprocess.run(
#             ["cabal", "exec", "--", "ghc", "-fno-code", "app/Main2.hs"],
#             cwd=base_path,
#             check=True,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#         )
#         return "True"
#     except subprocess.CalledProcessError as e:
#         error_output = e.stderr.decode().strip().splitlines()
#         first_4_lines = "\n".join(error_output[:4])  # Lấy 4 dòng đầu
#         return f"❌ Syntax check failed for \n{first_4_lines}\n\n"
#     finally:
#         if os.path.exists(test_file_path):
#             os.remove(test_file_path)



del_file = []
output_dir = "/data/scratch/projects/punim1928/NA/llms_results/scala/gpt5"
problem_path =  '/data/scratch/projects/punim1928/NA/LLM4FunctionalProgramming/in_out_ver2'
os.makedirs(output_dir, exist_ok=True)
def read_file(filename):
        private_path = f"{problem_path}/{filename}"
        meta_path = f'/data/scratch/projects/punim1928/NA/LLM4FunctionalProgramming/split_by_name/{filename}'
        llm = filename.replace('-','_')
        llms_path = f'/data/scratch/projects/punim1928/NA/LLM4FunctionalProgramming/output/scala/gpt-5/{llm}'
        
        with open(meta_path, 'r', encoding='utf-8') as f:
            row = json.load(f)
        with open(private_path, 'r', encoding='utf-8') as f:
            private_row = json.load(f)
        with open(llms_path, 'r', encoding='utf-8') as f:
            llms_mess = json.load(f)
        test_code = llms_mess['code_traces'][0]
        return row, private_row, test_code,meta_path

import pickle
common_files = []
# with open('/data/scratch/projects/punim1928/NA/RQ3/sampled_100_files.pkl', 'rb') as f:
#     common_files = pickle.load(f)
def save_haskell_files(problem_path, error_log_file="error.txt"):
    del_file = []
    err = []
    i = 0
    
    # start = 100*index
    # end = 100*(index+1)
    for filename in sorted(os.listdir(problem_path)):
        err_compile = 0
        try:
            i+=1
           
        
            if os.path.exists(f"{output_dir}/{filename}"):
                print(f"Continue {filename}")
                continue
            
            # if filename != 'count-the-number-of-vowel-strings-in-range.json':
            #     continue
           
            print(f"🙀Processing the problem {i}th {filename}")
            # print(i)  
            if i>=300:
                break
            
            problem_results = []
            problem_name = filename.replace(".json","")
            try:
                row, private_row, scala_code,meta_path = read_file(filename)
            except:
                del_file.append(problem_name)
                print("do not have file " + problem_name)
                continue
            env = main_code(scala_code)
            if isinstance(env, list):  # tức là là lỗi dạng [-4, ...]
                err_compile = 1
            if err_compile != 1:
                print("This is env: " + str(env))
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
                for case in public_cases:
                    count+=1
                    
                    haskell_code = generate_public_scala_test_file(question_title, func_name, case, python_template,0)
                    
                    try:
                        result = run_single_scala_test(env, haskell_code)
                        print("This is result:" + str(result))
                    except:
                        err.append("Output err")
                        continue
                    check_test_case = generate_public_scala_test_file(question_title, func_name, case, python_template,1)
                    problem_results.append({
                        "Test_num": count,
                        "Result": result})
                for ind in range(len(private_row)):

                    count +=1 
                    if count > 10:
                        break
                    private = private_row[ind]
                    haskell_code = generate_private_scala_test_file(question_title,func_name,private, python_template,0)
                    try:
                        print("ok")
                        result = run_single_scala_test(env, haskell_code)
                
                        if result[3] == "Timeout":
                            print(f"Timeout +{filename}")
                    
                    except:
                        err.append("Output err")
                

                
                    problem_results.append({
                            "Test_num": count,
                            "Result": result})
                if count < 10:
                    err.append(f"Numb of test {count}")
            else:
                print('ok')
                problem_results.append({
                            "Test_num": 1,
                            "Result": env})
            if problem_results:
                with open(os.path.join(output_dir, f"{problem_name}.json"), "w") as f:
                    json.dump(problem_results, f, indent=2)
            # if err:
            #     with open(os.path.join("/home/error_scala", f"{problem_name}.json"), "w") as f:
            #         json.dump(err, f, indent=2)
            gc.collect()
        except Exception as e:
            print(e)
            if problem_results:
                with open(os.path.join(output_dir, f"{problem_name}.json"), "w") as f:
                    json.dump(problem_results, f, indent=2)
            if len(problem_results)>3:
                with open(os.path.join(output_dir, f"{problem_name}.json"), "w") as f:
                    json.dump(problem_results, f, indent=2)
            gc.collect()
            continue
        


save_haskell_files(problem_path)
# export PATH="/data/scratch/projects/punim1928/NA/sbt/bin:$PATH"
