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

sys.path.append(os.path.abspath('/scratch/punim1928/NA/LLM4FunctionalProgramming'))
import subprocess

####### java EXECUTOR ############
from core.executor import JavaExecutor
executor = JavaExecutor()

def create_java_env_copy(file_name: str):
    base_env_path = "/scratch/punim1928/NA/LLM4FunctionalProgramming/envs/java"
    temp_env_dir = f"/scratch/punim1928/NA/tmp_4o/{file_name}"  # hoặc bạn có thể dùng /workspace/tmp_test
    shutil.copytree(base_env_path, temp_env_dir)
    return temp_env_dir


def main_code( java_code: str, file_name: str):
    env_dir = create_java_env_copy(file_name)
    print(java_code)
    base_env_path = "/scratch/punim1928/NA/LLM4FunctionalProgramming/envs/java/src/main/java/Main.java"
    main_file_path = os.path.join(env_dir, "src/main/java/Main.java")
    
    shutil.copyfile(base_env_path, main_file_path)
    try:

        executor = JavaExecutor()
        executor.apply_code(java_code, main_file_path)  
    except Exception as e:
        print(e.stderr)
        raise e
    return env_dir
def run_single_java_test(env_dir: str, test_code:str):
    test_file_path = os.path.join(env_dir, "src/test/java/MainTest.java")
    try:
        # Ghi code test

        with open(test_file_path, "w") as f:
            f.write(test_code)
       
        executor = JavaExecutor()
        # executor.apply_code(java_code, test_file_path)
        output = executor.execute(env_dir)
        return output
    except subprocess.CalledProcessError as e:
        return [-1,-1,-1, e]
        
        
        


# def run_java_testcase(test_code: str, source_file: str):
#     env_dir = create_java_env_copy()
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

def python_type_to_java(python_type):
    primitive_types = {
        "int": "int",
        "float": "double",
        "bool": "boolean",
        "str": "String",
    }
    wrapper_types = {
        "int": "Integer",
        "float": "Double",
        "bool": "Boolean",
        "str": "String",
    }

    def parse_type(py_type, inside_list=False):
        py_type = py_type.strip()
        if py_type in primitive_types:
            return wrapper_types[py_type] if inside_list else primitive_types[py_type]

        list_match = re.match(r"List\[(.+)\]", py_type)
        if list_match:
            inner_type = parse_type(list_match.group(1), inside_list=True)
            return f"List<{inner_type}>"

        return "Object"

    return parse_type(python_type)

def python_list_to_java(py_list_str):
    """
    Chuyển danh sách Python (ở dạng chuỗi) thành cú pháp Java Arrays.asList(...)
    """
    def replace_nested(match):
        content = match.group(1)
        return f"new ArrayList<>(Arrays.asList({content}))"

    # Xử lý list lồng nhau trước (tìm  `[...]` lặp lại nhiều lần)
    while re.search(r"\[([^\[\]]+)\]", py_list_str):
        py_list_str = re.sub(r"\[([^\[\]]+)\]", replace_nested, py_list_str)

    return f"{py_list_str}"


    def parse_type(py_type, inside_list=False):
        if py_type in primitive_types:
            return wrapper_types[py_type] if inside_list else primitive_types[py_type]

        list_match = re.match(r"List\[(.+)\]", py_type)
        if list_match:
            inner_type = parse_type(list_match.group(1), inside_list=True)
            return f"List<{inner_type}>"

        return "Object"

    return parse_type(python_type)
def extract_java_signature(python_template):

    function_match = re.search(r"def (\w+)\(", python_template)
    if not function_match:
        return None

    function_name = function_match.group(1)
    docstring_match = re.search(r'"""(.*?)"""', python_template, re.DOTALL)
    if not docstring_match:
        return None

    docstring = docstring_match.group(1)
    param_types, return_type = extract_types_from_docstring(docstring)
    for param in param_types:
        param_types[param] = python_type_to_java(param_types[param])
    java_params = [f"{python_type_to_java(param_types[param])} {param}" for param in param_types]
    java_return = python_type_to_java(return_type)

    java_function= f"public static {java_return} {function_name}({', '.join(java_params)}) "
    return java_function, list(param_types.keys()),list(param_types.values())

python_template = '''
class Solution(object):
    def maxContainers(self, n, w, maxWeight):
        """
        :type n: List[int]
        :type w: int
        :type maxWeight: int
        :rtype: int
        """
'''

java_return,params, param_types = extract_java_signature(python_template)

print(param_types)

# java_return,sig, fn,params,param_types = extract_java_signature(python_template)
# print(param_types)
# print(params)
# print(java_return)
# print(sig)  # maxContainers :: Int -> Int -> Int -> Int
# print(fn)   # maxContainers n w maxWeight
import re

#############################
with open('/scratch/punim1928/NA/LLM4FunctionalProgramming/leetcode_updated.json', 'r') as f:
    data = json.load(f)  # Không phải json.loads(f.read()) hay đọc từng dòng
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



def generate_public_java_test_file(question_title, func_name, case,  python_template,check_temp:int):
        java_function,params,_ = extract_java_signature(str(python_template))
   
        
        input_var=[]
        if case['output']=='' or case['output'] == 'None':
            return
        if case["output"] == "tru":
            case["output"] = 'true'
        if case["output"] == "fals":
            case['output'] = 'false'
       
        for j in range(len(params)):
            if j == (len(params)-1):
                inp = extract_value_between_params(case['input'], params[j])
                inp = python_list_to_java(str(inp))
                input_var.append(inp)
                break

            inp = extract_value_between_params(case['input'], params[j], params[j+1])
            # Check if inp is None before calling replace
            if inp is not None:
                inp = python_list_to_java(str(inp))
                if str(inp).startswith('[]'):
                    inp = 'new ArrayList<>()'
                input_var.append(inp)
            # Handle the case where inp is None, maybe append a default value?
            else:
                

                input_var.append("null")
        test_input = ", ".join(str(inp) for inp in input_var)
        test_output = " ".join(re.sub(r"\bConstraints.*", "", case['output']).strip().split("\n"))
        test_output = test_output.replace('True', "true").replace('False', "false").replace(" ","")
        if test_output.startswith('"'):
                test_output = f'\"'+test_output.replace('"', "")+'\"'
        else:
            test_output = python_list_to_java(test_output)
    
        return f"""
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;
import java.util.*;
public class MainTest {{
    // Program start

    // Program end
    @Test\npublic void test1() {{\n    assertEquals({test_output}, Main.{func_name}({test_input}));\n}}\n
}}
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

def generate_private_java_test_file(problem_name, func_name, case, python_template, check_temp:int):
    
    try:
        java_function,_,params = extract_java_signature(str(python_template))
        if check_temp == 0:
            temp_function = ''
        else: 
            temp_function = f"{java_function} {{ throw new UnsupportedOperationException(\"Function not implemented yet\"); }}\n"
        
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
            for i, types in enumerate(params):
                if 'int' in types or 'Integer' in types:
                    input_var[i] = input_var[i].replace("\"","").replace('"','').replace("\'", "").replace("'","")
                    input_var[i] = python_list_to_java(str(input_var[i]))
                if "String" in types:
                    if input_var[i].startswith('"'):
                        input_var[i] = f'\"'+input_var[i][1:-1]+'\"'
                    if not input_var[i].startswith('"') and not input_var[i].startswith('\\"'):
                        input_var[i] = '\"' + input_var[i] +'\"' 
                if input_var[i].startswith('[]'):
                    input_var[i] = 'new ArrayList<>()'
                input_var[i] = python_list_to_java(input_var[i])
            test_input = ",".join(inp for inp in input_var if inp)
            
        except:
                print("error test_input")
                return
     
        
        test_output = " ".join(re.sub(r"\bConstraints.*", "", str(case['output'])).strip().split("\n"))
        test_output = str(test_output).replace('True', "true").replace('False', "false").replace(" ","")
        
        if test_output.startswith('"'):
                test_output = f'\"'+test_output.replace('"', "")+'\"'
        else:
            test_output = python_list_to_java(test_output)
        
    except subprocess.CalledProcessError as e:
            print("heloo" + e)
    return f"""
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;
import java.util.*;
public class MainTest {{
    // Program start

    // Program end
    @Test\npublic void test1() {{\n    assertEquals({test_output}, Main.{func_name}({test_input}));\n}}\n
}}
"""


def check_java_test_syntax(project_dir: str) -> bool:
    """
    Kiểm tra cú pháp các file Java test trong một Maven project.
    
    Args:
        project_dir (str): Đường dẫn tới thư mục chứa `pom.xml`.

    Returns:
        bool: True nếu cú pháp hợp lệ, False nếu có lỗi biên dịch.
    """
    try:
        result = subprocess.run(
            ["mvn", "test-compile"],
            cwd=project_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode == 0:
            print("✅ Test syntax is valid.")
            return True
        else:
            print("STDOUT:")
            print(result.stdout)
            print("❌ Syntax error in test files.")
            print(result.stderr)
            return False

    except FileNotFoundError:

        print("❌ Maven (mvn) is not installed or not in PATH.")
        return False



del_file = []
output_dir = "/scratch/punim1928/NA/results/java/gpt-5"
problem_path =  '/scratch/punim1928/NA/LLM4FunctionalProgramming/in_out_ver2'
os.makedirs(output_dir, exist_ok=True)
def read_file(filename):
        private_path = f"{problem_path}/{filename}"
        meta_path = f'/scratch/punim1928/NA/LLM4FunctionalProgramming/split_by_name/{filename}'
        lm = filename.replace('-','_')
        llms_path = f'/data/scratch/projects/punim1928/NA/LLM4FunctionalProgramming/output/java/gpt-5/{lm}'
        try: 
            with open(meta_path, 'r', encoding='utf-8') as f:
                row = json.load(f) 
            with open(private_path, 'r', encoding='utf-8') as f:
                private_row = json.load(f)
            with open(llms_path, 'r', encoding='utf-8') as f:
                llms_mess = json.load(f)
            test_code = llms_mess['code_traces'][0]
        except:
            return None, None, None, None
        return row, private_row, test_code,meta_path
import pickle
common_files = []
with open('/scratch/punim1928/NA/common_files.pkl', 'rb') as f:
    common_files = pickle.load(f)
def save_haskell_files(problem_path, error_log_file="error.txt"):
    del_file = []
    err = []
    i = 0
    index = 0
    start = 100*index
    end = 100*(index+1)
    # env_dir = create_java_env_copy()
    for filename in sorted(os.listdir(problem_path)):
        try:
            # if filename != 'apply-bitwise-operations-to-make-strings-equal.json':
            #     continue
            i+=1
        
            if os.path.exists(f"{output_dir}/{filename}"):
                print(f"Continue {filename}")
                continue

            # if filename not in common_files:
            #     print(f"Continue {filename}")
            #     continue
           
            # if i >= 200:
            #     break
            # if i<200:
            #     continue
            print(f"🙀Processing the problem {i}th {filename}")
            # print(i)  
           
            problem_results = []
            problem_name = filename.replace(".json","")
            try:
                row, private_row, java_code,meta_path = read_file(filename)
            except:
                del_file.append(problem_name)
                print("do not have file " + problem_name)
                continue
            if java_code is None:
                del_file.append(problem_name)
                print("do not have code " + problem_name)
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
           
            env_dir = main_code(java_code, problem_name)
            print("This is env: " + str(env_dir))
            for case in public_cases:
                print("count")
                count+=1
               
                java_test = generate_public_java_test_file(question_title, func_name, case, python_template,0)
                if java_test is None:
                    print("Skip because java_test is None")
                    continue
                # check_test_case = generate_public_java_test_file(question_title, func_name, case, python_template,1)
                if check_java_test_syntax(env_dir):
                    try:
                        result = run_single_java_test(env_dir, java_test)
                        problem_results.append({
                                "Test_num": count,
                                "Result": result})
                    except Exception as e:
                        result = run_single_java_test(env_dir, java_test)
                        problem_results.append({
                                "Test_num": count,
                                "Result": result})
                        err.append("Output err")
                        continue
                else:
                    try:
                        result = run_single_java_test(env_dir, java_test)
                        problem_results.append({
                                "Test_num": count,
                                "Result": result})
                    except Exception as e:
                        result = run_single_java_test(env_dir, java_test)
                        problem_results.append({
                                "Test_num": count,
                                "Result": result})
                        err.append("Output err")
                    break
            err_count = 0
            for ind in range(len(private_row)):
               
                count +=1 
                print(count)
                if count > 10:
                    break
                private = private_row[ind]
                java_test = generate_private_java_test_file(question_title,func_name,private, python_template,0)
                if check_java_test_syntax(env_dir):
                    try:
                        result = run_single_java_test(env_dir, java_test)
                        problem_results.append({
                                "Test_num": count,
                                "Result": result})
                        if result[0] == -2 and result[1] == -1 and result[2] == -1:
                            print("\n\nTime out, need break.")
                            break
                    except Exception as e:

                        # print(e)
                        err.append("Output err")
                        continue
                else:
                    err_count +=1
                    err.append("Syntax err")
                    try:
                        result = run_single_java_test(env_dir, java_test)
                        problem_results.append({
                                "Test_num": count,
                                "Result": result})
                    except Exception as e:
                        result = run_single_java_test(env_dir, java_test)
                        problem_results.append({
                                "Test_num": count,
                                "Result": result})
                        err.append("Output err")
                    if err_count ==2:
                        break
                
            if count < 10:
                err.append(f"Numb of test {count}")

            if problem_results:
                with open(os.path.join(output_dir, f"{problem_name}.json"), "w") as f:
                    json.dump(problem_results, f, indent=2)
            if err:
                with open(os.path.join("/scratch/punim1928/NA/error_java_4o", f"{problem_name}.json"), "w") as f:
                    json.dump(err, f, indent=2)
            shutil.rmtree(env_dir)
            gc.collect()
           
        except subprocess.CalledProcessError as e:
        
            print(e)
            if problem_results:
                with open(os.path.join(output_dir, f"{problem_name}.json"), "w") as f:
                    json.dump(problem_results, f, indent=2)
            if err:
                with open(os.path.join("/scratch/punim1928/NA/error_java_4o", f"{problem_name}.json"), "w") as f:
                    json.dump(err, f, indent=2)
            gc.collect()
            # try:
            #     # shutil.rmtree(env_dir)
            # except:
            #     continue
            continue
        
        

save_haskell_files(problem_path)
print("Done!")