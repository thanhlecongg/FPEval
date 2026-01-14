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
sys.set_int_max_str_digits(5000)  # Tăng lên đủ lớn hơn số chữ số bạn cần

sys.path.append(os.path.abspath('/data/scratch/projects/punim1928/NA/LLM4FunctionalProgramming'))

####### OCAML EXECUTOR ############
from core.executor import OcamlExecutor
executor = OcamlExecutor()

def create_ocaml_env_copy():
    base_env_path = "/data/scratch/projects/punim1928/NA/LLM4FunctionalProgramming/envs/ocaml"
    temp_env_dir = f"/data/scratch/projects/punim1928/NA/tmp/ocaml_env_{uuid.uuid4().hex[:8]}"
    shutil.copytree(base_env_path, temp_env_dir)
    return temp_env_dir

def run_single_ocaml_test(env_dir:str,test_code: str, ocaml_code: str):
    main_file_path = os.path.join(env_dir, "bin", "main.ml")

    try:
       
        with open(main_file_path, "w") as f:
            f.write(ocaml_code)
       
        executor = OcamlExecutor()
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

###### CREAT OCAML TEST FILE ###########
def extract_types_from_docstring(docstring):
    """Trích xuất kiểu dữ liệu từ docstring, bao gồm cả danh sách."""
    type_pattern = r":type (\w+): ([\w\[\], ]+)"
    return_pattern = r":rtype: ([\w\[\], ]+)"

    param_types = re.findall(type_pattern, docstring)
    return_type_match = re.search(return_pattern, docstring)

    param_dict = {param: p_type for param, p_type in param_types}
    return_type = return_type_match.group(1) if return_type_match else "Unknown"
    return param_dict, return_type

def python_type_to_ocaml(python_type):
    base_type_mappings = {
        "int": "int",
        "float": "float",
        "str": "string",
        "bool": "bool",
    }

    def parse_type(py_type):
        if py_type in base_type_mappings:
            return base_type_mappings[py_type]

        list_match = re.match(r"List\[(.+)\]", py_type)
        if list_match:
            inner_type = list_match.group(1)
            return f"{parse_type(inner_type)} list"

        return "unknown"

    return parse_type(python_type)
# Function to extract function information
def extract_ocaml_signature(python_template):
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


    ocaml_params = [python_type_to_ocaml(param_types[param]) for param in param_types]
    ocaml_return = python_type_to_ocaml(return_type)
    params = list(param_types.keys())
    formatted_args = " ".join([f"({name}: {arg_type})" for name, arg_type in zip(params, ocaml_params)])
    ocaml_function = f"let {function_name} {formatted_args} : {ocaml_return} =  "


    return ocaml_params,ocaml_function, params
py = '''   
class Solution(object):
    def maxContainers(self, n, w, maxWeight):
        """
        :type n: List[int]
        :type w: int
        :type maxWeight: int
        :rtype: int
        """
'''
sig, fn,params= extract_ocaml_signature(py)
print(params)
print(sig)  # maxContainers :: Int -> Int -> Int -> Int
print(fn)

#############################
with open('/data/scratch/projects/punim1928/NA/LLM4FunctionalProgramming/leetcode_updated.json', 'r') as f:
    data = json.load(f)  # Không phải json.loads(f.read()) hay đọc từng dòng

####################################
import os
import json
import re

import subprocess
import gc
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



def generate_public_ocaml_test_file(question_title, func_name, case, python_template, check_temp:int):
        _,ocaml_function,params = extract_ocaml_signature(str(python_template))
        input_var=[]
        if check_temp == 0:
            temp_function = "" 
        else:
            temp_function = f"{ocaml_function} failwith \"Not implemented\"\n"
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
        print(f"This is test input var: {test_input_var}")
        cleaned_lines = [line.replace(',', ';') for line in test_input_var]
        test_input = "".join(cleaned_lines)
        test_input = re.sub(r'(^|\s)-(\d+)', r'\1(\g<0>)', test_input)
        test_output = case['output'].split("Cons", 1)[0].strip()
        test_output = " ".join(test_output.split("\n"))
        test_output =  re.sub(r'(^|[\s\[,])-\s*(\d+)', r'\1(-\2)', test_output)
        test_output = test_output.replace('True', "true").replace('False', "false").replace(',', ';')
        test_output = test_output.replace('"','\"')
        return f"""
module Main = struct
    open OUnit2

    (* Program start *)
    {temp_function}
    (* Program end *)

    (* Test case *)
    let test1 _ = assert_equal {test_output} ({func_name} {test_input})

    (* Grouping test cases *)
    let suite = "Test Suite for {func_name}" >::: [
        "test1" >:: test1;
    ]

    (* Running the tests *)
    let () = run_test_tt_main suite
end
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


def generate_private_ocaml_test_file(problem_name, func_name, case, python_template, check_temp:int):
    param_types,ocaml_function,params = extract_ocaml_signature(str(python_template))  # Assuming params are returned here
    print(param_types)
   
    if check_temp == 0:
        temp_function = ""
    else:
        temp_function = f"{ocaml_function} failwith \"Not implemented\"\n"
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
            if 'int' in types:
                input_var[i] = input_var[i].replace("\"","").replace('"','').replace("\'", "").replace("'","")
            elif "string" in types and "list" in types.lower():
                # xử lý list<string>
                content = input_var[i].strip()[1:-1]  # remove [ ]
                items = [x.strip() for x in content.split(',') if x.strip()]
                quoted_items = ['"{}"'.format(item.strip('"').strip("'")) for item in items]
                input_var[i] = "[" + ", ".join(quoted_items) + "]"
            elif types == "string":
                # xử lý string đơn
                if not input_var[i].startswith('"') and not input_var[i].startswith('\\"'):
                    input_var[i] = '"' + input_var[i] + '"'
                
        test_input_var = " ".join(inp for inp in input_var).strip()
    except Exception as e:
        print("error:", e)  # In thông báo lỗi chi tiết
        del_file.append(problem_name)
        return
    
    test_input_var = re.sub(r'(^|[\s\[,])-\s*(\d+)', r'\1(-\2)', test_input_var)

    # cleaned_lines = [line.replace('\"', " \\\"") for line in test_input_var]
    cleaned_lines = [line.replace(',', ';') for line in test_input_var]  # Handle escaped quotes
    test_input = "".join(cleaned_lines)
    test_input =  re.sub(r'(^|[\s\[,])-\s*(\d+)', r'\1(-\2)', test_input)
    test_input = test_input.replace("'",'"')
    test_output = str(case['output'])
    test_output = test_output.replace('\"','"').replace(" ","")
    test_output = " ".join(test_output.split("\n"))
    test_output = re.sub(r'(^|[\s\[,])-\s*(\d+)', r'\1(-\2)',test_output)
    
    test_output = test_output.replace('True', "true").replace('False', "false").replace('"','\"')
    test_output = test_output.replace('"','\"').replace(',',';')
    return f"""
module Main = struct
    open OUnit2

    (* Program start *)
    {temp_function}
    (* Program end *)

    (* Test case *)
    let test1 _ = assert_equal {test_output} ({func_name} {test_input})

    (* Grouping test cases *)
    let suite = "Test Suite for {func_name}" >::: [
        "test1" >:: test1;
    ]

    (* Running the tests *)
    let () = run_test_tt_main suite
end
"""

def check_ocaml_syntax_and_types(path: str, code: str) -> bool:
    test_file_path = os.path.join(path, "bin", "main2.ml")

    if code is None:
        return False

    # Ghi code vào file
    with open(test_file_path, "w") as f:
        f.write(code)

    try:
        # Compile OCaml file with ocamlfind and ounit2
        result = subprocess.run(
            ["ocamlfind", "ocamlopt", "-o", "test_bin", "-package", "ounit2", "-linkpkg", "bin/main2.ml"],
            cwd=path,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return True

    except subprocess.CalledProcessError as e:
        print("⚠️ Compilation failed with error:")
        print(e.stderr.decode("utf-8"))  # In lỗi cụ thể
        return False

del_file = []
output_dir = "/scratch/punim1928/NA/results_llm/ocaml/gpt5_reasoning"
problem_path = '/data/scratch/projects/punim1928/NA/LLM4FunctionalProgramming/remain2'
output_check_dir ='/scratch/punim1928/NA/results/ocaml/gpt4o'
os.makedirs(output_dir, exist_ok=True)
def read_file(filename):
    try:
        private_path = f"{problem_path}/{filename}"
        meta_path = f'/data/scratch/projects/punim1928/NA/LLM4FunctionalProgramming/split_by_name/{filename}'
        llm = filename.replace('-','_')
        llms_path = f'/data/scratch/projects/punim1928/NA/RQ3/result_llms/ocaml/gpt-5/{llm}'
        
        with open(meta_path, 'r', encoding='utf-8') as f:
            row = json.load(f)
        with open(private_path, 'r', encoding='utf-8') as f:
            private_row = json.load(f)
        with open(llms_path, 'r', encoding='utf-8') as f:
            llms_mess = json.load(f)
        test_code = llms_mess['code_traces'][0]
        return row, private_row, test_code,meta_path 
    except:
        return None, None, None, None

common_files = []
# with open('/data/scratch/projects/punim1928/NA/RQ3/sampled_100_files.pkl', 'rb') as f:
#     common_files = pickle.load(f)
def save_haskell_files(problem_path, error_log_file="error.txt"):
    err = []
    i = 0
    env_dir = create_ocaml_env_copy()
    for filename in sorted(os.listdir(problem_path)):
       
        try:  
           
            if os.path.exists(f"{output_dir}/{filename}"):
                print(f"Continue complete {filename}")
                continue
            # if not os.path.exists(f"{output_check_dir}/{filename}"):
            #     print(f"Continue not complete {filename}")
            #     continue
            # if filename not in common_files:
            #     print(f"Continue {filename} not in common files")
            #     continue
            
            # if filename != 'count-prefixes-of-a-given-string.json':
            #     continue
            # if i >= 2:
            
            #     break
            # if i < 21:
            #     continue
            i+=1
            # if i>=200:
            #    break
            print(f"🙀Processing the problem {i}th {filename}")
            problem_results = []
            problem_name = filename.replace(".json","")
            row, private_row, test_code,meta_path = read_file(filename)
            if test_code == None:
                print("do not have file " + problem_name)
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
            
            for case in public_cases:
                print(case)
                count+=1
              
                ocaml_code = generate_public_ocaml_test_file(question_title, func_name, case, python_template,0)
                ocaml_syntax = generate_public_ocaml_test_file(question_title, func_name, case, python_template,1)
                if ocaml_syntax == None:
                    err.append("Syntaxx == none")
                    break
                if check_ocaml_syntax_and_types(env_dir, ocaml_syntax):
                    print("✅ True Syntax")
                    try:
                        result = run_single_ocaml_test(env_dir, test_code, ocaml_code)
                        print(f"This is result {result}")
                        problem_results.append({
                    "Test_num": count,
                    "Result": result})
                    except subprocess.CalledProcessError as e:
                        print(e)
                        print("output err")
                        err.append("Output err")
                        continue
                else:
                     
                        print("❌False Syntax")
                        err.append("Syntax error")
                        break
                

            for ind in range(len(private_row)):
               
                count+=1
                if count >= 11:
                    break
                private = private_row[ind]
                ocaml_code = generate_private_ocaml_test_file(question_title,func_name,private, python_template,0)
               
                ocaml_syntax = generate_private_ocaml_test_file(question_title, func_name, private, python_template,1)
                if ocaml_syntax == None:
                    err.append("Syntaxx == none")
                    break
                if check_ocaml_syntax_and_types(env_dir, ocaml_syntax):
                    print("✅ True Syntax")
                   
                    try:
                        result = run_single_ocaml_test(env_dir, test_code, ocaml_code)
                        problem_results.append({
                    "Test_num": count,
                    "Result": result})
                    except:
                        print("output err")
                        err.append("Output err")
                        continue
                else:
                        
                        print("❌False Syntax")
                        err.append("Syntax error")
                        break
            
                if result[0] == -2 and result[1] == -1 and result[2] == -1:
                    print("\n\nTime out need break ")
                    break

            
            if problem_results:
                with open(os.path.join(output_dir, f"{problem_name}.json"), "w") as f:
                    json.dump(problem_results, f, indent=2)
            # if err:
            #     with open(os.path.join("/home/error_ocaml_3.5", f"{problem_name}.json"), "w") as f:
            #         json.dump(err, f, indent=2)


            gc.collect()
        except subprocess.CalledProcessError as e:
            if problem_results:
                with open(os.path.join(output_dir, f"{problem_name}.json"), "w") as f:
                    json.dump(problem_results, f, indent=2)
            print(e)
            continue
        


save_haskell_files(problem_path)
print("Done!")