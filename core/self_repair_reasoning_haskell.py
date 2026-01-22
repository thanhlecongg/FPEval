import os
import json
import re
import pickle as pkl
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

warning_fix_map = {
    "Avoid lambda": "Avoid using lambda expressions when a function or operator already exists.",
    "Eta reduce": "Remove unnecessary parameters by applying eta-reduction.",
    "Redundant bracket": "Eliminate redundant brackets to simplify the expression.",
    "Avoid lambda using infix": "Replace lambda expressions with infix operators when possible.",
    "Use zipWith": "Use `zipWith` for parallel list operations instead of combining `map` and `zip`.",
    "Avoid reverse": "Avoid using `reverse` before `head`, `last`, or indexing, as it is inefficient.",
    "Move brackets to avoid $": "Prefer parentheses over excessive use of the `$` operator for readability.",
    "Move filter": "Apply `filter` closer to the data source to reduce unnecessary computation.",
    "Use uncurry": "Use `uncurry` when applying a function to a tuple.",
    "Use infix": "Use infix notation for better readability when working with operators."
}

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
def get_warning_fixes(warnings):
    """
    Convert raw warnings into a formatted string with fixes
    for injection into LLM prompt.
    """
    formatted = []
    for w in warnings:
        _, _, msg = w.rpartition(":")
        msg = msg.strip()
        fix = None
        for key in warning_fix_map:
            if key in msg:
                fix = warning_fix_map[key]
                break
        formatted.append(f"- {msg}\n Fix: {fix or 'General Haskell style improvement needed.'}")
    return "\n".join(formatted) if formatted else "No style warnings."

class CodeRepairAssistant:
    def __init__(self, model_name, lang):
        self.lang = lang
        self.model_name = model_name
        if self.model_name == "gpt-5":
            self.llm = ChatOpenAI(
                model="gpt-5.1",
                temperature=1,
                model_kwargs={
                "reasoning_effort": "none" 
            },
                max_tokens=2048,
            )
        else:
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=0.7,
                max_tokens=2048,
            )
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system",
        "You are a helpful programming assistant and an expert Haskell functional programming developer.\n"
"The previously generated code has quality issues, fails some tests, and is not idiomatic functional programming.\n"
"Your task is to fully repair the code so that:\n"
 "When answering:\n"
        "Write the fully fixed program that passes all tests, inside a single ```haskell code block.\n"
        "Do not include explanations outside the code block.\n"
                    ),
            (
                "human",
                "{description}\n"
                "### PREVIOUS CODE:\n```{lang}\n{curr_code}\n```\n"
                "### COMPILE ERROR: {error}\n"
                "### WARNINGS:\n{quality_issues}\n"
                "{template}"
                "{answer}"
            ),
        ])

    def repair_code(self, description, curr_code, template, answer, error, quality_issues):
        prompt = self.prompt_template.invoke({
            "lang": self.lang,
            "description": description,
            "curr_code": curr_code,
            "template": template,
            "answer": answer,
            "error": error,
            "quality_issues": quality_issues
        })
        messages = prompt.to_messages()
        response = self.llm.predict_messages(messages)
        response_metadata = {}
        fixed_code = self.extract_code_from_response(response.content)
        return fixed_code, response.content, response_metadata, messages[0].content, messages[1].content

    @staticmethod
    def extract_code_from_response(text):
        code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", text, flags=re.DOTALL)
        return code_blocks[0].strip() if code_blocks else text.strip()

problem_path =  '/data/scratch/projects/punim1928/NA/LLM4FunctionalProgramming/in_out_ver2'
def main():
    # sub_dataset_path = Path("/data/scratch/projects/punim1928/NA/RQ3/sampled_100_files.pkl")
    # with open(sub_dataset_path, "rb") as f:
    #     sub_dataset = set(pkl.load(f))

    # Load warnings JSON
    warnings_path = Path("/data/scratch/projects/punim1928/NA/LLM4FunctionalProgramming/haskell_quality_results_5.json")
    with open(warnings_path, "r") as f:
        all_warnings = json.load(f)
    
    warnings_map = []
    
    # input_folder = Path("/data/scratch/projects/punim1928/NA/llms_results")
    #path for gpt5
    # input_folder = Path("/data/scratch/projects/punim1928/NA/LLM4FunctionalProgramming/output")
    langs = ['haskell']
    models = ["gpt-5"]
    results_folder =   Path("/data/scratch/projects/punim1928/NA/results/haskell/gpt5")
    for lang in langs:
        for model_name in models:
            output_folder = Path("/data/scratch/projects/punim1928/NA/RQ3/result_llms_reasoning") / lang / model_name
            output_folder.mkdir(parents=True, exist_ok=True)

            assistant = CodeRepairAssistant(model_name=model_name, lang=lang)

            for file_name in sorted(os.listdir(problem_path)):
                
                clean_file_name = file_name.replace("-", "_")
                if os.path.exists(f"{output_folder}/{clean_file_name}"):
                    print(f"Continue exist {file_name}")
                    continue
                input_path = input_folder / lang / model_name / clean_file_name
                result_path = results_folder / file_name

                for item in all_warnings:
                    if item["file"] == clean_file_name:
                        file_warnings = item["hlint_issues"]
                        warnings_map = [issue["hint"] for issue in file_warnings if "hint" in issue]
                        break
                try:
                    with open(result_path) as f:
                        result = json.load(f)
                except Exception as e:
                    print(f"Error reading test data for {file_name}: {e}")
                    continue        
                results_test = [d["Result"] for d in result if isinstance(d.get("Result", []), list)]
                classification = classify_test_results(results_test)
                print(classification)
                if classification in ['compile_error']:
                    error = [d["Result"] for d in result if isinstance(d.get("Result", []), list)][:1]
                else:
                    error = classification
                if not input_path.exists():
                    print(f"File not found: {input_path}")
                    continue

                with open(input_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                msg = data.get("messages", [])
                response = msg[1].get("response", "").split("###")

                if len(response) < 4:
                    print(f"Skipping {file_name}: malformed response sections.")
                    continue

                description = "###" + response[1]
                curr_code = data.get("code_traces", [""])[0]
                if curr_code == "":
                    print(f"Skipping {file_name} as no code is available.")
                    continue
                template = "###" + response[2]
                answer = "###" + response[3]
                file_warnings = warnings_map
                print(f"Processing {file_name} with {len(file_warnings)} warnings.")
                quality_issues = get_warning_fixes(file_warnings)

                fixed_code, ai_response, response_metadata, system_content, human_content = assistant.repair_code(
                    description, curr_code, template, answer, error, quality_issues=quality_issues
                )

                repaired_data = {
                    "code_traces": [fixed_code],
                    "messages": [
                        {
                            "idx": 0,
                            "role": "system",
                            "response": system_content,
                        
                        },
                        {
                            "idx": 1,
                            "role": "human",
                            "response": human_content,
                           
                        },
                        {
                            "idx": 2,
                            "role": "ai",
                            "full_response": ai_response,
                            "response": f"```{lang}\n{fixed_code}\n```",
                           
                        }
                    ]
                }

                output_path = output_folder / clean_file_name
                print(repaired_data)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(repaired_data, f, ensure_ascii=False, indent=4)

                print(f"Repaired code saved for {model_name} -> {output_path}")


if __name__ == "__main__":
    main()
