import os
import json
import re
import pickle as pkl
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from assistants import create_llm, test_llm_connection
class CodeRepairAssistant:
    def __init__(self, model_name, lang):
        self.lang = lang
        self.model_name = model_name
        # if self.model_name == "gpt-5":
        #     self.llm = ChatOpenAI(
        #         model="gpt-5.1",
        #         temperature=1,
        #         model_kwargs={
        #         "reasoning_effort": "none" 
        #     },
        #         max_tokens=2048,
        #     )
        # else:
        #     self.llm = ChatOpenAI(
        #         model=model_name,
        #         temperature=0.7,
        #         max_tokens=2048,
        #     )
        self.llm = create_llm(model_name)
    
        if not test_llm_connection(model_name):
            raise ValueError(f"LLM connection FAILED for model '{self.llm.model_name}'")
        self.prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful programming assistant and an expert {lang} programmer. The previously generated code has quality issues, does not pass the tests  and is not idiomatic functional programming. Please provide a better code implementation as expected by the task description and the function using idiomatic functional programming style.You must put the entired fixed program within code delimiters only for once.\n",
            ),
            (
                "human",
                "{description}\n"
                "### PREVIOUS CODE:\n```{lang}\n{curr_code}\n```\n"
                "{template}"
                "{answer}"
            ),
        ])

    def repair_code(self, description, curr_code, template, answer):
        prompt = self.prompt_template.invoke({
            "lang": self.lang,
            "description": description,
            "curr_code": curr_code,
            "template": template,
            "answer": answer
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

# problem_path =  '/workspace/dataset'
def main(problem_path=None, input_folder=None, output_folder=None, langs=None, models=None):
    if problem_path is None: problem_path = Path("/workspace/dataset")
    if input_folder is None: input_folder = Path("/workspace/output")
    if output_folder is None: output_folder = Path("/workspace/output_repaired")
    if langs is None: langs = ['haskell']
    if models is None: models = ["gpt-3.5-turbo"]
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    for lang in langs:
        for model_name in models:
            output_folder = Path(os.path.join(output_folder,lang,model_name))
            output_folder.mkdir(parents=True, exist_ok=True)

            assistant = CodeRepairAssistant(model_name=model_name, lang=lang)

            for file_name in sorted(os.listdir(problem_path)):
                if os.path.exists(f"{output_folder}/{file_name}"):
                    print(f"Continue exist {file_name}")
                    continue
                clean_file_name = file_name.replace("-", "_") + ".json"
                input_path = Path(os.path.join(input_folder,lang,model_name,clean_file_name))

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

                fixed_code, ai_response, response_metadata, system_content, human_content = assistant.repair_code(
                    description, curr_code, template, answer
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
    main(langs=['haskell'], models=['gpt-3.5-turbo'])