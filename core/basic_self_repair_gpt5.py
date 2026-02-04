import os
import json
import re
import pickle as pkl
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from config import logger


class CodeRepairAssistant:
    def __init__(self, model_name, lang):
        self.lang = lang
        self.model_name = model_name
        self.llm = ChatOpenAI(
            model="gpt-5.1",
            temperature=1,
            model_kwargs={
            "reasoning_effort": "none" 
        },
            max_tokens=2048,
        )
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
        response = self.llm.invoke(messages)

        fixed_code = self.extract_code_from_response(response.content)
        system_msg = messages[0].content
        human_msg = messages[1].content

        return fixed_code, response.content, {}, system_msg, human_msg


    @staticmethod
    def extract_code_from_response(text):
        code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", text, flags=re.DOTALL)
        return code_blocks[0].strip() if code_blocks else text.strip()

problem_path =  '/data/scratch/projects/punim1928/NA/LLM4FunctionalProgramming/remain2'
def main():
    # sub_dataset_path = Path("/data/scratch/projects/punim1928/NA/RQ3/sampled_100_files.pkl")
    # with open(sub_dataset_path, "rb") as f:
    #     sub_dataset = set(pkl.load(f))

    input_folder = Path("/data/scratch/projects/punim1928/NA/LLM4FunctionalProgramming/output")
    langs = ['ocaml']
    models = ["gpt-5"]
    for lang in langs:
        for model_name in models:
            output_folder = Path("/data/scratch/projects/punim1928/NA/RQ3/result_llms") / lang / model_name
            output_folder.mkdir(parents=True, exist_ok=True)

            assistant = CodeRepairAssistant(model_name=model_name, lang=lang)

            for file_name in sorted(os.listdir(problem_path)):
                if os.path.exists(f"{output_folder}/{file_name}"):
                    logger.info(f"Continue exist {file_name}")
                    continue
                clean_file_name = file_name.replace("-", "_")
                input_path = input_folder / lang / model_name / clean_file_name

                if not input_path.exists():
                    logger.warning(f"File not found: {input_path}")
                    continue

                with open(input_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                msg = data.get("messages", [])
                response = msg[1].get("response", "").split("###")

                if len(response) < 4:
                    logger.warning(f"Skipping {file_name}: malformed response sections.")
                    continue

                description = "###" + response[1]
                curr_code = data.get("code_traces", [""])[0]
                if curr_code == "":
                    logger.warning(f"Skipping {file_name} as no code is available.")
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
                logger.debug(f"Repaired data: {repaired_data}")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(repaired_data, f, ensure_ascii=False, indent=4)

                logger.info(f"Repaired code saved for {model_name} -> {output_path}")


if __name__ == "__main__":
    main()
