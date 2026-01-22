from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_huggingface.llms import HuggingFacePipeline
from abc import abstractmethod, ABC
from state import State
import torch
import re
from langchain_core.messages.human import HumanMessage
from utils import update_messages
from langchain_core.tracers.langchain import wait_for_all_tracers
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_anthropic import ChatAnthropic
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login
import gc
import os

def get_code_from_response(response: str, language: str) -> str:
    """
    Extracts code blocks from a response string.

    Parameters:
    response (str): The response string containing code.
    language (str): The programming language of the code.

    Returns:
    str: Extracted code as a string.
    """
    
    if "```" not in response:
        return response.strip()

    if f"```{language}" in response:
        pattern = rf'```{language}(.*?)```'
    else:
        pattern = r'```(.*?)```'

    code_blocks = re.findall(pattern, response, re.DOTALL)
    
    return ''.join(code_blocks)


def test_llm_connection(model_name: str = "gpt-5") -> bool:
    """
    Quickly test whether the configured LLM can be instantiated and respond to a
    trivial prompt.

    Returns:
        bool: True if the model responds without raising an exception, False otherwise.
    """
    
    try:
        if model_name is type(ChatOpenAI):
            llm = model_name
        else:
            llm = create_llm(model_name)
        messages = [
            (
                "system",
                "You are a helpful assistant that translates English to French. Translate the user sentence.",
            ),
            ("human", "I love programming."),
        ]
        ai_msg = llm.invoke(messages)
        print(ai_msg)
        print(f"LLM connection OK for model '{model_name}'.")
        return True
    except Exception as e:
        print(f"LLM connection FAILED for model '{model_name}': {e}")
        return False

def create_llm(model_name: str):
    if model_name == "gpt-5":
        print("Using gpt-5")
        return ChatOpenAI(
            model="gpt-5.1",
            temperature=1,
            model_kwargs={
            "reasoning_effort": "none" 
        },
            max_tokens=2048,
            base_url=os.environ.get('BASE_URL', ''),
            api_key=os.environ.get('OPENAI_API_KEY', '')
        )
    if model_name == "gpt-3.5-turbo":
        print("Using gpt-3.5-turbo")
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=2048,
            base_url=os.environ.get('BASE_URL', ''),
            api_key=os.environ.get('OPENAI_API_KEY', '')
        )
    elif model_name == "gpt-4o":
        print("Using gpt-4o")
        return ChatOpenAI(
            model="gpt-4o",
            temperature=0.7,
            max_tokens=2048,
            base_url=os.environ.get('BASE_URL', ''),
            api_key=os.environ.get('OPENAI_API_KEY', '')
        )
    elif model_name == "gpt-4o-mini" or model_name=='openai/gpt-4o-mini':
        print("Using gpt-4o")
        return ChatOpenAI(
            model=model_name,
            temperature=0.7,
            max_tokens=2048,
            base_url=os.environ.get('BASE_URL', ''),
            api_key=os.environ.get('OPENAI_API_KEY', '')
        )
    elif model_name == "gpt-4.1":
        print("Using gpt-4.1")
        return ChatOpenAI(
            model="gpt-4.1",
            temperature=0.7,
            max_tokens=2048,
            base_url=os.environ.get('BASE_URL', ''),
            api_key=os.environ.get('OPENAI_API_KEY', '')
        )
    elif model_name == "gpt-o1":
        print("Using gpt-o1")
        return ChatOpenAI(
            model="gpt-o1",
            temperature=1,
            max_tokens=2048,
            base_url=os.environ.get('BASE_URL', ''),
            api_key=os.environ.get('OPENAI_API_KEY', '')
        )
    elif model_name == "gpt-o3-mini":
        print("Using gpt-o3-mini")
        return ChatOpenAI(
            model="o3-mini",
            temperature=1,
            max_tokens=2048,
            base_url=os.environ.get('BASE_URL', ''),
            api_key=os.environ.get('OPENAI_API_KEY')
        )
    elif model_name == "sonnet":
        return ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            base_url=os.environ.get('BASE_URL', ''),
            # api_key=os.environ.get('OPENAI_API_KEY', '')
            temperature=0.7
        )
    elif model_name == "haiku":
        return ChatAnthropic(
            model="claude-3-5-haiku-20241022",
            max_tokens=2048,
            base_url=os.environ.get('BASE_URL', ''),
            # api_key=os.environ.get('OPENAI_API_KEY', '')
            temperature=0.7,
        )
    elif model_name == "opus":
        return ChatAnthropic(
            model="claude-3-5-opus-20240229",
            max_tokens=2048,
            base_url=os.environ.get('BASE_URL', ''),
            # api_key=os.environ.get('OPENAI_API_KEY', '')
            temperature=0.7
        )
    elif model_name == "deepseekv3":
        return ChatOpenAI(
            model="deepseek-chat", 
            api_key=os.environ.get("DEEPSEEK_API_KEY"), 
            base_url="https://api.deepseek.com/v1",
            temperature=0.7,
            max_tokens=2048,
            # base_url=os.environ.get('BASE_URL', ''),
            # api_key=os.environ.get('OPENAI_API_KEY', '')
        )
    elif model_name == "deepseekr1":
        return ChatOpenAI(
            model="deepseek-reasoner", 
            api_key=os.environ.get("DEEPSEEK_API_KEY"), 
            base_url="https://api.deepseek.com/v1",
            temperature=0.7,
            max_tokens=2048,
            # base_url=os.environ.get('BASE_URL', ''),
            # api_key=os.environ.get('OPENAI_API_KEY', '')
        )
    elif model_name in ["qwen2.5-coder", "mixtral", "llama3"]: # for the open source models
        # get the right model id from the model_name
        model_map = {
            "qwen2.5-coder": "Qwen/Qwen2.5-Coder-32B",
            "mixtral": "mistralai/Mixtral-8x22B-v0.1",
            "llama3": "meta-llama/Llama-3.3-70B",
        }

        repo_id = model_map[model_name]
        model_dir = f"./models/{repo_id}"

        # Check if model files exist locally
        if not os.path.exists(model_dir):
            print(f"{repo_id} model was not found locally - now downloading model from Huggingface")
            try: 
                # Save Tokenizer locally
                tokenizer = AutoTokenizer.from_pretrained(repo_id)
                tokenizer.save_pretrained(model_dir)

                # Save model locally
                model = AutoModelForCausalLM.from_pretrained(
                    repo_id,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
                model.save_pretrained(model_dir)
                print(f"Model saved to {model_dir}")

            except (RuntimeError, MemoryError) as e:
                # Check if error is memory-related (OOM)
                if "CUDA out of memory" in str(e) or "DefaultCPUAllocator" in str(e) or "memory" in str(e).lower():
                    print(f"ERROR: Insufficient memory during model initialization: {str(e)}")
                    # Try to free up memory
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    print("Attempting to load with reduced memory settings...")
                    tokenizer = AutoTokenizer.from_pretrained(repo_id, cache_dir=model_dir)
                    model = AutoModelForCausalLM.from_pretrained(
                            repo_id,
                            device_map="auto",
                            torch_dtype=torch.float16,
                            low_cpu_mem_usage=True,
                            offload_folder="offload",
                        )
                else:
                    # Re-raise non-memory errors
                    raise
        else:  
            print("found model locally")
            # Load local model and tokenizer
            try: 
                tokenizer = AutoTokenizer.from_pretrained(model_dir)
                model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    device_map="auto",  # Automatically uses GPU if available
                    torch_dtype=torch.float16  # Reduces memory usage
                )
            except (RuntimeError, MemoryError) as e:
                # Check if error is memory-related (OOM)
                if "CUDA out of memory" in str(e) or "DefaultCPUAllocator" in str(e) or "memory" in str(e).lower():
                    print(f"ERROR: Insufficient memory during model initialization: {str(e)}")
                    # Try to free up memory
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    print("Attempting to load with reduced memory settings...")
                    tokenizer = AutoTokenizer.from_pretrained(model_dir, cache_dir=model_dir)
                    model = AutoModelForCausalLM.from_pretrained(
                            model_dir,
                            device_map="auto",
                            torch_dtype=torch.float16,
                            low_cpu_mem_usage=True,
                            offload_folder="offload",
                        )
                else:
                    raise
        # Create text generation pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=2048,
        )
        
        # Create LangChain LLM
        llm = HuggingFacePipeline(pipeline=pipe)
        
        return ChatHuggingFace(llm=llm)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
class Assistant(ABC):
    """
    Abstract base class for assistants that interact with a language model.
    """

    def __init__(self, runnable: Runnable):
        """
        Initializes the assistant with a runnable task.

        Parameters:
        runnable (Runnable): The runnable task to be executed.
        """
        self.runnable = runnable

    def __call__(self,
                 state: State,
                 config: RunnableConfig,
                 max_attempt: int = 3):
        """
        Invokes the runnable task with the given state and configuration, retrying if necessary.

        Parameters:
        state (State): The current state of the assistant.
        config (RunnableConfig): Configuration for the runnable task.
        max_attempt (int): Maximum number of attempts to retry the task.

        Returns:
        tuple: Result of the runnable task and list of new messages.
        """

        new_messages = []
        while max_attempt >= 0:
            try:
                formatted_prompt = self.prompt_template.invoke(state)
                new_messages += formatted_prompt.to_messages()
                result = self.llm.invoke(formatted_prompt)
                new_messages.append(result)
            finally:
                wait_for_all_tracers()

            if not result.tool_calls and (
                    not result.content or isinstance(result.content, list)
                    and not result.content[0].get("text")):
                message = HumanMessage(
                    content="Invalid response from the model. Please try again."
                )
                state = update_messages(state, message)
                new_messages.append(message)
                max_attempt -= 1
            else:
                break

        return result, new_messages


class CodingAssistant(Assistant):
    """
    Specialized assistant for generating code based on a description and template.
    """

    def __init__(self, model_name):
        self.llm = create_llm(model_name)
        self.prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert {lang} programmer. You will be given a question (problem specification) and will generate a correct {lang} program that matches the specification and passes all tests. You will NOT return anything except for the program AND neccessary imports.\n",
            ),
            ("human", "### QUESTION:\n{description}\n"
            "### FORMAT: You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n{template}\n"
            "### ANSWER: (use the provided format with backticks)\n"),
        ])
        if not self.test_connection():
            raise ValueError(f"LLM connection FAILED for model '{self.llm.model_name}'")
    def test_connection(self):
        return test_llm_connection(self.llm.model_name)
    def __call__(self,
                 state: State,
                 config: RunnableConfig = None,
                 max_attempt: int = 3):
        """
        Invokes the runnable task and extracts code from the response.

        Parameters:
        state (State): The current state of the assistant.
        config (RunnableConfig): Configuration for the runnable task.
        max_attempt (int): Maximum number of attempts to retry the task.

        Returns:
        tuple: Extracted code and list of new messages.
        """

        res, new_messages = Assistant.__call__(self, state, config=config)
        code = get_code_from_response(res.content, state["lang"])
        return code, new_messages


class CodeRepairAssistant(Assistant):

    def __init__(self, model_name):
        self.llm = create_llm(model_name)
        self.prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful programming assistant and an expert {lang} programmer. You are helping a user write a program to solve a problem. The user has written some code, but it has some errors and is not passing the tests. You will help the user by first giving a concise (at most 2-3 sentences) textual explanation of what is wrong with the code. After you have pointed out what is wrong with the code, you will then generate a fixed version of the program. You must put the entired fixed program within code delimiters only for once.\n",
            ),
            ("human", "### QUESTION:\n{description}\n"
            "### ANSWER:\n```{lang}\n{curr_code}\n```\n"
            "### {curr_status}\n"
            "### FORMAT: You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n{template}\n"
            "### ANSWER: (use the provided format with backticks)\n"),
        ])
        

    def __call__(self, state: State, config: RunnableConfig = None):
        status, n_tests, n_errors, output = state["validation_traces"][-1]
        if status == 0:
            return state["curr_code"], []

        res, new_messages = Assistant.__call__(self, state, config=config)
        return get_code_from_response(res.content, state["lang"]), new_messages
        
class CodeBaseRepairAssistant(Assistant):

    def __init__(self, model_name):
        self.llm = create_llm(model_name)
        self.prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful programming assistant and an expert {lang} programmer. You are helping a user write a program to solve a problem. The user has written some code, but it has some errors and is not passing the tests. You will help the user by first giving a concise (at most 2-3 sentences) textual explanation of what is wrong with the code. After you have pointed out what is wrong with the code, you will then generate a fixed version of the program. You must put the entire fixed program within code delimiters only for once.\n",
            ),
            (
                "human",
                "### QUESTION:\n{description}\n"
                "### PREVIOUS CODE:\n```{lang}\n{curr_code}\n```\n"
                "### {curr_status}\n"
                "### FORMAT: You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n{template}\n"
                "### ANSWER: (use the provided format with backticks)\n"
            ),
        ])
    def __call__(self, state: State, config: RunnableConfig = None):
        status, n_tests, n_errors, output = state["validation_traces"][-1]
        if status == 0:
            return state["curr_code"], []

        res, new_messages = Assistant.__call__(self, state, config=config)
        return get_code_from_response(res.content, state["lang"]), new_messages

        
class TestOutputPredictionAssistant(Assistant):

    def __init__(self, model_name):
        self.llm = create_llm(model_name)
        self.prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a helpful programming assistant and an expert Python programmer. 
                You are helping a user to write a test case to help to check the correctness of the function. 
                The user has written an input for the testcase. You will calculate the output of the testcase 
                and write the whole assertion statement in the markdown code block with the correct output.""",
            ),
            ("human", 
                "### Problem:\n{description}\n\n"
                "### Function:\n"
                "```\n{template}\n```\n\n"
                "### Please complete the following test cases:\n"
                "{formatted_testcases}\n"
                "### Response:"
            ),
        ])
        print("\n\n\n Prompt variables:", self.prompt_template.input_variables)


    def __call__(self, state: State, config: RunnableConfig = None):
       
        
        res, new_messages = Assistant.__call__(self, state, config=config)

      

        predicted_assertion = get_code_from_response(res.content, state["lang"])

        return predicted_assertion, new_messages
    



class CodeExecutionAssistant(Assistant):
    def __init__(self, model_name):
        self.llm = create_llm(model_name)
        self.prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an expert {lang} programmer. 
                You will be given a function and an assertion containing an input to the function.
                → Complete the assertion with a literal (no unsimplified expressions, no function calls) 
                containing the exact output when executing the provided code on the given input.
                → Even if the function is incorrect or incomplete, predict the most likely output.
                → Execute the program step by step before arriving at an answer.
                → Return the full assertion with the correct output in [ANSWER] and [/ANSWER] tags.
                → Do NOT output any extra information outside of [ANSWER] and [/ANSWER] tags.
                """,
            ),
            ("human", 
                "[{lang}]\n"
                "{code}\n"
                "{formatted_testcases}\n"
                "[/{lang}]\n"
                "[THOUGHT]\n"
                "Explain how the code executes step by step.\n"
                "[/THOUGHT]\n"
                "[ANSWER]\n"
                "{formatted_testcases}\n"
                "[/ANSWER]"
            ),
        ])

    def __call__(self, state: State, config: RunnableConfig = None):
        """
        Predicts the execution output of the given code using CoT reasoning.
        """
        res, new_messages = Assistant.__call__(self, state, config=config)
        predicted_assertion = res.content.strip()
        return predicted_assertion, new_messages
