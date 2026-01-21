from state import State
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from assistants import CodingAssistant, CodeRepairAssistant, TestOutputPredictionAssistant, CodeExecutionAssistant
from langchain_core.messages import system, human, ai
from utils import save_workflow_to_image, _print_event
import json
import os
import torch
import gc
import shutil
from executor import HaskellExecutor, OcamlExecutor, ScalaExecutor, JavaExecutor

# Do not remove this line, it is required for setting up the environment
from config import *


_TMP_DIR = "tmp"


class CodeGenerator():
    """
    LLM-based agent for code generation using [LangGraph](https://langchain-ai.github.io/langgraph/)
    """

    def __init__(self, workflow="basic", language="java", max_iters: int = 5, model_name: str = "gpt-3.5-turbo", output_file: str ="output", timeout_limit: int = 300):
        """
        Initializes the CodeGenerator with the specified workflow, language, and maximum iterations.

        Parameters:
        workflow (str): The workflow to use for code generation.
        language (str): The programming language for code generation.
        max_iters (int): Maximum number of iterations for code generation.
        """

        self.state = State()
        self.workflow = self.basic_workflow()
        self.max_iters = max_iters
        self.language = language
        self.workflow_type = workflow
        self.output_file = output_file
        if workflow == "basic":
            self.workflow = self.basic_workflow()
        elif workflow == "self-repair":
            self.workflow = self.self_repair_workflow()
        elif workflow == "test-predict-output":
            self.workflow = self.test_output_predict_workflow()
        elif workflow == "code-execution":
            self.workflow = self.code_execution_workflow()
        elif workflow == "repair":
            self.workflow = self.repair_workflow()
        elif workflow == "gen_code_workflow":
            self.workflow = self.gen_code_workflow()
        else:
            return NotImplementedError(f"Workflow {workflow} not implemented")

        if self.language == "haskell":
            self.executable_dir = "envs/haskell"
            self.executor = HaskellExecutor(timeout_limit)
        elif self.language == "ocaml":
            self.executable_dir = "envs/ocaml"
            self.executor = OcamlExecutor(timeout_limit)
        elif self.language == "scala":
            self.executable_dir = "envs/scala"
            self.executor = ScalaExecutor(timeout_limit)
        elif self.language == "java":
            self.executable_dir = "envs/java"
            self.executor = JavaExecutor(timeout_limit)
        else:
            raise NotImplementedError(
                f"Language {self.language} not implemented")

        # attempt to initialize the assistants with error handling
        try: 
            self.env_dir = None
            self.source_file = None
            self.code_gen_assistant = CodingAssistant(model_name)
            self.code_repair_assistant = CodeRepairAssistant(model_name)
            self.test_output_prediction_assistant = TestOutputPredictionAssistant(model_name)
            self.code_execution_assistant = CodeExecutionAssistant(model_name)
        except (RuntimeError, MemoryError) as e:
            # Check if error is memory-related (OOM)
            if "CUDA out of memory" in str(e) or "DefaultCPUAllocator" in str(e) or "memory" in str(e).lower():
                print(f"ERROR: Insufficient memory during initialization: {str(e)}")
                
                return NotImplementedError("Insufficient memory during initialization")
            else:
                # Re-raise non-memory errors
                raise


    def gen_init_code(self, state: State, config: RunnableConfig = None):
        """
        Generates initial code using the CodingAssistant.

        Parameters:
        state (State): The current state of the code generation process.
        config (RunnableConfig): Configuration for the runnable task.

        Returns:
        State: The updated state with the generated code.
        """
        
        # run with error handling
        try: 
            new_state = state
            new_state["n_iters"] += 1
            new_state["curr_code"], new_messages = self.code_gen_assistant.__call__(
                state, config=config)
            new_state["code_traces"] = []
            new_state["code_traces"].append(new_state["curr_code"])
            new_state["messages"] = new_messages
            return new_state
        except (RuntimeError, MemoryError) as e:
            # Check if error is memory-related (OOM)
            if "CUDA out of memory" in str(e) or "DefaultCPUAllocator" in str(e) or "memory" in str(e).lower():
                print(f"ERROR: Insufficient memory during code generation: {str(e)}")
                
                # Try to free up memory
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Error response
                new_state = state
                new_state["n_iters"] += 1
                new_state["curr_code"] = "# Failed to generate code due to memory constraints"
                new_state["code_traces"] = []
                new_state["code_traces"].append(new_state["curr_code"])
                new_state["messages"] = [system("Memory error occurred"), human("Please try again with a smaller model")]
                return new_state
            else:
                # Re-raise non-memory errors
                raise

    def validate_code(self, state: State, config: RunnableConfig = None):
        """
        Validates the generated code by executing it in the specified environment.

        Parameters:
        state (State): The current state of the code generation process.
        config (RunnableConfig): Configuration for the runnable task.

        Returns:
        State: The updated state with the validation results.
        """

        assert self.env_dir is not None, "Environment directory not set"
        assert self.source_file is not None, "Source file not set"
        new_state = state
        curr_code = state.get("curr_code")
        validation_traces = state.get("validation_traces")
        if not validation_traces:
            new_state["validation_traces"] = []

        ### Insert code to source file. New code need to be inserted after line self.start_line and before self.end_line
        self.executor.apply_code(curr_code, self.source_file)
        status, n_tests, n_errors, output = self.executor.execute(self.env_dir)
        new_state["validation_traces"].append(
            (status, n_tests, n_errors, output))
        if status == -1:
            new_state["curr_status"] = "The above code is incorrect and got the following compilation error.\n" + output
        elif status == -2:
            new_state["curr_status"] = "The above code is incorrect and got time limit exceeded.\n" + output
        elif status == -3:
            new_state["curr_status"] = "The above code is incorrect and the following runtime error.\n" + output
        else:
            assert status == 0, f"Invalid status {status}"
            new_state["curr_status"] = "The above code is correct and passed all the tests."
        return new_state

    def code_repair(self, state: State, config: RunnableConfig = None):
        new_state = state
        new_state["n_iters"] += 1
        curr_spec = state.get("curr_code")
        assert curr_spec is not None, "Current code is not available"
        new_state["curr_code"], new_messages = self.code_repair_assistant.__call__(
            state, config=config)
        return new_state

    def test_output_prediction(self, state: State,  config: RunnableConfig = None) -> State:
        new_state = state
       
        
        new_state["curr_code"], new_messages = self.test_output_prediction_assistant.__call__(
            state, config=config)
        new_state["messages"] = new_messages 
        return state
    def execute_code(self, state: State, config: RunnableConfig = None):
      
        new_state = state
       
        new_state["curr_code"], new_messages = self.code_execution_assistant.__call__(
            state, config=config)
        new_state["messages"] = new_messages
        return new_state
    
    def analysis_condition(self, state: State):
        status, _, _, _ = state["validation_traces"][-1]
        if status == 0:
            print("Validation passed !!!")
            return "passed"

        if state["n_iters"] >= state["max_iters"]:
            print("Maximum iterations reached !!!")
            return "terminated"

        print("Validation failed !!!")
        return "failed"

    def basic_workflow(self):
        """
        Defines the basic workflow for code generation and validation.
        See worflows/basic.png for the workflow diagram.

        Returns:
        StateGraph: The compiled workflow.
        """

        builder = StateGraph(State)
        builder.add_node("gen_init_code", self.gen_init_code)
        builder.add_node("validate_code", self.validate_code)
        builder.add_edge(START, "gen_init_code")
        builder.add_edge("gen_init_code", "validate_code")
        builder.add_edge("validate_code", END)
        memory = MemorySaver()
        workflow = builder.compile(checkpointer=memory)

        return workflow
    
    def gen_code_workflow(self):    
        builder = StateGraph(State)
        builder.add_node("gen_init_code", self.gen_init_code)
        builder.add_edge(START, "gen_init_code")
        builder.add_edge("gen_init_code", END)
        memory = MemorySaver()
        workflow = builder.compile(checkpointer=memory)
        return workflow
        
    def self_repair_workflow(self):
        builder = StateGraph(State)
        builder.add_node("gen_init_code", self.gen_init_code)
        builder.add_node("validate_code", self.validate_code)
        builder.add_node("self-repair", self.code_repair)
        builder.add_edge(START, "gen_init_code")
        builder.add_edge("gen_init_code", "validate_code")
        builder.add_conditional_edges("validate_code", self.analysis_condition,
                                      {
                                          "passed": END,
                                          "failed": "self-repair",
                                          "terminated": END
                                      })
        builder.add_edge("self-repair", "validate_code")
        memory = MemorySaver()
        workflow = builder.compile(checkpointer=memory)
        return workflow
    
    def repair_workflow(self):
        builder = StateGraph(State)
        builder.add_node("validate_code", self.validate_code)
        builder.add_node("self-repair", self.code_repair)
        builder.add_edge(START, "validate_code")
        builder.add_conditional_edges(
            "validate_code", self.analysis_condition,  
            {
                "passed": END, 
                "failed": "self-repair",  
                "terminated": END  
            }
        )
        builder.add_edge("self-repair", "validate_code")
        memory = MemorySaver()
        workflow = builder.compile(checkpointer=memory)

        return workflow

    def test_output_predict_workflow(self):
    
        builder = StateGraph(State)
        builder.add_node("test_output_prediction", self.test_output_prediction) 
        builder.add_edge(START, "test_output_prediction")
        builder.add_edge("test_output_prediction", END)
        memory = MemorySaver()
        workflow = builder.compile(checkpointer=memory)
        return workflow

    def code_execution_workflow(self):
        """
        Workflow for code execution prediction.
        """
        builder = StateGraph(State)
        builder.add_node("execute_code", self.execute_code)
        builder.add_edge(START, "execute_code")
        builder.add_edge("execute_code", END)
        memory = MemorySaver()
        workflow = builder.compile(checkpointer=memory)
        return workflow




    def stream_query(self, query: dict, config: dict):
        """
        Streams the query through the workflow and prints events.

        Parameters:
        query (dict): The query to be processed.
        config (dict): Configuration for the workflow.
        """

        _printed = set()
        events = self.workflow.stream(query, config, stream_mode="values")
        for event in events:
            _print_event(event, _printed)
        
        last_event = event
        return last_event
    

    def model_invoke(self, description: str, template: str, function_name: str, test_cases: list, code: str, config: dict):
        """
        Invokes the model to generate code based on the description and template.

        Parameters:
        description (str): The description of the task.
        template (str): The template for the code.
        config (dict): Configuration for the workflow.
        """
        def extract_curr_code_from_output(path: str) -> str:
            with open(path) as f:
                data = json.load(f)
            print("Data loaded from file:", data)
            
            for msg in data.get("messages", []):
                if msg.get("idx") == 2 and msg.get("role") == "ai":
                    return msg["response"]
            
            raise ValueError("AI response with idx == 2 not found in output")

        if self.workflow_type == "repair":
            script_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(script_dir)
            file_path = os.path.join(parent_dir, self.output_file)
            curr_code = extract_curr_code_from_output(file_path)
        else:
            curr_code = None
        try: 
            formatted_testcases = "\n".join(
                    [f"```\nassert {function_name}({tc}) == # TODO\n```" for tc in test_cases]
                )
            query = {
                    "description": description,
                    "template": template,
                    "function_name": function_name,
                    "formatted_testcases": formatted_testcases,
                    "lang" : self.language,
                    "n_iters": 0,
                    "max_iters": self.max_iters,
                    "code" : code,
                    "curr_code":curr_code
                } 

            return self.stream_query(query, config)
        except (RuntimeError, MemoryError) as e:
            # Check if error is memory-related (OOM)
            if "CUDA out of memory" in str(e) or "DefaultCPUAllocator" in str(e) or "memory" in str(e).lower():
                print(f"ERROR: Insufficient memory during code generation: {str(e)}")
                print("Unable to complete model invocation due to memory constraints")
            
                # Create a minimal event structure to return
                error_event = {
                    "n_iters": 0,
                    "curr_code": "# Failed to generate code due to memory constraints",
                    "validation_traces": [],
                    "code_traces": ["# Failed to generate code due to memory constraints"],
                    "messages": [
                        system("Memory error occurred during model invocation"),
                        human("The operation could not be completed due to insufficient memory"),
                        ai("Please try again with a smaller model or reduce input complexity")
                    ]
                }
                return error_event
            else:
                # Re-raise non-memory errors
                raise

    def generate(self, path: str, config: dict):
        """
        Generates code by processing the metadata and invoking the model.

        Parameters:
        path (str): The path to the directory containing the metadata.
        config (dict): Configuration for the workflow.
        """

        print("generate is called")

        meta_path = os.path.join(path, "meta.json")
        with open(meta_path) as f:
            meta_data = json.load(f)

        tmp_dir = os.path.join(_TMP_DIR, meta_data["name"])
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)

        env_dir = os.path.join(tmp_dir, self.language)
        shutil.copytree(self.executable_dir, env_dir)
        self.env_dir = env_dir
    
        # Extract function name
        haskell_template = meta_data["haskell_template"]
        function_name = haskell_template.split("::")[0].strip()

        if self.language == "haskell":
            test_path = os.path.join(path, "haskell_tests/Main.hs")
            shutil.copy(test_path, os.path.join(env_dir, "app", "Main.hs"))
            self.source_file = os.path.join(env_dir, "app", "Main.hs")
        elif self.language == "ocaml":
            test_path = os.path.join(path, "ocaml_tests/main.ml")
            shutil.copy(test_path, os.path.join(env_dir, "bin", "main.ml"))
            self.source_file = os.path.join(env_dir, "bin", "main.ml")
        elif self.language == "scala":
            test_path = os.path.join(path, "scala_tests/MySuite.scala")
            shutil.copy(test_path, os.path.join(env_dir, "src/test/scala/MySuite.scala"))
            self.source_file = os.path.join(env_dir, "src/main/scala/Main.scala")
        elif self.language == "java":   
            test_path = os.path.join(path, "java_tests/Main.java")
            test_path_sol = os.path.join(env_dir, "src", "test", "java", "MainTest.java")
            shutil.copy(test_path, test_path_sol)
            self.source_file = os.path.join(env_dir, "src", "main", "java", "Main.java")
           
            #Update class name and the method
            with open(test_path_sol, "r") as file:
                test_code = file.read()
            updated_test_code = test_code.replace("Main ", "MainTest").replace(f" {function_name}",f"Main.{function_name}")
            with open(test_path_sol, "w") as file:
                file.write(updated_test_code)

        else:
            print("failed to find language")
            print(self.language)
            raise NotImplementedError(
                f"Language {self.language} not implemented")

        description = meta_data["task_description"]
        function_signature = meta_data[f"{self.language}_template"]
        function_name = function_signature.split()[0] if function_signature else "unknown_function"
        test_cases = [case["input"] for case in meta_data.get("public_test_cases", [])]
        code = ''
        if self.workflow_type == "code-execution":
            code =  meta_data[f"{self.language}_code"]
        print(f"Description: {description}")
        print(f"Function: {function_signature}")
        print("Language: ", self.language)

        print("invoking model")
        last_event = self.model_invoke(description, function_signature, function_name, test_cases, code, config)
        print("invoking model finished")

        messages = []

        for idx, mess in enumerate(last_event["messages"]):
            
            if type(mess) == system.SystemMessage:
                role = "system"
            elif type(mess) == human.HumanMessage:
                role = "human"
            elif type(mess) == ai.AIMessage:
                role = "ai"
            else:
                raise ValueError(f"Unknown message type: {type(mess)}")
            
            messages.append({
                "idx": idx,
                "response": mess.content,
                "response_metadata": mess.response_metadata,
                "role": role
            })
       
                
        return {
            "code_traces": last_event["code_traces"],
            "messages": messages
        }

   
