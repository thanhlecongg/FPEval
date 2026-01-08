from typing import Annotated, Literal, Optional, Tuple
# from typing_extensions import TypedDict
from typing import TypedDict
from langgraph.graph.message import AnyMessage, add_messages


class State(TypedDict):
    """
    Represents the state of the code generation process.

    Attributes:
    messages (list[AnyMessage]): List of messages exchanged during the process.
    description (str): Description of the task.
    lang (str): Programming language for code generation.
    curr_code (str): Current code being generated.
    curr_status (Tuple[int, int, str]): Current status of the code execution.
    n_iters (int): Number of iterations performed.
    template (str): Template for the code generation.
    validation_traces (list[Tuple[int, int, int, str]]): List of validation traces.
    code_traces (list[str]): List of code traces.
    code (str): The final generated code.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    description: str
    lang: str
    curr_code: str
    curr_status: Tuple[int, int, str]
    n_iters: int
    template: str
    validation_traces: list[Tuple[int, int, int, str]]
    code_traces: list[str]
    max_iters: int
    formatted_testcases: str
    code: str

