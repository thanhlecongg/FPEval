from langchain_core.messages.base import BaseMessage
from langchain_core.messages.human import HumanMessage
from state import State
import psutil
import subprocess
from config import logger


def save_workflow_to_image(workflow, filename: str):
    """
    Saves the workflow as an image file.

    Parameters:
    workflow: The workflow to be saved.
    filename (str): The name of the file to save the image as.
    """

    try:
        image_data = workflow.get_graph().draw_mermaid_png()

        with open(filename, "wb") as f:
            f.write(image_data)

        logger.info(f"Image saved successfully as {filename}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")


def _print_event(event: dict, _printed: set, max_length=1500):
    """
    Prints the event details.

    Parameters:
    event (dict): The event to be printed.
    _printed (set): Set of printed message IDs to avoid duplicates.
    max_length (int): Maximum length of the message to be printed.
    """

    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            logger.debug(msg_repr)
            _printed.add(message.id)

    curr_code = event.get("curr_code")
    n_iters = event.get("n_iters")
    curr_status = event.get("curr_status")
    if curr_status:
        status_id = f"status_{n_iters}"
        if status_id not in _printed:
            logger.info(f"\n@@@@ Validation Status - Iter {n_iters} @@@@")
            logger.info(curr_status)
            logger.info("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            _printed.add(status_id)
            
    if curr_code:
        code_id = f"code_{n_iters}"
        if code_id not in _printed:
            logger.info(f"@@@ Extracted Code - Iter {n_iters} @@@@")
            logger.info(curr_code)
            _printed.add(code_id)

def update_messages(state: State, new_message: BaseMessage) -> State:
    """
    Updates the messages in the state with a new message.

    Parameters:
    state (State): The current state.
    new_message (BaseMessage): The new message to be added.

    Returns:
    State: The updated state with the new message.
    """

    messages = state["messages"] + [new_message]
    return {**state, "messages": messages}


def kill_process_tree(pid: int):
    """
    Kills a process tree given a process ID.

    Parameters:
    pid (int): The process ID of the parent process.
    """

    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass


def execute_command(command, timeout=None):
    """
    Executes a shell command with an optional timeout.

    Parameters:
    command (str): The command to be executed.
    timeout (int, optional): The timeout for the command execution.

    Returns:
    str: The output of the command execution.
    """

    try:
        process = subprocess.Popen(command,
                                   shell=True,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        try:
            output, error = process.communicate(timeout=timeout)
            logger.debug(f"Command output: {output}")
            logger.debug(f"Command error: {error}")
        except subprocess.TimeoutExpired:
            logger.warning("Command execution timed out")
            kill_process_tree(process.pid)
            return "Timeout"

        final_output = ""
        if output:
            final_output += output.decode("utf-8")
        if error:
            final_output += error.decode("utf-8")
        logger.debug(f"Final output: {output}")
        return final_output

    except subprocess.CalledProcessError as e:
        return e.output.decode("utf-8")
