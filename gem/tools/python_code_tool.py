# Adapted from https://github.com/TIGER-AI-Lab/verl-tool
import os
import subprocess
import uuid
from typing import Tuple

import regex as re

from gem.tools.base_tool import BaseTool
from gem.utils.sandbox import check_forbidden_imports

# Timeout for code execution in seconds
TIMEOUT = 5

def get_python_output(
    code: str, timeout: int = TIMEOUT, return_traceback: bool = False
) -> Tuple[str, bool]:
    """
    Execute Python code with a timeout.
    Args: code: Python code string to execute
    Returns: String containing execution output or error message
    """
    # Check for forbidden imports first
    if check_forbidden_imports(code):
        return (
            "Execution blocked: Code contains potentially dangerous operations or imports.",
            True,
        )

    # Create a minimal environment instead of copying everything
    original_env = os.environ.copy()
    env = {}

    # Core system variables
    essential_vars = [
        "PATH",
        "HOME",
        "USER",
        "SHELL",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TERM",
        # Python-specific
        "PYTHONIOENCODING",
        "PYTHONUNBUFFERED",
        "PYTHONHASHSEED",
        "PYTHONDONTWRITEBYTECODE",
        # Runtime optimization
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        # Temp directories
        "TMPDIR",
        "TEMP",
        "TMP",
        # Display if needed
        "DISPLAY",
        "XAUTHORITY",
    ]

    # Copy only essential variables if they exist
    for var in essential_vars:
        if var in original_env:
            env[var] = original_env[var]

    # Explicitly set optimization variables
    env["OPENBLAS_NUM_THREADS"] = "1"

    if "PYTHONPATH" in env:
        del env["PYTHONPATH"]

    # set cwd to be a temp dir
    command = []
    cwd = "/tmp/python_code"
    if not os.path.exists(cwd):
        os.makedirs(cwd, exist_ok=True)
    # write code to a temp file
    # file_name = f"code_{hashlib.md5(code.encode()).hexdigest()}.py"
    file_name = f"code_{uuid.uuid4().hex}.py"
    file_path = os.path.join(cwd, file_name)
    with open(file_path, "w") as f:
        f.write(code)
    # command.extend(["python3", "-c", code])
    command.extend(["python3", file_path])
    has_error = False
    try:
        # Execute the command
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )

        stdout = result.stdout
        stderr = result.stderr.strip()
        if stderr:
            has_error = True
            if not return_traceback:
                # If we don't want the full traceback, just return the error message
                stderr = stderr.splitlines()[-1]

        result = f"{stdout}\n{stderr}" if stderr else stdout
        if result:
            result = result.strip()
    except subprocess.TimeoutExpired:
        has_error = True
        result = f"Execution timed out after {timeout} seconds.\n"
    # Clean up the temporary file
    try:
        os.remove(file_path)
    except Exception as e:
        pass
    return result, has_error


class PythonCodeTool(BaseTool):
    tool_type = "python_code"

    def __init__(self, timeout: int = TIMEOUT, return_traceback: bool = False):
        self.timeout = timeout
        self.return_traceback = return_traceback

    def _parse_action(self, action: str) -> Tuple[str, bool]:
        """
        Extract the first complete codeblock from the raw action string (llm response) (if possible).
        Args: action: Raw action string containing Python code
        Returns:
            parsed_code: First extracted Python code block as a string (or "" if no code block found).
            parsed_action: Text up to the end of the first code block (or whole action if no code block found).
            is_valid: A boolean indicating if a valid code block was found.
        """
        # Regex patterns to search for.
        patterns = [r"<python>(.*?)</python>", r"```\n?python(.*?)```"]

        parsed_code = None
        parsed_action = action
        is_valid = False
        prev_end = len(action)
        for pattern in patterns:
            # Search for the first occurrence of the pattern
            matches = re.search(pattern, action, re.DOTALL)
            if matches:
                is_valid = True
                if matches.end() <= prev_end:
                    parsed_code = matches.group(1).strip()
                    parsed_action = action[: matches.end()]
                    prev_end = matches.end()
        return parsed_code, parsed_action, is_valid

    def instruction_string(self) -> str:
        return (
            "You can execute Python code by wrapping it in <python>...</python> tags or "
            "using ```python...``` code blocks. "
        )

    def execute_action(self, action):
        """
        Execute the parsed action
        Args:
            trajectory_id: ID for tracking the action
            action: Raw action string
        Returns:
            Tuple containing observation, done flag, and validity flag
        """
        parsed_code, parsed_action, is_valid = self._parse_action(action)

        if not is_valid:
            # observation = "No valid Python code found. Please provide code in either <python>...</python> tags or ```python...``` code blocks."
            observation = ""
            done = False
        else:
            execution_result, has_error = get_python_output(
                parsed_code,
                timeout=self.timeout,
                return_traceback=self.return_traceback,
            )

            execution_result = execution_result.lstrip(" \n")

            # Format the result
            if "Execution timed out" in execution_result:
                observation = execution_result
            else:
                observation = f"{execution_result}"

            if action.endswith("```output"):
                observation = "\n" + observation + "\n```\n"
            elif action.endswith("</tool_call>"):
                observation = "\n```output\n" + observation + "\n```\n"
            elif action.endswith("<output>"):
                observation = "\n" + observation + "\n</output>\n"
            elif action.endswith("</python>") or "</python>" in action:
                observation = "\n<output>\n" + observation + "\n</output>\n"
            elif "<|calling system for feedback|>" in action:
                if "```python" in action:
                    observation = "\n```output\n" + observation + "\n```\n"
                elif "<python>" in action:
                    observation = "\n<output>\n" + observation + "\n</output>\n"
                else:
                    observation = "\n" + observation + "\n"
            elif action.strip(" \n").endswith("```") or "```python" in action:
                if action.count("```") % 2 == 0:
                    observation = "\n```output\n" + observation + "\n```\n"
                else:
                    observation = "output\n" + observation + "\n```\n"
            else:
                observation = "\n" + observation + "\n"

        return is_valid, observation, parsed_action
