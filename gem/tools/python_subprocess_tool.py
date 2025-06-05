# Adapted from https://github.com/TIGER-AI-Lab/verl-tool
from .base import BaseTool, register_tool
import regex as re
import subprocess
import os
import uuid
from typing import Tuple

# Timeout for code execution in seconds
TIMEOUT = 5

def check_forbidden_imports(code: str) -> bool:
    """
    Checks if the code contains imports of potentially dangerous packages.
    Args: code: Python code string to analyze
    Returns: Boolean indicating if the code contains forbidden imports
    """
    # List of potentially dangerous modules that could affect the host system
    forbidden_modules = [
        'subprocess', 'multiprocessing', 'threading',
        'socket', 'psutil', 'resource', 'ctypes'
    ]
    
    # Simple string-based check for import statements
    for module in forbidden_modules:
        if f"import {module}" in code or f"from {module}" in code:
            return True
    
    # Check for os.system, os.popen, and similar dangerous calls
    dangerous_patterns = [
        "os.system", "os.popen", "os.spawn", "os.fork", 
        "os.exec", "sys.exit", "os._exit", "os.kill"
    ]
    
    for pattern in dangerous_patterns:
        if pattern in code:
            return True
    
    return False
    
def execute_python(code: str, timeout: int=TIMEOUT) -> Tuple[str, bool]:
    """
    Execute Python code with a timeout.
    Args: code: Python code string to execute
    Returns: String containing execution output or error message
    """
    # Check for forbidden imports first
    if check_forbidden_imports(code):
        return "Execution blocked: Code contains potentially dangerous operations or imports.", True
    
    # Create a minimal environment instead of copying everything
    original_env = os.environ.copy()
    env = {}
    
    # Core system variables
    essential_vars = [
        "PATH", "HOME", "USER", "SHELL", 
        "LANG", "LC_ALL", "LC_CTYPE", "TERM",
        # Python-specific
        "PYTHONIOENCODING", "PYTHONUNBUFFERED", "PYTHONHASHSEED", "PYTHONDONTWRITEBYTECODE",
        # Runtime optimization
        "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS",
        # Temp directories
        "TMPDIR", "TEMP", "TMP",
        # Display if needed
        "DISPLAY", "XAUTHORITY"
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
        
        result = f"{stdout}\nError:\n{stderr}" if stderr else stdout
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

@register_tool
class PythonCodeTool(BaseTool):
    tool_type = "python_code"
    timeout = TIMEOUT
    
    def _parse_action(self, action: str) -> Tuple[str, bool]:
        """
        Parse the raw action string (which is the llm response) into an actual action and its contents.
        Ensures that the parsed code is valid and safe for execution.
        Args: action: Raw action string containing Python code
        Returns: Tuple containing the extracted code and a validity flag
        """
        # Try to find Python code in various formats
        all_valid_python_code = re.findall(r"<python>(.*?)</python>", action, re.DOTALL)
        
        if not all_valid_python_code:
            all_valid_python_code = re.findall(r"```\n?python(.*?)```", action, re.DOTALL)
        
        if len(all_valid_python_code) == 0:
            return "", False
        
        # use all the code blocks
        parsed_code = "\n".join([code.strip() for code in all_valid_python_code])
        
        return parsed_code, True
    
    def execute_action(self, action):
        """
        Execute the parsed action
        Args:
            trajectory_id: ID for tracking the action
            action: Raw action string
        Returns:
            Tuple containing observation, done flag, and validity flag
        """
        parsed_action, is_valid = self._parse_action(action)
        
        if not is_valid:
            # observation = "No valid Python code found. Please provide code in either <python>...</python> tags or ```python...``` code blocks."
            observation = ""
            done = False
            valid = False
        else:
            code_to_execute = parsed_action
            execution_result, has_error = execute_python(code_to_execute, self.timeout)
                
            execution_result = execution_result.lstrip(' \n')
                        
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
            elif action.strip(' \n').endswith("```") or "```python" in action:
                if action.count("```") % 2 == 0:
                    observation = "\n```output\n" + observation + "\n```\n"
                else:
                    observation = "output\n" + observation + "\n```\n"
            else:
                observation = "\n" + observation + "\n"

            done = False
            valid = True
        
        return observation, done, valid
        