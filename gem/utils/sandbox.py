import copy
import shlex
import subprocess
import sys
import uuid
from tempfile import NamedTemporaryFile
from typing import List, Optional

from gem.utils.constants import BASE_IMPORTS

DEFAULT_TIMEOUT = 10
CLI_ARG_SIZE_LIMIT = 1024 * 3
ERROR_MSG_PREFIX = "Failed to execute program: "


def subprocess_run(
    code: str,
    cmd_list: List[str],
    stdin: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
):
    if len(code) < CLI_ARG_SIZE_LIMIT:
        cmd_list.extend(["python", "-c", code])
        result = subprocess.run(
            cmd_list,
            input=stdin.encode() if stdin else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    else:
        with NamedTemporaryFile(
            mode="wb", prefix=uuid.uuid4().hex, suffix=".py"
        ) as tmp:
            tmp.write(code.encode())
            tmp.flush()
            cmd_list.extend(["--ro-bind", tmp.name, tmp.name, "python", tmp.name])
            result = subprocess.run(
                cmd_list,
                input=stdin.encode() if stdin else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=timeout,
            )
    stderr = result.stderr.decode().strip()
    stdout = result.stdout.decode()
    if result.returncode == 0:
        return True, stdout, stderr
    return False, stdout, stderr


def run_python(code: str, stdin: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT):
    command = """bwrap \
  --unshare-all \
  --ro-bind {python_env} /python_env \
  --setenv PATH "/python_env/bin:/usr/bin:/bin" \
  --ro-bind /lib /lib \
  --ro-bind /lib64 /lib64 \
  --ro-bind /usr/lib /usr/lib \
  --proc /proc \
  --dev /dev \
"""
    cmd_list = shlex.split(command.format(python_env=sys.prefix))
    try:
        # 1) Run the code without extra imports first
        run_success, stdout, stderr = subprocess_run(
            code, copy.deepcopy(cmd_list), stdin
        )
        if not run_success and "is not defined" in stderr:
            # 2) Fix the missing imports and run again
            code = BASE_IMPORTS + "\n" + code
            run_success, stdout, stderr = subprocess_run(code, cmd_list, stdin)
    except subprocess.TimeoutExpired:
        run_success, stdout, stderr = (
            False,
            "",
            f"Execution timed out after {timeout} seconds.\n",
        )
    return run_success, stdout, stderr
