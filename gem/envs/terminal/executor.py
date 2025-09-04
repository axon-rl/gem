import asyncio
from typing import Tuple


class DockerExecutor:
    """Execute commands using docker exec."""

    def __init__(self, container_name: str):
        self.container_name = container_name

    async def execute(self, cmd: str, timeout: int = 30) -> Tuple[str, int]:
        """Execute a command in the Docker container and return (output, return_code)."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "exec",
                self.container_name,
                "bash",
                "-c",
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            output = stdout.decode("utf-8", errors="replace")
            exit_code = proc.returncode or 0

            return output, exit_code

        except asyncio.TimeoutError:
            if proc:
                proc.kill()
            return (
                f"Command timed out after {timeout} seconds",
                124,
            )  # 124 is the standard timeout exit code
        except Exception as e:
            return f"Error executing command: {str(e)}", 1

    async def execute_background(self, cmd: str) -> None:
        """Execute a command in background in the Docker container."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "exec", "-d", self.container_name, "bash", "-c", cmd
            )
            await proc.wait()
        except Exception:
            # Background execution failures are silently ignored
            pass
