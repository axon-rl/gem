import json
import os

import fire

from gem.envs.terminal.docker_env import ContainerConfig, DockerEnv, TaskConfig


def test_hello_world():
    task_id = "./tasks/_eval/csv-to-parquet"
    env = DockerEnv(
        task_configs=[
            TaskConfig(
                task_name="test",
                task_path=task_id,
                instruction="Convert the file '/app/data.csv' into a Parquet file named '/app/data.parquet'. The CSV file contains sample data with headers.",
                test_weights=json.load(
                    open(os.path.join(task_id, "test_weights.json"))
                ),
            )
        ],
        container_config=ContainerConfig(),
    )
    obs, _ = env.reset()
    print("OBS", obs)
    dummy_action = open(os.path.join(task_id, "solution.sh")).read()
    dummy_action = f"I will write a bash script for it.\n<bash>{dummy_action}</bash>"
    print("ACT", dummy_action)
    next_obs, reward, _, _, info = env.step(dummy_action)
    print("NEXT_OBS", next_obs)
    print("REWARD", reward)
    print("INFO", info)

    next_obs, reward, _, _, info = env.step("<summary>I have done the job.</summary>")
    print("NEXT_OBS", next_obs)
    print("REWARD", reward)
    print("INFO", info)

    env.close()


if __name__ == "__main__":
    fire.Fire({"hello_world": test_hello_world})

    """Run with:
    python -m tests.test_env.test_terminal hello_world
    """
