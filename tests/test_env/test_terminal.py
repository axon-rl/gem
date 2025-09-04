import json
import os
import time

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
    dummy_action = dummy_action.replace("uv run convert.py", "uv run convert_.py")
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


def test_llm_inference(model: str = "Qwen/Qwen2.5-14B-Instruct"):
    import pdb

    import requests

    url = "http://localhost:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": "Bearer token"}
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

    with open("tests/test_env/terminal_agent_prompt.md", "r") as file:
        SYSTEM_PROMPT = file.read()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": obs,
                },
            ],
        },
    ]

    done = False
    while not done:
        data = {"model": model, "messages": messages, "temperature": 0.15}

        response = requests.post(url, headers=headers, data=json.dumps(data))
        action = response.json()["choices"][0]["message"]["content"]
        print("ACT", action)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print("NEXT_OBS", next_obs)
        pdb.set_trace()

        messages.append({"role": "assistant", "content": action})
        messages.append({"role": "user", "content": next_obs})

    env.close()


def test_openai(
    model: str = "gpt-5-nano",
    task_name: str = "csv-to-parquet",
    use_open_router: bool = False,
):
    import os

    from openai import OpenAI

    _api_key = (
        os.environ.get("OPENROUTER_API_KEY")
        if use_open_router
        else os.environ.get("OPENAI_API_KEY")
    )
    assert (
        _api_key
    ), "Please provide valid api key via env var: OPENROUTER_API_KEY | OPENAI_API_KEY"
    client = OpenAI(
        base_url=(
            "https://openrouter.ai/api/v1"
            if use_open_router
            else "https://api.openai.com/v1"
        ),
        api_key=_api_key,
    )

    task_path = f"./tasks/_eval/{task_name}"

    try:
        test_weights = json.load(open(os.path.join(task_path, "test_weights.json")))
    except Exception:
        test_weights = None

    env = DockerEnv(
        task_configs=[
            TaskConfig(
                task_name=f"test_{task_name}",
                task_path=task_path,
                test_weights=test_weights,
                max_retry=20,
            )
        ],
        container_config=ContainerConfig(),
    )
    obs, _ = env.reset()

    with open("tests/test_env/terminal_agent_prompt.md", "r") as file:
        SYSTEM_PROMPT = file.read()

    messages = [
        {"role": "developer", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": obs,
        },
    ]

    try:
        done = False
        episode = []
        while not done:
            if use_open_router:
                completion = client.chat.completions.create(
                    model=model, messages=messages
                )
                action = completion.choices[0].message.content
            else:
                response = client.responses.create(model=model, input=messages)
                action = response.output_text
            print("ACT", action)
            next_obs, reward, terminated, truncated, info = env.step(action)
            episode.append(
                {"observation": obs, "action": action, "reward": reward, "info": info}
            )
            done = terminated or truncated
            obs = next_obs
            print("NEXT_OBS", next_obs)
            print("REW", reward)
            print("-" * 20)
            messages.append({"role": "assistant", "content": action})
            messages.append({"role": "user", "content": next_obs})
        env.close()
        model = model.replace("/", "_")
        save_path = os.path.join(task_path, f"{model}-episode-{int(time.time())}.json")
        json.dump(episode, open(save_path, "w"), indent=4)
    except Exception as e:
        env.close()
        if episode:
            model = model.replace("/", "_")
            save_path = os.path.join(
                task_path, f"{model}-episode-{int(time.time())}.json"
            )
            json.dump(episode, open(save_path, "w"), indent=4)
        raise e


if __name__ == "__main__":
    fire.Fire(
        {
            "hello_world": test_hello_world,
            "llm_inference": test_llm_inference,
            "openai": test_openai,
        }
    )

    """Run with:
    python -m tests.test_env.test_terminal hello_world
    python -m tests.test_env.test_terminal llm_inference
    python -m tests.test_env.test_terminal openai
    python -m tests.test_env.test_terminal openai --model google/gemini-2.5-flash-lite --use_open_router
    """
