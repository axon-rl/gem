<div align="center">

# GEM: A Gym for Generalist LLMs


[![Notion blog](https://img.shields.io/badge/Notion-000000?style=for-the-badge&logo=notion&logoColor=white)](https://axon-rl.notion.site/gem) 
[![🌐 Axon-RL](https://img.shields.io/badge/-AxonRL%20project-5865F2?style=for-the-badge)](https://axon-rl.github.io/) 
[![Hugging Face Collection](https://img.shields.io/badge/AxonRL-fcd022?style=for-the-badge&logo=huggingface&logoColor=000&labelColor)](https://huggingface.co/axon-rl) 
[![Documentation](https://img.shields.io/badge/Documentation-blue?style=for-the-badge&logo=readthedocs&logoColor=white)](https://axon-rl.github.io/gem/)

<div align="center" style="font-family: Arial, sans-serif;">
  <p>
    <a href="#links" style="text-decoration: none; font-weight: bold;">Links</a> •
    <a href="#installation" style="text-decoration: none; font-weight: bold;">Installation</a> •
    <a href="#interface" style="text-decoration: none; font-weight: bold;">Interface</a> •
    <a href="#integration-examples" style="text-decoration: none; font-weight: bold;">Integration Examples</a> •
    <a href="#acknowledgement" style="text-decoration: none; font-weight: bold;">Acknowledgement</a> •
    <a href="#citation" style="text-decoration: none; font-weight: bold;">Citation</a> 
  </p>
</div>

</div>

We’re entering the era of experience, where LLM training moves beyond static datasets, towards LLM agents learning from experience gathered in complex, expressive environments. As a step towards this we introduce **GEM**, our open-source **G**eneral **E**xperience **M**aker.

Like OpenAI [Gym](https://github.com/openai/gym) for traditional RL, GEM is a dedicated environment simulator for the age of LLMs. GEM offers a diverse range of environments with clean, standardized interfaces, making it easy to integrate with existing RL training frameworks (Oat, Verl, etc.). In addition, GEM features tool integration, flexible and easy-to-modify wrappers, async vectorized environment execution to maximize throughput, multi-environment training, and more … everything you need to make LLM agent RL training simple.


## Links
* **GEM: Gym for Generalist LLMs**
  * 📜 [Blog](https://axon-rl.notion.site/gem)
  * 🚀 [Release tweet](https://x.com)
  * 📄 [Documentation](https://axon-rl.github.io/gem/)

## Installation

We recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management:

```bash
uv pip install -e .
```

To use the `search` tool, run: 
```bash
uv pip install -e .[search]
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```

## Interface
GEM's interface closely follows Gym's API. Here's an example using the "game:GuessTheNumber-v0" environment: 

```python 
import gem

# List all supported environments
gem.print_envs()

# Initialize the environment
env = gem.make("game:GuessTheNumber-v0")

# Reset the environment to generate the first observation
observation, info = env.reset()

# Start the agent-environment loop
while True:
    action = env.sample_random_action() # insert policy here, e.g.,
    # (pseudocode) action = llm.generate(observation)

    # apply action and receive next observation, reward
    # and whether the episode has ended
    next_observation, reward, terminated, truncated, info = env.step(action)
    print("OBS", observation)
    print("ACT", action)

    # update the policy (online) here
    # e.g., policy = learn(policy, observation, action, reward, info)

    observation = next_observation
    # Exit when the episode terminates
    if terminated or truncated:
        break
```

### Tool Integration Examples

Below are examples for enabling tools within environments.

**Example using the Python tool:**
```python
from transformers import AutoTokenizer

import gem
from gem.tools.python_code_tool import PythonCodeTool
from gem.tools.tool_env_wrapper import ToolEnvWrapper
from gem.wrappers.wrapper_factory import WRAPPER_FACTORY

env = gem.make("math:GSM8K")
tool = PythonCodeTool()
wrapped_env = ToolEnvWrapper(env, tools=[tool])
wrapped_env = WRAPPER_FACTORY["concat_chat"](
    wrapped_env, tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
)
obs, info = wrapped_env.reset()

# we ignore the obs and use a dummy action
dummy_action = "<think>Let me compare 9.9 and 0.11 using python.</think><python>print('9.9 > 9.11?', 9.9 > 9.11)</python>"
obs, reward, terminated, truncated, info = wrapped_env.step(dummy_action)
print(obs)
# continue to sample the next response given the tool results ...

wrapped_env.close()
```

**Example using the search tool:**
```python
# assume you have search server running

env = gem.make("game:GuessTheNumber-v0", max_turns=2)
tool = SearchTool(search_url="http://localhost:8000/retrieve", topk=2)
wrapped_env = ToolEnvWrapper(env, tools=[tool], max_tool_uses=1)
wrapped_env = WRAPPER_FACTORY['concat_chat'](wrapped_env, tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B"))
wrapped_env.reset()

dummy_action = "<think>I need to search for Python list comprehension examples</think><search>Python list comprehension examples</search>"
obs, reward, terminated, truncated, info = wrapped_env.step(dummy_action)
print(obs)
```

<details>
<summary>Click to get the complete runnable code</summary>

```python
import subprocess
import time

from transformers import AutoTokenizer

import gem
from gem.tools.search_tool import SearchTool
from gem.tools.tool_env_wrapper import ToolEnvWrapper
from gem.wrappers.wrapper_factory import WRAPPER_FACTORY

# start the search server
serp_api_key = "add you api key" # get api at https://serpapi.com/manage-api-key
server_process = subprocess.Popen([
    'python', '-m', 'gem.tools.search_engine.serp_search_server',
    '--search_url', 'https://serpapi.com/search',
    '--topk', '2', '--serp_api_key', serp_api_key
])
time.sleep(5)

# interact using search tool
env = gem.make("game:GuessTheNumber-v0", max_turns=2)
tool = SearchTool(search_url="http://localhost:8000/retrieve", topk=2)
wrapped_env = ToolEnvWrapper(env, tools=[tool], max_tool_uses=1)
wrapped_env = WRAPPER_FACTORY['concat_chat'](wrapped_env, tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B"))
wrapped_env.reset()

dummy_action = "<think>I need to search for Python list comprehension examples</think><search>Python list comprehension examples</search>"
obs, reward, terminated, truncated, info = wrapped_env.step(dummy_action)
print(obs)
```
</details>

## Integration Examples

We demonstrate how to leverage existing LLM RL infrastructure to train agents with GEM. First, we show how to train game agents using [Oat](https://github.com/sail-sg/oat). 

Before running the training, ensure you set up the development environment by following the [instructions](https://github.com/axon-rl/gem/tree/main/examples#training-with-oat). 

Run the following command to train an agent for the game environment `game:GuessTheNumber-v0`: 

```python 
python train.py \
    --env_id game:GuessTheNumber-v0 \
    --wrappers concat \
    --gamma 0.9 \
    --norm_adv \
    --gpus 8 \
    --gradient-checkpointing \
    --num_samples 1 \
    --rollout_batch_size 128 \
    --num_envs 2 \
    --rollout_batch_size_per_device 16 \
    --pi_buffer_maxlen_per_device 16 \
    --pretrain Qwen/Qwen3-1.7B-Base \
    --enable_prefix_caching \
    --collocate \
    --vllm_sleep \
    --vllm_gpu_ratio 0.45 \
    --rnd-seed \
    --learning_rate 0.000001 \
    --lr_scheduler constant \
    --lr_warmup_ratio 0 \
    --num_ppo_epochs 2 \
    --train_batch_size 128 \
    --train_batch_size_per_device 1 \
    --beta 0 \
    --max_model_len 12800 \
    --generate_max_length 4096 \
    --temperature 1.0 \
    --top_p 1 \
    --eval_steps -1 \
    --save_steps -1 \
    --eval_temperature 0.6 \
    --eval_top_p 0.95 \
    --eval_generate_max_length 4096 \
    --max_train 65000 \
    --max_save_num 30 \
    --use-wb \
    --wb-run-name oat-qwen3-1.7b-base-game:GuessTheNumber-v0 \
    --wb_project gem \
    --debug
```


We also provide sample code for math, code, and general QA in the [examples](https://github.com/axon-rl/gem/tree/main/examples) directory. In addition to Oat integration, you can find examples of RL training with Verl [here](https://github.com/axon-rl/gem/tree/main/examples#training-with-verl). 

## Roadmap

We plan to integrate (but not limited) the following environments soon:
- [ ] Terminal-bench
- [ ] SWE-Gym

## Contributing

We welcome all forms of contribution — from adding new environments to integrating additional training frameworks. We're planning to write a community-driven technical report, and major contributors will be recognized with authorship.

## Acknowledgement
* This work is supported by [Sea AI Lab](https://sail.sea.com/) for computing resources.
* Our code learns from and builds on several awesome projects such as [gym](https://github.com/openai/gym), [rllm](https://github.com/rllm-org/rllm), [TextArena](https://github.com/LeonGuertler/TextArena), [Search-R1](https://github.com/PeterGriffinJin/Search-R1), [ReasoningGym](https://github.com/open-thought/reasoning-gym).
* The training example code is built on [Oat](https://github.com/sail-sg/oat) and [Verl](https://github.com/volcengine/verl).
