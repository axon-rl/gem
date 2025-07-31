<div align="center">

# GEM: Gym for Generalist LLMs

</div>

We’re entering the era of experience, where LLM training moves beyond static datasets, towards LLM agents learning from experience gathered in complex, expressive environments. As a step towards this we introduce **GEM**, our open-source **G**eneral **E**xperience **M**aker.

Like OpenAI [Gym](https://github.com/openai/gym) for traditional RL, GEM is a dedicated environment simulator for the age of LLMs. GEM offers a diverse range of environments with clean, standardized interfaces, making it easy to integrate with existing RL training frameworks (Oat, Verl, etc.). In addition, GEM features tool integration, flexible and easy-to-modify wrappers, async vectorized environment execution to maximize throughput, multi-environment training, and more … everything you need to make LLM agent RL training simple.


## Links
* **GEM: Gym for Generalist LLMs**
  * 📄 [Blog](https://axon-rl.notion.site/gem)
  * 🚀 [Release tweet](https://x.com)

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

Example using the Python tool: 
```python

```

Example using the search tool: 
```python

```

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


We also provide sample code for math, code, and general QA in [examples](https://github.com/axon-rl/gem/tree/main/examples). In addition to Oat integration, you can find examples of RL training with Verl [here](https://github.com/axon-rl/gem/tree/main/examples#training-with-verl). 


## Acknowledgements
* This work is supported by [Sea AI Lab](https://sail.sea.com/) for computing resources.
* Our code learns from and builds on several awesome projects such as [gym](https://github.com/openai/gym), (math ref project), (code ref project), [TextArena](https://github.com/LeonGuertler/TextArena), [Search-R1](https://github.com/PeterGriffinJin/Search-R1), [ReasoningGym](https://github.com/open-thought/reasoning-gym).
* The training example code is built on [Oat](https://github.com/sail-sg/oat) and [Verl](https://github.com/volcengine/verl).

## Citation
If you find our work useful for your research, please consider citing:

- Our blog that releases GEM: 
    ```bibtex
    @misc{liu2025gem,
    title={GEM: A Gym for Generalist LLMs},
    author={},
    year={2025},
    howpublished={\url{https://axon-rl.notion.site/gem}},
    note={Notion Blog},
    }
    ```

- The training framework for implementing the baseline algorithm: 
    ```bibtex
    @misc{liu2024oat,
    title={OAT: A research-friendly framework for LLM online alignment},
    author={Liu, Zichen and Chen, Changyu and Wan, Xinyi and Du, Chao and Lee, Wee Sun and Lin, Min},
    year={2024},
    howpublished={\url{https://github.com/sail-sg/oat}},
    }
    ```