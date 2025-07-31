<div align="center">

# GEM: Gym for Generalist LLMs

</div>

We’re entering the era of experience, where LLM training moves beyond static datasets, towards LLM agents learning from experience gathered in complex, expressive environments. As a step towards this we introduce **GEM**, our open-source **G**eneral **E**xperience **M**aker.

Like OpenAI [Gym](https://github.com/openai/gym) for traditional RL, GEM is a dedicated environment simulator for the age of LLMs. GEM offers a diverse range of environments with clean, standardized interfaces, making it easy to integrate with existing RL training frameworks (Oat, Verl, etc.). In addition, GEM features tool integration, flexible and easy-to-modify wrappers, async vectorized environment execution to maximize throughput, multi-environment training, and more … everything you need to make LLM agent RL training simple.

## Updates
* 01/08/2025: 🎉 We release our GEM codebase, along with a [blog](https://axon-rl.notion.site/gem) elaborating its key features, a baseline algorithm of multi-turn RL training, and a set of baselines. 

## Links
* **GEM: Gym for Generalist LLMs**
  * 🤗 [Blog](https://axon-rl.notion.site/gem)
  * 🚀 [Release tweet](https://x.com)

* **OAT: A research-friendly framework for LLM online alignment**
  * 💻 [Codebase](https://github.com/sail-sg/oat)

## Installation

We recommand using [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependency management:

```bash
uv pip install -e .

# sandbox: this is for code environments
conda install bubblewrap
```

To use `search` tool, do: 
```bash
pip install -e .[search]
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```

## Interface
GEM's interface closely follows Gym's API - here's an example using "game:GuessTheNumber-v0" environment: 

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

## Acknowledgements
* This work is supported by [Sea AI Lab](https://sail.sea.com/) for computing resources.
* Our code learns and builds based on several awesome projects such as [gym](https://github.com/openai/gym), (math ref project), (code ref project), [TextArena](https://github.com/LeonGuertler/TextArena), [Search-R1](https://github.com/PeterGriffinJin/Search-R1), [ReasoningGym](https://github.com/open-thought/reasoning-gym).
* The training example codes are built on [Oat](https://github.com/sail-sg/oat) and [Verl](https://github.com/volcengine/verl).

## Citation
If you find our works useful for your research, please consider citing:

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