<div align="center">

# 🌍 GEM: A Gym for Agentic LLMs


[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2510.01051) [![Notion blog](https://img.shields.io/badge/Notion-000000?style=for-the-badge&logo=notion&logoColor=white)](https://axon-rl.notion.site/gem) 
[![🌐 Axon-RL](https://img.shields.io/badge/-AxonRL%20project-5865F2?style=for-the-badge)](https://axon-rl.github.io/) 
[![Hugging Face Collection](https://img.shields.io/badge/AxonRL-fcd022?style=for-the-badge&logo=huggingface&logoColor=000&labelColor)](https://huggingface.co/axon-rl) 
[![Documentation](https://img.shields.io/badge/Documentation-blue?style=for-the-badge&logo=readthedocs&logoColor=white)](https://axon-rl.github.io/gem/)

</div>

## Overview

We’re entering the **era of experience**, where large language models (LLMs) learn not just from static datasets, but from *interactive experience* gathered in complex, expressive environments.

As a step toward this, we introduce **GEM** — a **G**eneral **E**xperience **M**aker for LLMs — an open-source environment suite designed for training *agentic LLMs* via online reinforcement learning.

Like [OpenAI Gym](https://github.com/openai/gym) for traditional RL, GEM provides a standardized API and a growing collection of diverse environments. It is **training framework-agnostic** and supports seamless integration with six popular RL training frameworks including [Oat](https://github.com/sail-sg/oat) and [Tinker](https://github.com/thinking-machines-lab/tinker), offering:

* 🧩 Clean, composable environment APIs
* ⚙️ Async vectorized execution for high-throughput simulation
* 🔧 Tool integration & custom wrappers
* 🧠 Multi-environment training
* 🎈 Ready-to-use benchmark environments and algorithms

## Links
  * 📜 [Initial Blog](https://axon-rl.notion.site/gem)
  * 🚀 [Blog release tweet](https://x.com/zzlccc/status/1951358948587741295)
  * 📄 [Paper](https://arxiv.org/pdf/2510.01051)
  * 📘 [Documentation](https://axon-rl.github.io/gem/)

## Installation

```bash
pip install -U gem-llm
```

Or install from source for the latest version:

```bash
git clone https://github.com/axon-rl/gem.git
cd gem
pip install -e .
```

Please check [Getting Started](./GETTING_STARTED.md) for more setup details.

🔥 You can jump into [examples](./examples/) to quickly start your agentic RL training with GEM & your favorite training framework.

## Interface
GEM's interface closely follows OpenAI-Gym's API. Here's an example using the `game:GuessTheNumber-v0` environment: 

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

## Features

1. Environments consist of tasks and (optional) tools. Tool-calling is achieved via an environment wrapper, as demonstrated [here](./GETTING_STARTED.md#tool-integration-examples).
2. GEM is training framework-agnostic, and we demonstrate its integration with six popular RL training frameworks.
3. We provide implementations and benchmarking results for different algorithms across a diverse set of environments.

### Supported Tasks

<div align="center">

| Category                   | Example Environments                              | Description                                      |
| -------------------------- | ------------------------------------------------- | ------------------------------------------------ |
| **Games**                  | `game:GuessTheNumber-v0-hard`, `game:Sudoku-v0-easy`        | Classic language games   |
| **Math**           | `math:Math12K`, `math:DeepScaleR40K`        | Mathematical reasoning |
| **Code**           | `code:CodeContest`, `code:Taco8k`        | Competitive coding |
| **QA**           | `qa:NaturalQuestions`, `qa:HotpotQA`            | Knowledge-intensive question answering             |
| **ReasoningGym**   | `rg:arc_1d`, `rg:letter_counting`           | Diverse synthetic reasoning tasks       |

</div>

### Supported Tools

<div align="center">

| Tool                            | Description                                      |
| -------------------------- | ------------------------------------------------ |
| **Python**                         | Python code executor that parses code blocks, executes them, and returns outputs   |
| **Search**                | Calls a search engine to retrieve documents for any query
| **MCP**            | Calls the general MCP API to train tool-use agents |

</div>


### Supported Frameworks

<div align="center">

| Framework                            | Description                                      |
| -------------------------- | ------------------------------------------------ |
| **[Oat](https://github.com/sail-sg/oat)**                         | vLLM + DeepSpeed, modular, no ray   |
| **[Tinker](https://github.com/thinking-machines-lab/tinker)**                | SDK provided by Thinking Machines, frees you from infra issues |
| **[Verl](https://github.com/volcengine/verl)**            | Support diverse backends, models, and algorithms |
| **[RL2](https://github.com/ChenmienTan/RL2)**            | SGLang + FSDP, no ray, easy to hack |
| **[ROLL](https://github.com/alibaba/ROLL)**            | Support diverse backends, models, and algorithms |
| **[OpenRLHF](https://github.com/alibaba/ROLL)**            | Support diverse backends, models, and algorithms |

</div>

Examples of training agents on GEM environments with all above frameworks can be found in [here](./examples/)!


### Supported Algorithms

<div align="center">

| Algorithm                            | Description                                      |
| -------------------------- | ------------------------------------------------ |
| **REINFORCE**                         | A general policy gradient algorithm that can be applied to single- and multi-turn environments |
| **GRPO**                | Mainly for bandits (single-turn), using group advantage normalization |
| **PPO**            | Learns a turn-level critic to compute generalized advantage estimation (GAE) |
| **REINFORCE + ReBN**            | REINFORCE with return batch normalization as introduced in our paper |

</div>

Please check out [our paper](https://arxiv.org/pdf/2510.01051) for a more detailed description for each algorithm and empirical results showing their tradeoffs.

## Contributing

We welcome all forms of contribution — from adding new environments to integrating additional training frameworks. We're planning to write a community-driven technical report, and major contributors will be recognized with authorship. Join [discord](https://discord.gg/AfXVkEphzD) to discuss more!

## Acknowledgement
* This work is supported by [Sea AI Lab](https://sail.sea.com/) for computing resources.
* Our code learns from and builds on several awesome projects such as [gym](https://github.com/openai/gym), [rllm](https://github.com/rllm-org/rllm), [TextArena](https://github.com/LeonGuertler/TextArena), [Search-R1](https://github.com/PeterGriffinJin/Search-R1), [ReasoningGym](https://github.com/open-thought/reasoning-gym).
* The training example code is built on [Oat](https://github.com/sail-sg/oat), [Tinker](https://github.com/thinking-machines-lab/tinker), [Verl](https://github.com/volcengine/verl), [RL2](https://github.com/ChenmienTan/RL2), [ROLL](https://github.com/alibaba/ROLL), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF).


## Citation
If you find our works useful for your research, please consider citing:

* GEM paper (please prioritize citing the paper unless you believe the blog is a better fit):
  ```bibtex
  @article{liu2025gem,
    title={GEM: A Gym for Agentic LLMs},
    author={Liu, Zichen and Sims, Anya and Duan, Keyu and Chen, Changyu and Yu, Simon and Zhou, Xiangxin and Xu, Haotian and Xiong, Shaopan and Liu, Bo and Tan, Chenmien and others},
    journal={arXiv preprint arXiv:2510.01051},
    year={2025}
  }
  ```

* GEM blog:
  ```bibtex
  @misc{liu2025gemblog,
    title={GEM: A Gym for Generalist LLMs},
    author={Liu, Zichen and Sims, Anya and Duan, Keyu and Chen, Changyu and Yang, Diyi and Lee, Wee Sun and Lin, Min},
    year={2025},
    howpublished={\url{https://axon-rl.notion.site/gem}},
    note={Notion Blog},
  }
  ```
