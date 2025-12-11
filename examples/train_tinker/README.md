## Training with Tinker and Tinker Cookbook

The [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) provides a practical framework for fine-tuning LLMs through [Tinker](https://thinkingmachines.ai/tinker/), an API that abstracts away the complexities of distributed training. Tinker manages backend training workflows, while the Cookbook delivers a suite of high-level recipes and utilities to accelerate experimentation for researchers, developers, and practitioners. 

In this guide, we present two implementations leveraging Tinker:
1. Standard Training with Cookbook Utilities:
   We introduce an adapter layer that enables GEM to work seamlessly with the Tinker Cookbook’s training loop.
2. Custom Training via Low-level API:
   For advanced use cases, we provide a basic RL implementation that interacts directly with Tinker’s low-level API, enabling maximum customization.


### Getting Started

Before running the experiments, install the required libraries as follows:
```bash
# install gem
pip install -U gem-llm

# install tinker
pip install tinker

# install util libs
pip install transformers wandb termcolor chz

# export tinker api key
export TINKER_API_KEY=<your-tinker-api-key>

# (optional: install tinker-cookbook if you opt for the first approach)
git clone git@github.com:thinking-machines-lab/tinker-cookbook.git
cd ./tinker-cookbook
pip install -e .
```

### Using Tinker Cookbook
> Find the training curves in [[📈 WandB Logs](https://wandb.ai/cameron_chen/gem-tinker-cookbook)].

To train agents with the Tinker Cookbook, we implemented an adaption layer (`tinker_cookbook_adapter.py`) that exposes a GEM-compatible interface. The entry point is `tinker_cookbook_train.py`.

**Example 1: Training on Math Environments**

```bash 
python -m examples.train_tinker.tinker_cookbook_train env_id=math:Math12K groups_per_batch=64 group_size=16 learning_rate=2e-5 max_tokens=2048 model_name=Qwen/Qwen3-8B-Base env_kwargs_json='{"use_mp": false}'
```

Note:
- You may train on different math environments by simply changing the `env_id` argument.
- `env_kwargs_json='{"use_mp": false}'` is only required for math environments.

**Example 2: Training on Reasoning Gym**

```bash 
python -m examples.train_tinker.tinker_cookbook_train env_id=rg:simple_equations groups_per_batch=64 group_size=8 learning_rate=2e-5 max_tokens=2048 model_name=Qwen/Qwen3-8B-Base
```

### Using Tinker
> Find the training curves in [[📈 WandB Logs](https://wandb.ai/lkevinzc/gem-tinker_train)]

We can also directly integrate GEM with Tinker, bypassing the abstraction layer defined by tinker-cookbook. The entry point is `tinker_train.py`, which implements REINFORCE with Return Batch Normalization (introduced in our [paper](https://arxiv.org/pdf/2510.01051)).

**Example 1: Training on Math Environments**

```bash 
python -m examples.train_tinker.tinker_train model_name=Qwen/Qwen3-8B-Base env_id=math:DeepScaleR40K num_env=128 max_tokens=8192
```
* In this example we observe the increasing response length phenomenon (as in DeepSeek-R1-Zero) with LoRA training with rank 32!

**Example 2: Training on Math with Python Tool**

```bash
python -m examples.train_tinker.tinker_train env_id=math:Math12K num_env=128 max_tokens=2048 template=no model_name=meta-llama/Llama-3.1-8B-Instruct env_wrappers=python_tool_no_int_reward_last_line_error,concat_chat gamma=1
```

**Example 3: Training on Multi-turn Language Games**

```bash
python -m examples.train_tinker.tinker_train model_name=Qwen/Qwen3-8B-Base num_env=64 env_id=game:Sudoku-v0-easy max_tokens=1024 template=qwen3_game model_name=Qwen/Qwen3-8B-Base env_wrappers=concat
```

* Training is currently slow because in each turn the agent needs to think then act, and this process repeats sequentially. We can optimize it via prefix sharing, async sample/learn, etc.
