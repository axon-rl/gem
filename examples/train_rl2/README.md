## Training with RL2

[RL2](https://github.com/ChenmienTan/RL2) is supported as the RL framework to integrate with GEM.

Next, we provide example command lines to run experiments for training LLMs to perform math, code, language games, and general QA, as well as to use tools like Python or search for them.

> **_NOTE_**: All scripts below assume a single-node (8 GPUs) setup. You should modify the arguments following the example below to customize the training on different hardware setups.

You can use the scripts:
- `run_count_letter.sh`
- `run_guess_number.sh`

for reasoning-gym and game environments respectively.

### Installation

```bash
pip install -U gem-llm

pip install rl_square
```