# OPENRLHF with GEM

In this document, we demonstrate how to integrate [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) with GEM to train your own agents. One of our key goals is to enable seamless integration with all major frameworks, allowing researchers to use their preferred tools while **easily comparing results across different setups**. We believe this level of interoperability is central to the mission of building a standardized suite of environments.

## Training with OpenRLHF

[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) is supported as the RL framework to integrate with GEM.

Before you start the experiments, you can build the docker-image following the instructions in the [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) repository.

Next, we provide example command lines to run experiments for training LLMs to perform math, code, language games, and general QA, as well as to use tools like Python or search for them.

> **_NOTE_**: All scripts below assume a single-node (1 GPU) setup. You should modify the arguments following the example below to customize the training on different hardware setups.

You can use the scripts:
- `run_count_letter.sh`
- `run_guess_number.sh`

for reasoning-gym and game environments respectively.

The corresponding agent function files are:
- `agent_func_single_turn.py`
- `agent_func_multi_turn.py`

Since [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) requires training data files, we also use GEM to generate prompt datasets specifically for integration with OpenRLHF. All prompts used for training are entirely sourced from GEM.