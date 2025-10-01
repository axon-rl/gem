# TAU-BENCH Retail Integration for GEM

## Overview

This is the official integration of TAU-BENCH Retail benchmark into GEM (Gym for LLM Agents). TAU-BENCH evaluates tool-augmented LLM agents on realistic customer service tasks in a retail environment.

## Directory Structure

```
multiagent/
└── tau_bench_retail/
    ├── assets/                # Original TAU-bench assets
    │   ├── data/              # users.json, orders.json, products.json
    │   ├── tools/             # Tool implementations for evaluation
    │   ├── tasks_test.py      # Test tasks (115 tasks)
    │   ├── tasks_train.py     # Training tasks (350 tasks)
    │   ├── tasks_dev.py       # Development tasks (25 tasks)
    │   ├── wiki.md            # Agent policy documentation
    │   └── rules.py           # Evaluation rules
    ├── tau_bench_env.py       # TAU-bench environment for GEM
    ├── tau_benchmark.py       # Benchmark runner with OpenAI API
    └── run_benchmark.sh       # Execution script
```

## Quick Start

```bash
cd tau_bench_retail
export OPENAI_API_KEY="your-key-here"
./run_benchmark.sh
```

## Key Features

- **Real TAU-BENCH Tasks**: Uses actual TAU-bench retail tasks with 115 test tasks
- **GEM-Native Integration**: Clean integration without external TAU-bench dependencies
- **User Clarity Analysis**: Measures impact of clear vs vague user instructions
- **Pass@k Evaluation**: Standard Pass@1, Pass@2, Pass@3, Pass@4 metrics
- **OpenAI API Compatible**: Works with GPT-4o and other OpenAI models

## Implementation Details

### Architecture

1. **tau_bench_env.py**:
   - Defines Task and Action dataclasses locally
   - Loads TAU-bench tasks using monkey-patching for imports
   - Provides OpenAI-compatible tool definitions
   - Uses TAU-bench wiki as system prompt
   - Evaluates tool calls against expected actions

2. **tau_benchmark.py**:
   - Uses OpenAI API for LLM agent
   - Tests with clear and vague user instructions
   - Computes Pass@k metrics
   - Generates visualization of results

### User Clarity Impact

The benchmark tests two user types:
- **Clear Users**: Original TAU-bench instructions with full details
- **Vague Users**: Modified instructions with partial information removed

Example transformation:
```
Clear: "Cancel order #W6619432 because it's no longer needed"
Vague: "Cancel order #W661... and need help with something"
```

## Expected Results

| Model | Clear Users Pass@1 | Vague Users Pass@1 | Performance Drop |
|-------|-------------------|-------------------|------------------|
| GPT-4o | ~0.60-0.70 | ~0.35-0.45 | ~35-40% |
| Claude-3.5 (Paper) | 0.692 | - | - |

## TAU-BENCH Assets Used

- **Data Files**: Users, orders, and products JSON files
- **Task Definitions**: Test (115), train (350), and dev (25) tasks
- **Tool Implementations**: 16 customer service tools for evaluation
- **Wiki & Rules**: Agent policy and evaluation criteria

## Tools Available

The retail environment provides 16 tools:
- **Order Management**: cancel_pending_order, return_delivered_order_items, exchange_delivered_order_items
- **User Identification**: find_user_id_by_email, find_user_id_by_name_zip
- **Information Retrieval**: get_order_details, get_product_details, get_user_details
- **Order Modification**: modify_pending_order_address, modify_pending_order_items, modify_pending_order_payment
- **User Management**: modify_user_address
- **Support**: transfer_to_human_agents, list_all_product_types
- **Utilities**: think, calculate

## Running Custom Evaluations

```python
from tau_bench_env import TauRetailGEMEnv
from tau_benchmark import TauBenchmark

# Load environment
env = TauRetailGEMEnv(task_split="test")  # or "train", "dev"
print(f"Loaded {env.get_task_count()} tasks")

# Run benchmark
benchmark = TauBenchmark(model="gpt-4o")
results = benchmark.run_benchmark(
    num_tasks=20,  # Number of tasks to evaluate
    k_attempts=4   # Attempts per task for Pass@k
)
```

## Citation

If you use this benchmark, please cite:

```bibtex
@article{taubench2024,
  title={TAU-bench: A Benchmark for Tool-Agent-User Interaction},
  year={2024}
}

@article{gem2024,
  title={GEM: Gym for LLM Agents},
  year={2024}
}
```

## License

This integration follows the licenses of both GEM and TAU-BENCH projects.