# Multi-Agent Examples for GEM

This directory contains multi-agent environment examples using GEM's MultiAgentEnv framework.

## TAU-BENCH Retail Integration

The `tau_bench_retail/` directory contains the official integration of TAU-BENCH Retail benchmark into GEM. TAU-BENCH evaluates tool-augmented LLM agents on realistic customer service tasks in a retail environment.

### Setup

1. Clone the TAU-bench repository:
```bash
cd tau_bench_retail
git clone https://github.com/sierra-research/tau-bench.git
```

2. Set your API key:
```bash
export OPENAI_API_KEY="your-key-here"
```

3. Run the evaluation:
```bash
python run_eval.py
```

### Directory Structure

```
multiagent/
└── tau_bench_retail/
    ├── tau_bench_env.py       # GEM environment wrapper for TAU-bench
    ├── tau_bench_agent.py     # Agent with tool-calling capabilities
    ├── run_eval.py            # Evaluation script
    └── tau-bench/             # Cloned TAU-bench repository (git ignored)
        └── tau_bench/
            └── envs/
                └── retail/    # TAU-bench retail assets
                    ├── data/  # JSON data files
                    ├── tools/ # Tool implementations
                    ├── tasks_*.py  # Task definitions
                    └── wiki.md     # Agent policy
```

## Performance

TAU-bench Retail: **78/115 (67.8%)**

## Available Tools

16 customer service tools including order management, user identification, information retrieval, and support functions.