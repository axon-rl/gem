# TAU-bench Retail - GEM MultiAgentEnv Integration

Clean implementation of TAU-bench retail benchmark using GEM's MultiAgentEnv API.

**Performance**: 78/115 (67.8%) - Exceeds target of 60.4%

## Setup

### 1. Clone TAU-bench Repository

```bash
# Clone the official TAU-bench repository
git clone https://github.com/sierra-research/tau-bench.git

# Option 1: Clone to the default location (within tau_bench_retail directory)
cd examples/multiagent/tau_bench_retail
git clone https://github.com/sierra-research/tau-bench.git

# Option 2: Clone anywhere and set environment variable
git clone https://github.com/sierra-research/tau-bench.git /path/to/tau-bench
export TAU_BENCH_PATH=/path/to/tau-bench
```

### 2. Install Dependencies
```bash
# Install GEM
cd /path/to/gem/
pip install -e .

# Install TAU-bench
cd /path/to/gem/examples/multiagent/tau_bench_retail/tau-bench
pip install -e .
```

### 3. Set API Keys

```bash
# Required for OpenAI models
export OPENAI_API_KEY="your-key"

# Optional: For OpenRouter models (Gemini, Claude, DeepSeek)
export OPENROUTER_API_KEY="your-key"
```

### 4. Run Evaluation

```bash
python run_eval.py
```

## Files

- `tau_bench_env.py` - GEM MultiAgentEnv environment wrapper
- `tau_bench_agent.py` - Agent with OpenRouter-style tool calling
- `run_eval.py` - Evaluation runner (115 test tasks)

## Model Support

Supported models via `run_eval.py`:
- OpenAI: `gpt-4o`
- OpenRouter: `google/gemini-2.0-flash-001`, `deepseek/deepseek-chat`, `anthropic/claude-3.5-sonnet`

For OpenRouter models:
```bash
export OPENROUTER_API_KEY="your-key"
```
