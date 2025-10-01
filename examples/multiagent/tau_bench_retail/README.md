# TAU-bench Retail - GEM MultiAgentEnv

Clean implementation of TAU-bench retail benchmark using GEM's MultiAgentEnv API.

**Performance**: 78/115 (67.8%) - Exceeds target of 60.4%

## Quick Start

```bash
export OPENAI_API_KEY="your-key"
python run_eval.py
```

## Files

- `tau_bench_env.py` - GEM MultiAgentEnv environment
- `tau_bench_agent.py` - Agent with OpenRouter-style tool calling
- `run_eval.py` - Evaluation runner (115 tasks)
- `assets/` - Data, tools, wiki, tasks
- `experiments/` - Research experiments (9 model combinations)

## Model Support

Edit `run_eval.py` to change models (lines 33-36):

```python
# OpenAI
model = "gpt-4o"
provider = "openai"

# Gemini via OpenRouter
model = "google/gemini-2.0-flash-001"
provider = "openrouter"

# DeepSeek via OpenRouter
model = "deepseek/deepseek-chat"
provider = "openrouter"

# Claude via OpenRouter
model = "anthropic/claude-3.5-sonnet"
provider = "openrouter"
```

For OpenRouter models:
```bash
export OPENROUTER_API_KEY="your-key"
```

## Architecture

**Environment** (`tau_bench_env.py`):
- Inherits from `gem.envs.multiagent.MultiAgentEnv`
- Single agent: "assistant" (user simulator managed internally)
- Implements `_process_actions()` and `observe()`
- Reward calculation matches original tau-bench exactly

**Agent** (`tau_bench_agent.py`):
- OpenRouter-style tool calling pattern
- Multi-provider support via litellm
- Handles tool calls and respond actions

**Tool Calling Pattern**:
```python
request = {
    "model": model,
    "tools": tools,
    "messages": messages
}

response = completion(custom_llm_provider=provider, **request)
messages.append(response.choices[0].message.model_dump())

for tool_call in response.choices[0].message.tool_calls:
    tool_name = tool_call.function.name
    tool_args = json.loads(tool_call.function.arguments)
    # Execute in environment
    result = env.step({"assistant": json.dumps({"name": tool_name, "kwargs": tool_args})})
    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
```

## Research Experiments

See `experiments/` directory for multi-agent experiments studying how user model strength affects agent performance.

```bash
cd experiments
./run_experiments.sh
```

Runs 9 experiments (gpt-4o, gpt-4o-mini, gemini-2.0-flash) and generates visualizations.
