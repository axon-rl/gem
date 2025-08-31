# Multi-Agent Environment Examples

This directory contains examples demonstrating the multi-agent environment capabilities in GEM.

## Examples

### 1. User-Tool Interaction (`user_tool_interaction.py`)

Demonstrates a sequential (AEC) multi-agent environment where a user agent interacts with a tool agent to accomplish tasks.

**Features:**
- Sequential turn-based interaction
- User agent with multiple strategies (scripted, LLM, human)
- Tool agent with search, Python execution, and response capabilities
- Conversation management and termination handling

**Run:**
```bash
python user_tool_interaction.py
```

### 2. Collaborative Question Answering (`collaborative_qa.py`)

Shows a parallel multi-agent environment where multiple specialized agents work together to answer questions.

**Features:**
- Parallel execution with all agents acting simultaneously
- Three specialized agents: Researcher, Validator, and Synthesizer
- Shared information board for collaboration
- Reward shaping for successful collaboration

**Run:**
```bash
python collaborative_qa.py
```

## Key Concepts

### AEC (Agent Environment Cycle) Environments

In AEC environments, agents take turns acting sequentially:

```python
from gem.multiagent import AECEnv

class MyAECEnv(AECEnv):
    def step(self, action):
        # Process action for current agent
        # Automatically advance to next agent
        pass
    
    def observe(self, agent):
        # Get observation for specific agent
        pass
```

Usage:
```python
env = MyAECEnv()
obs, info = env.reset()

for agent in env.agent_iter():
    observation, reward, terminated, truncated, info = env.last()
    if terminated or truncated:
        action = None
    else:
        action = policy(observation, agent)
    env.step(action)
```

### Parallel Environments

In Parallel environments, all agents act simultaneously:

```python
from gem.multiagent import ParallelEnv

class MyParallelEnv(ParallelEnv):
    def step(self, actions):
        # Process actions from all agents at once
        # Return results for all agents
        pass
```

Usage:
```python
env = MyParallelEnv()
observations, infos = env.reset()

while env.agents:
    actions = {agent: policy(observations[agent]) for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
```

### Agent Types

GEM provides base agent classes for building custom agents:

```python
from gem.multiagent.agents import BaseAgent

class MyAgent(BaseAgent):
    def act(self, observation):
        # Generate action based on observation
        return action
    
    def reset(self):
        # Reset agent state
        pass
```

Pre-built agents:
- `UserAgent`: Simulates user interactions with various strategies
- `ToolAgent`: Executes tools and APIs based on requests

### Environment Registration

Register multi-agent environments just like single-agent ones:

```python
from gem import register, make

register(
    "MyMultiAgentEnv-v0",
    entry_point="path.to:MyMultiAgentEnv",
    kwargs={"param": value}
)

env = make("MyMultiAgentEnv-v0")
```

### Conversion Between APIs

Convert between AEC and Parallel interfaces:

```python
from gem.multiagent.conversions import aec_to_parallel, parallel_to_aec

# Convert AEC to Parallel
aec_env = MyAECEnv()
parallel_env = aec_to_parallel(aec_env)

# Convert Parallel to AEC
parallel_env = MyParallelEnv()
aec_env = parallel_to_aec(parallel_env)
```

## Creating Your Own Multi-Agent Environment

1. **Choose the appropriate API**: 
   - Use AEC for turn-based, sequential scenarios
   - Use Parallel for simultaneous action scenarios

2. **Define agents and their roles**:
   - Set `possible_agents` and `agents` lists
   - Define observation and action spaces per agent

3. **Implement core methods**:
   - `reset()`: Initialize environment state
   - `step()`: Process agent actions
   - `observe()` (AEC only): Get agent observations

4. **Manage agent lifecycle**:
   - Track terminations and truncations
   - Remove dead agents when appropriate
   - Handle rewards and information

5. **Test your environment**:
   - Ensure proper agent coordination
   - Verify termination conditions
   - Check reward distribution

## Advanced Features

### Inter-Agent Communication

Agents can communicate through:
- Shared state/board (as in collaborative_qa.py)
- Direct message passing
- Environment-mediated observations

### Dynamic Agent Management

- Add/remove agents during episodes
- Handle variable numbers of agents
- Support heterogeneous agent types

### Integration with GEM Tools

Multi-agent environments can use existing GEM tools:
- Search tools for information gathering
- Python execution for computation
- Custom tools for domain-specific tasks

## Best Practices

1. **Clear Agent Roles**: Define specific responsibilities for each agent
2. **Proper Termination**: Handle both individual and collective termination
3. **Reward Design**: Shape rewards to encourage desired collaboration
4. **State Management**: Maintain consistent state across agents
5. **Testing**: Thoroughly test agent interactions and edge cases

## Further Reading

- [Multi-Agent Design Document](../../docs/multi_agent_design.md)
- [PettingZoo Documentation](https://pettingzoo.farama.org/)
- [GEM Core Documentation](../../README.md)