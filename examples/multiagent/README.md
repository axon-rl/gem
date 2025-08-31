# Multi-Agent Examples

This directory contains two examples demonstrating different multi-agent scenarios in GEM (General Experience Maker).

## Examples

### 1. conversation.py - User-Assistant Dialogue
Demonstrates a **conversational scenario** between a user and an assistant with tool capabilities.

**Key Features:**
- Turn-based conversation (uses AEC/sequential execution)
- Assistant can execute Python code via PythonCodeTool
- Assistant can search for information via SearchTool
- Natural dialogue flow with tool integration

**Run:**
```bash
python conversation.py
```

**Environment Class:** `ConversationEnv`

**Use Cases:**
- Chatbots with tool use
- Interactive coding assistants
- Q&A systems with external capabilities

### 2. collaboration.py - Multi-Agent Team Task
Demonstrates **collaborative problem-solving** where multiple agents work together simultaneously.

**Key Features:**
- Agents work in parallel on shared task
- Shared memory for information exchange
- Collective decision making
- All agents contribute simultaneously each round

**Run:**
```bash
python collaboration.py
```

**Environment Class:** `CollaborationEnv`

**Use Cases:**
- Research teams analyzing problems
- Distributed problem solving
- Multi-perspective analysis
- Consensus building systems

## Key Differences

| Aspect | Conversation | Collaboration |
|--------|--------------|---------------|
| Scenario | User-Assistant dialogue | Team problem solving |
| Agents | 2 (user, assistant) | 3 (researcher, analyst, reviewer) |
| Execution | Turn-based (AEC) | Simultaneous (Parallel) |
| Communication | Direct dialogue | Shared memory |
| Tools | Python, Search | Information sharing |
| Goal | Answer user queries | Solve complex problems |

## Architecture

Both examples use GEM's multi-agent infrastructure:
```
gem/multiagent/
├── __init__.py
├── multi_agent_env.py    # Base class for all multi-agent environments
├── aec_env.py            # Sequential execution (used by conversation.py)
├── parallel_env.py       # Parallel execution (used by collaboration.py)
└── utils.py              # AgentSelector and conversion utilities
```

## Creating Your Own Multi-Agent Environment

### For Conversational Scenarios (like conversation.py):
```python
from gem.multiagent.aec_env import AECEnv
from gem.multiagent.multi_agent_env import MultiAgentEnv
from gem.multiagent.utils import AgentSelector

class MyConversationEnv(AECEnv):
    def __init__(self):
        super().__init__()
        self.possible_agents = ["user", "assistant"]
        self.agents = self.possible_agents.copy()
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.selected
        
    def step(self, action):
        # Process one agent's action at a time
        self._agent_selector.next()
        self.agent_selection = self._agent_selector.selected
        
    def reset(self, seed=None):
        # Important: Call MultiAgentEnv.reset() directly
        MultiAgentEnv.reset(self, seed)
        # Reset your environment state
        return initial_observation, {}
```

### For Collaborative Scenarios (like collaboration.py):
```python
from gem.multiagent.parallel_env import ParallelEnv
from gem.multiagent.multi_agent_env import MultiAgentEnv

class MyCollaborationEnv(ParallelEnv):
    def __init__(self):
        super().__init__()
        self.possible_agents = ["agent1", "agent2", "agent3"]
        self.agents = self.possible_agents.copy()
        
    def step(self, actions):
        # Process all agents' actions simultaneously
        observations = {}
        rewards = {}
        for agent in self.agents:
            observations[agent] = process(actions[agent])
            rewards[agent] = calculate_reward(agent)
        return observations, rewards, terminations, truncations, infos
        
    def reset(self, seed=None):
        # Important: Call MultiAgentEnv.reset() directly
        MultiAgentEnv.reset(self, seed)
        # Reset your environment state
        return observations, infos
```

## Important Implementation Notes

1. **Reset Method**: Always call `MultiAgentEnv.reset(self, seed)` directly in your reset method, not `super().reset()` to avoid NotImplementedError.

2. **Agent Management**: The framework automatically handles agent removal when they terminate. Don't manually remove agents in your step() method.

3. **Tool Integration**: Use GEM's existing tools from `gem.tools` for agent capabilities.

## Testing

Run all multi-agent tests:
```bash
pytest -xvs tests/test_multiagent/
cd examples/multiagent && python conversation.py
cd examples/multiagent && python collaboration.py
```

## Learn More

- [Multi-Agent Design Document](../../docs/multi_agent_design.md)
- [GEM Documentation](../../README.md)
- [PettingZoo Documentation](https://pettingzoo.farama.org/) (inspiration for the API)