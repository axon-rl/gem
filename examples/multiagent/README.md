# Multi-Agent Environment Examples

This directory contains example implementations demonstrating the multi-agent capabilities of GEM (General Experience Maker).

## Examples

### 1. Conversation Environment (`conversation.py`)
- **Mode**: Sequential (turn-based)
- **Agents**: User and Assistant
- **Features**:
  - Turn-based dialogue between agents
  - Tool integration (Python code execution, search)
  - Message history tracking
  - Reward system based on action quality
  
**Usage**:
```bash
python conversation.py
```

### 2. Collaboration Environment (`collaboration.py`)
- **Mode**: Simultaneous (parallel)
- **Agents**: Researcher, Analyst, Reviewer
- **Features**:
  - All agents act simultaneously each round
  - Shared memory for inter-agent communication
  - Python code tool for data analysis
  - Multi-round collaboration with termination conditions

**Usage**:
```bash
python collaboration.py
```

## Key Concepts

### Sequential vs Simultaneous Modes

GEM's unified `MultiAgentEnv` supports two modes:

1. **Sequential Mode** (`simultaneous=False`):
   - Agents take turns acting one at a time
   - Uses `AgentSelector` to manage turn order
   - Single action string input to `step()`
   - Suitable for dialogue, games, and turn-based scenarios

2. **Simultaneous Mode** (`simultaneous=True`):
   - All agents act at the same time
   - Dictionary of actions input to `step()`
   - No turn order management
   - Suitable for collaborative tasks and parallel processing

### Creating Custom Environments

To create your own multi-agent environment:

```python
from gem.multiagent import MultiAgentEnv

class MyCustomEnv(MultiAgentEnv):
    def __init__(self):
        super().__init__(simultaneous=True)  # or False for sequential
        self.possible_agents = ["agent1", "agent2"]
        
    def observe(self, agent: str) -> str:
        return f"Observation for {agent}"
    
    def _step_simultaneous(self, actions: Dict[str, str]) -> Tuple:
        # Implement simultaneous step logic
        self._validate_actions(actions)
        
        observations = {}
        rewards = {}
        
        for agent in self.agents:
            # Process action for each agent
            observations[agent] = self.observe(agent)
            rewards[agent] = calculate_reward(actions[agent])
            
        return observations, rewards, self.terminations, self.truncations, self.infos
    
    def _step_sequential(self, action: str) -> Tuple:
        # Implement sequential step logic
        current = self.current_agent
        
        # Process action for current agent
        reward = calculate_reward(action)
        
        # Advance to next agent
        if self._agent_selector:
            self._agent_selector.next()
            self.agent_selection = self._agent_selector.selected
        
        obs = self.observe(self.current_agent) if self.current_agent else ""
        
        return obs, reward, self.terminations[current], self.truncations[current], {}
```

### Tool Integration

Both examples demonstrate integration with GEM's tool system:
- `PythonCodeTool`: Execute Python code within the environment
- `SearchTool`: Perform searches (simulated in examples)

Tools can be used to enhance agent capabilities and create more realistic LLM training scenarios.

## Key Differences Between Examples

| Aspect | Conversation | Collaboration |
|--------|--------------|---------------|
| Mode | Sequential (`simultaneous=False`) | Simultaneous (`simultaneous=True`) |
| Agents | 2 (user, assistant) | 3 (researcher, analyst, reviewer) |
| Execution | Turn-based | All agents act together |
| Communication | Direct dialogue | Shared memory |
| Tools | Python, Search | Python analysis |
| Goal | Answer user queries | Solve complex problems |

## Architecture

The multi-agent system uses a unified architecture:
```
gem/multiagent/
├── __init__.py
├── multi_agent_env.py    # Unified base class with both modes
└── utils.py              # AgentSelector for sequential mode
```

## Running Tests

To test the multi-agent examples:

```bash
# Run unit tests
make test-multiagent

# Run all tests and examples
make test-multiagent-all
```

## Important Implementation Notes

1. **Unified API**: Use `MultiAgentEnv` with `simultaneous` flag to select mode
2. **Step Method**: Accepts single action (sequential) or dict of actions (simultaneous)
3. **Reset Method**: Call `super().reset(seed)` to initialize properly
4. **Agent Management**: Framework handles agent lifecycle automatically
5. **Tool Integration**: Use GEM's existing tools from `gem.tools`

## Learn More

- [Multi-Agent Design Document](../../docs/multi_agent_design.md)
- [GEM Documentation](../../README.md)
- [PettingZoo Documentation](https://pettingzoo.farama.org/) (inspiration for the API)