# Multi-Agent Environment Design for GEM

## Overview

This document describes the implemented multi-agent environment support in GEM (Gym for LLMs). The design follows PettingZoo's proven multi-agent APIs while maintaining compatibility with GEM's existing architecture.

## Design Principles

1. **Separation of Concerns**: GEM provides environment infrastructure only; agent implementations belong in examples
2. **Compatibility**: Seamlessly integrates with GEM's existing `Env` base class and registration system
3. **Flexibility**: Supports both sequential (AEC) and parallel agent execution models
4. **Simplicity**: Clean API without comments in implementation
5. **Type Safety**: Clear interfaces with type hints

## Architecture

### File Structure

```
gem/
├── multiagent/
│   ├── __init__.py
│   ├── multi_agent_env.py    # Base class for all multi-agent environments
│   ├── aec_env.py            # Sequential execution environment
│   ├── parallel_env.py       # Parallel execution environment
│   └── utils.py              # AgentSelector and conversion utilities
├── tests/
│   └── test_multiagent/
│       ├── test_aec_env.py
│       ├── test_parallel_env.py
│       ├── test_core.py
│       ├── test_agent_selector.py
│       └── test_conversions.py
└── examples/
    └── multiagent/
        ├── conversation.py    # User-assistant dialogue example
        ├── collaboration.py   # Multi-agent collaboration example
        └── README.md
```

### Core Components

#### 1. MultiAgentEnv Base Class

Located in `gem/multiagent/multi_agent_env.py`:

```python
class MultiAgentEnv(Env):
    @property
    def agents(self) -> List[str]:
        """Currently active agents."""
        
    @property
    def possible_agents(self) -> List[str]:
        """All possible agents that could be in the environment."""
        
    def observation_space(self, agent: str) -> Any:
        """Returns observation space for a specific agent."""
        
    def action_space(self, agent: str) -> Any:
        """Returns action space for a specific agent."""
```

Key features:
- Extends GEM's base `Env` class
- Manages per-agent states (rewards, terminations, truncations)
- Provides reward accumulation
- Implements dead step detection

#### 2. AEC (Agent Environment Cycle) API

Sequential execution where agents take turns (`gem/multiagent/aec_env.py`):

```python
class AECEnv(MultiAgentEnv):
    @property
    def agent_selection(self) -> str:
        """Currently selected agent that should take an action."""
        
    def observe(self, agent: str) -> str:
        """Get observation for specific agent."""
        
    def last(self) -> Tuple[str, float, bool, bool, dict]:
        """Returns observation, reward, terminated, truncated, info."""
        
    def step(self, action: Optional[str]) -> None:
        """Process action for current agent."""
        
    def agent_iter(self, max_iter: int = 2**63) -> AECIterable:
        """Returns an iterator over agents."""
```

Usage pattern:
```python
for agent in env.agent_iter():
    observation, reward, terminated, truncated, info = env.last()
    action = policy(observation, agent)
    env.step(action)
```

#### 3. Parallel API

Simultaneous execution where all agents act at once (`gem/multiagent/parallel_env.py`):

```python
class ParallelEnv(MultiAgentEnv):
    def step(self, actions: Dict[str, str]) -> Tuple[
        Dict[str, str],      # observations
        Dict[str, float],    # rewards
        Dict[str, bool],     # terminated
        Dict[str, bool],     # truncated
        Dict[str, dict]      # infos
    ]:
        """Execute actions for all agents simultaneously."""
```

Usage pattern:
```python
while env.agents:
    actions = {agent: policy(obs[agent]) for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
```

### Utilities

Located in `gem/multiagent/utils.py`:

#### AgentSelector
Manages agent turn order in AEC environments:
```python
class AgentSelector:
    def __init__(self, agents: List[str])
    def next(self) -> str
    def is_first(self) -> bool
    def is_last(self) -> bool
    def remove_agent(self, agent: str)
```

#### Environment Converters
- `AECToParallelWrapper`: Converts AEC to Parallel interface
- `ParallelToAECWrapper`: Converts Parallel to AEC interface
- `aec_to_parallel()`: Convenience function
- `parallel_to_aec()`: Convenience function

## Implementation Examples

### Example 1: conversation.py

Demonstrates user-assistant dialogue with tool use:
- Uses AEC (sequential) execution
- Integrates PythonCodeTool and SearchTool from gem.tools
- Natural turn-based conversation

### Example 2: collaboration.py

Demonstrates multi-agent team collaboration:
- Uses Parallel execution
- Shared memory for information exchange
- All agents work simultaneously

## Key Implementation Details

### 1. Reset Method Pattern

To avoid NotImplementedError, subclasses must call `MultiAgentEnv.reset()` directly:

```python
def reset(self, seed=None):
    from gem.multiagent.multi_agent_env import MultiAgentEnv
    MultiAgentEnv.reset(self, seed)
    # Your reset logic here
    return observations, infos
```

### 2. Agent Lifecycle Management

- Agents are automatically removed when terminated
- Use `_was_dead_step()` to detect actions for terminated agents
- Maintain separate `agents` (active) and `possible_agents` (all) lists

### 3. Tool Integration

Examples show integration with GEM's tool infrastructure:
```python
from gem.tools.python_code_tool import PythonCodeTool
from gem.tools.search_tool import SearchTool

self.python_tool = PythonCodeTool()
is_valid, has_error, result, _ = self.python_tool.execute_action(action)
```

## Testing

Comprehensive test suite with 63 tests covering:
- Core MultiAgentEnv functionality
- AEC environment and iteration
- Parallel environment
- AgentSelector utility
- Environment converters
- Edge cases and error handling

Run tests:
```bash
make test-multiagent      # Run tests only
make test-multiagent-all  # Run tests and examples
```

## API Comparison with PettingZoo

| Feature | PettingZoo | GEM Multi-Agent |
|---------|------------|-----------------|
| AEC API | ✓ | ✓ |
| Parallel API | ✓ | ✓ |
| Agent Management | ✓ | ✓ |
| Reward Accumulation | ✓ | ✓ |
| Dead Step Detection | ✓ | ✓ |
| Environment Converters | ✓ | ✓ |
| Registration System | ✓ | ✓ (uses GEM's) |
| Tool Integration | - | ✓ (gem.tools) |

## Design Decisions

1. **No Agent Code in Core**: All agent implementations are in examples, keeping the core library focused on environment infrastructure.

2. **Clean Code**: No comments in implementation files for cleaner codebase.

3. **Scenario-Based Examples**: Examples named by their scenarios (conversation, collaboration) rather than technical patterns.

4. **Direct Reset Call**: Avoiding super().reset() pattern to prevent NotImplementedError.

5. **Automatic Agent Management**: Framework handles agent removal on termination.

## Future Extensions

Potential areas for enhancement:
- Hierarchical agents
- Dynamic agent spawning
- Advanced communication protocols
- Large-scale multi-agent support (10+ agents)
- Integration with LLM providers for agent policies

## Migration from Single-Agent

To convert a single-agent GEM environment to multi-agent:

1. Choose execution model (AEC or Parallel)
2. Extend appropriate base class
3. Define `possible_agents` list
4. Implement per-agent observation/action spaces
5. Update step() to handle agent-specific logic
6. Use AgentSelector for turn management (AEC only)

## Conclusion

The multi-agent framework successfully extends GEM with robust multi-agent capabilities while maintaining clean architecture and separation of concerns. The implementation provides a solid foundation for multi-agent LLM environments with tool integration.