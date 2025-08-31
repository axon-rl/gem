# Multi-Agent Environment Design for GEM

## Overview

This document outlines the design for adding multi-agent environment support to GEM (Gym for LLMs). The design draws inspiration from PettingZoo's proven multi-agent APIs and tau-bench's practical implementation patterns, while maintaining compatibility with GEM's existing architecture.

## Design Principles

1. **Compatibility**: Seamlessly integrate with GEM's existing `Env` base class and registration system
2. **Flexibility**: Support both sequential (AEC) and parallel agent execution models
3. **Simplicity**: Maintain GEM's clean API while adding multi-agent capabilities
4. **Extensibility**: Easy to add new multi-agent environments and agent types
5. **Type Safety**: Leverage Python's type system for clear interfaces

## Architecture

### Core Components

#### 1. Base Multi-Agent Environment Classes

```python
# gem/multiagent/core.py

class MultiAgentEnv(Env):
    """Base class for multi-agent environments in GEM."""
    
    @property
    @abstractmethod
    def agents(self) -> list[str]:
        """List of currently active agent IDs."""
        
    @property
    @abstractmethod
    def possible_agents(self) -> list[str]:
        """List of all possible agents that could be in the environment."""
        
    @abstractmethod
    def observation_space(self, agent: str) -> Any:
        """Returns observation space for a specific agent."""
        
    @abstractmethod
    def action_space(self, agent: str) -> Any:
        """Returns action space for a specific agent."""
```

#### 2. AEC (Agent Environment Cycle) API

Sequential execution where agents take turns:

```python
class AECEnv(MultiAgentEnv):
    """Sequential multi-agent environment following PettingZoo's AEC pattern."""
    
    @property
    @abstractmethod
    def agent_selection(self) -> str:
        """Currently selected agent that should take an action."""
        
    @abstractmethod
    def observe(self, agent: str) -> str:
        """Get observation for specific agent."""
        
    @abstractmethod
    def last(self) -> Tuple[str, float, bool, bool, dict]:
        """Returns observation, reward, terminated, truncated, info for current agent."""
```

#### 3. Parallel API

Simultaneous execution where all agents act at once:

```python
class ParallelEnv(MultiAgentEnv):
    """Parallel multi-agent environment where agents act simultaneously."""
    
    @abstractmethod
    def step(self, actions: dict[str, str]) -> Tuple[
        dict[str, str],      # observations
        dict[str, float],    # rewards
        dict[str, bool],     # terminated
        dict[str, bool],     # truncated
        dict[str, dict]      # infos
    ]:
        """Execute actions for all agents simultaneously."""
```

### Agent Types

Following tau-bench's pattern, we'll support different agent roles:

#### 1. User Agent
- Simulates user interactions
- Provides natural language inputs
- Can use different strategies (LLM, scripted, human)

#### 2. Tool Agent
- Executes tools and APIs
- Processes user requests
- Returns structured responses

#### 3. Collaborative Agent
- Works with other agents toward shared goals
- Can share information and coordinate actions

### Communication Patterns

#### 1. Message Passing
```python
class Message:
    sender: str
    receiver: str
    content: str
    metadata: dict
```

#### 2. Shared State
```python
class SharedState:
    """Shared memory accessible by all agents."""
    def get(self, key: str) -> Any
    def set(self, key: str, value: Any)
    def update(self, agent: str, updates: dict)
```

### Conversion Utilities

Support conversion between AEC and Parallel environments:

```python
# gem/multiagent/conversions.py

def aec_to_parallel(env: AECEnv) -> ParallelEnv:
    """Convert AEC environment to Parallel interface."""
    
def parallel_to_aec(env: ParallelEnv) -> AECEnv:
    """Convert Parallel environment to AEC interface."""
```

## Implementation Plan

### Phase 1: Core Infrastructure
1. Create `gem/multiagent/` module
2. Implement base classes (`MultiAgentEnv`, `AECEnv`, `ParallelEnv`)
3. Add agent management utilities
4. Implement environment converters

### Phase 2: Agent Components
1. Create agent base classes
2. Implement user agent with multiple strategies
3. Implement tool agent with GEM's existing tools
4. Add communication mechanisms

### Phase 3: Example Environments
1. **Collaborative QA**: Multiple agents work together to answer questions
2. **Tool Delegation**: User agent delegates tasks to specialized tool agents
3. **Negotiation Game**: Agents negotiate to reach agreements

### Phase 4: Integration
1. Update GEM's registration system for multi-agent envs
2. Add multi-agent wrappers
3. Create evaluation utilities
4. Write comprehensive tests

## Example Usage

### Creating a Multi-Agent Environment

```python
from gem.multiagent import AECEnv
from gem import register, make

class CollaborativeQAEnv(AECEnv):
    def __init__(self):
        self.possible_agents = ["researcher", "validator", "synthesizer"]
        self.agents = self.possible_agents.copy()
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()
        
    def step(self, action: str):
        # Process action for current agent
        agent = self.agent_selection
        
        # Update state based on agent's action
        if agent == "researcher":
            self._process_research(action)
        elif agent == "validator":
            self._process_validation(action)
        elif agent == "synthesizer":
            self._process_synthesis(action)
            
        # Move to next agent
        self._agent_selector.next()
        self.agent_selection = self._agent_selector.selected
        
    def reset(self, seed=None):
        # Reset environment state
        self.agents = self.possible_agents.copy()
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        return self.observe(self.agent_selection), {}

# Register the environment
register("CollaborativeQA-v0", CollaborativeQAEnv)

# Use the environment
env = make("CollaborativeQA-v0")
obs, info = env.reset()

for agent in env.agent_iter():
    observation, reward, terminated, truncated, info = env.last()
    if terminated or truncated:
        action = None
    else:
        action = policy(observation, agent)  # Your policy here
    env.step(action)
```

### Using Parallel Execution

```python
from gem.multiagent import ParallelEnv

class MultiToolEnv(ParallelEnv):
    def __init__(self):
        self.possible_agents = ["search_agent", "code_agent", "math_agent"]
        self.agents = self.possible_agents.copy()
        
    def step(self, actions: dict[str, str]):
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}
        
        for agent, action in actions.items():
            # Process each agent's action
            obs, reward, term, trunc, info = self._process_agent_action(agent, action)
            observations[agent] = obs
            rewards[agent] = reward
            terminations[agent] = term
            truncations[agent] = trunc
            infos[agent] = info
            
        return observations, rewards, terminations, truncations, infos

# Use the environment
env = MultiToolEnv()
observations, infos = env.reset()

while env.agents:
    actions = {agent: policy(observations[agent]) for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    
    # Remove terminated agents
    env.agents = [a for a in env.agents if not terminations[a]]
```

## File Structure

```
gem/
├── multiagent/
│   ├── __init__.py
│   ├── core.py              # Base classes
│   ├── aec_env.py           # AEC environment implementation
│   ├── parallel_env.py      # Parallel environment implementation
│   ├── agent_selector.py    # Agent selection utilities
│   ├── conversions.py       # Environment converters
│   ├── communication.py     # Inter-agent communication
│   └── agents/
│       ├── __init__.py
│       ├── base_agent.py
│       ├── user_agent.py
│       └── tool_agent.py
├── envs/
│   └── multiagent/
│       ├── __init__.py
│       ├── collaborative_qa.py
│       ├── tool_delegation.py
│       └── negotiation.py
└── examples/
    └── multiagent/
        ├── train_collaborative_agents.py
        ├── user_tool_interaction.py
        └── README.md
```

## Compatibility with Existing GEM Features

### Wrappers
Multi-agent environments will work with existing GEM wrappers through adapter patterns:

```python
class MultiAgentWrapperAdapter(EnvWrapper):
    """Adapts multi-agent environments to work with single-agent wrappers."""
    
    def __init__(self, env: MultiAgentEnv, wrapper_cls: type[EnvWrapper]):
        self.env = env
        self.wrapped_envs = {
            agent: wrapper_cls(SingleAgentView(env, agent))
            for agent in env.possible_agents
        }
```

### Tools
Existing GEM tools can be used by tool agents:

```python
from gem.tools import SearchTool, PythonCodeTool

class ToolAgent:
    def __init__(self):
        self.tools = {
            "search": SearchTool(),
            "python": PythonCodeTool()
        }
        
    def execute_action(self, action: str):
        tool_name, tool_input = parse_action(action)
        if tool_name in self.tools:
            return self.tools[tool_name].execute(tool_input)
```

### Registration
Multi-agent environments will use the same registration system:

```python
from gem import register

# Register AEC environment
register(
    "CollaborativeQA-v0",
    entry_point="gem.envs.multiagent:CollaborativeQAEnv",
    kwargs={"max_agents": 3}
)

# Register Parallel environment  
register(
    "ParallelTools-v0",
    entry_point="gem.envs.multiagent:ParallelToolsEnv",
    kwargs={"tools": ["search", "python", "math"]}
)
```

## Evaluation Metrics

### Multi-Agent Specific Metrics
1. **Coordination Efficiency**: How well agents work together
2. **Communication Overhead**: Amount of inter-agent messages
3. **Task Completion Rate**: Success rate for collaborative tasks
4. **Individual vs Collective Performance**: Compare solo vs team performance

### Integration with Existing Metrics
- Per-agent rewards and metrics
- Aggregate team performance
- Cost tracking across all agents

## Testing Strategy

1. **Unit Tests**: Test each component in isolation
2. **Integration Tests**: Test agent interactions
3. **Environment Tests**: Validate environment dynamics
4. **Performance Tests**: Benchmark multi-agent overhead

## Future Extensions

1. **Hierarchical Agents**: Agents that can spawn sub-agents
2. **Dynamic Agent Creation**: Environments that add/remove agents during episodes
3. **Competitive Environments**: Adversarial multi-agent scenarios
4. **Large-Scale Multi-Agent**: Support for 10+ agents
5. **Heterogeneous Agents**: Different model types/sizes for different agents

## Conclusion

This design provides a robust foundation for multi-agent environments in GEM, combining the best practices from PettingZoo and tau-bench while maintaining GEM's simplicity and extensibility. The modular architecture allows for easy extension and customization while preserving compatibility with existing GEM features.