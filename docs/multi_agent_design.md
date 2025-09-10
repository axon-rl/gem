# Multi-Agent Environment Design for GEM

## Overview

This document describes the multi-agent environment support in GEM (General Experience Maker), designed specifically for LLM-based multi-agent scenarios. The design provides a unified API that naturally extends GEM's text-based environment paradigm to multiple agents.

## Core Design Principles

1. **LLM-First**: Optimized for text-based observations and actions
2. **Unified API**: Single class handles both sequential and simultaneous interactions
3. **Natural Language**: Agents communicate through text, matching LLM capabilities
4. **Simple**: Minimal abstraction layers, easy to understand and extend

## Architecture

### File Structure

```
gem/
├── multiagent/
│   ├── __init__.py
│   ├── multiagent_env.py    # Unified multi-agent environment
│   └── utils.py              # AgentSelector and helpers
```

### Core MultiAgentEnv Class

```python
from typing import Dict, List, Optional, Tuple, Union
from gem.core import Env

class MultiAgentEnv(Env):
    """
    Unified multi-agent environment for LLM agents.
    Supports both sequential (turn-based) and simultaneous interactions.
    """
    
    def __init__(self, simultaneous: bool = True):
        super().__init__()
        
        # Agent configuration
        self.agents: List[str] = []
        self.possible_agents: List[str] = []
        
        # Interaction mode
        self.simultaneous = simultaneous
        
        # Agent states
        self.terminations: Dict[str, bool] = {}
        self.truncations: Dict[str, bool] = {}
        self.rewards: Dict[str, float] = {}
        self.infos: Dict[str, dict] = {}
        
        # For sequential mode
        self._agent_selector: Optional[AgentSelector] = None
        
    def step(self, action: Union[str, Dict[str, str]]) -> Tuple:
        """
        Execute one step in the environment.
        
        Args:
            action: 
                - str: Single action for current agent (sequential mode)
                - Dict[str, str]: Actions for all agents (simultaneous mode)
                
        Returns:
            Sequential mode: (obs, reward, terminated, truncated, info)
            Simultaneous mode: (obs_dict, rewards_dict, terminations_dict, truncations_dict, infos_dict)
        """
        
    def reset(self, seed: Optional[int] = None) -> Tuple:
        """
        Reset the environment.
        
        Returns:
            Sequential mode: (observation, info) for first agent
            Simultaneous mode: (observations_dict, infos_dict) for all agents
        """
        
    def observe(self, agent: str) -> str:
        """Get text observation for specific agent."""
        
    @property
    def current_agent(self) -> Optional[str]:
        """Current agent in sequential mode."""
```

## LLM-Optimized Features

### 1. Text-Based Communication

All observations and actions are strings, perfect for LLM agents:

```python
class ConversationEnv(MultiAgentEnv):
    def observe(self, agent: str) -> str:
        if agent == "user":
            return "You are chatting with an AI assistant. Ask a question."
        else:
            return f"User said: {self.last_message}. Please respond helpfully."
```

### 2. Natural Language Actions

Actions are text strings that can be:
- Direct messages: `"Hello, how can I help?"`
- Tool calls: `"search: quantum computing"`
- Commands: `"terminate_conversation"`

```python
def step(self, action: Union[str, Dict[str, str]]):
    if isinstance(action, str):
        # Sequential: process single agent's text action
        message = action
        tool_call = self.parse_tool_call(message)
        if tool_call:
            result = self.execute_tool(tool_call)
    else:
        # Simultaneous: process all agents' text actions
        for agent, message in action.items():
            self.process_message(agent, message)
```

### 3. Shared Context

Multi-agent LLM environments often need shared context:

```python
class MultiAgentEnv(Env):
    def __init__(self):
        super().__init__()
        self.shared_memory: List[str] = []  # Conversation history
        self.global_context: str = ""       # Shared world state
        
    def observe(self, agent: str) -> str:
        # Each agent sees shared context + agent-specific view
        return f"{self.global_context}\n\n{self.agent_view(agent)}"
```

## Interaction Patterns

### Sequential Mode (Turn-Based)

Perfect for conversations, negotiations, or any turn-based interaction:

```python
class DialogueEnv(MultiAgentEnv):
    def __init__(self):
        super().__init__(simultaneous=False)
        self.possible_agents = ["user", "assistant"]
        self.conversation_history = []
        
    def observe(self, agent: str) -> str:
        # Show conversation history
        history = "\n".join(self.conversation_history[-10:])
        return f"Conversation:\n{history}\n\nYour turn ({agent}):"
        
    def step(self, action: str) -> Tuple:
        # Add message to history
        current = self.current_agent
        self.conversation_history.append(f"{current}: {action}")
        
        # Check for conversation end
        terminated = "goodbye" in action.lower()
        
        # Move to next agent
        self._agent_selector.next()
        
        # Return next observation
        next_obs = self.observe(self.current_agent)
        return next_obs, 0.0, terminated, False, {}
```

### Simultaneous Mode

For collaborative problem-solving where agents work in parallel:

```python
class TeamProblemSolvingEnv(MultiAgentEnv):
    def __init__(self):
        super().__init__(simultaneous=True)
        self.possible_agents = ["researcher", "coder", "reviewer"]
        self.shared_workspace = {}
        
    def step(self, actions: Dict[str, str]) -> Tuple:
        observations = {}
        rewards = {}
        
        # All agents contribute simultaneously
        for agent, action in actions.items():
            if agent == "researcher":
                self.shared_workspace["research"] = action
            elif agent == "coder":
                self.shared_workspace["code"] = action
            elif agent == "reviewer":
                feedback = self.review(action)
                self.shared_workspace["feedback"] = feedback
                
        # Everyone sees the updated workspace
        for agent in self.agents:
            observations[agent] = str(self.shared_workspace)
            rewards[agent] = self.evaluate_progress()
            
        return observations, rewards, self.terminations, self.truncations, self.infos
```

## Agent Management

### Dynamic Agent Addition/Removal

```python
def add_agent(self, agent_id: str, role: str = "participant"):
    """Add new agent to environment."""
    if agent_id not in self.possible_agents:
        self.possible_agents.append(agent_id)
    if agent_id not in self.agents:
        self.agents.append(agent_id)
        self.terminations[agent_id] = False
        self.truncations[agent_id] = False
        self.rewards[agent_id] = 0.0
        self.infos[agent_id] = {"role": role}

def remove_agent(self, agent_id: str):
    """Remove agent from active list."""
    if agent_id in self.agents:
        self.agents.remove(agent_id)
        self.terminations[agent_id] = True
```

### Agent Roles and Capabilities

```python
class MultiAgentEnv(Env):
    def __init__(self):
        super().__init__()
        self.agent_capabilities = {}  # What each agent can do
        self.agent_roles = {}         # Agent's role in the environment
        
    def register_agent(self, agent_id: str, capabilities: List[str], role: str):
        self.agent_capabilities[agent_id] = capabilities
        self.agent_roles[agent_id] = role
```

## Tool Integration

Multi-agent LLM environments often need tool access:

```python
from gem.tools import PythonCodeTool, SearchTool

class ToolEnabledMultiAgentEnv(MultiAgentEnv):
    def __init__(self):
        super().__init__()
        self.tools = {
            "python": PythonCodeTool(),
            "search": SearchTool(),
        }
        
    def step(self, action: Union[str, Dict[str, str]]):
        # Parse tool calls from action
        if "execute:" in action:
            tool_name, tool_input = action.split("execute:", 1)
            tool_result = self.tools[tool_name].execute_action(tool_input)
            observation = f"Tool result: {tool_result}"
        else:
            observation = self.process_dialogue(action)
```

## Example Implementations

### 1. User-Assistant Conversation

```python
class ConversationEnv(MultiAgentEnv):
    def __init__(self):
        super().__init__(simultaneous=False)
        self.possible_agents = ["user", "assistant"]
        self.agents = self.possible_agents.copy()
        self._agent_selector = AgentSelector(self.agents)
        self.messages = []
        
    def observe(self, agent: str) -> str:
        if not self.messages:
            return "Start the conversation"
        return f"Conversation history:\n" + "\n".join(self.messages[-5:])
        
    def step(self, action: str):
        current = self.current_agent
        self.messages.append(f"{current}: {action}")
        
        # Simple reward based on response quality
        reward = len(action.split()) / 100.0  # Reward longer responses
        
        # Check termination
        terminated = any(word in action.lower() for word in ["goodbye", "exit", "quit"])
        
        # Next agent's turn
        self._agent_selector.next()
        
        return self.observe(self.current_agent), reward, terminated, False, {}
```

### 2. Multi-Agent Collaboration

```python
class CollaborationEnv(MultiAgentEnv):
    def __init__(self):
        super().__init__(simultaneous=True)
        self.possible_agents = ["researcher", "analyst", "reviewer"]
        self.agents = self.possible_agents.copy()
        self.shared_doc = ""
        self.iteration = 0
        
    def step(self, actions: Dict[str, str]):
        observations = {}
        rewards = {}
        
        # Process all contributions
        contributions = []
        for agent, action in actions.items():
            contributions.append(f"[{agent}]: {action}")
            
        # Update shared document
        self.shared_doc = "\n".join(contributions)
        self.iteration += 1
        
        # Everyone sees the combined work
        for agent in self.agents:
            observations[agent] = f"Iteration {self.iteration}:\n{self.shared_doc}"
            rewards[agent] = self.evaluate_quality(self.shared_doc)
            
        # Terminate after 10 iterations
        terminated = self.iteration >= 10
        for agent in self.agents:
            self.terminations[agent] = terminated
            
        return observations, rewards, self.terminations, self.truncations, self.infos
```

## Usage Patterns

### Sequential Usage

```python
env = ConversationEnv()
obs, info = env.reset()

for agent in env.agent_iter(max_iter=100):
    obs = env.observe(agent)
    
    # Get action from LLM
    if agent == "user":
        action = get_user_input()
    else:
        action = llm.generate(obs)
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        break
```

### Simultaneous Usage

```python
env = CollaborationEnv()
observations, infos = env.reset()

while env.agents:
    actions = {}
    for agent in env.agents:
        # Each agent acts based on their observation
        actions[agent] = llm.generate(observations[agent], role=agent)
    
    observations, rewards, terminations, truncations, infos = env.step(actions)
    
    # Remove terminated agents
    env.agents = [a for a in env.agents if not terminations[a]]
```

## Advanced Features

### Message Passing

```python
class MultiAgentEnv(Env):
    def __init__(self):
        super().__init__()
        self.message_buffer = defaultdict(list)
        
    def send_message(self, from_agent: str, to_agent: str, message: str):
        self.message_buffer[to_agent].append({
            "from": from_agent,
            "message": message,
            "timestamp": self.current_step
        })
        
    def get_messages(self, agent: str) -> List[Dict]:
        messages = self.message_buffer[agent]
        self.message_buffer[agent] = []  # Clear after reading
        return messages
```

### Hierarchical Agents

```python
class HierarchicalMultiAgentEnv(MultiAgentEnv):
    def __init__(self):
        super().__init__()
        self.agent_hierarchy = {
            "manager": ["worker1", "worker2", "worker3"],
            "reviewer": ["manager"]
        }
        
    def step(self, actions):
        # Process in hierarchical order
        for supervisor, subordinates in self.agent_hierarchy.items():
            if supervisor in actions:
                # Supervisor action affects subordinates
                for sub in subordinates:
                    self.assign_task(sub, actions[supervisor])
```

## Testing

The unified API makes testing straightforward:

```python
def test_multiagent_env():
    # Test sequential mode
    env = ConversationEnv()
    obs, info = env.reset()
    assert isinstance(obs, str)
    
    obs, reward, term, trunc, info = env.step("Hello")
    assert env.current_agent == "assistant"
    
    # Test simultaneous mode
    env = CollaborationEnv()
    obs, info = env.reset()
    assert isinstance(obs, dict)
    
    actions = {agent: f"Action from {agent}" for agent in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
    assert len(obs) == len(env.agents)
```

## Conclusion

This unified multi-agent design for GEM provides a clean, LLM-optimized API for building multi-agent environments. By focusing on text-based interaction and providing both sequential and simultaneous modes in a single class, we make it easy to create sophisticated multi-agent scenarios for LLM research and applications.