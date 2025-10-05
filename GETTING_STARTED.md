- [Installation](#installation)
- [Tool Integration Examples](#tool-integration-examples)

## Installation

Install `GEM` from PyPI:

```bash
pip install -U gem-llm
```

or from source (in editable mode):

```bash
git clone https://github.com/axon-rl/gem.git && cd gem
pip install -e .
```

To use the `search` tool, run the following to install extra dependencies: 
```bash
pip install -U 'gem-llm[search]'
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```

To use the `mcp` tool and [MCPMark](https://mcpmark.ai/) environment, run the following to install extra dependencies: 
```bash
pip install -U `gem-llm[mcp]`

# install MCPMark
git clone git@github.com:axon-rl/mcpmark.git; cd mcpmark
pip install -e .
playwright install # If you'll use browser-based tasks, install Playwright browsers first
```

## Tool Integration Examples

Below are examples for enabling tools within environments.

**Example using the Python tool:**
```python
from transformers import AutoTokenizer

import gem
from gem.tools.python_code_tool import PythonCodeTool
from gem.tools.tool_env_wrapper import ToolEnvWrapper
from gem.wrappers.wrapper_factory import WRAPPER_FACTORY

env = gem.make("math:GSM8K")
tool = PythonCodeTool()
wrapped_env = ToolEnvWrapper(env, tools=[tool])
wrapped_env = WRAPPER_FACTORY["concat_chat"](
    wrapped_env, tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
)
obs, info = wrapped_env.reset()

# we ignore the obs and use a dummy action
dummy_action = "<think>Let me compare 9.9 and 9.11 using python.</think><python>print('9.9 > 9.11?', 9.9 > 9.11)</python>"
obs, reward, terminated, truncated, info = wrapped_env.step(dummy_action)
print(obs)
# continue to sample the next response given the tool results ...

wrapped_env.close()
```

**Example using the search tool:**
```python
# assume you have search server running

env = gem.make("game:GuessTheNumber-v0", max_turns=2)
tool = SearchTool(search_url="http://localhost:8000/retrieve", topk=2)
wrapped_env = ToolEnvWrapper(env, tools=[tool], max_tool_uses=1)
wrapped_env = WRAPPER_FACTORY['concat_chat'](wrapped_env, tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B"))
wrapped_env.reset()

dummy_action = "<think>I need to search for Python list comprehension examples</think><search>Python list comprehension examples</search>"
obs, reward, terminated, truncated, info = wrapped_env.step(dummy_action)
print(obs)
```

<details>
<summary>Click to get the complete runnable code</summary>

```python
import subprocess
import time

from transformers import AutoTokenizer

import gem
from gem.tools.search_tool import SearchTool
from gem.tools.tool_env_wrapper import ToolEnvWrapper
from gem.wrappers.wrapper_factory import WRAPPER_FACTORY

# start the search server
serp_api_key = "add you api key" # get api at https://serpapi.com/manage-api-key
server_process = subprocess.Popen([
    'python', '-m', 'gem.tools.search_engine.serp_search_server',
    '--search_url', 'https://serpapi.com/search',
    '--topk', '2', '--serp_api_key', serp_api_key
])
time.sleep(5)

# interact using search tool
env = gem.make("game:GuessTheNumber-v0", max_turns=2)
tool = SearchTool(search_url="http://localhost:8000/retrieve", topk=2)
wrapped_env = ToolEnvWrapper(env, tools=[tool], max_tool_uses=1)
wrapped_env = WRAPPER_FACTORY['concat_chat'](wrapped_env, tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B"))
wrapped_env.reset()

dummy_action = "<think>I need to search for Python list comprehension examples</think><search>Python list comprehension examples</search>"
obs, reward, terminated, truncated, info = wrapped_env.step(dummy_action)
print(obs)
```
</details>
