from functools import partial

from gem.tools.python_code_tool import PythonCodeTool
from gem.tools.search_tool import SearchTool
from gem.tools.tool_env_wrapper import ToolEnvWrapper
from gem.wrappers.complete_traj_verify_wrapper import TrajVerifyWrapper
from gem.wrappers.episode_tracking_wrapper import EpisodeTrackingWrapper
from gem.wrappers.observation_wrapper import ObservationWrapper

# TODO refactor later

### Note: Order is important!
WRAPPER_FACTORY = {
    ### 1. Frist, optionlly add the tool wrapper
    "python_tool": partial(
        ToolEnvWrapper,
        tools=[PythonCodeTool(timeout=5)],
        tool_reward=0.05,
        tool_success_reward=0.25,
        max_tool_uses=5,
    ),
    "python_tool_no_int_reward": partial(
        ToolEnvWrapper,
        tools=[PythonCodeTool(timeout=5)],
        tool_reward=0.0,
        tool_success_reward=0.0,
        max_tool_uses=5,
    ),
    "python_tool_no_reward": partial(
        ToolEnvWrapper,
        tools=[PythonCodeTool(timeout=5)],
        tool_reward=0.0,
        max_tool_uses=10,
    ),
    "python_tool_1_reward": partial(
        ToolEnvWrapper,
        tools=[PythonCodeTool(timeout=5)],
        tool_reward=1.0,
        max_tool_uses=10,
    ),
    "python_tool_for_dummy_env": partial(
        ToolEnvWrapper,
        tools=[PythonCodeTool(timeout=5)],
        tool_reward=0.1,
        max_tool_uses=10,
        obs_suffix="Great! Now output another example python code block to print something.",
    ),
    "search_tool": partial(
        ToolEnvWrapper,
        tools=[SearchTool(topk=3, timeout=5)],
        tool_reward=0.05,
        tool_success_reward=0.25,
        max_tool_uses=5,
    ),
    "search_tool_no_reward": partial(
        ToolEnvWrapper,
        tools=[SearchTool(topk=3, timeout=5)],
        tool_reward=0.0,
        max_tool_uses=10,
    ),
    "search_tool_03_reward": partial(
        ToolEnvWrapper,
        tools=[SearchTool(topk=3, timeout=5)],
        tool_reward=0.3,
        max_tool_uses=10,
    ),
    "search_tool_05_reward": partial(
        ToolEnvWrapper,
        tools=[SearchTool(topk=3, timeout=5)],
        tool_reward=0.5,
        max_tool_uses=10,
    ),
    ### 2. Then choose an observation wrapper
    "concat": partial(
        ObservationWrapper,
        include_action=False,
        include_chat_template=False,
    ),
    "concat_chat": partial(
        ObservationWrapper,
        include_action=True,
        include_chat_template=True,
        # Requires tokenizer to be passed later
    ),
    "concat_with_action": partial(
        ObservationWrapper,
        include_action=True,
        include_chat_template=False,
    ),
    "concat_chat_on_reset": partial(
        ObservationWrapper,
        include_action=True,
        include_chat_template=False,
        apply_chat_template_on_reset=True,
    ),
    ### 3. Then choose a trajectory verification wrapper
    "traj_verify": partial(
        TrajVerifyWrapper,
        formatted_reward=0.2,
    ),
    ### 4. Finally, optionally add the episode tracking wrapper
    "episode_tracking": EpisodeTrackingWrapper,
}

TOKENIZER_REQUIRED = ["concat_chat", "concat_chat_on_reset"]


def get_wrapper_fns(wrappers: str, tokenizer=None):
    """Get a list of wrapper functions based on the provided wrapper names."""
    wrapper_fns = []
    print(f"Wrappers requested: {wrappers}")
    if wrappers:
        wrappers = wrappers.split(",")
        print(f"Wrappers: {wrapper_fns}")
        for w in wrappers:
            wrapper_fn = WRAPPER_FACTORY[w]
            if w in TOKENIZER_REQUIRED and tokenizer is not None:
                wrapper_fn = partial(wrapper_fn, tokenizer=tokenizer)
            wrapper_fns.append(wrapper_fn)
    return wrapper_fns
