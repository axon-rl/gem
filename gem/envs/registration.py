"""Environment registration."""

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Union

from gem import Env

ENV_REGISTRY: Dict[str, Callable] = {}


@dataclass
class EnvSpec:
    """A specification for creating environments."""

    id: str
    entry_point: Callable
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def make(self, **kwargs) -> Any:
        """Create an environment instance."""
        all_kwargs = {**self.kwargs, **kwargs}
        return self.entry_point(**all_kwargs)


def register(id: str, entry_point: Callable, **kwargs: Any):
    """Register an environment with a given ID."""
    if id in ENV_REGISTRY:
        raise ValueError(f"Environment {id} already registered.")
    ENV_REGISTRY[id] = EnvSpec(id=id, entry_point=entry_point, kwargs=kwargs)


def pprint_registry_detailed():
    if not ENV_REGISTRY:
        print("No environments registered.")
    else:
        print("Detailed Registered Environments:")
        for env_id, env_spec in ENV_REGISTRY.items():
            print(f"  - {env_id}:")
            print(f"      Entry Point: {env_spec.entry_point}")
            print(f"      Kwargs: {env_spec.kwargs}")


def make(env_id: Union[str, List[str]], **kwargs) -> Env:
    if env_id not in ENV_REGISTRY:
        raise ValueError(f"Environment {env_id} not found in registry.")

    env_spec = ENV_REGISTRY[env_id]

    if isinstance(env_spec.entry_point, str):
        module_path, class_name = env_spec.entry_point.split(":")
        try:
            module = importlib.import_module(module_path)
            env_class = getattr(module, class_name)
        except (ModuleNotFoundError, AttributeError) as e:
            raise ImportError(
                f"Could not import {module_path}.{class_name}. Error: {e}"
            )
    else:
        env_class = env_spec.entry_point

    env = env_class(**{**env_spec.kwargs, **kwargs})

    env.env_id = env_id
    env.entry_point = env_spec.entry_point

    return env
