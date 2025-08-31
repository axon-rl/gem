# Copyright 2025 AxonRL Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multi-agent environment support for GEM."""

from gem.multiagent.aec_env import AECEnv
from gem.multiagent.agent_selector import AgentSelector
from gem.multiagent.conversions import aec_to_parallel, parallel_to_aec
from gem.multiagent.core import MultiAgentEnv
from gem.multiagent.parallel_env import ParallelEnv

__all__ = [
    "MultiAgentEnv",
    "AECEnv",
    "ParallelEnv",
    "AgentSelector",
    "aec_to_parallel",
    "parallel_to_aec",
]
