"""
Minimal Tool base class for TAU-bench tools
"""

from typing import Any, Dict


class Tool:
    """Base Tool class for TAU-bench tools"""

    @staticmethod
    def invoke(data: Dict[str, Any], **kwargs) -> str:
        raise NotImplementedError

    @staticmethod
    def get_info() -> Dict[str, Any]:
        raise NotImplementedError
