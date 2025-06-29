# Adapted from https://github.com/PeterGriffinJin/Search-R1

import os
import re
from typing import Tuple

import msgspec
import requests

from gem.tools.base_tool import BaseTool

# Timeout for search request in seconds
TIMEOUT = 5


class SearchTool(BaseTool):
    tool_type = "search"

    def __init__(self, num_workers=1, search_url=None, topk=3, timeout=TIMEOUT):
        super().__init__(num_workers)
        self.search_url = search_url
        self.topk = topk
        self.timeout = timeout
        self._search_url_resolved = self.search_url is not None

    def _parse_action(self, action: str) -> Tuple[str, str, bool]:
        """
        Parse the action string to extract the <search> content and the full matched tag.
        Returns (content, parsed_action, is_valid)
        """
        # only take the first match
        pattern = r"<search>(.*?)</search>"
        match = re.search(pattern, action, re.DOTALL)
        if match:
            parsed_query = match.group(1).strip()
            parsed_action = action[: match.end()]  # including thinking process
            return parsed_query, parsed_action, True
        else:
            return "", "", False

    def _search(self, query: str):
        """
        Perform a search using the configured search_url.
        Returns a formatted string of search results.
        """
        if not self._search_url_resolved:
            self.search_url = self.search_url or os.environ.get("SEARCH_URL")
            self._search_url_resolved = True

        if not self.search_url:
            raise ValueError("search_url must be provided for SearchTool.")

        payload = {"queries": [query], "topk": self.topk, "return_scores": True}
        try:
            response = requests.post(
                self.search_url,
                data=msgspec.msgpack.encode(payload),
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = msgspec.msgpack.decode(response.content)["result"][0]
            return self._passages2string(result)
        except Exception as e:
            return f"[SearchTool Error: {e}]"

    def _passages2string(self, result):
        format_reference = ""
        for idx, doc_item in enumerate(result):
            content = doc_item["document"]["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    def instruction_string(self) -> str:
        return (
            "You are provided with a search engine to help answer questions.\n\n"
            "Instructions:\n"
            "- Always conduct reasoning inside:\n"
            "  <think> your reasoning here </think>\n"
            "- After reasoning, if knowledge is missing, issue a search query:\n"
            "  <search> your query </search>\n"
            "- The search engine returns results inside:\n"
            "  <information> ... </information>\n"
            "- You can search as many times as needed.\n"
            "- When ready, give the final concise answer using:\n"
            "  <answer> your answer </answer>\n\n"
            "Example:\n"
            "<think> I need to find the capital of China. </think>\n"
            "<search> capital of China </search>\n"
            "<information> Beijing is the capital of China. </information>\n"
            "<think> The capital is Beijing. </think>\n"
            "<answer> Beijing </answer>"
        )

    def execute_action(self, action: str):
        """
        Execute the parsed action for the SearchTool.

        Args:
            action: The raw action string, typically containing a search query
                within <search>...</search> tags.

        Returns:
            observation: The formatted search result, or an empty string if invalid.
            done: Always False for search tool (search does not terminate the episode).
            valid: True if a valid search query was found and executed, False otherwise.
        """
        parsed_query, parsed_action, is_valid = self._parse_action(action)
        if not is_valid:
            # observation = "No valid search query found. Please provide your query within <search>...</search> tags."
            observation = ""
            valid = False
        else:
            search_result = self._search(parsed_query)
            observation = f"\n\n<information>{search_result}</information>\n\n"
            valid = True
        return valid, observation, parsed_action
