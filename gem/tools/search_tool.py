# Adapted from https://github.com/PeterGriffinJin/Search-R1

import re
from typing import Tuple

import requests

from gem.tools.base_tool import BaseTool


class SearchTool(BaseTool):
    tool_type = "search"
    # TODO: add a timeout 

    def __init__(self, num_workers=1, search_url=None, topk=3):
        super().__init__(num_workers)
        if not search_url:
            raise ValueError("search_url must be provided for SearchTool.")
        self.search_url = search_url
        self.topk = topk

    def _parse_action(self, action: str) -> Tuple[str, bool]:
        """
        Parse the action string to extract the <search> content only.
        Returns (content, is_valid)
        """
        # TODO: @changyu handle multiple search matches
        pattern = r'<search>(.*?)</search>'
        match = re.search(pattern, action, re.DOTALL)
        if match:
            content = match.group(1).strip()
            return content, True
        else:
            return '', False

    def _search(self, query: str):
        """
        Perform a search using the configured search_url.
        Returns a formatted string of search results.
        """
        payload = {
            "queries": [query],
            "topk": self.topk,
            "return_scores": True
        }
        try:
            response = requests.post(self.search_url, json=payload)
            response.raise_for_status()
            result = response.json()['result'][0]
            return self._passages2string(result)
        except Exception as e:
            return f"[SearchTool Error: {e}]"

    def _passages2string(self, result):
        format_reference = ''
        for idx, doc_item in enumerate(result):
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

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
        query, is_valid = self._parse_action(action)
        if not is_valid:
            # observation = "No valid search query found. Please provide your query within <search>...</search> tags."
            observation = ""
            valid = False
        else:
            search_result = self._search(query)
            observation = f'\n\n<information>{search_result}</information>\n\n'
            valid = True
        return observation, valid