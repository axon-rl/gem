import argparse
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import requests
from mosec import Runtime, Server, Worker
from mosec.mixin import TypedMsgPackMixin
from msgspec import Struct

parser = argparse.ArgumentParser(description="Launch online search server.")
parser.add_argument(
    "--search_url",
    type=str,
    required=True,
    help="URL for search engine (e.g. https://serpapi.com/search)",
)
parser.add_argument(
    "--topk", type=int, default=3, help="Number of results to return per query"
)
parser.add_argument(
    "--serp_api_key", type=str, default=None, help="SerpAPI key for online search"
)
parser.add_argument(
    "--serp_engine", type=str, default="google", help="SerpAPI engine for online search"
)
parser.add_argument(
    "--max_wait_time",
    type=int,
    default=10,
    help="Maximum wait time for batching requests",
)
args = parser.parse_args()


class SearchRequest(Struct, kw_only=True):
    queries: List[str]


class SearchResponse(Struct, kw_only=True):
    result: List[List[Dict]]


# --- Config ---
class OnlineSearchConfig:
    def __init__(
        self,
        search_url: str = "https://serpapi.com/search",
        topk: int = 3,
        serp_api_key: Optional[str] = None,
        serp_engine: Optional[str] = None,
    ):
        self.search_url = search_url
        self.topk = topk
        self.serp_api_key = serp_api_key
        self.serp_engine = serp_engine


# --- Online Search Wrapper ---
class OnlineSearchEngine:
    def __init__(self, config: OnlineSearchConfig):
        self.config = config

    def _search_query(self, query: str):
        params = {
            "engine": self.config.serp_engine,
            "q": query,
            "api_key": self.config.serp_api_key,
        }
        response = requests.get(self.config.search_url, params=params)
        return response.json()

    def batch_search(self, queries: List[str]):
        # TODO: @changyu maybe add per request topk
        results = []
        with ThreadPoolExecutor() as executor:
            for result in executor.map(self._search_query, queries):
                results.append(self._process_result(result))
        return results

    def _process_result(self, search_result: Dict):
        results = []

        answer_box = search_result.get("answer_box", {})
        if answer_box:
            title = answer_box.get("title", "No title.")
            snippet = answer_box.get("snippet", "No snippet available.")
            results.append(
                {
                    "document": {"contents": f'"{title}"\n{snippet}'},
                }
            )

        organic_results = search_result.get("organic_results", [])
        for _, result in enumerate(organic_results[: self.config.topk]):
            title = result.get("title", "No title.")
            snippet = result.get("snippet", "No snippet available.")
            results.append(
                {
                    "document": {"contents": f'"{title}"\n{snippet}'},
                }
            )

        related_results = search_result.get("related_questions", [])
        for _, result in enumerate(related_results[: self.config.topk]):
            title = result.get("question", "No title.")  # question is the title here
            snippet = result.get("snippet", "No snippet available.")
            results.append(
                {
                    "document": {"contents": f'"{title}"\n{snippet}'},
                }
            )

        return results


class SearchWorker(TypedMsgPackMixin, Worker):
    def __init__(self):
        super().__init__()
        self.config = OnlineSearchConfig(
            search_url=args.search_url,
            topk=args.topk,
            serp_api_key=args.serp_api_key,
            serp_engine=args.serp_engine,
        )
        self.engine = OnlineSearchEngine(self.config)

    def forward(self, request: SearchRequest) -> SearchResponse:
        results = self.engine.batch_search(request.queries)
        return SearchResponse(result=results)


if __name__ == "__main__":
    server = Server()
    runtime = Runtime(
        worker=SearchWorker,
        num=1,
        max_batch_size=1,
        max_wait_time=args.max_wait_time,
        timeout=30,
    )

    server.register_runtime(
        {
            "/retrieve": [runtime],
        }
    )

    server.run()
