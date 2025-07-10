import asyncio
import statistics
import time

import fire
import httpx
import msgspec


async def make_request(client, url, payload_bytes):
    """Sends a single request and returns (latency, success_boolean)."""
    start_time = time.monotonic()
    try:
        response = await client.post(url, content=payload_bytes, timeout=30.0)
        response.raise_for_status()
        # Decode msgpack response to ensure it's valid
        msgspec.msgpack.decode(response.content)
        latency = time.monotonic() - start_time
        return (latency, True)
    except Exception:
        latency = time.monotonic() - start_time
        return (latency, False)


async def run_test(url: str, concurrent_requests: int = 10, total_requests: int = 100):
    """
    Tests the reliability of the search engine.

    Args:
        url: The URL of the search engine's retrieve endpoint.
        concurrent_requests: The number of concurrent requests to send.
        total_requests: The total number of requests to send.
    """
    print("Starting reliability test with:")
    print(f"  URL: {url}")
    print(f"  Concurrent Requests: {concurrent_requests}")
    print(f"  Total Requests: {total_requests}")
    print("-" * 30)

    # Following the client.py pattern - use msgpack encoding, not JSON
    payload = {
        "query": "What is the capital of France?",
        "topk": 3,
        "return_scores": True,
    }
    payload_bytes = msgspec.msgpack.encode(payload)

    limiter = asyncio.Semaphore(concurrent_requests)

    async def limited_request(client):
        async with limiter:
            return await make_request(client, url, payload_bytes)

    async with httpx.AsyncClient() as client:
        tasks = [limited_request(client) for _ in range(total_requests)]
        results = await asyncio.gather(*tasks)

    latencies = [latency for latency, success in results if success]
    success_count = len(latencies)
    failure_count = total_requests - success_count
    error_rate = (failure_count / total_requests) * 100 if total_requests > 0 else 0

    print("\n--- Test Results ---")
    print(f"Total Requests: {total_requests}")
    print(f"Successful Requests: {success_count}")
    print(f"Failed Requests: {failure_count}")
    print(f"Error Rate: {error_rate:.2f}%")

    if latencies:
        print("\n--- Latency (for successful requests) ---")
        print(f"Average: {statistics.mean(latencies):.4f} seconds")
        print(f"Median: {statistics.median(latencies):.4f} seconds")
        if len(latencies) > 1:
            print(f"Standard Deviation: {statistics.stdev(latencies):.4f} seconds")
        print(f"Min: {min(latencies):.4f} seconds")
        print(f"Max: {max(latencies):.4f} seconds")
    else:
        print("\n--- Latency ---")
        print("No successful requests to calculate latency.")

    print("-" * 30)


def main():
    """
    Run with:
    python -m tests.test_tool.test_search_engine_reliability --url http://search-engine/retrieve --concurrent_requests 50 --total_requests 1000
    python -m tests.test_tool.test_search_engine_reliability --url http://localhost:8000/retrieve --concurrent_requests 10 --total_requests 100
    """
    fire.Fire(run_test)


if __name__ == "__main__":
    main()
