#!/usr/bin/env python3
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
from tau_bench_agent import TauBenchAgent
from tau_bench_env import TauBenchEnv


def eval_task(args):
    task_idx, model, provider, user_model, user_provider = args
    try:
        env = TauBenchEnv(
            task_split="test", user_model=user_model, user_provider=user_provider
        )
        agent = TauBenchAgent(model=model, provider=provider, temperature=0.0)
        result = agent.solve(env, task_index=task_idx)
        return task_idx, result["reward"]
    except Exception as e:
        print(f"Task {task_idx} error: {e}")
        return task_idx, 0.0


if __name__ == "__main__":
    # OpenAI: model="gpt-4o", provider="openai"
    # Gemini: model="google/gemini-2.0-flash-001", provider="openrouter"
    # DeepSeek: model="deepseek/deepseek-chat", provider="openrouter"
    # Claude: model="anthropic/claude-3.5-sonnet", provider="openrouter"

    model = "gpt-4o"
    provider = "openai"
    user_model = "gpt-4o"
    user_provider = "openai"

    print(f"Running 115 tasks with {model} via {provider}")
    print(f"User simulator: {user_model} via {user_provider}")
    print("=" * 60)

    tasks = [(i, model, provider, user_model, user_provider) for i in range(115)]
    results = []
    passed = 0

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(eval_task, args): args[0] for args in tasks}

        for future in as_completed(futures):
            task_idx, reward = future.result()
            results.append((task_idx, reward))

            if reward > 0:
                passed += 1

            completed = len(results)
            print(
                f"Task {task_idx}: {'✓' if reward > 0 else '✗'} | "
                f"{completed}/115 | Pass@1: {passed}/{completed} ({100*passed/completed:.1f}%)"
            )

    print(f"\n{'='*60}")
    print(f"FINAL: {passed}/115 ({100*passed/115:.1f}%)")
    print(f"Target: 60.4%")
    print(f"{'='*60}")
