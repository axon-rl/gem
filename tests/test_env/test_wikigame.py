import fire

import gem
from gem.utils.debug import run_and_print_episode
from gem.wrappers.wrapper_factory import WRAPPER_FACTORY

from typing import Literal

def test_correct_termination_behavior(backend: Literal["mw", "kiwix"] = "kiwix"):
    '''
    WikiGame should terminate in only 2 ways:
    1. The agent reaches the target page
    2. The agent runs out of turns
    Any other termination condition is a bug and deserves closer inspection.
    '''
    if backend == "kiwix":
        trawler_kwargs = {
            "url": "http://localhost:8080",
            "zimfile": "wikipedia_en_simple_all_nopic_2025-09",
            "query_delay_ms": 0,
        }
    else:
        trawler_kwargs = {
            "url": "https://en.wikipedia.org/w/api.php",
            "query_delay_ms": 25,
        }

    for env_name in [
        "game:WikiGame-v0-easy",
        "game:WikiGame-v0-hard",
    ]:
        env = gem.make(env_name, backend = backend, trawler_kwargs = trawler_kwargs)
        wrapped_env = WRAPPER_FACTORY["concat"](env)
        run_and_print_episode(
            wrapped_env,
            lambda _: env.sample_random_action(),
            ignore_done=False,
        )
        assert env.turn_count <= env.max_turns, f"Turn count limits violated in {env_name}"
        assert (
            env.current_page.title == env.target_page.title 
            or not(env.current_page.links)
            or env.turn_count == env.max_turns
        ), f"Episode met unexpected termination condition in {env_name}"
        print(f"Test passed for {env_name}")


def test():
    print("TESTING IF WIKIGAME TERMINATES CORRECTLY")
    for _ in range(50):
        test_correct_termination_behavior(backend = "mw")
        test_correct_termination_behavior(backend = "kiwix")

if __name__ == "__main__":
    fire.Fire(test)

    print(f"\n\nAll tests run.\n\n")