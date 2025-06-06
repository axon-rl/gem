from transformers import AutoTokenizer
import fire

import gem
from gem.envs.multi_turn import MultiTurnEnv
from gem.utils.debug import run_and_print_episode
from gem.wrappers.stateful_observation import (ChatTemplatedObservation,
                                               ConcatenatedObservation)


def test(env_name: str = "ta:GuessTheNumber-v0"):
    env: MultiTurnEnv = gem.make(env_name, max_turns=3)
    policy = lambda _: env.sample_random_action()
    run_and_print_episode(env, policy)

    print("\n" * 5)
    wrapped_env = ConcatenatedObservation(env)
    run_and_print_episode(wrapped_env, policy)

    print("\n" * 5)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    wrapped_env = ChatTemplatedObservation(env, tokenizer)
    run_and_print_episode(wrapped_env, policy)


if __name__ == "__main__":
    fire.Fire(test)

    """Run with:
        python -m tests.test_env.test_textarena --env_name ta:GuessTheNumber-v0
        python -m tests.test_env.test_textarena --env_name ta:Mastermind-v0
        python -m tests.test_env.test_textarena --env_name ta:Minesweeper-v0
    """

