from transformers import AutoTokenizer

import gem
from gem.envs.multi_turn import MultiTurnEnv
from gem.utils.debug import run_and_print_episode
from gem.wrappers.stateful_observation import (ChatTemplatedObservation,
                                               ConcatenatedObservation)


def test():
    env: MultiTurnEnv = gem.make("ta:GuessTheNumber-v0", max_turns=5)
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
    test()
