from transformers import AutoTokenizer

import gem
from gem.envs.multi_turn import MultiTurnEnv
from gem.utils.debug import run_and_print_episode
from gem.wrappers.stateful_observation import (ChatTemplatedObservation,
                                               ConcatenatedObservation)


def test():
    env: MultiTurnEnv = gem.make("ta:GuessTheNumber-v0", max_turns=3)
    policy = lambda _: env.sample_random_action()

    print("\n" * 5, "EPISODE 1: DEFAULT OBSERVATION")
    run_and_print_episode(env, policy)

    print("\n" * 5, "EPISODE 2: CONCATENATED OBSERVATION")
    wrapped_env = ConcatenatedObservation(env)
    run_and_print_episode(wrapped_env, policy)

    print("\n" * 5, "EPISODE 3: CHAT TEMPLATE OBSERVATION")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    wrapped_env = ChatTemplatedObservation(env, tokenizer)
    run_and_print_episode(wrapped_env, policy)


if __name__ == "__main__":
    test()
