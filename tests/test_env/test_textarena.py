from functools import partial

import fire
from transformers import AutoTokenizer

import gem
from gem.utils.debug import run_and_print_episode
from gem.wrappers.wrapper_factory import WRAPPER_FACTORY


def test(env_name: str = "ta:GuessTheNumber-v0"):

    env = gem.make(env_name, max_turns=3)
    policy = lambda _: env.sample_random_action()

    print("\n" * 5, "EPISODE 1: DEFAULT OBSERVATION")
    run_and_print_episode(env, policy)

    print("\n" * 5, "EPISODE 2: CONCATENATED OBSERVATION")
    wrapped_env = WRAPPER_FACTORY["concat"](env)
    run_and_print_episode(wrapped_env, policy)

    print("\n" * 5, "EPISODE 3: CHAT TEMPLATE OBSERVATION")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    wrapped_env = WRAPPER_FACTORY["concat_chat"](env, tokenizer=tokenizer)
    run_and_print_episode(wrapped_env, policy)

    print("\n" * 5, "BATCH EPISODE: VECTORIZED ENV")
    num_envs = 3
    wrapper_fn = partial(WRAPPER_FACTORY["concat_chat"], tokenizer=tokenizer)
    ta_vec_env = gem.make_vec(
        env_name,
        num_envs=num_envs,
        wrappers=[wrapper_fn],
        max_turns=3,
    )
    run_and_print_episode(
        ta_vec_env,
        lambda _: [env.sample_random_action() for _ in range(num_envs)],
        ignore_done=True,
        max_steps=5,
    )


if __name__ == "__main__":
    fire.Fire(test)
    print(f"\n\nAll tests run.\n\n")

    """Run with:
        python -m tests.test_env.test_textarena --env_name ta:GuessTheNumber-v0-sanitycheck
        python -m tests.test_env.test_textarena --env_name ta:GuessTheNumber-v0
        python -m tests.test_env.test_textarena --env_name ta:Mastermind-v0
        python -m tests.test_env.test_textarena --env_name ta:Minesweeper-v0
        python -m tests.test_env.test_textarena --env_name ta:Wordle-v0
        python -m tests.test_env.test_textarena --env_name eval:MATH500
    """
