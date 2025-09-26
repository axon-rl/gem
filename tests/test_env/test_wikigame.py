import fire

import gem
from gem.utils.debug import run_and_print_episode
from gem.wrappers.wrapper_factory import WRAPPER_FACTORY

def test_correct_termination_behavior():
    '''
    WikiGame should terminate in only 2 ways:
    1. The agent reaches the target page
    2. The agent runs out of turns
    Any other termination condition is a bug and deserves closer inspection.
    '''
    for env_name in [
        "game:WikiGame-v0-easy",
        "game:WikiGame-v0-hard",
    ]:
        env = gem.make(env_name)
        wrapped_env = WRAPPER_FACTORY["concat"](env)
        run_and_print_episode(
            wrapped_env,
            lambda _: env.sample_random_action(),
            ignore_done=False,
        )
        assert env.turn_count <= env.max_turns, f"Turn count limits violated in {env_name}"
        assert (
            env.current_page.title == env.target_page.title 
            or env.turn_count == env.max_turns
        ), f"Episode met unexpected termination condition in {env_name}"
        print(f"Test passed for {env_name}")


def test():
    print("TESTING IF WIKIGAME TERMINATES CORRECTLY")
    for _ in range(5):
        test_correct_termination_behavior()

if __name__ == "__main__":
    fire.Fire(test)

    print(f"\n\nAll tests run.\n\n")