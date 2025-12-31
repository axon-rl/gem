import fire

import gem
from gem.utils.debug import run_and_print_episode
from gem.wrappers.wrapper_factory import WRAPPER_FACTORY
from gem.envs.game_env.wikigame.rewards import WikiGameReward

from typing import Literal

TA_BACKENDS = Literal["mw", "kiwix"]

def get_wikigame_env(backend: TA_BACKENDS, difficulty = 'easy', page_summary_length = (150, 'characters'), variant: str = 'noregrets'):
    if backend == "kiwix":
        trawler_kwargs = {
            "url": "http://localhost:8080",
            "zimfile": "wikipedia_en_simple_all_nopic_2025-11", # Change as needed
            "query_delay_ms": 0,
        }
    else:
        trawler_kwargs = {
            "url": "https://simple.wikipedia.org/w/api.php",
            "query_delay_ms": 25,
        }
    
    env = gem.make(
        f"game:WikiGame-v0-{difficulty}", 
        backend = backend, 
        trawler_kwargs = trawler_kwargs,
        page_summary_length = page_summary_length,
        variant = variant,
    )
    return env

def test_transition_correctness():
    '''
    Test that the transitions in the WikiGame environment are correct.

    Note that we ALWAYS use the Kiwix backend for these tests, so we don't torture the live MediaWiki servers.
    '''

    # Test 1: Default, noregrets kiwix backend.
    env = get_wikigame_env(backend = "kiwix", difficulty = "easy", page_summary_length = (150, 'characters'), variant = 'noregrets')
    
    # Restart on Python page.
    env._reset_fixed_page("Python (programming language)", "Artificial intelligence")
    assert env.current_page.title == "Python (programming language)", "Failed to reset to fixed start page."
    assert env.target_page.title == "Artificial intelligence", "Failed to reset to fixed target page."

    # 1a. Test valid transitions
    print("=" * 10 + "\n\nTEST 1A: Valid transition to Guido van Rossum" + "\n\n" + "=" * 10)
    ortti = env.step(r"\\boxed{Guido_van_Rossum}")
    assert "Guido van Rossum" in ortti[0], (
        f"Valid transition message not found. Message: {ortti[0]}"
    )
    assert env.current_page.title == "Guido van Rossum", (
        f"Current page not updated correctly after valid transition. "
        f"Expected 'Guido van Rossum', got '{env.current_page.title}'."
    )
    assert not (ortti[2] or ortti[3]), "Episode terminated prematurely after valid transition."
    assert ortti[1] == WikiGameReward.internal_step_reward, (
        f"Incorrect reward for VALID transition. Expected {WikiGameReward.internal_step_reward}, got {ortti[1]}."
    )

    # 1b. Test invalid transition
    print("=" * 10 + "\n\nTEST 1B: Invalid transition to non-existent page" + "\n\n" + "=" * 10)
    env._reset_fixed_page("Python (programming language)", "Artificial intelligence")
    ortti = env.step(r"\\boxed{I Think I Am A Wikipedia Page}") # Obvious bogus page
    assert env.current_page.title == "Python (programming language)", (
        f"Current page changed after invalid transition, which is incorrect. "
        f"Expected 'Python (programming language)', got '{env.current_page.title}'."
    )
    assert not (ortti[2] or ortti[3]), "Episode terminated prematurely after invalid transition."
    assert ortti[1] == WikiGameReward.invalid_action_reward, "Incorrect penalty for INVALID transition."

    # 1c. Test format error transition
    print("=" * 10 + "\n\nTEST 1C: Format error transition" + "\n\n" + "=" * 10)
    env._reset_fixed_page("Python (programming language)", "Artificial intelligence")
    ortti = env.step(r"Guido_van_Rossum")  # Missing \boxed{}
    assert env.current_page.title == "Python (programming language)", (
        f"Current page changed after format error, which is incorrect. "
        f"Expected 'Python (programming language)', got '{env.current_page.title}'."
    )
    assert not (ortti[2] or ortti[3]), "Episode terminated prematurely after format error."
    assert ortti[1] == WikiGameReward.format_error_reward, "Incorrect penalty for FORMAT ERROR transition."

    # Test 2: Default, adjacent to target, kiwix backend.

    env = get_wikigame_env(backend = "kiwix", difficulty = "easy", page_summary_length = (150, 'characters'), variant = 'noregrets')
    env._reset_fixed_page("Python (programming language)", "Guido van Rossum")
    assert env.current_page.title == "Python (programming language)", "Failed to reset to fixed start page."
    assert env.target_page.title == "Guido van Rossum", "Failed to reset to fixed target page."

    # 2a. Move to target page
    print("=" * 10 + "\n\nTEST 2A: Valid transition to target page Guido van Rossum" + "\n\n" + "=" * 10)
    ortti = env.step(r"\\boxed{Guido_van_Rossum}")
    assert env.current_page.title == "Guido van Rossum", (
        f"Current page not updated correctly after valid transition to target. "
        f"Expected 'Guido van Rossum', got '{env.current_page.title}'."
    )
    assert (ortti[2] or ortti[3]), "Episode did not terminate upon reaching target page."
    assert ortti[1] == WikiGameReward.success_reward, (
        f"Incorrect reward for reaching target page. Expected {WikiGameReward.success_reward}, got {ortti[1]}."
    )

    # Test 3: Dead-end page, kiwix backend.
    env = get_wikigame_env(backend = "kiwix", difficulty = "easy", page_summary_length = (150, 'characters'), variant = 'noregrets')
    env._reset_fixed_page("Euler's totient function", "Artificial intelligence")
    assert env.current_page.title == "Euler's totient function", "Failed to reset to fixed start page."
    assert env.target_page.title == "Artificial intelligence", "Failed to reset to fixed target page."

    # 3a. Make valid transition to our identified dead-end page
    # Funnily enough, the ring-theoretic concept of a unit is a dead-end page (as far as the Nov 2025 edition of Simple Wikipedia goes).
    print("=" * 10 + "\n\nTEST 3A: Valid transition to dead-end page Unit (ring theory)" + "\n\n" + "=" * 10)
    ortti = env.step(r"\\boxed{Unit_(ring_theory)}")
    assert env.current_page.title == "Unit (ring theory)", (
        f"Current page not updated correctly after valid transition. "
        f"Expected 'Unit (ring theory)', got '{env.current_page.title}'."
    )
    assert (ortti[2] or ortti[3]), "Episode did not terminate upon reaching dead-end page."
    assert ortti[1] == WikiGameReward.fail_reward, (
        f"Incorrect reward for reaching dead-end page. Expected {WikiGameReward.fail_reward}, got {ortti[1]}."
    )

    # 3b. Do the same with another variant. This should yield different behaviors.
    env = get_wikigame_env(backend = "kiwix", difficulty = "easy", page_summary_length = (150, 'characters'), variant = 'freenav')
    env._reset_fixed_page("Euler's totient function", "Artificial intelligence")
    assert env.current_page.title == "Euler's totient function", "Failed to reset to fixed start page."
    assert env.target_page.title == "Artificial intelligence", "Failed to reset to fixed target page."  

    # 3b.i Make valid transition to our identified dead-end page
    print("=" * 10 + "\n\nTEST 3B: Valid transition to dead-end page Unit (ring theory) in freenav variant" + "\n\n" + "=" * 10)
    ortti = env.step(r"\\boxed{Unit_(ring_theory)}")
    assert env.current_page.title == "Euler's totient function", (
        f"Did not bounce back to previous page after hitting a dead-end."
        f"Expected 'Euler's totient function', got '{env.current_page.title}'. {ortti}"
    )
    assert not (ortti[2] or ortti[3]), "Episode terminated prematurely in freenav variant."
    assert ortti[1] == 2 * WikiGameReward.internal_step_reward, (
        f"Incorrect reward for VALID transition in freenav variant. Expected {WikiGameReward.internal_step_reward}, got {ortti[1]}."
    )

    # Test 4: Check that turn limits are enforced correctly.

    for difficulty in [
        "easy",
        "hard",
    ]:
        env = get_wikigame_env(backend = "kiwix", difficulty = difficulty, page_summary_length = (150, 'characters'), variant = 'noregrets')
        run_and_print_episode(
            env,
            lambda _: env.sample_random_action(),
            ignore_done=False,
        )
        assert env.turn_count <= env.max_turns, f"Turn count limits violated in difficulty {difficulty}"
        assert (
            env.current_page.title == env.target_page.title 
            or not(env.current_page.links)
            or env.turn_count == env.max_turns
        ), f"Episode met unexpected termination condition in difficulty {difficulty}"
    
    print("All transition correctness tests passed.")

def test_oneback():
    '''
    Test that the oneback variant of the WikiGame environment works as intended.
    '''
    env = get_wikigame_env(backend = "kiwix", difficulty = "easy", page_summary_length = (150, 'characters'), variant = 'oneback')

    # Test: Basically, we are just here to check that you cannot go back to TWICE in a row.
    # BUT, you can go back twice if you first go to another page.

    # Test 1: Cannot go back twice in a row.
    print("=" * 10 + "\n\nTEST 1: Cannot go back twice in a row." + "\n\n" + "=" * 10)
    env._reset_fixed_page("Python (programming language)", "Artificial intelligence")

    # Step 1: Go to Guido van Rossum
    ortti = env.step(r"\\boxed{Guido_van_Rossum}")
    print(f"T1.1: After going to Guido van Rossum, {ortti}")

    # Step 2: Go to Programmer
    ortti = env.step(r"\\boxed{Programmer}")
    print(f"T1.2: After going to Programmer, {ortti}")

    # Step 3: Go back to Guido van Rossum
    ortti = env.step(r"\\boxed{<PREV_PAGE>}")
    print(f"T1.3: After going back, {ortti}")
    assert env.current_page.title == "Guido van Rossum", "Failed to go back to previous page."

    # Step 4: Try to go back to Python (programming language) - should be disallowed.
    ortti = env.step(r"\\boxed{<PREV_PAGE>}")
    print(f"T1.4: After trying to go back twice, {ortti}")
    assert env.current_page.title == "Guido van Rossum", "Went back twice in a row, which should be disallowed."
    assert ortti[1] == WikiGameReward.invalid_action_reward, "Incorrect reward for INVALID action when trying to go back twice in a row."

    # Test 2: Can go back twice if we first go to another page.
    print("=" * 10 + "\n\nTEST 2: Can go back twice if we first go to another page." + "\n\n" + "=" * 10)
    env._reset_fixed_page("Python (programming language)", "Artificial intelligence")

    # Step 1: Go to Guido van Rossum
    ortti = env.step(r"\\boxed{Guido_van_Rossum}")  
    print(f"T2.1: After going to Guido van Rossum, {ortti}")

    # Step 2: Go to Programmer
    ortti = env.step(r"\\boxed{Programmer}")
    print(f"T2.2: After going to Programmer, {ortti}")

    # Step 3: Go back to Guido van Rossum
    ortti = env.step(r"\\boxed{<PREV_PAGE>}")
    print(f"T2.3: After going back, {ortti}")
    assert env.current_page.title == "Guido van Rossum", "Failed to go back to previous page."

    # Step 4: Go to Netherlands
    ortti = env.step(r"\\boxed{Netherlands}")
    print(f"T2.4: After going to Netherlands, {ortti}")
    assert env.current_page.title == "Netherlands", "Failed to go to Netherlands page."

    # Step 5: Now go back to Guido van Rossum again
    ortti = env.step(r"\\boxed{<PREV_PAGE>}")
    print(f"T2.5: After going back again, {ortti}")
    assert env.current_page.title == "Guido van Rossum", "Failed to go back to previous page after visiting another page."

    # Test 3: For obvious reasons, cannot go back on first move.
    print("=" * 10 + "\n\nTEST 3: Cannot go back on first move." + "\n\n" + "=" * 10)
    env._reset_fixed_page("Python (programming language)", "Artificial intelligence")
    ortti = env.step(r"\\boxed{<PREV_PAGE>}")
    print(f"T3.1: After trying to go back on first move, {ortti}")
    assert env.current_page.title == "Python (programming language)", "Went back on first move, which should be disallowed in oneback variant."
    assert ortti[1] == WikiGameReward.invalid_action_reward, "Incorrect reward for INVALID action when trying to go back on first move."
    
    # Test 4: Ensure that we automatically get bounced if we reach a dead-end page.
    print("=" * 10 + "\n\nTEST 4: Automatically get bounced if we reach a dead-end page." + "\n\n" + "=" * 10)
    env._reset_fixed_page("Euler's totient function", "Artificial intelligence")
    ortti = env.step(r"\\boxed{Unit_(ring_theory)}")
    print(f"T4.1: After going to dead-end page, {ortti}")
    assert not (ortti[2] or ortti[3]), "We should NOT be done yet in oneback variant."
    assert env.current_page.title == "Euler's totient function", "After reaching dead-end, we should be bounced back to previous page."
    assert ortti[1] == 2 * WikiGameReward.internal_step_reward, "Incorrect reward after being bounced back from dead-end."
    assert env.backtracked, "Backtracked flag not set after being bounced back from dead-end."
    assert env.turn_count == 2, "Turn count not incremented correctly after being bounced back from dead-end."

    # Test 5: Ensure after being bounced back, we cannot go back again immediately.
    env._reset_fixed_page("RSA algorithm", "Artificial intelligence")
    ortti = env.step(r"\\boxed{Euler's_totient_function}")
    print(f"T5.1: After going to Euler's totient function, {ortti}")
    assert env.current_page.title == "Euler's totient function", "Failed to go to Euler's totient function."
    ortti = env.step(r"\\boxed{Unit_(ring_theory)}")
    print(f"T5.2: After going to dead-end page, {ortti}")
    assert env.current_page.title == "Euler's totient function", "After reaching dead-end, we should be bounced back to previous page."
    ortti = env.step(r"\\boxed{<PREV_PAGE>}")
    print(f"T5.3: After trying to go back again, {ortti}")
    assert env.current_page.title == "Euler's totient function", "Trying to go back after being bounced should be disallowed in oneback variant."
    assert ortti[1] == WikiGameReward.invalid_action_reward, "Incorrect reward for INVALID action when trying to go back after being bounced."

    print("Oneback variant tests passed.")

def test_freenav():
    '''
    Test that the freenav variant of the WikiGame environment works as intended.
    '''
    env = get_wikigame_env(backend = "kiwix", difficulty = "easy", page_summary_length = (150, 'characters'), variant = 'freenav')

    # Test: Nearly the same as oneback, but we should always be allowed to go back unless on first move.

    # Test 1: Can go back twice in a row.
    env._reset_fixed_page("Python (programming language)", "Artificial intelligence")

    # Step 1: Go to Guido van Rossum
    ortti = env.step(r"\\boxed{Guido_van_Rossum}")
    print(f"T1.1: After going to Guido van Rossum, {ortti}")

    # Step 2: Go to Programmer
    ortti = env.step(r"\\boxed{Programmer}")
    print(f"T1.2: After going to Programmer, {ortti}")

    # Step 3: Go back to Guido van Rossum
    ortti = env.step(r"\\boxed{<PREV_PAGE>}")
    print(f"T1.3: After going back, {ortti}")
    assert env.current_page.title == "Guido van Rossum", "Failed to go back to previous page."

    # Step 4: Try to go back to Python (programming language) - should be allowed.
    ortti = env.step(r"\\boxed{<PREV_PAGE>}")
    print(f"T1.4: After trying to go back twice, {ortti}")
    assert env.current_page.title == "Python (programming language)", "Failed to go back twice in a row, which should be allowed in freenav variant."
    assert ortti[1] == WikiGameReward.internal_step_reward, "Incorrect reward for going back twice in a row."

    # Test 2: Can go back twice if we first go to another page.
    env._reset_fixed_page("Python (programming language)", "Artificial intelligence")

    # Step 1: Go to Guido van Rossum
    ortti = env.step(r"\\boxed{Guido_van_Rossum}")  
    print(f"T2.1: After going to Guido van Rossum, {ortti}")

    # Step 2: Go to Programmer
    ortti = env.step(r"\\boxed{Programmer}")
    print(f"T2.2: After going to Programmer, {ortti}")

    # Step 3: Go back to Guido van Rossum
    ortti = env.step(r"\\boxed{<PREV_PAGE>}")
    print(f"T2.3: After going back, {ortti}")
    assert env.current_page.title == "Guido van Rossum", "Failed to go back to previous page."

    # Step 4: Go to Netherlands
    ortti = env.step(r"\\boxed{Netherlands}")
    print(f"T2.4: After going to Netherlands, {ortti}")
    assert env.current_page.title == "Netherlands", "Failed to go to Netherlands page."

    # Step 5: Now go back to Guido van Rossum again
    ortti = env.step(r"\\boxed{<PREV_PAGE>}")
    print(f"T2.5: After going back again, {ortti}")
    assert env.current_page.title == "Guido van Rossum", "Failed to go back to previous page after visiting another page."

    # Test 3: For obvious reasons, cannot go back on first move.
    env._reset_fixed_page("Python (programming language)", "Artificial intelligence")
    ortti = env.step(r"\\boxed{<PREV_PAGE>}")
    print(f"T3.1: After trying to go back on first move, {ortti}")
    assert env.current_page.title == "Python (programming language)", "Went back on first move, which should be disallowed."
    assert ortti[1] == WikiGameReward.invalid_action_reward, "Incorrect reward for INVALID action when trying to go back on first move."
    
    # Test 4: Ensure that we automatically get bounced if we reach a dead-end page.
    env._reset_fixed_page("Euler's totient function", "Artificial intelligence")
    ortti = env.step(r"\\boxed{Unit_(ring_theory)}")
    print(f"T4.1: After going to dead-end page, {ortti}")
    assert not (ortti[2] or ortti[3]), "We should NOT be done yet in oneback variant."
    assert env.current_page.title == "Euler's totient function", "After reaching dead-end, we should be bounced back to previous page."
    assert ortti[1] == 2 * WikiGameReward.internal_step_reward, "Incorrect reward after being bounced back from dead-end."
    assert env.backtracked, "Backtracked flag not set after being bounced back from dead-end."
    assert env.turn_count == 2, "Turn count not incremented correctly after being bounced back from dead-end."

    # Test 5: Ensure after being bounced back, we can still go back.
    env._reset_fixed_page("RSA algorithm", "Artificial intelligence")
    ortti = env.step(r"\\boxed{Euler's_totient_function}")
    print(f"T5.1: After going to Euler's totient function, {ortti}")
    assert env.current_page.title == "Euler's totient function", "Failed to go to Euler's totient function."
    ortti = env.step(r"\\boxed{Unit_(ring_theory)}")
    print(f"T5.2: After going to dead-end page, {ortti}")
    assert env.current_page.title == "Euler's totient function", "After reaching dead-end, we should be bounced back to previous page."
    ortti = env.step(r"\\boxed{<PREV_PAGE>}")
    print(f"T5.3: After trying to go back again, {ortti}")
    assert env.current_page.title == "RSA algorithm", "Failed to go back to previous page after being bounced."
    assert ortti[1] == WikiGameReward.internal_step_reward, "Incorrect reward for going back after being bounced within freenav variant."

    print("Freenav variant tests passed.")

def kiwix_stress_test():
    ''''
    Run a stress test on a Kiwix backend with massively parallel requests.
    '''
    # env = gem.make("game:WikiGame-v0-easy", backend = "kiwix", trawler_kwargs = {
    #     "url": "http://localhost:8080",
    #     "zimfile": "wikipedia_en_simple_all_nopic_2025-09",
    #     "query_delay_ms": 0,
    # })
    NUM_ENVS = 64
    vec_env = gem.make_vec(
        ["game:WikiGame-v0-easy"] * 64,
        vec_kwargs=[{
            "backend": "kiwix",
            "trawler_kwargs": {
                "url": "http://localhost:8080",
                "zimfile": "wikipedia_en_simple_all_nopic_2025-11", # Change as needed
                "query_delay_ms": 0,
                "query_use_cache": True,
            },
        }] * NUM_ENVS,
        seed = int(1e9),
    )
    run_and_print_episode(
        vec_env,
        lambda _: [vec_env.envs[i].sample_random_action() for i in range(NUM_ENVS)],
        ignore_done = True,
    )

def test():
    test_transition_correctness()
    test_oneback()
    test_freenav()
    kiwix_stress_test()

if __name__ == "__main__":
    fire.Fire({
        'full_test': test,
        'test_transition': test_transition_correctness,
        'kiwix_stress_test': kiwix_stress_test,
        'test_oneback': test_oneback,
        'test_freenav': test_freenav,
    })

    print(f"\n\nAll tests run.\n\n")
    ''''
    Run with:
        python -m tests.test_env.test_wikigame test_transition
        python -m tests.test_env.test_wikigame <YOUR_FUNCTION_NAME>

    FAIR WARNING:
    Unfortunately, the Wikigame backends all rely on shaky assumptions about the layout
    of either the live MediaWiki servers or the Kiwix ZIM files. 
    - For MediaWiki, this is due to the API specification.
    - For Kiwix, this is due to the fact that the ZIM files are periodically updated
    and there is no guarantee that the link structure remains the same.

    Therefore, it is difficult to write tests that will remain valid across time.
    As of December '25, these tests are valid for the Simple English Wikipedia
    as hosted on both the live MediaWiki servers and the Kiwix ZIM file
    "wikipedia_en_simple_all_nopic_2025-11".
    Future changes to either the live servers or the ZIM file may cause these tests to fail
    even if the code is conceptually correct.
    '''