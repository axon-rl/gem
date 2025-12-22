from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List
from .rewards import WikiGameReward

class WikiGameDynamics(ABC):
    """
    Strategy interface for Wikipedia Game dynamics (variants).
    Handles instructions, backtracking rules, and dead-end behavior.
    """

    @abstractmethod
    def reset(self) -> None:
        """Resets any dynamics-specific state if necessary."""
        pass

    @abstractmethod
    def get_instruction_snippet(self) -> str:
        """Returns the text describing the movement rules for the prompt."""
        pass

    @abstractmethod
    def valid_pages(self, current_page) -> List[str]:
        """Returns the set of valid neighboring page titles from the current page."""
        pass

    def construct_invalid_obs(self, turn_count, current_page, next_page_title):
        # We try to fuzzy match for the cases where the model KIND OF
        # knows the neighboring page's title, but made a small mistake which
        # resulted in an invalid action.
        fuzzy_matches = (
            link for link in self.valid_pages(current_page)
            if next_page_title.lower() in link.lower()
        )
        snippet = None
        try:
            snippet = f"Did you mean '{next(fuzzy_matches)}'?"
        except StopIteration:
            snippet = "and neither could it be construed as a valid neighboring page title of it. "

        return (
            f"At turn {turn_count}, you guessed '{next_page_title}', "
            f"which is neither a neighboring page of '{current_page.title}', {snippet}"
            f"You must match **exactly** one of the neighboring page titles to navigate there."
        )

    @abstractmethod
    def handle_backtrack(self, env) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Handles the <PREV_PAGE> action.
        Returns the standard step tuple: (obs, reward, done, truncated, info)
        """
        pass

    @abstractmethod
    def handle_dead_end(self, env, next_page) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Handles reaching a page with no links.
        Returns the standard step tuple: (obs, reward, done, truncated, info)
        """
        pass


class NoRegretsDynamics(WikiGameDynamics):

    def reset(self) -> None:
        pass

    def valid_pages(self, current_page) -> List[str]:
        return current_page.links

    def get_instruction_snippet(self) -> str:
        return (
            "Choose wisely, as you may NOT backtrack; "
            "you can only visit neighboring pages (though you may revisit pages). "
            "Note that trying to backtrack is an illegal move and will "
            "still count towards your maximum number of turns.\n"
        )
    
    def handle_backtrack(self, env) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        next_obs = (
            f"At turn {env.turn_count}, you attempted to backtrack "
            "to the previous page, but backtracking is not allowed "
            "in this variant of the Wikipedia Game."
        )
        return (
            next_obs, 
            WikiGameReward.invalid_action_reward, 
            False, 
            env.turn_count == env.max_turns, 
            {"suffix": env.get_task_suffix()}
        )

    def handle_dead_end(self, env, next_page) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        env.current_page = next_page
        terminate_obs = f"At turn {env.turn_count}, you navigated to the '{env.current_page.title}' page, which is a dead-end page with no neighboring pages."
        return (
            terminate_obs, 
            WikiGameReward.fail_reward, 
            True, 
            False, 
            {"suffix": env.get_task_suffix()}
        )

class EideticDynamics(NoRegretsDynamics):
    '''
    Actually, none of the prompts need to change.
    From the model's POV it can always see all the links it's seen before,
    and doesn't know this is because we are keeping tabs for it. 
    '''

    def __init__(self):
        super().__init__()
        self.perfect_memory = set()

    def reset(self) -> None:
        self.perfect_memory.clear()

    def valid_pages(self, current_page) -> List[str]:
        self.perfect_memory.update(current_page.links)
        return list(self.perfect_memory)

class OneBackDynamics(WikiGameDynamics):

    def reset(self) -> None:
        pass
    
    def valid_pages(self, current_page) -> List[str]:
        return current_page.links

    def get_instruction_snippet(self) -> str:
        return (
            "If you wish, you may respond \\boxed{<PREV_PAGE>} to return "
            "to the previous page. Do note, however, that this counts "
            "towards your maximum number of turns. "
            "Note that you may NOT backtrack twice in a row. \n"
        )

    def handle_backtrack(self, env) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        # Step 1a: If no last page (such as during 1st turn), invalid action.
        if len(env.page_history) < 2:
            next_obs = (
                f"At turn {env.turn_count}, you attempted to backtrack "
                "to the previous page, but there is no previous page "
                "that you can return to."
            )
            reward = WikiGameReward.invalid_action_reward
        # Step 1b: If already backtracked in 'oneback' variant, invalid action.
        elif env.backtracked:
            next_obs = (
                f"At turn {env.turn_count}, you attempted to backtrack "
                "to the previous page, but you have already backtracked "
                "once and cannot do so again."
            )
            reward = WikiGameReward.invalid_action_reward
        # Step 1c: Valid backtrack.
        else:
            env.page_history.pop()
            env.current_page = env.page_history[-1]
            env.backtracked = True
            next_obs = (
                f"At turn {env.turn_count}, you backtracked to the "
                f"previous page '{env.current_page.title}'. "
                f"Here is a summary of the page:\n{env._page_summary(env.current_page)}...\n"
            )
            reward = WikiGameReward.internal_step_reward
            
        return (
            next_obs, 
            reward, 
            False, 
            env.turn_count == env.max_turns, 
            {"suffix": env.get_task_suffix()}
        )

    def handle_dead_end(self, env, next_page) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        env.backtracked = True
        next_obs = (
            f"At turn {env.turn_count}, you reached a dead-end page "
            f"'{next_page.title}' with no neighboring pages. "
            f"You have been automatically backtracked to the previous page "
            f"'{env.current_page.title}' at the cost of an additional turn. "
            f"So now, you are at turn {env.turn_count + 1}. "
            f"Here is a summary of the page:\n{env._page_summary(env.current_page)}...\n"
        )
        env.turn_count += 1
        return (
            next_obs, 
            2 * WikiGameReward.internal_step_reward, 
            False, 
            env.turn_count >= env.max_turns, 
            {"suffix": env.get_task_suffix()}
        )


class FreeNavDynamics(OneBackDynamics):

    def reset(self) -> None:
        pass
    
    def valid_pages(self, current_page) -> List[str]:
        return current_page.links

    def get_instruction_snippet(self) -> str:
        return (
            "If you wish, you may respond \\boxed{<PREV_PAGE>} to return "
            "to the previous page. Do note, however, that this counts "
            "towards your maximum number of turns. \n"
        )

    def handle_backtrack(self, env) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        # Override OneBack to remove the 'already backtracked' check
        if len(env.page_history) < 2:
            next_obs = (
                f"At turn {env.turn_count}, you attempted to backtrack "
                "to the previous page, but there is no previous page "
                "that you can return to."
            )
            reward = WikiGameReward.invalid_action_reward
        else:
            env.page_history.pop()
            env.current_page = env.page_history[-1]
            env.backtracked = True
            next_obs = (
                f"At turn {env.turn_count}, you backtracked to the "
                f"previous page '{env.current_page.title}'. "
                f"Here is a summary of the page:\n{env._page_summary(env.current_page)}...\n"
            )
            reward = WikiGameReward.internal_step_reward
            
        return (
            next_obs, 
            reward, 
            False, 
            env.turn_count == env.max_turns, 
            {"suffix": env.get_task_suffix()}
        )