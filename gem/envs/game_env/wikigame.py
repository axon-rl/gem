# Copyright 2025 AxonRL Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from time import sleep
import datetime
import re
import wikipedia
from wikipedia import WikipediaPage, PageError, DisambiguationError, HTTPTimeoutError
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from gem.core import Env
from gem.utils.constants import LanguageGameReward

WIKIPEDIA_DELAY_SECONDS = 0.01

class WikiGameEnv(Env):
    '''
    Implements a fully text-based Wikipedia Game environment.

    **WARNINGS**: 
    1. By default, this environment does NOT concatenate the 
    full history of observations and actions to the current observation.
    You can do so by wrapping this environment with the `ObservationWrapper`
    in `gem.wrappers.observation_wrapper`.
    2. This environment makes live API calls to Wikipedia.
        a. Excessive usage may get your IP address rate-limited or blocked by Wikimedia.
        b. The environment may be slow due to network latency.
        c. The environment may use a wide range of characters which may exceed some models' vocabularies.

    Environment Description:

    The Wikipedia Game is a fully observable, static, (almost) deterministic, single-agent environment
    where the agent's objective is to navigate from a starting Wikipedia page to a target Wikipedia page
    within a limited number of hyperlink visits (turns).

    State: (p_current, p_target, step_ct):
        - (p_current) The title and plain-text body of the current Wikipedia page, with the exception of hyperlinks,
            whose HTML tags are retained to identify neighboring pages.
        - (p_target) The title of the target Wikipedia page.
        - (step_ct) The current turn count, starting from 0.

    Initial State:
        - (p_current) A random Wikipedia page as the starting page
        - (p_target) A random Wikipedia page as the target page

    Action & Transition: 
        - The title of a neighboring Wikipedia page, specified within \\boxed{} tags.
        - p_current <- any Wikipedia page whose title matches that of the action.
        - If the action does not correspond to a valid neighboring page, p_current remains unchanged.
        - Regardless of the validity of the action, step_ct increments by 1.

    Terminal State:
        - The game **terminates** when the agent reaches the target page (p_current == p_target).
        - The game **truncates** when the agent exhausts the maximum number of allowed turns without reaching the target page
    '''
    def __init__(self, max_turns: int = 10, **_):
        super().__init__()
        # Trivial rate-limiting in order to not get blocked by Wikimedia
        wikipedia.set_rate_limiting(True, min_wait = datetime.timedelta(seconds = WIKIPEDIA_DELAY_SECONDS))
        self.max_turns = max_turns
        self._resolve_page_cache = {}
        self.reset()

    def _get_instructions(self) -> str:
        '''
        Since the model's performance is being judged based on the number of steps taken to reach the target page,
        we help the model perform by making it clear that the model should use its general knowledge
        instead of inefficient backtracking or random exploration.

        Some preliminaries that guide prompting decisions:
        - Concise and least-ambiguous specification of the game
            - e.g. by defining terms like "page", "neighboring page"
        - Clear instructions on how to provide actions
            - e.g. by specifying the format of the action (in box tags)
        - Wikipedia Game is not a game where pure navigation skills can let you excel,
            but rather, general knowledge and familiarity with Wikipedia's structure is key.
        '''
        return (
            f"You are playing the Wikipedia Game, and must reach the target Wikipedia page within {self.max_turns} turns.\n"
            "This game tests your general knowledge, as well as familiarity with how Wikipedia pages are interlinked.\n"
            "A page refers to a webpage.\n"
            "A Wikipedia page is a webpage on Wikipedia containing articles on a specific topic, as identified by its title.\n"
            "A neighboring page is defined as any page accessible via a hyperlink from the current page.\n"
            "You will start at a random Wikipedia page, and must navigate to a target Wikipedia page by visiting neighboring pages.\n"
            "To visit a neighboring page, enter its title wrapped in \\boxed{} tags (e.g., '\\boxed{Python (programming language)}').\n"
            f"You can visit up to {self.max_turns} neighboring pages (excluding the starting page) to reach the target page.\n"
            "As you play, the history of your moves will be appended below. Use the information to navigate to the target page before you run out of turns.\n"
            f"Lastly, you started at '{self.current_page.title}' and your target page is '{self.target_page.title}'.\n"
            f"Here is a summary of the target page:\n{self._page_summary(self.target_page)}...\n"
            "Enter your first guess to start the game.\n"
        )

    def _page_summary(self, page: WikipediaPage) -> str:
        return page.content[:250]
    
    def _get_neighboring_pages_formatted(self, page: WikipediaPage) -> list[str]:
        return '\n- '.join(page.links)

    def _construct_current_page_summary(self) -> str:
        return (
            f"Here is a summary of the page:\n{self._page_summary(self.current_page)}...\n"
            f"This page has the following neighboring pages: {self._get_neighboring_pages_formatted(self.current_page)}"
        )

    def get_task_suffix(self) -> str:
        '''
        Maintain consistent terminology with the instructions to avoid confusion.
        '''
        return (
            f"You are currently on the '{self.current_page.title}' page. Your target page is '{self.target_page.title}'.\n"
            f"{self._construct_current_page_summary()}\n"
            "Enter the title of the neighboring page you want to navigate to."
        )

    def _try_resolve_page(self, page_name: str) -> Optional[WikipediaPage]:
        '''
        Tries to resolve a page title to a WikipediaPage object.
        Returns None if the page cannot be resolved.
        '''
        if page_name in self._resolve_page_cache:
            return self._resolve_page_cache[page_name]

        next_page = None
        attempts = 0
        while next_page is None and attempts < 5:
            try:
                next_page = wikipedia.page(page_name, auto_suggest = False, redirect = True)
            except DisambiguationError as e:
                # This is unlikely to happen since most Wikipedia links should be well-formed.
                # But we handle it just in case.
                next_page = wikipedia.page(e.options[0], auto_suggest = False, redirect = True)
            except PageError:
                # Unfortunately Wikipedia articles get renamed or deleted over time.
                # In these cases, try to resolve the page via search.
                potential_pages = wikipedia.search(page_name, results = 1)
                if len(potential_pages) and potential_pages[0] not in self._resolve_page_cache:
                    next_page = self._try_resolve_page(potential_pages[0])
                else:
                    next_page = None
            except HTTPTimeoutError:
                # Try again
                pass
            # Pokemon Exception Handling :( 
            # But this keeps the environment from crashing which is arguably more annoying.
            except Exception:
                next_page = None
            attempts += 1
        
        self._resolve_page_cache[page_name] = next_page

        return next_page

    def _random_page(self) -> WikipediaPage:
        '''
        Returns a random Wikipedia page.
        '''
        return self._try_resolve_page(wikipedia.random(1))

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        self.start_page: WikipediaPage = None
        self.target_page: WikipediaPage = None

        while self.start_page == self.target_page:
            self.start_page = self._random_page()
            self.target_page = self._random_page()

        self.current_page: WikipediaPage = self.start_page
        self.turn_count: int = 0
        self._resolve_page_cache = {}
        return (self._get_instructions(), {"suffix": self.get_task_suffix()})

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        '''
        Generally obeys the flow of other GEM environments' step functions.
        '''

        self.turn_count += 1
        reward = 0

        try:
            action_search_pattern = re.compile(r"\\boxed{([^}]+)}")
            matches = list(action_search_pattern.finditer(action))
            clean_action = matches[-1] if matches else None
            next_page_title = clean_action.group(1).strip()
        except AttributeError:
            next_page_title = None
        
        if not next_page_title:
            next_obs = f"At turn {self.turn_count}, you did not provide a valid neighboring page title."
            reward = LanguageGameReward.format_error_reward
            return (
                next_obs,
                reward,
                False,
                self.turn_count == self.max_turns,
                {"suffix": self.get_task_suffix()},
            )
        
        # Step 1: Check the title exists on the current page
        if next_page_title not in self.current_page.links:
            next_obs = f"At turn {self.turn_count}, you guessed '{next_page_title}', which is not a neighboring page of '{self.current_page}'."
            reward = LanguageGameReward.invalid_action_reward
        
        # Step 2: Try to load the page
        # As a default, error handling behavior is taken from ARENA 3.0, Chapter 3, Part 4.
        # Source: https://github.com/callummcdougall/ARENA_3.0/blob/main/chapter3_llm_evals/exercises/part4_llm_agents/3.4_LLM_Agents_solutions.ipynb
        next_page = self._try_resolve_page(next_page_title)
        
        if next_page is None:
            next_obs = f"At turn {self.turn_count}, you guessed '{next_page_title}', which could not be loaded due to repeated errors."
            reward = LanguageGameReward.invalid_action_reward

        # Step 3a: Check if we are on the target page.
        elif next_page.title == self.target_page:
            self.current_page = next_page
            terminate_obs = f"Congratulations! You have reached the target page '{self.target_page.title}' in {self.turn_count} turns."
            reward = LanguageGameReward.success_reward
            return (
                terminate_obs,
                reward,
                True,
                False,
                {"suffix": self.get_task_suffix()},
            )
        # Step 3b: Check for maximum turns reached
        elif self.turn_count >= self.max_turns:
            terminate_obs = f"At turn {self.turn_count}, you have reached the maximum number of turns without reaching the target page '{self.target_page.title}'."
            reward = LanguageGameReward.fail_reward
            return (
                terminate_obs,
                reward,
                True,
                True,
                {"suffix": self.get_task_suffix()},
            )
        # Step 3c: Valid action, but not the target page.
        else:
            self.current_page = next_page
            next_obs = (
                f"At turn {self.turn_count}, you navigated to the '{self.current_page.title}' page. "
                f"Here is a summary of the page:\n{self.current_page.content[:500]}...\n"
            )
            reward = LanguageGameReward.internal_step_reward
        
        return next_obs, reward, False, False, {"suffix": self.get_task_suffix()}

    def sample_random_action(self) -> str:
        '''
        Samples a random neighboring page
        '''
        return f"\\boxed{{{random.choice(self.current_page.links)}}}"