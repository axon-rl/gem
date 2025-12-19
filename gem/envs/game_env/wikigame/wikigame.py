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
import re
from time import sleep
from typing import Any, Dict, Optional, Tuple

from gem.core import Env
# from gem.utils.constants import LanguageGameReward

from .backend import BaseWikiTrawler, MediaWikiTrawler, KiwixWikiTrawler
from .errors import BackendFailureException
from .wikipage import WikipediaPage
from .rewards import WikiGameReward

class WikiGameEnv(Env):
    VALID_BACKENDS = {
        "mw": MediaWikiTrawler,
        "kiwix": KiwixWikiTrawler
    }
    VALID_SUMMARY_UNITS = [
        'characters',
        'words',
        'sentences',
    ]
    VALID_VARIANTS = [
        # (Default) No backwards navigation; 
        # can only click neighbor pages (may revisit pages)
        'noregrets', 
        # Allows backwards navigation (ONCE)
        'oneback',
        # Allows backwards navigation (free)
        'freenav',
    ]

    '''
    Implements a fully text-based Wikipedia Game environment.

    **WARNINGS**: 
    1. By default, this environment does NOT concatenate the 
    full history of observations and actions to the current observation.
    You can do so by wrapping this environment with the `ObservationWrapper`
    in `gem.wrappers.observation_wrapper`.
    2. This environment CAN use a local Kiwix server (if backend = 'kiwix') or 
        MediaWiki API (if backend = 'mw') to fetch Wikipedia pages.
    3. Alternatively, this environment MAY make live API calls to Wikipedia 
        (if provided URL leads to 'wikipedia.org')
        a. Excessive usage may get your IP address rate-limited or blocked 
            by Wikimedia.
        b. The environment may be slow due to network latency.
        c. The environment may use a wide range of characters which may 
            exceed some models' vocabularies.

    Environment Description:

    The Wikipedia Game is a fully observable, static, (almost) deterministic, 
        single-agent environment where the agent's objective is to navigate 
        from a starting Wikipedia page to a target Wikipedia page.

    There exists many variations of the Wikipedia Game. In light of this,
        this environment is designed to be extensible to different rule-sets.
        For simplicity, however, this environment by default implements a 
        "least-clicks, no-regrets" variant, where the agent must reach the target 
        page within a limited number of hyperlink visits (turns), and is not 
        allowed to backtrack to previously visited pages.

    State: (p_current, p_target, step_ct):
        - (p_current) The title and plain-text body of the current 
            Wikipedia page, with the exception of hyperlinks, whose HTML tags 
            are retained to identify neighboring pages.
        - (p_target) The title of the target Wikipedia page.
        - (step_ct) The current turn count, starting from 0.

    Initial State:
        - (p_current) A random Wikipedia page as the starting page
        - (p_target) A random Wikipedia page as the target page

    Action & Transition: 
        - The title of a neighboring Wikipedia page, specified within \\boxed{} 
            tags.
        - p_current <- any Wikipedia page whose title matches that of the action.
        - If the action does not correspond to a valid neighboring page, p_current 
            remains unchanged.
        - Regardless of the validity of the action, step_ct increments by 1.

    Terminal State:
        - The game **terminates** when the agent reaches the target page 
            (p_current == p_target), or when the agent reaches a "dead-end" page
            (with no outgoing hyperlinks).
        - The game **truncates** when the agent exhausts the maximum number of 
            allowed turns without reaching the target page.
    
    Reward Function:
        - Customizable via `wikigame.rewards.WikiGameReward`.
    '''
    
    def __init__(self, 
        max_turns: int = 10, 
        backend = 'mw', 
        trawler_kwargs = None, 
        page_summary_length: tuple[int, str] = (100, 'characters'), 
        variant = 'noregrets',
        **_
    ):
        super().__init__()
        
        self.max_turns = max_turns
        self.page_summary_length, self.page_summary_length_unit = page_summary_length
        assert self.page_summary_length_unit in self.VALID_SUMMARY_UNITS, (
            f"Invalid page_summary_length unit '{self.page_summary_length_unit}'. "
            f"Valid options are: {self.VALID_SUMMARY_UNITS}"
        )

        try:
            self.trawler: BaseWikiTrawler = self.VALID_BACKENDS[backend](**(trawler_kwargs or {}))
        except KeyError:
            raise ValueError(f"Invalid backend '{backend}'. Valid options are: {list(self.VALID_BACKENDS.keys())}")
        
        if not isinstance(self.trawler, BaseWikiTrawler):
            raise ValueError(f"Backend '{backend}' has to subclass BaseWikiTrawler.")

        self.variant = variant
        assert self.variant in self.VALID_VARIANTS, (
            f"Invalid variant '{self.variant}'. "
            f"Valid options are: {self.VALID_VARIANTS}"
        )

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
        instr = (
            f"You are playing the Wikipedia Game, and must reach the target Wikipedia page within {self.max_turns} turns.\n"
            "This game tests your general knowledge, as well as familiarity with how Wikipedia pages are interlinked.\n"
            "A page refers to a webpage.\n"
            "A Wikipedia page is a webpage on Wikipedia containing articles on a specific topic, as identified by its title.\n"
            "A neighboring page is defined as any page accessible via a hyperlink from the current page.\n"
            "You will start at a random Wikipedia page, and must navigate to a target Wikipedia page by visiting neighboring pages.\n"
            "To visit a neighboring page, enter its EXACT title, wrapped in \\boxed{} tags (for example, \\boxed{Python_(programming_language)}).\n"
        )

        if self.variant in ['oneback', 'freenav']:
            instr += (
                "If you wish, you may respond \\boxed{<PREV_PAGE>} to return "
                "to the previous page. Do note, however, that this counts "
                "towards your maximum number of turns. "
            )
            if self.variant == 'oneback':
                instr += (
                    "Note that you may NOT backtrack twice in a row. "
                )
            instr += "\n"
        else:
            instr += (
                "Choose wisely, as you may not backtrack; "
                "you can only visit neighboring pages (though you may revisit pages). "
                "Note that trying to backtrack is an illegal move and will "
                "still count towards your maximum number of turns.\n"
            )

        return instr + (
            f"You can visit up to {self.max_turns} neighboring pages (excluding the starting page) to reach the target page.\n"
            "As you play, the history of your moves will be appended below. Use the information to navigate to the target page before you run out of turns.\n"
            f"Lastly, you started at '{self.current_page.title}' and your target page is '{self.target_page.title}'.\n"
            f"Here is a summary of the target page:\n{self._page_summary(self.target_page)}...\n"
            "Enter your first guess to start the game.\n"
        )

    def _page_summary(self, page: WikipediaPage) -> str:
        if self.page_summary_length_unit == 'characters':
            summ_length = min(self.page_summary_length, len(page.content))
            return page.content[:summ_length]
        
        elif self.page_summary_length_unit == 'words':
            summ_length = min(self.page_summary_length, len(page.content.split()))
            return ' '.join(page.content.split()[:summ_length])
        
        elif self.page_summary_length_unit == 'sentences':
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', page.content)
            return ' '.join(sentences[:self.page_summary_length])

    # Facilitate copying by formatting as \\boxed tags.
    # This allows models to leverage their induction heads (which most LMs possess)
    # and succeed at navigating to **some page** more often.
    # https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
    def _get_neighboring_pages_formatted(self, page: WikipediaPage) -> list[str]:
        return '\n- ' + '\n- '.join(f"\\boxed{{{link}}}" for link in page.links)

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
    
    def _random_page(self) -> WikipediaPage:
        '''
        Returns a random, disambiguated Wikipedia page.

        Page is guaranteed not to be a dead end (i.e. has at least one outgoing link).
        For obvious reasons, it is unfortunately not guaranteed 
        that we can reach all pages from this one.
        '''
        page: WikipediaPage = None
        # Exponential backoff on backend failure to minimize errors.
        for i in range(10):
            page = self.trawler.random()
            if not isinstance(page, WikipediaPage):
                # Delay for factor of 1.2^i before retrying
                sleep(0.2 * 1.2 ** i)
            elif 'disambiguation' in page.title.lower():
                sleep(0.2)
            else:
                break

        if not isinstance(page, WikipediaPage):
            raise BackendFailureException()
        return page

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed)
        self.start_page: WikipediaPage = None
        self.target_page: WikipediaPage = None

        while self.start_page == self.target_page:
            self.start_page = self._random_page()
            self.target_page = self._random_page()

        self.page_history: list[WikipediaPage] = [self.start_page]
        self.backtracked: bool = False

        self.current_page: WikipediaPage = self.start_page
        self.turn_count: int = 0
        self.trawler.empty_cache()
        return (self._get_instructions(), {"suffix": self.get_task_suffix()})
    
    def _reset_fixed_page(self, start_page_title: str, target_page_title: str, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        '''
        Resets the environment to a fixed start and target page.
        Mostly for testing purposes, not intended for general use.
        '''
        super().reset(seed)
        self.start_page = self.trawler.get_page(start_page_title)
        self.target_page = self.trawler.get_page(target_page_title)

        self.page_history: list[WikipediaPage] = [self.start_page]
        self.backtracked: bool = False

        self.current_page: WikipediaPage = self.start_page
        self.turn_count: int = 0
        self.trawler.empty_cache()
        return (self._get_instructions(), {"suffix": self.get_task_suffix()})


    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        '''
        Generally obeys the flow of other GEM environments' step functions.
        '''

        self.turn_count += 1
        reward = 0

        try:
            matches = list(re.finditer(r"\\boxed{([^}]+)}", action))
            clean_action = matches[-1] if matches else None
            next_page_title = clean_action.group(1).strip()
        except AttributeError:
            next_page_title = None
        
        if not next_page_title:
            next_obs = f"At turn {self.turn_count}, you did not provide a valid neighboring page title."
            reward = WikiGameReward.format_error_reward
            return (
                next_obs,
                reward,
                False,
                self.turn_count == self.max_turns,
                {"suffix": self.get_task_suffix()},
            )
       
        # Step 1: If allowed, handle backtracking.
        if self.variant in ['oneback', 'freenav'] and next_page_title == '<PREV_PAGE>':
            # Step 1a: If no last page (such as during 1st turn), invalid action.
            if len(self.page_history) < 2:
                next_obs = (
                    f"At turn {self.turn_count}, you attempted to backtrack "
                    "to the previous page, but there is no previous page "
                    "that you can return to."
                )
                reward = WikiGameReward.invalid_action_reward
            # Step 1b: If already backtracked in 'oneback' variant, invalid action.
            elif self.variant == 'oneback' and self.backtracked:
                next_obs = (
                    f"At turn {self.turn_count}, you attempted to backtrack "
                    "to the previous page, but you have already backtracked "
                    "once and cannot do so again."
                )
                reward = WikiGameReward.invalid_action_reward
            # Step 1c: Valid backtrack.
            else:
                self.page_history.pop()
                self.current_page = self.page_history[-1]
                self.backtracked = True
                next_obs = (
                    f"At turn {self.turn_count}, you backtracked to the "
                    f"previous page '{self.current_page.title}'. "
                    f"Here is a summary of the page:\n{self._page_summary(self.current_page)}...\n"
                )
                reward = WikiGameReward.internal_step_reward
        
        # Step 2: If model attempted to backtrack on the wrong variant,
        # treat as invalid action.
        elif next_page_title == '<PREV_PAGE>':
            next_obs = (
                f"At turn {self.turn_count}, you attempted to backtrack "
                "to the previous page, but backtracking is not allowed "
                "in this variant of the Wikipedia Game."
            )
            reward = WikiGameReward.invalid_action_reward
        # Step 2b: If the title does not exist, it is an invalid action. Resolve immediately.
        elif next_page_title not in self.current_page.links:
            # We try to fuzzy match for the cases where the model KIND OF
            # knows the neighboring page's title, but made a small mistake which
            # resulted in an invalid action.
            fuzzy_matches = (
                link for link in self.current_page.links
                if next_page_title.lower() in link.lower()
            )
            try:
                next_obs = (
                    f"At turn {self.turn_count}, you guessed '{next_page_title}'. "
                    f"which is not an exact match for any neighboring page of '{self.current_page.title}'. "
                    f"Did you mean '{next(fuzzy_matches)}'? "
                    f"You must match **exactly** one of the neighboring page titles to navigate there."
                )
            except StopIteration:
                next_obs = (
                    f"At turn {self.turn_count}, you guessed '{next_page_title}', "
                    f"which is neither a neighboring page of '{self.current_page.title}', "
                    f"nor could it be construed as a valid neighboring page title of it. "
                    f"You must match **exactly** one of the neighboring page titles to navigate there."
                )

            reward = WikiGameReward.invalid_action_reward
        else:
            # Otherwise, we try to figure out the exact reward and next observation.
            # Step 3: Try to load the page
            next_page = self.trawler.get_page(next_page_title)
            
            # Step 3a: Check if the page could not be loaded
            if next_page is None:
                next_obs = f"At turn {self.turn_count}, you guessed '{next_page_title}', which could not be loaded due to repeated errors."

                reward = WikiGameReward.invalid_action_reward

            # Step 3b: Check if we are on the target page.
            # IMPORTANT NOTE FOR BACKTRACKING VARIANTS:
            # We only remove the backtrack tag if the action is valid and
            # the agent successfully navigates to a new page.
            elif next_page.title == self.target_page.title:
                self.current_page = next_page
                self.page_history.append(next_page)
                self.backtracked = False
                terminate_obs = f"Congratulations! You have reached the target page '{self.target_page.title}' in {self.turn_count} turns."
                reward = WikiGameReward.success_reward
                return (
                    terminate_obs,
                    reward,
                    True,
                    False,
                    {"suffix": self.get_task_suffix()},
                )
            
            # Step 3c: Valid action, but dead-end page.
            elif not next_page.links:
                if self.variant in ['oneback', 'freenav']:
                    self.backtracked = True
                    next_obs = (
                        f"At turn {self.turn_count}, you reached a dead-end page "
                        f"'{next_page.title}' with no neighboring pages. "
                        f"You have been automatically backtracked to the previous page "
                        f"'{self.current_page.title}' at the cost of an additional turn. "
                        f"So now, you are at turn {self.turn_count + 1}. "
                        f"Here is a summary of the page:\n{self._page_summary(self.current_page)}...\n"
                    )
                    self.turn_count += 1
                    reward = 2 * WikiGameReward.internal_step_reward # This was two steps.
                else:                    
                    self.current_page = next_page
                    terminate_obs = f"At turn {self.turn_count}, you navigated to the '{self.current_page.title}' page, which is a dead-end page with no neighboring pages."

                    reward = WikiGameReward.fail_reward
                    return (
                        terminate_obs,
                        reward,
                        True,
                        False,
                        {"suffix": self.get_task_suffix()},
                    )
            # Step 3d: Valid action, but not the target page.
            # IMPORTANT NOTE FOR BACKTRACKING VARIANTS:
            # We only remove the backtrack tag if the action is valid and
            # the agent successfully navigates to a new page.
            else:
                self.current_page = next_page
                self.page_history.append(next_page)
                self.backtracked = False
                next_obs = (
                    f"At turn {self.turn_count}, you navigated to the '{self.current_page.title}' page. "
                    f"Here is a summary of the page:\n{self._page_summary(self.current_page)}...\n"
                )
                reward = WikiGameReward.internal_step_reward
        
        # Step 4: If we exhausted max turns, truncate the episode IMMEDIATELY.
        if self.turn_count >= self.max_turns:
            terminate_obs = f"At turn {self.turn_count}, you have reached the maximum number of turns without reaching the target page '{self.target_page.title}'."
            reward = WikiGameReward.fail_reward
            return (
                terminate_obs,
                reward,
                True,
                True,
                {"suffix": self.get_task_suffix()},
            )

        return (
            next_obs,
            reward, 
            False, 
            False, 
            {"suffix": self.get_task_suffix()},
        )

    def sample_random_action(self) -> str:
        '''
        Samples a random neighboring page
        '''
        return f"\\boxed{{{random.choice(self.current_page.links)}}}"