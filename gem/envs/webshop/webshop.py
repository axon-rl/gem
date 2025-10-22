import random
import re
import string
from typing import Any, Optional, Tuple

import torch
from bs4 import BeautifulSoup
from bs4.element import Comment

from gem.core import Env
from gem.envs.webshop.utils import SimBrowser, SimServer
from gem.envs.webshop.web_agent_site.engine.engine import parse_action

FORMAT_ERROR_REWARD = -0.1


class WebshopEnv(Env):
    def __init__(
        self,
        observation_mode="text",  # "html", "text", "text_rich", "url"
        split="train",
        max_turns=100,
        show_attrs=False,
        session=None,
        session_prefix=None,
        stop_when_error=True,
        **_,
    ):
        super().__init__()
        self.observation_mode = observation_mode
        self.max_turns = max_turns
        self.base_url = "https://127.0.0.1:3000"
        self.stop_when_error = stop_when_error
        self.server = SimServer(self.base_url, split, show_attrs)
        self.browser = SimBrowser(self.server)
        self.session = session
        self.session_prefix = session_prefix
        self.reset()

    def _get_instructions(self) -> str:
        return (
            "You are shopping online using a web browser.\n"
            "You need to buy a designated product given an instruction on our amazon shopping site.\n"
            "You can perform two types of actions: search[keywords] or click[value].\n"
            "keywords is the search query you want to search.\n"
            "value for click could be as follows:\n"
            "- search[Buy Now] to purchase the item immediately.\n"
            "- click[Next >] to go to the next page of search results.\n"
            "- click[< Prev] to go to the previous page of search results.\n"
            "- click[Back to search] to go back to the search page.\n"
            "- click[${product_id}] to view the product details.\n"
            "- click[${option_value}] to select a buying option (e.g., size, color).\n"
            "You may provide your response in any manner. Only the action that is wrapped inside \\boxed{} will be considered as your action.\n"
            "For example, \\boxed{search[laptop]} or \\boxed{click[Buy Now]}.\n"
            "As you play, the history of your actions will be appended below."
            "The initial page of the webshop is as follows. Enter your first action to start.\n"
        )

    def reset(
        self, seed: Optional[int] = None, session=None, instruction_text=None
    ) -> Tuple[str, dict[str, Any]]:
        """Create a new session and reset environment variables."""
        super().reset(seed)
        session_int = None
        if session is not None:
            self.session = str(session)
            if isinstance(session, int):
                session_int = session
        else:
            self.session = "".join(
                random.choices(random.choices(string.ascii_lowercase, k=10))
            )

        if self.session_prefix is not None:
            self.session = self.session_prefix + self.session

        init_url = f"{self.base_url}/session={self.session}"
        self.browser.get(init_url, session_id=self.session, session_int=session_int)

        self.text_to_clickable = None
        self.instruction_text = (
            self.get_instruction_text()
            if instruction_text is None
            else instruction_text
        )
        self.turn_count = 0
        return self._get_instructions() + self.observation, {}

    def parse_action(self, action: str) -> Tuple[str, Optional[str]]:
        action_search_pattern = re.compile(r"\\boxed{(.+)}")  # e.g. click[value]
        matches = list(action_search_pattern.finditer(action))
        if matches:
            return parse_action(matches[-1].group(1))
        else:
            return None, None

    def step(self, action: str) -> Tuple[str, float, bool, bool, dict[str, Any]]:
        """
        action should be of the following structure:
        - "click[value]"
        - "search[keywords]"
        """
        self.turn_count += 1
        info = {}
        self.get_available_actions()
        action_name, action_arg = self.parse_action(action)
        info["parsed_action"] = (
            f"{action_name}[{action_arg}]" if action_arg else f"{action_name}"
        )

        if action_name == "search" and action_arg is not None:
            status = self.browser.search(action_arg)
        elif (
            action_name == "click"
            and action_arg in self.text_to_clickable.keys()
            and action_arg != "search"
        ):
            status = self.browser.click(action_arg, self.text_to_clickable)
        else:  # invalid action
            obs = "Invalid action format. Please use \\boxed{search[keywords]} or \\boxed{click[value]}."
            return (
                obs,
                FORMAT_ERROR_REWARD,
                self.stop_when_error,
                self.turn_count == self.max_turns,
                info,
            )

        if self.turn_count >= self.max_turns:
            return self.observation, status["reward"], True, True, info

        return self.observation, status["reward"], status["done"], False, info

    def get_available_actions(self):
        """Returns list of available actions at the current step"""
        html_obj = self._parse_html()

        # Collect search bar, buttons, links, and options as clickables
        search_bar = html_obj.find(id="search_input")
        has_search_bar = True if search_bar is not None else False
        buttons = html_obj.find_all(class_="btn")
        product_links = html_obj.find_all(class_="product-link")
        buying_options = html_obj.select('input[type="radio"]')

        self.text_to_clickable = {
            f"{b.get_text()}".lower(): b for b in buttons + product_links
        }
        for opt in buying_options:
            opt_value = opt.get("value")
            self.text_to_clickable[f"{opt_value}"] = opt
        return dict(
            has_search_bar=has_search_bar,
            clickables=list(self.text_to_clickable.keys()),
        )

    def get_image(self):
        """Scrape image from page HTML and return as a list of pixel values"""
        html_obj = self._parse_html(self.browser.page_source)
        image_url = html_obj.find(id="product-image")
        if image_url is not None:
            image_url = image_url["src"]
            if image_url in self.ids:
                image_idx = self.ids[image_url]
                image = self.feats[image_idx]
                return image
        return torch.zeros(512)

    def get_instruction_text(self):
        """Get corresponding instruction text for current environment session"""
        html_obj = self._parse_html(self.browser.page_source)
        instruction_text = html_obj.find(id="instruction-text").h4.text
        return instruction_text

    def _parse_html(self, html=None):
        """
        Returns web request result wrapped in BeautifulSoup object

        Arguments:
        url (`str`): If no url or html is provided, use the current
            observation (HTML) for parsing.
        """
        if html is None:
            html = self.state["html"]
        html_obj = BeautifulSoup(html, "html.parser")
        return html_obj

    @property
    def observation(self):
        """Compiles state into either the `html` or `text` observation mode"""
        html = self.state["html"]
        if self.observation_mode == "html":
            return html
        elif self.observation_mode == "text":
            return self.convert_html_to_text(html, simple=True)
        elif self.observation_mode == "text_rich":
            return self.convert_html_to_text(html, simple=False)
        elif self.observation_mode == "url":
            return self.state["url"]
        else:
            raise ValueError(f"Observation mode {self.observation_mode} not supported.")

    @property
    def state(self):
        """
        State that includes all information. The actual observation are
        likely to be a subset or reduced form of the state.
        """
        return dict(
            url=self.browser.current_url,
            html=self.browser.page_source,
            instruction_text=self.instruction_text,
        )

    def convert_html_to_text(self, html, simple=False):
        """Strip HTML of tags and add separators to convert observation into simple mode"""
        texts = self._parse_html(html).findAll(text=True)
        visible_texts = filter(tag_visible, texts)
        if simple:
            # For `simple` mode, return just [SEP] separators
            return " [SEP] ".join(t.strip() for t in visible_texts if t != "\n")
        else:
            # Otherwise, return an observation with tags mapped to specific, unique separators
            observation = ""
            for t in visible_texts:
                if t == "\n":
                    continue
                if t.parent.name == "button":  # button
                    processed_t = f"[button] {t} [button_]"
                elif t.parent.name == "label":  # options
                    if f'"{t}"' in self.state["url"]:
                        processed_t = f"  [clicked button] {t} [clicked button_]"
                        observation = f"You have clicked {t}.\n" + observation
                    else:
                        processed_t = f"  [button] {t} [button_]"
                elif t.parent.get("class") == ["product-link"]:  # product asins
                    if f"{t}" in self.server.user_sessions[self.session]["asins"]:
                        processed_t = f"\n[clicked button] {t} [clicked button_]"
                    else:
                        processed_t = f"\n[button] {t} [button_]"
                else:  # regular, unclickable text
                    processed_t = str(t)
                observation += processed_t + "\n"
            return observation


def tag_visible(element):
    ignore = {"style", "script", "head", "title", "meta", "[document]"}
    return element.parent.name not in ignore and not isinstance(element, Comment)
