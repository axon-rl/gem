from dataclasses import dataclass

@dataclass
class WikipediaPage:
    page_id: str
    title: str
    content: str
    links: list[str]