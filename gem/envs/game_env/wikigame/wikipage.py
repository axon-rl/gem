from dataclasses import dataclass

@dataclass
class WikipediaPage:
    '''
    Platform-agnostic Wikipedia Page representation.

    Attributes:
        page_id (str): The unique identifier of the Wikipedia page.
        title (str): The title of the Wikipedia page.
        content (str): The full text content of the Wikipedia page.
        links (list[str]): A list of titles of Wikipedia pages linked from this page.
    '''
    page_id: str
    title: str
    content: str
    links: list[str]