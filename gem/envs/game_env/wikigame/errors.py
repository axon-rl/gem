'''
Some nifty exceptions with reference from https://github.com/goldsmith/Wikipedia
'''

class DisambiguationException(Exception):
    '''
    Raised when a disambiguation page is encountered.

    In the Wikipedia game, disambiguation pages are not allowed
    since they are not considered articles. (https://en.wikipedia.org/wiki/Wikipedia:Wiki_Game)
    '''
    def __init__(self, title, options):
        self.title = title
        self.options = options
        option_string = '\n'.join(self.options)
        super().__init__(
            f"{self.title} points to a disambiguation page. "
            f"It may refer to: {option_string}"
        )

class QueryPageNotFoundException(Exception):
    '''
    Raised when a requested Wikipedia page is not found
    within the specified backend.

    Basically a 404 for Wikipedia pages, and can 
    potentially be used as a signal for future search-based policies.
    '''
    def __init__(self, title):
        self.title = title
        super().__init__(
            f"'{self.title}' does not correspond to a real Wikipedia page. "
            "Consider using the search tool if available."
        )


class BackendFailureException(Exception):
    '''
    Raised when the backend itself appears to have failed.

    Here we do NOT know whether the page exists or not,
    just that the backend could not retrieve it.
    '''
    def __init__(self):
        super().__init__(
            "ValueError("
                "Failed to fetch a valid Wikipedia page, "
                "perhaps due to repeated backend failures."
            ")"
        )