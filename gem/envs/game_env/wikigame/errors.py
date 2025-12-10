'''
Some nifty exceptions with reference from https://github.com/goldsmith/Wikipedia
'''

class DisambiguationException(Exception):
    def __init__(self, title, options):
        self.title = title
        self.options = options
        option_string = '\n'.join(self.options)
        super().__init__(
            f"{self.title} points to a disambiguation page. "
            f"It may refer to: {option_string}"
        )

class QueryPageNotFoundException(Exception):
    def __init__(self, title):
        self.title = title
        super().__init__(
            f"'{self.title}' does not correspond to a real Wikipedia page. "
            "Consider using the search tool if available."
        )


class BackendFailureException(Exception):
    def __init__(self, title):
        super().__init__(
            "ValueError("
                "Failed to fetch a valid Wikipedia page, "
                "perhaps due to repeated backend failures."
            ")"
        )