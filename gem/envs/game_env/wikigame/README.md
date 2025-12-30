# WikiGame

This directory contains the implementation of the WikiGame environment, whose details can be found with `wikigame.py` as an entry point.

## Structure
```gem/
└── gem/
    └── envs/
        └── game_env/
            └── wikigame/
                ├── __init__.py
                ├── wikigame.py (abstract environment implementation)
                ├── backend.py (actual interface with Wikipedia data)
                ├── dynamics.py (environment dynamics fill-in-the-blanks)
                ├── rewards.py (reward structure definitions)
                ├── wikipage.py (page representation)
                ├── errors.py (custom exceptions)
                ├── setup_kiwix.sh (script to set up Kiwix backend (MacOS only))
                └── README.md (You are here!)
```

## How to Use

### Quickstart
[This (somewhat disorganized) repository](https://github.com/N00bcak/playing-wikigame) contains a **relatively** up-to-date project
which can run the WikiGame environment, though some hiccups may occur due to platform and hardware differences.

### Manual Setup

#### Kiwix
Go to [setup_kiwix.sh](./setup_kiwix.sh) for a script that automates the setup of the Kiwix backend on MacOS.
(You will also need this script for testing!)

For Linux, there is a [quickstart repository](https://github.com/N00bcak/playing-wikigame) containing a Linux flavor of the setup script.

Ensure that:
- You have enough disk space to store the ZIM file.
- You have `kiwix-serve` installed and accessible from your PATH / (in the case of MacOS, a Docker container with Kiwix installed and the ZIM file mounted).
- You have started the Kiwix server before running any code that uses the WikiGame environment.

#### Customization
The present gem registry codes for the environments are:
- `game:WikiGame-v0-easy` (30 turns, no-regrets, short summaries, kiwix backend)
- `game:WikiGame-v0-hard` (15 turns, no-regrets, short summaries, kiwix backend)

These are decidedly NOT the full extent of possible configurations. Keyword arguments may be passed to `gem.make()` 
to modify the environment behavior. See the docstring of `wikigame.WikiGameEnv.__init__` for a full list of options.

For example, to create a oneback variant on the live MediaWiki backend, where the summaries are 500 characters long, one can write:
```python
gem.make(
    "game:WikiGame-v0-easy",
    variant = "oneback",
    backend = "mw",
    page_summary_length = (500, 'characters'),
    trawler_kwargs = {
        "url": "https://en.wikipedia.org/w/api.php", # MediaWiki API endpoint
        "query_delay_ms": 25, # Be nice to the servers
        "query_use_cache": True, # Cache queries locally
    }
)
```

For extra power, one can directly instantiate the `wikigame.WikiGameEnv` class with any desired configuration:
```python
from gem.envs.game_env.wikigame.wikigame import WikiGameEnv

env = WikiGameEnv(
    max_turns = 20,
    variant = "freenav",
    backend = "kiwix",
    page_summary_length = (300, 'words'),
    trawler_kwargs = {
        "url": "http://localhost:8080", # Kiwix server endpoint
        "zimfile": "wikipedia_en_simple_all_nopic_2025-11", # Change as needed
        "query_delay_ms": 0,
        "query_use_cache": True,
    }
)
```

## Customization Quick Reference
| Parameter               | Description                                                  | Values                    |
|-------------------------|--------------------------------------------------------------|------------------------------------|
| `max_turns`             | Maximum number of turns per episode                          | Positive Integer                   |
| `variant`               | Game variant (ruleset)                                       | `"freenav"`, `"oneback"`, `"noregrets"`, `"eidetic"` |
| `backend`               | Wikipedia data backend                                       | `"kiwix"`, `"mw"` (MediaWiki)        |
| `page_summary_length`   | Length of page summaries                                     | (`<Non-negative integer>`, `<'words', 'characters', 'sentences'>`)  |
| `trawler_kwargs`        | Keyword arguments for the trawler (data fetcher)             | See above for details               |

### Variant Descriptions
- **noregrets** (default): No backtracking allowed, the agent can only navigate via visible links.
    - Note that if two pages reference each other, the agent can still "backtrack" by navigating forward through the link. This is not considered backtracking.
- **oneback**: The agent cannot backtrack more than once consecutively.
- **freenav**: The agent can backtrack freely without restrictions.
- **eidetic**: The agent can visit ANY page whose link it has ever seen before, regardless of current position.
    - Note that the agent still cannot backtrack.