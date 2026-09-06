"""AgentGod - one permanent agent that builds, runs and retires other agents.

The command line lives in `cli`; `main.cli_main` is the console-script entry
point. Everything the product does is a module beside this one.
"""

from .cli import __version__

__all__ = ["__version__"]
