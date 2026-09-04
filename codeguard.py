"""Static safety and structure checks for LLM-generated agent code.

Generated code is executed, so it is checked before it ever reaches disk:

- it must parse, and must expose the `run(task, previous_outputs)` contract;
- it may import only vetted modules (an allowlist, not a denylist);
- it may not shell out, touch the filesystem for writing, or eval strings.

No AI and no I/O here, so this module is fully unit-testable.
"""

from __future__ import annotations

import ast
import sys

# Standard-library modules an agent may NOT import. Everything else in the
# standard library is allowed, because a curated subset turned out to refuse
# ordinary work - `csv`, `sqlite3`, `zipfile`, `hashlib`, `difflib` and three
# hundred others were rejected for no reason but absence from a list.
#
# What stays blocked is only what would defeat a check this module already
# makes elsewhere: shelling out, and turning data into running code. Banning
# `subprocess` while banning `eval()` is one rule, not two.
BLOCKED_STDLIB = frozenset(
    {
        # spawns a process or loads native code
        "subprocess",
        "multiprocessing",
        "ctypes",
        "pty",
        "socket",
        # executes code chosen at runtime, which is what `eval` is banned for
        "importlib",
        "runpy",
        "code",
        "codeop",
        "pickle",
        "marshal",
        "shelve",
        # rewrites the filesystem wholesale
        "shutil",
    }
)

ALLOWED_STDLIB = frozenset(
    name for name in sys.stdlib_module_names if not name.startswith("_")
) - BLOCKED_STDLIB

# Vetted pip packages a generated agent may use, mapped to their import names.
# A model-invented package name is refused rather than installed: hallucinated
# names are a supply-chain vector, not a typo to be helpfully resolved. This is
# the single source of truth - executor.py installs from it, and the import
# check below derives from it, so the two can never drift apart.
ALLOWED_PACKAGES: dict[str, str] = {
    # web and HTTP
    "aiohttp": "aiohttp",
    "beautifulsoup4": "bs4",
    "fastapi": "fastapi",
    "feedparser": "feedparser",
    "flask": "flask",
    "html5lib": "html5lib",
    "httpx": "httpx",
    "jinja2": "jinja2",
    "lxml": "lxml",
    "requests": "requests",
    "starlette": "starlette",
    "urllib3": "urllib3",
    "uvicorn": "uvicorn",
    "werkzeug": "werkzeug",
    # data and maths
    "matplotlib": "matplotlib",
    "networkx": "networkx",
    "numpy": "numpy",
    "openpyxl": "openpyxl",
    "pandas": "pandas",
    "plotly": "plotly",
    "pyarrow": "pyarrow",
    "scikit-learn": "sklearn",
    "scipy": "scipy",
    "seaborn": "seaborn",
    "statsmodels": "statsmodels",
    "sympy": "sympy",
    "tabulate": "tabulate",
    "xlsxwriter": "xlsxwriter",
    # documents and text
    "chardet": "chardet",
    "ftfy": "ftfy",
    "inflect": "inflect",
    "markdown": "markdown",
    "nltk": "nltk",
    "num2words": "num2words",
    "pdfplumber": "pdfplumber",
    "pygments": "pygments",
    "pypdf": "pypdf",
    "python-docx": "docx",
    "python-pptx": "pptx",
    "python-slugify": "slugify",
    "rapidfuzz": "rapidfuzz",
    "regex": "regex",
    "reportlab": "reportlab",
    "textstat": "textstat",
    "unidecode": "unidecode",
    # configuration and serialisation
    "jsonschema": "jsonschema",
    "orjson": "orjson",
    "python-dotenv": "dotenv",
    "pyyaml": "yaml",
    "toml": "toml",
    # images and media
    "imageio": "imageio",
    "opencv-python": "cv2",
    "pillow": "PIL",
    "python-barcode": "barcode",
    "qrcode": "qrcode",
    "svgwrite": "svgwrite",
    # dates, places and locales
    "arrow": "arrow",
    "babel": "babel",
    "humanize": "humanize",
    "python-dateutil": "dateutil",
    "pytz": "pytz",
    # models and validation
    "attrs": "attrs",
    "email-validator": "email_validator",
    "marshmallow": "marshmallow",
    "phonenumbers": "phonenumbers",
    "pydantic": "pydantic",
    "validators": "validators",
    # terminal output
    "click": "click",
    "colorama": "colorama",
    "rich": "rich",
    "tqdm": "tqdm",
    "typer": "typer",
    # cryptography
    "bcrypt": "bcrypt",
    "cryptography": "cryptography",
    "passlib": "passlib",
    # databases
    "pymongo": "pymongo",
    "redis": "redis",
    "sqlalchemy": "sqlalchemy",
    # odds and ends
    "emoji": "emoji",
    "faker": "faker",
    "pint": "pint",
}

# Import names unlocked by the packages above.
ALLOWED_THIRD_PARTY = frozenset(ALLOWED_PACKAGES.values())

ALLOWED_MODULES = ALLOWED_STDLIB | ALLOWED_THIRD_PARTY

# Builtins that turn data into code, or block on a human.
BANNED_CALLS = frozenset({"eval", "exec", "compile", "__import__", "input", "breakpoint"})

# Attribute calls that escape the process or mutate the machine. None of these
# is also a method on a builtin type, so the name alone is enough to condemn.
BANNED_ATTRIBUTES = frozenset(
    {
        "system",
        "popen",
        "spawn",
        "spawnl",
        "spawnv",
        "execv",
        "execve",
        "execl",
        "unlink",
        "rmdir",
        "removedirs",
        "rmtree",
        "chmod",
        "chown",
        "kill",
        "fork",
    }
)

# Names that destroy a file on `os`, `shutil` or a `Path`, and are ordinary
# string or list methods on anything else.
#
# These were once banned outright, which was too blunt to live with: building
# a prompt is string work, and `prompt.replace(...)` is the most natural line
# in it. Rejecting that rejected whole agents over a method that cannot touch
# a disk - so the receiver decides, not the name.
FILESYSTEM_ATTRIBUTES = frozenset({"remove", "rename", "replace", "truncate"})

# Roots that make one of the names above a filesystem call.
FILESYSTEM_RECEIVERS = frozenset({"os", "shutil", "pathlib", "Path", "path", "io"})

# open() modes that write.
_WRITE_MODES = frozenset("wax+")

REQUIRED_FUNCTION = "run"


def _root_module(name: str) -> str:
    return name.split(".", 1)[0]


def _receiver_root(node: ast.expr) -> str | None:
    """The leftmost name a call's receiver is built from.

    `os.replace` -> "os", `os.path.replace` -> "os", `Path(p).replace` ->
    "Path", `prompt.replace` -> "prompt". A receiver that is not built from a
    plain name at all (a literal, a subscript of a call) yields None and is
    treated as ordinary data.
    """
    current: ast.expr | None = node
    while True:
        if isinstance(current, ast.Attribute):
            current = current.value
        elif isinstance(current, ast.Call):
            current = current.func
        elif isinstance(current, ast.Subscript):
            current = current.value
        else:
            break
    return current.id if isinstance(current, ast.Name) else None


class _Inspector(ast.NodeVisitor):
    """Walks the tree once, collecting every rule violation it sees."""

    def __init__(self, allowed_modules: frozenset[str]) -> None:
        self.allowed_modules = allowed_modules
        self.problems: list[str] = []

    def _refuse(self, lineno: int, name: str) -> str:
        """Why this import was refused, in terms the model can act on.

        The allowlist is now most of the standard library, so naming its
        members is no longer a hint - it is three hundred words of noise in a
        repair prompt. The reason is the useful part.
        """
        reason = (
            "it shells out, writes the filesystem or runs code chosen at runtime"
            if _root_module(name) in BLOCKED_STDLIB
            else "it is neither the standard library nor a vetted package"
        )
        return f"line {lineno}: import of {name!r} is not allowed - {reason}"

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if _root_module(alias.name) not in self.allowed_modules:
                self.problems.append(self._refuse(node.lineno, alias.name))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            self.problems.append(f"line {node.lineno}: relative imports are not allowed")
        elif node.module and _root_module(node.module) not in self.allowed_modules:
            self.problems.append(self._refuse(node.lineno, node.module))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in BANNED_CALLS:
                self.problems.append(f"line {node.lineno}: {func.id}() is not allowed")
            elif func.id == "open" and self._opens_for_writing(node):
                self.problems.append(f"line {node.lineno}: open() for writing is not allowed")
        elif isinstance(func, ast.Attribute):
            if func.attr in BANNED_ATTRIBUTES:
                self.problems.append(f"line {node.lineno}: .{func.attr}() is not allowed")
            elif func.attr in FILESYSTEM_ATTRIBUTES:
                receiver = _receiver_root(func.value)
                if receiver in FILESYSTEM_RECEIVERS:
                    self.problems.append(
                        f"line {node.lineno}: {receiver}.{func.attr}() is not allowed"
                    )
        self.generic_visit(node)

    @staticmethod
    def _opens_for_writing(node: ast.Call) -> bool:
        mode = next(
            (kw.value for kw in node.keywords if kw.arg == "mode"),
            node.args[1] if len(node.args) > 1 else None,
        )
        if isinstance(mode, ast.Constant) and isinstance(mode.value, str):
            return any(char in _WRITE_MODES for char in mode.value)
        return mode is not None  # a non-literal mode is not provably read-only


def _check_contract(tree: ast.Module) -> list[str]:
    """The executor calls run(task, previous_outputs); make sure it exists."""
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    run = functions.get(REQUIRED_FUNCTION)
    if run is None:
        return [f"no top-level {REQUIRED_FUNCTION}(task, previous_outputs) function"]
    if isinstance(run, ast.AsyncFunctionDef):
        return [f"{REQUIRED_FUNCTION}() must not be async"]

    args = run.args
    positional = len(args.posonlyargs) + len(args.args)
    if positional != 2:
        return [
            f"{REQUIRED_FUNCTION}() takes {positional} positional argument(s); "
            "the contract is run(task, previous_outputs)"
        ]
    return []


def check_agent_source(source: str, allowed_modules: frozenset[str] = ALLOWED_MODULES) -> list[str]:
    """Return a list of problems with `source`. An empty list means it is safe to run."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return [f"line {error.lineno}: syntax error: {error.msg}"]

    problems = _check_contract(tree)
    inspector = _Inspector(allowed_modules)
    inspector.visit(tree)
    return problems + inspector.problems
