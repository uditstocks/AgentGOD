"""Static safety and structure checks for LLM-generated agent code.

Generated code is executed, so it is checked before it ever reaches disk:

- it must parse, and must expose the `run(task, previous_outputs)` contract;
- it may import only vetted modules (an allowlist, not a denylist);
- it may not shell out, touch the filesystem for writing, or eval strings.

No AI and no I/O here, so this module is fully unit-testable.
"""

from __future__ import annotations

import ast

# Standard-library modules a single-LLM-call agent could legitimately need.
ALLOWED_STDLIB = frozenset(
    {
        "base64",
        "collections",
        "dataclasses",
        "datetime",
        "decimal",
        "enum",
        "fractions",
        "functools",
        "html",
        "itertools",
        "json",
        "math",
        "os",
        "pathlib",
        "random",
        "re",
        "statistics",
        "string",
        "sys",
        "textwrap",
        "time",
        "typing",
        "unicodedata",
        "urllib",
        "uuid",
    }
)

# Import names unlocked by the vetted pip packages in executor.ALLOWED_PACKAGES.
ALLOWED_THIRD_PARTY = frozenset(
    {"bs4", "dateutil", "lxml", "markdown", "numpy", "pandas", "requests", "tabulate", "yaml"}
)

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

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            module = _root_module(alias.name)
            if module not in self.allowed_modules:
                self.problems.append(
                    f"line {node.lineno}: import of {alias.name!r} is not allowed "
                    f"(permitted: {', '.join(sorted(self.allowed_modules))})"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            self.problems.append(f"line {node.lineno}: relative imports are not allowed")
        elif node.module and _root_module(node.module) not in self.allowed_modules:
            self.problems.append(f"line {node.lineno}: import from {node.module!r} is not allowed")
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
