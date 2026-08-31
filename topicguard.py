"""The check that keeps a kept agent reusable.

codeguard asks whether generated code is safe to run. This asks something
different: whether it will still do its job tomorrow, on a task about
something else.

It exists because of a real failure. A `summary_agent` was generated for the
task "name one benefit of code review in one line", and the model wrote the
subject straight into the prompt:

    "You are a summary agent. Condense the findings into a single line
     that states one benefit of code review."

That agent was correct once. Then it was kept, and reused sixteen times, and
every task after it - including "hi" - came back as an essay about code
review. The generator's instructions had said, in plain words, not to do
that. Instructions are not enforcement.

So this module enforces it: it compares the generated source against the task
it was generated for, and reports any of that task's subject that ended up
frozen in a string literal. The generator feeds the report back and tries
again, so poisoned code never reaches the disk, let alone the library.

No AI and no I/O, so every rule here is unit-testable.
"""

from __future__ import annotations

import ast
import re
from itertools import pairwise

# Vocabulary for describing the work rather than the subject. A generated
# prompt is expected to be full of these words; they carry no topic, so a
# literal echoing one proves nothing about reusability.
_CRAFT_VOCABULARY = """
    agent agents task tasks user users prompt prompts output outputs input inputs
    previous result results response responses following below above supplied given
    provided based using use used step steps call calls return returns role
    write writes writing written draft drafts compose composed create creates
    produce produces generate generates make makes made instructions responsibility
    summary summarise summarize summarised summarized condense condenses condensed
    brief briefly concise short shorter long longer expand expands detailed detail
    line lines word words sentence sentences paragraph paragraphs page pages
    section sections list lists bullet bullets point points item items table format
    formats formatted structure structured outline outlines heading headings
    report reports essay essays memo memos article articles post posts email emails
    letter note notes document documents text texts content piece pieces answer
    research gather gathers gathering facts fact information info data details
    findings finding insight insights takeaway takeaways conclusion conclusions
    overview background context material materials source sources evidence
    main important relevant accurate clear clearly complete comprehensive
    analyse analyze analysis analysing analyzing examine evaluate assess assessment
    review reviews reviewing critique critiques criticism feedback weaknesses
    compare compares comparison comparing contrast options criteria pros cons
    translate translates translation translating language languages phrase phrases
    code coding program programs function functions script scripts snippet snippets
    explain explains explanation describe describes description tell give gives
    provide provides name names state states single each every some more most other
    first second third final last only both all any new good best better well
    please help need want ask question subject topic topics about here there
    this that these those with from into their there then than such very just
    what which who whom when where does will should would could been being
"""

CRAFT_WORDS = frozenset(_CRAFT_VOCABULARY.split())

# Connectives dropped before phrases are formed, so "one benefit of code
# review" still matches "states one benefit of code review": the phrase
# survives whatever wording is wrapped around it.
_CONNECTIVE_VOCABULARY = """
    a an the and or of in into to for with on at by as from is are was were be
    that this it its their there then than so not no such your my our we you i
"""

CONNECTIVES = frozenset(_CONNECTIVE_VOCABULARY.split())

# Words too short to identify a subject on their own.
MIN_SUBJECT_LENGTH = 3

# How many of the task's own two-word phrases may reappear in the source
# before it is a copy of the task rather than a description of a role. One is
# a coincidence ("translate the phrase"); two is the subject.
MAX_ECHOED_PHRASES = 1

_WORDS = re.compile(r"[A-Za-z][A-Za-z'-]*")
_NUMBERS = re.compile(r"\d+")


def _words(text: str) -> list[str]:
    return [word.lower() for word in _WORDS.findall(text)]


def _numbers(text: str) -> set[str]:
    """Digit runs in `text`. A task's numbers are part of its subject.

    "60-word" is as much this task's alone as its topic is: an agent that
    freezes it answers every later request at sixty words, whatever was
    asked. The runtime's constraints() helper exists to read the limit from
    the task at run time instead.
    """
    return set(_NUMBERS.findall(text))


def subject_words(words: list[str]) -> set[str]:
    """The words in `words` that actually name a subject."""
    return {
        word for word in words if len(word) > MIN_SUBJECT_LENGTH and word not in CRAFT_WORDS
    }


def task_subjects(task: str) -> list[str]:
    """The subject words in a task, sorted - the words an agent may not contain.

    Handing this list to the generator up front turns a rule it kept breaking
    ("do not hardcode the subject") into one it can actually check itself
    against, and saves the regeneration attempts that discovering it late costs.
    """
    return sorted(subject_words(_words(task)))


def phrases(words: list[str]) -> set[tuple[str, str]]:
    """Adjacent meaningful word pairs, connectives removed first."""
    meaningful = [word for word in words if word not in CONNECTIVES]
    return set(pairwise(meaningful))


def _docstring_ids(tree: ast.Module) -> set[int]:
    """Identity of every docstring node, which is exempt from these rules.

    A docstring describes the agent to a human reader. Only the strings the
    agent actually sends to a model can steer it on a later, unrelated task.
    """
    found: set[int] = set()
    holders = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    for node in ast.walk(tree):
        if not isinstance(node, holders):
            continue
        body = getattr(node, "body", None)
        if not body or not isinstance(body[0], ast.Expr):
            continue
        value = body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            found.add(id(value))
    return found


# The trusted runtime that generator.py wraps around every agent. No model
# wrote it, it is identical in every file, and it is full of words like
# "system" and "usage" that belong to the plumbing rather than to any
# task. Checking it produced a page of false accusations the first time a
# task mentioned one of them.
RUNTIME_HELPERS = frozenset(
    {
        "api_key",
        "_post",
        "answer_text",
        "call_llm",
        "format_previous",
        "upstream",
        "chunk",
        "constraints",
        "word_count",
    }
)


def _generated_functions(tree: ast.Module) -> list[ast.AST]:
    """The parts of the file the model actually wrote."""
    return [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name not in RUNTIME_HELPERS
    ]


def literals(tree: ast.Module) -> list[tuple[int, str]]:
    """String literals the model wrote, except docstrings, with line numbers.

    Module-level constants and the runtime helpers are skipped: they are the
    trusted header, and holding the model responsible for them is how a task
    that merely mentioned "usage" got its agent rejected twenty times over.
    """
    exempt = _docstring_ids(tree)
    found: list[tuple[int, str]] = []
    for function in _generated_functions(tree):
        for node in ast.walk(function):
            if isinstance(node, ast.Constant):
                if isinstance(node.value, str) and id(node) not in exempt:
                    found.append((node.lineno, node.value))
            elif isinstance(node, ast.JoinedStr):
                # An f-string's fixed halves are literal; the {task} placeholder
                # inside it is the runtime value allowed to carry a subject.
                for part in node.values:
                    if isinstance(part, ast.Constant) and isinstance(part.value, str):
                        found.append((node.lineno, part.value))
    return found


def check_topic_leakage(source: str, task: str) -> list[str]:
    """Report any of `task`'s subject frozen into `source` as a string literal.

    An empty list means the agent describes only its role and will do the same
    job on any future task. Anything else is a reason to regenerate.
    """
    if not task.strip():
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []  # a parse error is codeguard's complaint to make, not this one's

    task_words = _words(task)
    subjects = subject_words(task_words)
    task_phrases = phrases(task_words)
    task_numbers = _numbers(task)
    if not subjects and not task_phrases and not task_numbers:
        return []

    problems: list[str] = []
    for lineno, literal in literals(tree):
        literal_words = _words(literal)

        figures = sorted(
            number
            for number in task_numbers.intersection(_numbers(literal))
            if re.search(rf"(?<!\d){re.escape(number)}(?!\d)", literal)
        )
        if figures:
            listed = ", ".join(figures)
            problems.append(
                f"line {lineno}: a string literal hardcodes this task's figures "
                f"({listed}). A later task will ask for a different number - read it "
                "from the task at runtime with constraints(task) instead."
            )
            continue

        if not literal_words:
            continue

        leaked = sorted(subjects.intersection(literal_words))
        if leaked:
            listed = ", ".join(repr(word) for word in leaked[:4])
            problems.append(
                f"line {lineno}: a string literal hardcodes this task's subject ({listed}). "
                "This agent is kept and reused on unrelated tasks, so its prompt must "
                "describe only its role - the subject has to arrive through the `task` "
                "argument at runtime."
            )
            continue

        echoed = task_phrases.intersection(phrases(literal_words))
        if len(echoed) > MAX_ECHOED_PHRASES:
            sample = ", ".join(repr(" ".join(pair)) for pair in sorted(echoed)[:3])
            problems.append(
                f"line {lineno}: a string literal repeats the wording of this specific "
                f"task ({sample}). Describe the agent's role instead, and let the task "
                "text arrive through the `task` argument."
            )
    return problems


# Calls that put a value into the text the agent sends to the model. A `task`
# that reaches none of them never reaches the model.
_STRING_BUILDING_METHODS = frozenset({"format", "join", "replace", "strip", "lstrip", "rstrip"})


def _run_function(tree: ast.Module) -> ast.FunctionDef | None:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            return node
    return None


# Calls that hand back what they were given. Everything else returns something
# *derived* from the task - constraints(task) returns a list of length limits -
# so the task's own words never make it through.
_PASSTHROUGH_CALLS = frozenset({"call_llm", "str", "repr"})


def _task_reaches(node: ast.AST) -> bool:
    """Whether `task` contributes its own text to this expression.

    Mentioning `task` is not using it. `" ".join(constraints(task))` mentions
    it and passes on nothing but word counts, which is precisely how the inert
    agent this rule was written for slipped through.
    """
    if isinstance(node, ast.Name):
        return node.id == "task" and isinstance(node.ctx, ast.Load)
    if isinstance(node, ast.Call):
        called = node.func
        if isinstance(called, ast.Name) and called.id not in _PASSTHROUGH_CALLS:
            return False
    return any(_task_reaches(child) for child in ast.iter_child_nodes(node))


def check_task_is_used(source: str) -> list[str]:
    """Report an agent whose prompt never contains the user's task.

    A subtler failure than hardcoding a subject, and a worse one. One
    generated `writer_agent` built its whole prompt from string literals and
    passed `task` only to `constraints(task)` - so it never saw what the user
    actually asked for, and returned the same paragraph no matter what. It
    read as careful, structured code, and it was inert.

    The rule: `task` must reach a string that is built, not merely be
    mentioned. Passing it to a helper that returns a list of length limits
    does not count.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []  # codeguard's complaint, not this one's

    run = _run_function(tree)
    if run is None:
        return []  # likewise: the contract check owns this

    for node in ast.walk(run):
        builds_a_string = (
            isinstance(node, (ast.JoinedStr, ast.BinOp))
            # "...".join(...) / "...".format(...) and friends
            or (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _STRING_BUILDING_METHODS
            )
            # call_llm(task) - the task being the entire prompt is fine
            or (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "call_llm"
            )
        )
        if builds_a_string and _task_reaches(node):
            return []

    return [
        "the prompt never includes `task`, so this agent ignores what the user "
        "asked for and would return the same thing for every request. Put the "
        "task itself into the prompt you build - passing it to a helper such as "
        "constraints(task) is not enough."
    ]


def is_reusable(source: str, task: str) -> bool:
    """True when `source` is safe to keep in the library for later tasks."""
    return not check_topic_leakage(source, task) and not check_task_is_used(source)
