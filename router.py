"""Which questions are tasks, and which are just conversation.

Every line a user types used to become a five-phase pipeline run: a plan, a
code-generation call per agent, a subprocess each, and a merge. That is the
right machinery for "write me an investor memo". It is the wrong machinery
for "hi", and for "what can you do?" it is worse than wrong - the generated
agents know nothing about this system, so they answer by inventing one.

So the session asks this module first. It recognises the handful of things a
person says to a program that are *about* the program rather than work for
it, and those are answered directly, for free, from facts this project
actually knows about itself.

Pure string work: no AI, no I/O, no imports from the rest of the project, so
every rule here is unit-testable and costs nothing to apply.
"""

from __future__ import annotations

import difflib
import re
from enum import Enum


class Intent(str, Enum):
    """What a line of user input actually is."""

    TASK = "task"
    GREETING = "greeting"
    THANKS = "thanks"
    FAREWELL = "farewell"
    CAPABILITY = "capability"
    IDENTITY = "identity"
    HELP = "help"
    LIVE_DATA = "live_data"

    @property
    def is_task(self) -> bool:
        return self is Intent.TASK


# A conversational turn is short. Anything long enough to carry real
# instructions is treated as work no matter which words it contains - this
# single guard is what stops "write a poem about how you are feeling today"
# from being mistaken for small talk.
MAX_CONVERSATIONAL_WORDS = 12

# Politeness and address that carries no instruction, stripped before matching
# so "so, bhai, what can you do?" reduces to "what can you do".
_LEADING_FILLER = frozenset(
    {
        "so", "ok", "okay", "k", "well", "um", "uh", "hmm", "please", "plz", "pls",
        "just", "actually", "btw", "anyway", "hey", "yo", "bro", "bhai", "yaar",
        "dude", "man", "sir", "maam", "buddy", "mate", "and", "but", "then", "now",
    }
)

# Only forms of address may be stripped from the end. Time words must survive:
# "now" is filler in "what can you do now" but the whole question in
# "whats the weather right now".
_TRAILING_FILLER = frozenset(
    {
        "please", "plz", "pls", "bro", "bhai", "yaar", "dude", "man", "sir",
        "maam", "buddy", "mate", "ok", "okay",
    }
)

_PUNCTUATION = re.compile(r"[^\w\s']+")
_WHITESPACE = re.compile(r"\s+")
_REPEATS = re.compile(r"(.)\1+")


def normalise(text: str) -> str:
    """Lowercase, unpunctuated, filler-free form used by every rule below."""
    lowered = _PUNCTUATION.sub(" ", text.lower())
    words = _WHITESPACE.sub(" ", lowered).strip().split()
    # Never strip the last word: "hey" is filler in front of a question and a
    # greeting on its own, and stripping it to nothing loses the message.
    while len(words) > 1 and words[0] in _LEADING_FILLER:
        words.pop(0)
    while len(words) > 1 and words[-1] in _TRAILING_FILLER:
        words.pop()
    return " ".join(words)


def _flatten(text: str) -> str:
    """Collapse repeated letters, so 'hiii' and 'helloooo' greet like the rest."""
    return _REPEATS.sub(r"\1", text)


# Whole-input phrases. Matching is exact after normalisation, because a
# greeting is the entire message - "hello" is a greeting, "hello, summarise
# this contract" is work.
_GREETINGS = frozenset(
    {
        "hi", "hello", "hey", "yo", "hiya", "howdy", "sup", "wassup", "whats up",
        "hi there", "hello there", "good morning", "good afternoon", "good evening",
        "greetings", "namaste", "hola", "bonjour", "salaam", "how are you",
        "how are you doing", "hows it going", "how is it going", "you there",
        "are you there", "anyone there", "test", "testing", "ping",
    }
)

_THANKS = frozenset(
    {
        "thanks", "thank you", "thanks a lot", "thank you so much", "thanks so much",
        "thx", "tysm", "ty", "cheers", "appreciate it", "much appreciated",
        "nice", "cool", "great", "awesome", "perfect", "good job", "well done",
        "nice one", "brilliant", "lovely", "got it", "understood", "makes sense",
        "ok thanks", "okay thanks", "shukriya", "dhanyavaad",
    }
)

_FAREWELLS = frozenset(
    {
        "bye", "goodbye", "good bye", "see you", "see ya", "cya", "later",
        "see you later", "catch you later", "good night", "goodnight", "gn",
        "im done", "i am done", "thats all", "that is all", "nothing else",
    }
)

# Collapsed forms, so the elongations people actually type still land.
_GREETINGS_FLAT = frozenset(_flatten(phrase) for phrase in _GREETINGS)
_THANKS_FLAT = frozenset(_flatten(phrase) for phrase in _THANKS)
_FAREWELLS_FLAT = frozenset(_flatten(phrase) for phrase in _FAREWELLS)


def _anchored(*patterns: str) -> list[re.Pattern[str]]:
    """Compile patterns that must match the whole normalised input."""
    return [re.compile("^(?:" + pattern + ")$") for pattern in patterns]


# "you" in every spelling a terminal sees.
_YOU = r"(?:you|u|ya)"
_YOUR = r"(?:your|ur|yours)"
_YOURSELF = r"(?:yourself|urself|youself)"
_THIS = r"(?:this|it|that|agent ?god|the (?:project|tool|program|app|system|thing))"

_CAPABILITY_PATTERNS = _anchored(
    # what can you do / what all can you do / what else can you do
    r"what (?:all |else |other )?(?:can|could|do|does|will) " + _YOU
    + r"(?: do| help(?: me)?(?: with)?| offer| provide)?",
    r"what (?:can|could) " + _THIS + r" do",
    # what do you do / what does this do
    r"what (?:do|does) " + _THIS + r" do",
    # what are your capabilities / features / services / skills
    r"what (?:are|is) " + _YOUR
    + r" (?:capabilities|capability|abilities|features|services|skills|strengths|uses)",
    r"(?:list|show|tell me) " + _YOUR
    + r"? ?(?:capabilities|features|services|skills|abilities)",
    # what services do you provide / offer
    r"what (?:services?|features?|tasks?|things?) (?:do|does|can|could) " + _YOU
    + r" (?:provide|offer|do|support|handle|perform)",
    r"what (?:kind|type|sort)s? of (?:tasks?|work|things?|jobs?) (?:can|do) " + _YOU
    + r" (?:do|handle|help with|support)",
    # what are you good at / useful for
    r"what (?:are|r) " + _YOU + r" (?:good|best|useful) (?:at|for)",
    r"what (?:can|could) " + _YOU + r" be used for",
    # how do you work
    r"how (?:do|does|dose) " + _THIS + r" work",
    r"how (?:do|does|dose) " + _YOU + r" work",
    r"how (?:do|does) " + _YOU + r" (?:do|work) (?:this|that|it)",
    # capability nouns on their own
    r"(?:capabilities|features|services|commands|options)",
)

_IDENTITY_PATTERNS = _anchored(
    r"who (?:are|r|is) " + _YOU,
    r"what (?:are|r|is) " + _YOU,
    r"what (?:is|are) " + _THIS,
    r"tell me about " + _YOURSELF,
    r"tell me about " + _THIS,
    r"(?:explain|describe) " + _YOURSELF,
    r"(?:explain|describe) " + _THIS,
    r"introduce " + _YOURSELF,
    r"what(?:s| is) " + _YOUR + r" (?:name|purpose|deal|story)",
    r"who (?:made|built|created|wrote) (?:you|u|this)",
    r"what (?:model|llm|ai) (?:are|r) (?:you|u)(?: using| running)?",
    r"which (?:model|llm) (?:are|r) (?:you|u)(?: using| running)?",
    r"are (?:you|u) (?:an? )?(?:ai|bot|robot|human|chatgpt|gpt|llm|claude)",
)

_HELP_PATTERNS = _anchored(
    r"help",
    r"help me",
    r"(?:i need|need) help(?: (?:from|with) " + _YOU + r")?",
    r"how (?:do i|to) use (?:this|you|it|agent ?god)",
    r"how (?:do i|to) (?:start|begin)",
    r"(?:show|give) me (?:the )?(?:help|usage|instructions|manual|docs)",
    r"what (?:do i|should i) (?:type|do|say|write)",
    r"how does this (?:cli|tool|thing) work",
    r"usage",
    r"commands",
)

# Questions whose only correct answer is a value this program cannot see.
# Deliberately narrow: it fires on a request for a live reading, never on
# "write a report on current AI trends", which is real, answerable work.
_LIVE_SUBJECT = (
    r"(?:stock|share|market)? ?price|stock|shares?|weather|temperature|forecast|"
    r"news|headlines|scores?|rates?|exchange rate|time|date|day|traffic|standings"
)
_LIVE_QUALIFIER = (
    r"(?:today|todays|now|right now|current|currently|latest|live|"
    r"at the moment|this (?:morning|evening|week))"
)

_LIVE_DATA_PATTERNS = [
    # "what is the stock price of itc today", "current weather in delhi"
    re.compile(
        r"^(?:what(?:s| is| are)?|hows?|tell me) .*\b(?:" + _LIVE_SUBJECT
        + r")\b.*\b" + _LIVE_QUALIFIER + r"\b"
    ),
    re.compile(
        r"^(?:what(?:s| is| are)?|hows?|tell me) .*\b" + _LIVE_QUALIFIER
        + r"\b.*\b(?:" + _LIVE_SUBJECT + r")\b"
    ),
    # "what time is it", "what is todays date"
    re.compile(r"^what(?:s| is)? (?:the )?(?:time|date|day)(?: is it| today| right now| now)?$"),
    re.compile(r"^what time is it"),
    # "itc share price today"
    re.compile(r"^[\w\s]{0,40}\b(?:share|stock) price\b.*\b" + _LIVE_QUALIFIER + r"\b"),
]

# Verbs that turn a sentence into an instruction. Their presence means the
# user wants work done, whatever else the sentence happens to contain - the
# guard that keeps "can you write a poem" out of the capability bucket.
_TASK_VERBS = frozenset(
    {
        "write", "draft", "compose", "make", "create", "build", "generate", "produce",
        "summarise", "summarize", "condense", "shorten", "expand", "translate",
        "explain", "compare", "analyse", "analyze", "review", "critique", "outline",
        "plan", "design", "code", "program", "implement", "fix", "debug", "refactor",
        "convert", "rewrite", "edit", "proofread", "research", "find", "search",
        "calculate", "compute", "solve", "count", "sort", "rank", "suggest",
        "recommend", "brainstorm", "email", "reply", "respond", "name", "give",
    }
)

# Words that make a short question a question about this program.
_SYSTEM_REFS = frozenset({"you", "u", "your", "ur", "yourself", "urself", "agentgod"})
_CAPABILITY_WORDS = frozenset(
    {
        "do", "can", "could", "capable", "capabilities", "capability", "abilities",
        "ability", "features", "feature", "services", "service", "offer", "offers",
        "provide", "provides", "support", "supports", "handle", "skills", "work",
        "works", "purpose", "useful", "use", "used", "good",
    }
)
_QUESTION_STARTS = frozenset(
    {"what", "who", "how", "which", "tell", "explain", "describe", "list", "show"}
)


def _matches(patterns: list[re.Pattern[str]], text: str) -> bool:
    return any(pattern.match(text) for pattern in patterns)


def _opens_like_a_question(word: str) -> bool:
    """True for 'what', and for the way it arrives when typed at speed.

    Anchoring the fallback on an exact spelling would defeat its whole
    purpose: the questions that reach it are the ones with typos in them.
    """
    if word in _QUESTION_STARTS:
        return True
    return bool(difflib.get_close_matches(word, _QUESTION_STARTS, n=1, cutoff=0.75))


def _looks_like_a_question_about_us(words: list[str]) -> bool:
    """Last-resort rule for short, misspelt questions the patterns missed.

    "wnat service you propvide" never matches a pattern, but it is plainly not
    a task: it is short, it opens like a question, it points at us, and it
    carries no instruction. Requiring all four keeps real work out.
    """
    if not words or not _opens_like_a_question(words[0]):
        return False
    unique = set(words)
    if unique & _TASK_VERBS:
        return False
    return bool(unique & _SYSTEM_REFS) and bool(unique & _CAPABILITY_WORDS)


def classify(text: str) -> Intent:
    """Decide what one line of user input is. Anything unrecognised is work."""
    normalised = normalise(text)
    if not normalised:
        return Intent.TASK

    words = normalised.split()
    # Long input is work by definition; no conversational rule may claim it.
    if len(words) > MAX_CONVERSATIONAL_WORDS:
        return Intent.TASK

    flattened = _flatten(normalised)
    if flattened in _GREETINGS_FLAT:
        return Intent.GREETING
    if flattened in _THANKS_FLAT:
        return Intent.THANKS
    if flattened in _FAREWELLS_FLAT:
        return Intent.FAREWELL

    if _matches(_HELP_PATTERNS, normalised):
        return Intent.HELP
    if _matches(_CAPABILITY_PATTERNS, normalised):
        return Intent.CAPABILITY
    if _matches(_IDENTITY_PATTERNS, normalised):
        return Intent.IDENTITY
    if _matches(_LIVE_DATA_PATTERNS, normalised):
        return Intent.LIVE_DATA

    if _looks_like_a_question_about_us(words):
        return Intent.CAPABILITY

    return Intent.TASK
