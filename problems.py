"""Turning failures into sentences a person can act on.

When a run dies, the user is owed two things before any stack detail: WHAT
happened, in plain language, and WHAT TO DO next. The SDK raises precise,
typed exceptions - and until now the product printed their class names and
raw JSON bodies verbatim, which told a developer everything and a normal
person nothing. ("Error code: 401 - {'type': 'error', ...}" is not a
sentence; "Your API key was rejected" is.)

This module is the single place a raised exception becomes a message. The
interface renders whatever it is given; deciding what a failure *means*
is not presentation, so it lives here, next to the knowledge of which
exceptions exist.

The SDK import is deliberately guarded: explaining a failure must never be
the thing that fails.
"""

from __future__ import annotations

from dataclasses import dataclass

CONSOLE_URL = "https://console.anthropic.com/settings/keys"


@dataclass(frozen=True)
class Problem:
    """One failure, translated: what happened, and what to do about it."""

    headline: str
    advice: str
    # The raw exception text, for the user who wants it. Kept separate so
    # renderers can dim it instead of leading with it.
    technical: str = ""


def _first_line(text: str, limit: int = 200) -> str:
    stripped = text.strip()
    line = stripped.splitlines()[0] if stripped else ""
    return line if len(line) <= limit else line[: limit - 3] + "..."


def _looks_like(error: BaseException, *names: str) -> bool:
    """Match an exception by class name across its whole MRO.

    Matching on names instead of imported types keeps this module free of
    the SDK: the translation still works whatever version raised the error,
    and explaining a failure can never itself fail on an import.
    """
    mro = {cls.__name__ for cls in type(error).__mro__}
    return any(name in mro for name in names)


def explain(error: BaseException) -> Problem:
    """The plain-language reading of one failure.

    Ordered from most to least specific; every branch names the thing the
    user can actually change. The fallback still explains itself instead of
    printing a bare class name.
    """
    from config import MODEL

    text = str(error)
    technical = _first_line(text)

    if _looks_like(error, "AuthenticationError", "PermissionDeniedError"):
        return Problem(
            headline="Your API key was rejected.",
            advice=(
                "Check ANTHROPIC_API_KEY in .env - the key may be mistyped, "
                f"revoked, or from another provider. Get a key at {CONSOLE_URL}"
            ),
            technical=technical,
        )
    if _looks_like(error, "RateLimitError"):
        return Problem(
            headline="The API is rate-limiting this key.",
            advice=(
                "Wait a minute and try again. If this keeps happening, your "
                "plan's request limit is the ceiling - space tasks out or "
                "raise the limit in the Anthropic console."
            ),
            technical=technical,
        )
    if _looks_like(error, "NotFoundError"):
        return Problem(
            headline=f"The model '{MODEL}' was not accepted by the API.",
            advice=(
                "Check the MODEL setting (in .env, or the --model flag) "
                "against the models your key can access."
            ),
            technical=technical,
        )
    if _looks_like(error, "BadRequestError") and (
        "credit" in text.lower() or "billing" in text.lower()
    ):
        return Problem(
            headline="The API refused the request over billing.",
            advice=(
                "Your Anthropic account looks out of credit. Top up at "
                "console.anthropic.com and run the task again."
            ),
            technical=technical,
        )
    if _looks_like(error, "APITimeoutError", "APIConnectionError"):
        return Problem(
            headline="Could not reach the Anthropic API.",
            advice=(
                "Check your internet connection (and any proxy or VPN), "
                "then run the task again - nothing was billed for this."
            ),
            technical=technical,
        )
    if _looks_like(
        error, "InternalServerError", "OverloadedError", "ServiceUnavailableError"
    ):
        return Problem(
            headline="The Anthropic API is having trouble right now.",
            advice="This is on their side, not yours. Wait a moment and try again.",
            technical=technical,
        )

    if isinstance(error, RuntimeError) and "every agent failed" in text:
        return _explain_every_agent_failed(text)
    if isinstance(error, ValueError) and "could not generate valid code" in text:
        return Problem(
            headline="The code for one agent could not be written correctly.",
            advice=(
                "This is usually a one-off model slip - run the task again. "
                "If it repeats, rephrase the task or split it into smaller asks."
            ),
            technical=technical,
        )

    return Problem(
        headline="The task failed.",
        advice="Run it again; if it keeps failing, the detail below says where it stopped.",
        technical=f"{type(error).__name__}: {technical}" if technical else type(error).__name__,
    )


def _explain_every_agent_failed(text: str) -> Problem:
    """Read the shared cause out of an every-agent-failed report, if there is one.

    The orchestrator's message embeds each agent's own error. When they all
    died the same way - all timeouts, all auth, all network - the shared
    cause is the story, not the fact that four things failed.
    """
    lowered = text.lower()
    if "timed out after" in lowered:
        return Problem(
            headline="Every agent ran out of time.",
            advice=(
                "The task may be heavier than the per-agent deadline allows. "
                "Raise AGENT_TIMEOUT_SECONDS in .env, or ask for less in one task."
            ),
            technical=_first_line(text.replace("every agent failed:", "").strip()),
        )
    if "401" in lowered or "authentication" in lowered or "api key" in lowered:
        return Problem(
            headline="Every agent was refused by the API - the key is the likely cause.",
            advice=f"Check ANTHROPIC_API_KEY in .env, or get a key at {CONSOLE_URL}",
            technical=_first_line(text.replace("every agent failed:", "").strip()),
        )
    if "api request failed" in lowered or "http" in lowered:
        return Problem(
            headline="Every agent failed to reach the API.",
            advice="Check your connection, then run the task again.",
            technical=_first_line(text.replace("every agent failed:", "").strip()),
        )
    return Problem(
        headline="Every agent this task built failed to finish.",
        advice=(
            "Run the task again - repairs usually catch this. If it repeats, "
            "the per-agent errors below say what each one hit."
        ),
        technical=_first_line(text.replace("every agent failed:", "").strip()),
    )
