"""Local, rule-based classifier for taxonomy-tagged "determination" fields -- fields whose value
is a yes/no/unclear judgment about something the caller said, rather than an extractable span
(see `gliner_pipeline.FIELD_TAXONOMY`).

GLiNER's zero-shot span extraction surfaces no signal at all for a determination like
`permission_to_enter`, even down to a very low threshold: the caller's answer is rarely phrased
as an extractable span ("yes") and is usually an indirect statement instead (e.g. call 005:
"I guess it's fine if I'm not there... fix the leak", never the word "yes"). `classify` runs over
a plain transcript window independently of whatever GLiNER itself reports for the field that
tick -- it is never gated on GLiNER having located anything first.

Purely local pattern matching -- no LLM call, consistent with the project's
no-LLM-in-the-runtime-loop premise for pipeline stages that run every tick.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Determination = Literal["yes", "no", "uncertain"]


@dataclass(frozen=True)
class _DeterminationRules:
    """Compiled affirmative/negative patterns for one determination field.

    `classify` matches both lists against the same window and only reports a definite `"yes"`/
    `"no"` when exactly one side hit -- either no match or a hit on both sides (a genuinely mixed
    or ambiguous window) falls back to `"uncertain"` rather than guessing.
    """

    yes: tuple[re.Pattern[str], ...]
    no: tuple[re.Pattern[str], ...]


def _compile(patterns: list[str]) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)


# Each pattern here is deliberately specific to entry-permission phrasing rather than a bare
# "yes"/"no" -- a caller's window is full of unrelated affirmatives/negatives ("yeah, that's
# right", "no, that's all") that would otherwise swamp the signal for this one field.
_PERMISSION_TO_ENTER_RULES = _DeterminationRules(
    yes=_compile(
        [
            # Negative lookbehind excludes "no permission to enter" -- without it, that phrase
            # (an explicit refusal) would also match this bare positive pattern, so `classify`
            # would see both a yes-hit and a no-hit and fall back to "uncertain" instead of "no".
            r"(?<!no )permission to enter",
            r"guess it'?s fine",
            r"fine if i'?m not (?:there|home|around)",
            r"go ahead and (?:enter|come (?:in|on in)|let (?:yourself|yourselves) in)",
            r"(?:you|they) can (?:enter|come (?:in|by|over)|go in|let (?:yourself|yourselves) in)",
            r"feel free to (?:enter|come (?:in|by|over)|go in)",
            r"no need to call (?:first|ahead|before)",
            r"don'?t need to (?:call|be (?:home|there|present))",
            r"ok(?:ay)? to enter",
            r"just come (?:by|over|on by|on over)",
        ]
    ),
    no=_compile(
        [
            # Requires a first-person subject immediately before the verb, not just the phrase
            # "need to be there" anywhere in the window -- otherwise the caller's own question
            # ("should I... need to be there when they come over?") false-positives as a refusal
            # before their actual answer even arrives, matching on the question itself.
            r"\bi (?:need|have|must) to be (?:home|there|present)\b",
            r"(?:do not|don'?t) (?:let|allow) (?:them|anyone) (?:in|enter)",
            r"not (?:comfortable|okay|ok) with (?:them|anyone) (?:entering|coming in)",
            r"no permission to enter",
            r"can'?t enter without me",
            r"not (?:giving|going to give) permission",
        ]
    ),
)

# Registry keyed by field name, not a single global ruleset -- adding a future determination
# field (per `gliner_pipeline.FIELD_TAXONOMY`) only needs a new `_DeterminationRules` entry here;
# `classify`'s dispatch and its `gliner_pipeline` call site are already field-agnostic.
_RULES_BY_FIELD: dict[str, _DeterminationRules] = {
    "permission_to_enter": _PERMISSION_TO_ENTER_RULES,
}


def classify(field_name: str, window_text: str) -> Determination:
    """Classify `window_text` to `"yes"`/`"no"`/`"uncertain"` for `field_name`.

    Returns `"uncertain"` for a field with no registered rules (not a taxonomy-tagged
    determination field, or a future one not yet given its own rules), or when the window's
    signal for a registered field is absent or contradictory.
    """
    rules = _RULES_BY_FIELD.get(field_name)
    if rules is None or not window_text:
        return "uncertain"
    yes_hit = any(pattern.search(window_text) for pattern in rules.yes)
    no_hit = any(pattern.search(window_text) for pattern in rules.no)
    if yes_hit and not no_hit:
        return "yes"
    if no_hit and not yes_hit:
        return "no"
    return "uncertain"
