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
class DeterminationResult:
    """`classify_with_confidence`'s return value -- the bare `Determination` plus how confident
    the matched rule pattern is and which pattern produced it.

    `confidence` is `0.0` exactly when `value == "uncertain"` (no signal, or contradictory
    yes/no signal). For a real `"yes"`/`"no"` match it reflects the matched pattern's hand-assigned
    tier, minus a hedge-word penalty when the window hedges its phrasing (see `_HEDGE_PATTERN`).
    """

    value: Determination
    confidence: float
    matched_pattern: str | None


@dataclass(frozen=True)
class _PatternRule:
    """One compiled pattern plus its hand-assigned confidence tier.

    Tiers are provisional starting-point estimates grouped by phrasing directness
    (direct/explicit ~0.90, clear-but-indirect ~0.75, hedged ~0.55, strong first-person negative
    ~0.85), not empirically validated against the real corpus.
    """

    pattern: re.Pattern[str]
    confidence: float


@dataclass(frozen=True)
class _DeterminationRules:
    """Compiled affirmative/negative patterns for one determination field.

    `classify` matches both lists against the same window and only reports a definite `"yes"`/
    `"no"` when exactly one side hit -- either no match or a hit on both sides (a genuinely mixed
    or ambiguous window) falls back to `"uncertain"` rather than guessing.
    """

    yes: tuple[_PatternRule, ...]
    no: tuple[_PatternRule, ...]


def _compile(patterns: list[tuple[str, float]]) -> tuple[_PatternRule, ...]:
    return tuple(
        _PatternRule(re.compile(pattern, re.IGNORECASE), confidence) for pattern, confidence in patterns
    )


# Words/phrases that soften an otherwise clear determination ("maybe go ahead and enter") --
# detected across the whole window rather than tied to a specific pattern, since a caller can
# hedge anywhere in their answer, not just adjacent to the matched phrase.
_HEDGE_PATTERN = re.compile(r"\b(?:maybe|probably|i think|not sure|possibly)\b", re.IGNORECASE)

# Subtracted from the matched pattern's confidence when a hedge word is present. The floor keeps
# a hedged-but-real match's confidence strictly above "uncertain"'s 0.0 -- a hedge lowers
# confidence, it never erases the signal entirely.
_HEDGE_PENALTY = 0.25
_MIN_CONFIDENCE = 0.1


# Each pattern here is deliberately specific to entry-permission phrasing rather than a bare
# "yes"/"no" -- a caller's window is full of unrelated affirmatives/negatives ("yeah, that's
# right", "no, that's all") that would otherwise swamp the signal for this one field.
_PERMISSION_TO_ENTER_RULES = _DeterminationRules(
    yes=_compile(
        [
            # Negative lookbehind excludes "no permission to enter" -- without it, that phrase
            # (an explicit refusal) would also match this bare positive pattern, so `classify`
            # would see both a yes-hit and a no-hit and fall back to "uncertain" instead of "no".
            (r"(?<!no )permission to enter", 0.90),
            (r"guess it'?s fine", 0.55),
            (r"fine if i'?m not (?:there|home|around)", 0.55),
            (r"go ahead and (?:enter|come (?:in|on in)|let (?:yourself|yourselves) in)", 0.90),
            (r"(?:you|they) can (?:enter|come (?:in|by|over)|go in|let (?:yourself|yourselves) in)", 0.75),
            (r"feel free to (?:enter|come (?:in|by|over)|go in)", 0.75),
            (r"no need to call (?:first|ahead|before)", 0.75),
            (r"don'?t need to (?:call|be (?:home|there|present))", 0.75),
            (r"ok(?:ay)? to enter", 0.90),
            (r"just come (?:by|over|on by|on over)", 0.75),
        ]
    ),
    no=_compile(
        [
            # Requires a first-person subject immediately before the verb, not just the phrase
            # "need to be there" anywhere in the window -- otherwise the caller's own question
            # ("should I... need to be there when they come over?") false-positives as a refusal
            # before their actual answer even arrives, matching on the question itself.
            (r"\bi (?:need|have|must) to be (?:home|there|present)\b", 0.85),
            (r"(?:do not|don'?t) (?:let|allow) (?:them|anyone) (?:in|enter)", 0.90),
            (r"not (?:comfortable|okay|ok) with (?:them|anyone) (?:entering|coming in)", 0.75),
            (r"no permission to enter", 0.90),
            (r"can'?t enter without me", 0.85),
            (r"not (?:giving|going to give) permission", 0.90),
        ]
    ),
)

# Registry keyed by field name, not a single global ruleset -- adding a future determination
# field (per `gliner_pipeline.FIELD_TAXONOMY`) only needs a new `_DeterminationRules` entry here;
# `classify`'s dispatch and its `gliner_pipeline` call site are already field-agnostic.
_RULES_BY_FIELD: dict[str, _DeterminationRules] = {
    "permission_to_enter": _PERMISSION_TO_ENTER_RULES,
}


def classify_with_confidence(field_name: str, window_text: str) -> DeterminationResult:
    """Classify `window_text` for `field_name`, same rules as `classify`, plus a confidence float
    and the matched pattern.

    Returns `DeterminationResult("uncertain", 0.0, None)` for a field with no registered rules, or
    when the window's signal is absent or contradictory (mirrors `classify`'s fallback cases).
    Otherwise picks the highest-confidence pattern among those that matched on the winning side,
    then applies `_HEDGE_PENALTY` (floored at `_MIN_CONFIDENCE`) if the window hedges its phrasing
    -- the hedge never changes `value`, only `confidence`.
    """
    rules = _RULES_BY_FIELD.get(field_name)
    if rules is None or not window_text:
        return DeterminationResult("uncertain", 0.0, None)

    yes_hits = [rule for rule in rules.yes if rule.pattern.search(window_text)]
    no_hits = [rule for rule in rules.no if rule.pattern.search(window_text)]

    if yes_hits and not no_hits:
        value: Determination = "yes"
        matched = max(yes_hits, key=lambda rule: rule.confidence)
    elif no_hits and not yes_hits:
        value = "no"
        matched = max(no_hits, key=lambda rule: rule.confidence)
    else:
        return DeterminationResult("uncertain", 0.0, None)

    confidence = matched.confidence
    if _HEDGE_PATTERN.search(window_text):
        confidence = max(confidence - _HEDGE_PENALTY, _MIN_CONFIDENCE)
    return DeterminationResult(value, confidence, matched.pattern.pattern)


def classify(field_name: str, window_text: str) -> Determination:
    """Classify `window_text` to `"yes"`/`"no"`/`"uncertain"` for `field_name`.

    Returns `"uncertain"` for a field with no registered rules (not a taxonomy-tagged
    determination field, or a future one not yet given its own rules), or when the window's
    signal for a registered field is absent or contradictory.
    """
    return classify_with_confidence(field_name, window_text).value
