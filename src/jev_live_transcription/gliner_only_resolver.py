"""gliner_only pipeline arm: a fully local, synchronous commit-policy resolver.

Mirrors `jev_pipeline.JevFieldResolver`'s public shape (`resolve_field`) for accumulating and
disambiguating GLiNER candidate spans for one field across the life of a call, but makes every
commit decision from GLiNER's own candidate scores alone -- no `await`, no network call to
jev/typesafe.ai. Reuses `jev_pipeline.normalize_candidate`/`normalize_email_span` for candidate
dedup and `settle_gate.SettleGate` for gating the multi-candidate path unchanged, so both pipeline
arms dedup and settle candidates identically and only diverge on how a settled set gets scored.

Per (call_id, field_name), distinct candidate values (deduped by `normalize_candidate`) accumulate
across ticks and never shrink, same as `JevFieldResolver`:

- 0 distinct candidates: no decision is made.
- Exactly 1 distinct candidate, not yet committed: commit it if its GLiNER score clears the
  field's commit floor (`_commit_floor`).
- Exactly 1 distinct candidate, already committed: nothing new to resolve.
- 2+ distinct candidates ever seen: gated by a `SettleGate` exactly like jev's Choice path -- only
  re-decided once the distinct-candidate set has persisted `config.GLINER_ONLY_SETTLE_TICKS`
  observations. Once a set has been decided once, the gate keeps it locked (no further re-decision
  for that exact set) with one exception: a set already locked in as `"none_of_these"` is reopened
  on any later tick whose `context_window` contains a self-correction cue, even though the set
  itself hasn't changed -- a locked real-candidate commit (`margin_winner`) is never reopened this
  way, since re-litigating an already-confident answer isn't what the cue is for. Every (re-)decision
  ranks candidates by score plus `config.GLINER_ONLY_SELF_CORRECTION_SCORE_BONUS` for whichever one
  this tick's cue (if any) points to. The top-ranked candidate commits if it clears its floor and
  its margin over the runner-up clears `config.GLINER_ONLY_MARGIN_THRESHOLD`; otherwise the set
  (re-)commits to `"none_of_these"` -- a settled-but-ambiguous set is a confident rejection, not an
  inconclusive no-op, so this always pairs `is_none_of_these=True` with `is_committed=True`.

Each candidate's GLiNER score is tracked per normalized value as "freshest known": a tick that
re-reports the candidate updates it, a tick where it ages out of GLiNER's report leaves the last
known score in place rather than dropping the candidate.
"""

import re
from dataclasses import dataclass
from typing import Literal

from jev_live_transcription import config
from jev_live_transcription.gliner_pipeline import FIELD_TAXONOMY
from jev_live_transcription.jev_pipeline import CallId, normalize_candidate, normalize_email_span
from jev_live_transcription.settle_gate import SettleGate

_NONE_OF_THESE = "none_of_these"

# Fixed lexical cues for a caller self-correcting mid-answer. Deliberately small and literal
# rather than a learned classifier, matching this module's fully local, no-LLM-in-the-runtime-loop
# premise (`determination_classifier`'s module docstring documents the same choice).
_SELF_CORRECTION_CUE_RE = re.compile(
    r"\b(?:actually|no wait|i mean|sorry i meant|scratch that|correction)\b",
    re.IGNORECASE,
)

DecisionReason = Literal[
    "single_above_floor",
    "single_below_floor",
    "margin_winner",
    "margin_too_close",
    "below_floor",
]


@dataclass(frozen=True)
class GlinerOnlyResolution:
    """The outcome of one local resolve_field decision for one field at one tick.

    Mirrors `jev_pipeline.JevResolution`'s shape, minus the jev-call-specific fields (no question
    type or token usage, since no network call is made), plus `decision_reason` explaining which
    branch of the commit policy produced this result.

    `candidate` holds the resolver's raw selection, which for a settled multi-candidate set can be
    the synthetic `"none_of_these"` label rather than an actual candidate value -- callers must
    check `is_none_of_these` before treating `candidate` as a real field value.
    """

    call_id: CallId
    field_name: str
    candidate: str
    confidence: float
    is_committed: bool
    is_none_of_these: bool
    distinct_candidates: tuple[str, ...]
    decision_reason: DecisionReason


def _commit_floor(field_name: str) -> float:
    """The minimum GLiNER score a top candidate must clear to commit for `field_name`.

    Determination-taxonomy fields use their own floor (their score comes from
    `determination_classifier`'s confidence scale, not GLiNER's span-score distribution);
    every other field uses its `GLINER_ONLY_PER_FIELD_COMMIT_THRESHOLDS` override if one exists,
    else `GLINER_ONLY_COMMIT_THRESHOLD`.
    """
    if FIELD_TAXONOMY.get(field_name) == "determination":
        return config.GLINER_ONLY_DETERMINATION_COMMIT_FLOOR
    return config.GLINER_ONLY_PER_FIELD_COMMIT_THRESHOLDS.get(
        field_name, config.GLINER_ONLY_COMMIT_THRESHOLD
    )


def _normalize_span_text(field_name: str, text: str) -> str:
    value = text.strip()
    if field_name == "email":
        value = normalize_email_span(value)
    return normalize_candidate(field_name, value)


class GlinerOnlyResolver:
    """Resolves candidate spans for fields across the life of one or more calls, fully locally."""

    def __init__(self, *, settle_ticks: int = config.GLINER_ONLY_SETTLE_TICKS) -> None:
        self._candidates: dict[tuple[CallId, str], dict[str, str]] = {}
        # Each distinct candidate's freshest-known GLiNER score, keyed the same way as
        # `_candidates` -- see module docstring for the aging-out fallback this enables.
        self._scores: dict[tuple[CallId, str], dict[str, float]] = {}
        self._committed: dict[tuple[CallId, str], bool] = {}
        self._settle_gate: SettleGate[tuple[CallId, str], frozenset[str]] = SettleGate(settle_ticks)
        # The most recently (re-)decided candidate set for the 2+-candidate path, and whether that
        # decision was `"none_of_these"` -- lets a later tick's self-correction cue reopen a locked
        # ambiguous rejection (see `resolve_field`) without the `SettleGate` itself needing to know
        # about cues at all.
        self._last_settled_set: dict[tuple[CallId, str], frozenset[str]] = {}
        self._last_settled_was_none_of_these: dict[tuple[CallId, str], bool] = {}

    def resolve_field(
        self,
        call_id: CallId,
        field_name: str,
        candidate_spans: list[dict],
        context_window: str,
    ) -> GlinerOnlyResolution | None:
        """Fold this tick's `candidate_spans` into the field's dedup set and resolve if warranted.

        `candidate_spans` is shaped like `gliner_pipeline.extract_candidates`'s per-field output:
        each entry a dict with `"text"`, `"score"`, and (for span/list_span fields) `"start"`/
        `"end"` offsets. Determination-taxonomy fields' synthetic candidates carry `start`/`end` of
        `None`, matching `gliner_pipeline._apply_determination_classifier`. Where offsets are
        present, they are read as positions within `context_window` -- this resolver has no
        visibility into any other coordinate system, so callers must slice `candidate_spans` and
        `context_window` consistently with each other.

        Returns `None` when no decision is warranted this tick: no candidates seen yet, a single
        already-committed candidate with nothing new to resolve, or -- for 2+ distinct candidates
        -- the set hasn't yet settled per `SettleGate` and isn't a locked `"none_of_these"` set
        that this tick's self-correction cue reopens (see module docstring).
        """
        key = (call_id, field_name)
        seen = self._candidates.setdefault(key, {})
        scores = self._scores.setdefault(key, {})

        for span in candidate_spans:
            raw = span.get("text")
            if not raw or not raw.strip():
                continue
            value = raw.strip()
            display = normalize_email_span(value) if field_name == "email" else value
            normalized = normalize_candidate(field_name, display)
            if not normalized:
                continue
            if normalized not in seen:
                seen[normalized] = display
            score = span.get("score")
            if score is not None:
                scores[normalized] = float(score)

        distinct = list(seen.keys())
        if not distinct:
            return None

        if len(distinct) == 1:
            normalized = distinct[0]
            if self._committed.get(key, False):
                return None
            result = self._resolve_single(call_id, field_name, normalized, seen, scores)
        else:
            candidate_set = frozenset(seen.keys())
            settled = self._settle_gate.observe(key, candidate_set)
            if not settled and not self._reopen_on_correction_cue(
                key, candidate_set, context_window
            ):
                return None
            result = self._resolve_settled(
                call_id, field_name, seen, scores, candidate_spans, context_window
            )
            if result.is_committed:
                self._settle_gate.record_resolved(key, candidate_set)
                self._last_settled_set[key] = candidate_set
                self._last_settled_was_none_of_these[key] = result.is_none_of_these

        self._committed[key] = result.is_committed
        return result

    def _reopen_on_correction_cue(
        self, key: tuple[CallId, str], candidate_set: frozenset[str], context_window: str
    ) -> bool:
        """Whether a candidate set the `SettleGate` reports as not (newly) settled should still be
        re-decided this tick because it's the same set already locked in as `"none_of_these"`, and
        this tick's `context_window` carries a self-correction cue -- see module docstring.
        """
        if self._last_settled_set.get(key) != candidate_set:
            return False
        if not self._last_settled_was_none_of_these.get(key, False):
            return False
        return bool(context_window) and _SELF_CORRECTION_CUE_RE.search(context_window) is not None

    @staticmethod
    def _resolve_single(
        call_id: CallId,
        field_name: str,
        normalized: str,
        seen: dict[str, str],
        scores: dict[str, float],
    ) -> GlinerOnlyResolution:
        candidate = seen[normalized]
        score = scores.get(normalized, 0.0)
        floor = _commit_floor(field_name)
        is_committed = score >= floor
        return GlinerOnlyResolution(
            call_id=call_id,
            field_name=field_name,
            candidate=candidate,
            confidence=score,
            is_committed=is_committed,
            is_none_of_these=False,
            distinct_candidates=(candidate,),
            decision_reason="single_above_floor" if is_committed else "single_below_floor",
        )

    @staticmethod
    def _resolve_settled(
        call_id: CallId,
        field_name: str,
        seen: dict[str, str],
        scores: dict[str, float],
        candidate_spans: list[dict],
        context_window: str,
    ) -> GlinerOnlyResolution:
        distinct_candidates = tuple(seen.values())
        bonus_target = _self_correction_target(field_name, candidate_spans, context_window, seen)

        def scored(normalized: str) -> float:
            bonus = (
                config.GLINER_ONLY_SELF_CORRECTION_SCORE_BONUS
                if normalized == bonus_target
                else 0.0
            )
            return scores.get(normalized, 0.0) + bonus

        ranked = sorted(seen.keys(), key=scored, reverse=True)
        top, runner_up = ranked[0], ranked[1]
        top_score = scored(top)
        runner_up_score = scored(runner_up)
        floor = _commit_floor(field_name)
        margin = top_score - runner_up_score

        if top_score >= floor and margin >= config.GLINER_ONLY_MARGIN_THRESHOLD:
            return GlinerOnlyResolution(
                call_id=call_id,
                field_name=field_name,
                candidate=seen[top],
                confidence=top_score,
                is_committed=True,
                is_none_of_these=False,
                distinct_candidates=distinct_candidates,
                decision_reason="margin_winner",
            )
        return GlinerOnlyResolution(
            call_id=call_id,
            field_name=field_name,
            candidate=_NONE_OF_THESE,
            confidence=top_score,
            is_committed=True,
            is_none_of_these=True,
            distinct_candidates=distinct_candidates,
            decision_reason="margin_too_close" if top_score >= floor else "below_floor",
        )


def _self_correction_target(
    field_name: str,
    candidate_spans: list[dict],
    context_window: str,
    seen: dict[str, str],
) -> str | None:
    """The normalized candidate value this tick's self-correction cue points to, if any.

    Returns `None` when no cue is present in `context_window`, or when the cue is present but
    points to nothing usable (no candidate anywhere in `context_window` after the cue, or the
    pointed-to candidate isn't one of the field's known distinct values).
    """
    if not context_window:
        return None
    match = _SELF_CORRECTION_CUE_RE.search(context_window)
    if match is None:
        return None

    if FIELD_TAXONOMY.get(field_name) == "determination":
        # Synthetic determination candidates carry no real span offsets -- bias toward this
        # tick's freshest classification instead, i.e. the last entry this tick's own
        # `candidate_spans` reports for the field.
        fresh = [span.get("text") for span in candidate_spans if span.get("text")]
        if not fresh:
            return None
        normalized = normalize_candidate(field_name, fresh[-1].strip())
        return normalized if normalized in seen else None

    # Span/list_span fields: bias toward the candidate whose span starts soonest after the cue
    # (closest to it), matching a caller's "no wait, it's <corrected value>" phrasing where the
    # correction directly follows the cue.
    cue_end = match.end()
    best_normalized: str | None = None
    best_start: int | None = None

    def consider(normalized: str, start: int) -> None:
        nonlocal best_normalized, best_start
        if start < cue_end or normalized not in seen:
            return
        if best_start is None or start < best_start:
            best_normalized = normalized
            best_start = start

    for span in candidate_spans:
        start = span.get("start")
        raw = span.get("text")
        if start is not None and raw:
            consider(_normalize_span_text(field_name, raw), start)

    # A candidate already known from an earlier tick but absent from this tick's own
    # `candidate_spans` (aged out of the extraction stage's window) can still be what the cue
    # points to, if its own text literally reappears in `context_window` after the cue -- e.g. the
    # caller just re-spoke it as part of the correction itself. Only consulted when no live span
    # already answered, since a live span's own offset is the more precise signal.
    if best_normalized is None:
        for normalized, display in seen.items():
            start = context_window.find(display, cue_end)
            if start != -1:
                consider(normalized, start)

    return best_normalized
