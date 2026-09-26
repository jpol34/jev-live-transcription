"""`distinctUntilChanged` plus a short settle window, generalized as a reusable gate.

A `SettleGate` decides whether a per-key observed value is "real enough" to act on: it tracks, per
key, the value a caller last actually resolved and the value most recently observed (plus how many
consecutive observations have reported that same value). An observation only triggers once it both
differs from what was last resolved and has persisted for `settle_ticks` consecutive observations
-- long enough that a single glitchy observation doesn't cause action, but a genuine, sustained
change does.

This is deliberately independent of what "resolving" means to a caller (an API call, a state
transition, anything) and of what the observed value represents -- callers supply a hashable
value per observation and call `record_resolved` once they've actually acted on a triggered one.
"""

from dataclasses import dataclass
from typing import Generic, Hashable, TypeVar

Key = TypeVar("Key", bound=Hashable)
Value = TypeVar("Value", bound=Hashable)


@dataclass
class _KeyState(Generic[Value]):
    last_resolved: Value | None = None
    pending_value: Value | None = None
    pending_streak: int = 0


class SettleGate(Generic[Key, Value]):
    """Gates per-key observations behind a value that has genuinely changed and settled."""

    def __init__(self, settle_ticks: int) -> None:
        if settle_ticks < 1:
            raise ValueError(f"settle_ticks must be >= 1, got {settle_ticks!r}")
        self._settle_ticks = settle_ticks
        self._states: dict[Key, _KeyState[Value]] = {}

    def observe(self, key: Key, value: Value) -> bool:
        """Record one observation of `value` for `key` and report whether it should trigger.

        Returns `True` only once a `value` distinct from `key`'s last-resolved value has been
        observed on `settle_ticks` consecutive `observe` calls for that key. A value that reverts
        before persisting that long never triggers, and a value equal to what's already resolved
        never re-triggers no matter how many times it's observed.
        """
        state = self._states.setdefault(key, _KeyState())
        if value == state.pending_value:
            state.pending_streak += 1
        else:
            state.pending_value = value
            state.pending_streak = 1
        if value == state.last_resolved:
            return False
        return state.pending_streak >= self._settle_ticks

    def record_resolved(self, key: Key, value: Value) -> None:
        """Mark `value` as `key`'s last-resolved value, silencing further triggers for it."""
        state = self._states.setdefault(key, _KeyState())
        state.last_resolved = value
