"""Unit tests for `SettleGate` in isolation -- no jev/GLiNER involved."""

import pytest

from jev_live_transcription.settle_gate import SettleGate


def test_settle_ticks_must_be_positive():
    with pytest.raises(ValueError):
        SettleGate(0)


def test_value_that_never_changes_never_retriggers_after_first_resolution():
    gate: SettleGate[str, frozenset[str]] = SettleGate(settle_ticks=2)
    value = frozenset({"a", "b"})

    # Not yet settled on the first two observations, then triggers once settled.
    assert gate.observe("k", value) is False
    assert gate.observe("k", value) is True
    gate.record_resolved("k", value)

    # The same value repeated many more times must never re-trigger.
    for _ in range(10):
        assert gate.observe("k", value) is False


def test_value_that_reverts_after_one_tick_never_triggers():
    gate: SettleGate[str, frozenset[str]] = SettleGate(settle_ticks=2)
    original = frozenset({"a"})
    changed = frozenset({"a", "b"})

    gate.observe("k", original)
    gate.record_resolved("k", original)

    # `changed` is observed for only 1 tick, then reverts to `original` -- never persists for the
    # settle window, so it must never trigger.
    assert gate.observe("k", changed) is False
    assert gate.observe("k", original) is False
    assert gate.observe("k", original) is False


def test_value_that_persists_for_settle_window_triggers_exactly_once():
    gate: SettleGate[str, frozenset[str]] = SettleGate(settle_ticks=3)
    original = frozenset({"a"})
    changed = frozenset({"a", "b"})

    gate.observe("k", original)
    gate.record_resolved("k", original)

    assert gate.observe("k", changed) is False
    assert gate.observe("k", changed) is False
    assert gate.observe("k", changed) is True  # 3rd consecutive observation: settled.
    gate.record_resolved("k", changed)

    # Once resolved, further identical observations don't re-trigger.
    assert gate.observe("k", changed) is False
    assert gate.observe("k", changed) is False


def test_settle_ticks_of_one_triggers_on_first_change():
    gate: SettleGate[str, frozenset[str]] = SettleGate(settle_ticks=1)
    original = frozenset({"a"})
    changed = frozenset({"a", "b"})

    gate.observe("k", original)
    gate.record_resolved("k", original)

    assert gate.observe("k", changed) is True


def test_keys_are_tracked_independently():
    gate: SettleGate[str, frozenset[str]] = SettleGate(settle_ticks=2)
    value = frozenset({"a", "b"})

    assert gate.observe("field_a", value) is False
    assert gate.observe("field_b", value) is False
    assert gate.observe("field_b", value) is True
    # field_a's own streak is unaffected by field_b's observations.
    assert gate.observe("field_a", value) is True


def test_pending_streak_resets_when_value_changes_again_before_settling():
    gate: SettleGate[str, frozenset[str]] = SettleGate(settle_ticks=2)
    a = frozenset({"a"})
    b = frozenset({"a", "b"})
    c = frozenset({"a", "c"})

    gate.observe("k", a)
    gate.record_resolved("k", a)

    assert gate.observe("k", b) is False  # streak=1 for b
    assert gate.observe("k", c) is False  # value changed again: streak resets to 1 for c
    assert gate.observe("k", c) is True  # streak=2 for c: settled
