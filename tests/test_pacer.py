from jev_live_transcription import config
from jev_live_transcription.pacer import CallPacer

# A small, deterministic synthetic transcript — not a real generated call —
# with enough words for the average-wpm check to converge tightly.
_SYNTHETIC_TURNS = [
    {"speaker": "Agent", "text": " ".join(f"agentword{i}" for i in range(40))},
    {"speaker": "Caller", "text": " ".join(f"callerword{i}" for i in range(40))},
    {"speaker": "Agent", "text": " ".join(f"closingword{i}" for i in range(20))},
]

_TINY_TURNS = [
    {"speaker": "Agent", "text": "hello there friend"},
    {"speaker": "Caller", "text": "hi back"},
]


def _make_pacer(call_id=1, turns=_SYNTHETIC_TURNS):
    return CallPacer(call_id=call_id, transcript_turns=turns)


def test_release_timestamps_are_monotonic():
    pacer = _make_pacer()

    times = [event.release_time for event in pacer.events]

    assert times == sorted(times)
    assert len(set(times)) == len(times)  # strictly increasing, no ties


def test_average_wpm_within_configured_range():
    pacer = _make_pacer()

    assert config.WPM_RANGE[0] <= pacer.average_wpm <= config.WPM_RANGE[1]


def test_deterministic_across_runs():
    pacer_a = _make_pacer(call_id=7)
    pacer_b = _make_pacer(call_id=7)

    assert [e.release_time for e in pacer_a.events] == [e.release_time for e in pacer_b.events]


def test_snapshot_empty_at_tick_zero():
    pacer = _make_pacer(turns=_TINY_TURNS)

    text, offset = pacer.snapshot_at(0)

    assert text == ""
    assert offset == 0


def test_snapshot_full_at_total_ticks():
    pacer = _make_pacer(turns=_TINY_TURNS)

    text, offset = pacer.snapshot_at(pacer.total_ticks)

    assert text == pacer.full_text()
    assert offset == len(pacer.full_text())


def test_snapshot_mid_call_is_a_strict_prefix_of_the_full_text():
    pacer = _make_pacer(turns=_TINY_TURNS)
    full_text = pacer.full_text()
    mid_tick = pacer.total_ticks // 2

    text, offset = pacer.snapshot_at(mid_tick)

    assert 0 < len(text) < len(full_text)
    assert full_text.startswith(text)
    assert offset == len(text)


def test_snapshot_offset_is_non_decreasing_across_ticks():
    pacer = _make_pacer(turns=_SYNTHETIC_TURNS)

    offsets = [pacer.snapshot_at(tick)[1] for tick in range(pacer.total_ticks + 1)]

    assert offsets == sorted(offsets)
    assert offsets[0] == 0
    assert offsets[-1] == len(pacer.full_text())
