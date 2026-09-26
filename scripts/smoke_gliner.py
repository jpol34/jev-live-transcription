"""Smoke-tests the local GLiNER candidate-extraction pipeline against real
transcripts, simulating growing per-tick transcript snapshots and printing
per-field candidates alongside ground truth for human review.
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jev_live_transcription import corpus, gliner_pipeline  # noqa: E402

TICK_TURN_STRIDE = 2  # simulate one tick every N transcript turns
GROUND_TRUTH_SKIP_KEYS = {"outcome", "fields_disclosed", "fields_withheld_or_unknown", "estimated_call_seconds"}


def _select_demo_calls(calls: dict[int, dict], count: int = 2) -> list[int]:
    """Pick the calls with the richest ground truth for a varied, informative demo."""

    def field_richness(call: dict) -> int:
        return sum(1 for value in call["ground_truth"].values() if value not in (None, [], ""))

    ranked = sorted(calls.items(), key=lambda item: field_richness(item[1]), reverse=True)
    return [call_id for call_id, _ in ranked[:count]]


async def _run_call(call_id: int, call: dict) -> None:
    turns = call["transcript_turns"]
    ground_truth = call["ground_truth"]
    session_id = f"call-{call_id}"

    print(f"\n{'=' * 80}\nCall {call_id} — {call['scenario']['subtype']}\n{'=' * 80}")
    print("Ground truth:")
    for field, value in ground_truth.items():
        if field in GROUND_TRUTH_SKIP_KEYS:
            continue
        print(f"  {field}: {value!r}")

    # Wrap the zero-shot tick to capture its per-tick latency for display below -- it re-encodes
    # only a bounded trailing window of the transcript (config.GLINER_ZERO_SHOT_WINDOW_CHARS), so
    # this is the number to watch for confirming that latency stays flat as the call gets longer.
    original_zero_shot_tick = gliner_pipeline._run_zero_shot_tick
    zero_shot_latency_s = 0.0

    def _timed_zero_shot_tick(transcript_snapshot: str) -> dict:
        nonlocal zero_shot_latency_s
        start = time.perf_counter()
        result = original_zero_shot_tick(transcript_snapshot)
        zero_shot_latency_s = time.perf_counter() - start
        return result

    gliner_pipeline._run_zero_shot_tick = _timed_zero_shot_tick
    try:
        tick_ends = range(TICK_TURN_STRIDE, len(turns) + TICK_TURN_STRIDE, TICK_TURN_STRIDE)
        for tick, end in enumerate(tick_ends, start=1):
            snapshot = corpus.render_turns(turns[:end])
            candidates = await gliner_pipeline.extract_candidates(snapshot, call_id=session_id)
            shown_turns = min(end, len(turns))
            print(
                f"\n--- tick {tick} (turns 1-{shown_turns}, "
                f"zero-shot latency {zero_shot_latency_s * 1000:.0f}ms) ---"
            )
            any_candidates = False
            for field, spans in candidates.items():
                if not spans:
                    continue
                any_candidates = True
                rendered = ", ".join(f"{span['text']!r} ({span['score']:.2f})" for span in spans)
                print(f"  {field}: {rendered}")
            if not any_candidates:
                print("  (no candidates yet)")
    finally:
        gliner_pipeline._run_zero_shot_tick = original_zero_shot_tick
        gliner_pipeline.reset_call(session_id)


async def main() -> None:
    calls = corpus.load_all()
    demo_call_ids = _select_demo_calls(calls)
    print(f"Running GLiNER smoke test against calls: {demo_call_ids}")
    for call_id in demo_call_ids:
        await _run_call(call_id, calls[call_id])


if __name__ == "__main__":
    asyncio.run(main())
