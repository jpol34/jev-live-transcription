"""Runs the pacer (batch mode) against a few real transcripts and prints the
simulated call duration alongside each call's `estimated_call_seconds`, as a
sanity comparison against the ground-truth metadata.
"""

from jev_live_transcription import config, corpus
from jev_live_transcription.pacer import CallPacer, iter_batch_ticks

SAMPLE_CALL_IDS = [1, 2, 3]


def main() -> None:
    calls = corpus.load_all()

    for call_id in SAMPLE_CALL_IDS:
        call = calls[call_id]
        pacer = CallPacer(call_id=call_id, transcript_turns=call["transcript_turns"])

        final_tick = 0
        final_text = ""
        for tick, text, _offset in iter_batch_ticks(pacer):
            final_tick = tick
            final_text = text
        simulated_seconds = final_tick * config.TICK_SECONDS

        assert final_text == pacer.full_text(), "batch replay did not reach the full transcript"

        estimated_seconds = call["ground_truth"].get("estimated_call_seconds")
        print(
            f"call {call_id:03d} ({call['scenario']['subtype']}): "
            f"simulated={simulated_seconds}s estimated_call_seconds={estimated_seconds}s "
            f"avg_wpm={pacer.average_wpm:.1f}"
        )


if __name__ == "__main__":
    main()
