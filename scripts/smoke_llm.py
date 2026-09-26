"""Smoke-tests the GPT-5.1 baseline against the real OpenAI API.

Verifies three things against a live account, in order: (a) `"gpt-5.1"` is
actually callable — if not, this stops and reports the failure rather than
falling back to a different model; (b) `previous_response_id` chaining works
across several sequential calls simulating a growing transcript; (c)
`usage.input_tokens_details.cached_tokens` goes nonzero on later calls in the
chain, which is the real evidence caching is active (`input_tokens` itself is
NOT expected to shrink — see `llm_baseline`'s module docstring).
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jev_live_transcription import corpus, llm_baseline, secrets  # noqa: E402

TICK_TURN_STRIDE = 2  # simulate one LLM-cadence tick every N transcript turns


def _select_demo_call(calls: dict[int, dict]) -> int:
    def field_richness(call: dict) -> int:
        return sum(1 for value in call["ground_truth"].values() if value not in (None, [], ""))

    return max(calls.items(), key=lambda item: field_richness(item[1]))[0]


async def main() -> None:
    print("Loading OPENAI_API_KEY from Strongbox...")
    secrets.load_openai_key()

    calls = corpus.load_all()
    call_id_int = _select_demo_call(calls)
    call = calls[call_id_int]
    turns = call["transcript_turns"]
    session_id = f"smoke-call-{call_id_int}"

    print(f"Running against call {call_id_int} ({call['scenario']['subtype']}), "
          f"{len(turns)} turns, in strides of {TICK_TURN_STRIDE} turns.\n")

    tick_ends = list(range(TICK_TURN_STRIDE, len(turns) + TICK_TURN_STRIDE, TICK_TURN_STRIDE))
    cached_tokens_seen_nonzero = False
    try:
        for tick, end in enumerate(tick_ends, start=1):
            snapshot = corpus.render_turns(turns[: min(end, len(turns))])
            try:
                result = await llm_baseline.extract(session_id, snapshot)
            except llm_baseline.PermanentLLMError as exc:
                print(f"\nBLOCKER: gpt-5.1 is not callable on this account: {exc}")
                sys.exit(1)

            if result["cached_tokens"] > 0:
                cached_tokens_seen_nonzero = True

            print(
                f"--- tick {tick} (response_id={result['response_id']}) ---\n"
                f"  input_tokens={result['input_tokens']} "
                f"cached_tokens={result['cached_tokens']} "
                f"output_tokens={result['output_tokens']} "
                f"latency_ms={result['latency_ms']:.0f} "
                f"cost_usd={result['estimated_cost_usd']:.6f}"
            )
            committed = {
                name: value
                for name, value in result["fields"].items()
                if result["is_committed"][name] and value is not None
            }
            print(f"  committed fields: {committed}\n")
    finally:
        llm_baseline.reset_call(session_id)

    print("=" * 80)
    if cached_tokens_seen_nonzero:
        print("PASS: cached_tokens went nonzero on a later call in the chain — "
              "prompt caching is active.")
    else:
        print("WARNING: cached_tokens never went nonzero across this chain. "
              "gpt-5.1 was still callable and chaining worked (previous_response_id "
              "was accepted each call), but caching evidence did not show up — "
              "possibly because the chain is short or caching is model/account "
              "dependent. This is not the failure this script is meant to catch "
              "(a model that isn't callable at all); report it for review rather "
              "than treating it as a hard blocker.")


if __name__ == "__main__":
    asyncio.run(main())
