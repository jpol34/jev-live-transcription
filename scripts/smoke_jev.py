"""Exercises `jev_pipeline` against the live typesafe.ai API.

Run manually: `uv run python scripts/smoke_jev.py`. Requires network access; sources
`TYPESAFE_API_KEY` from Strongbox via `secrets.load_typesafe_key()`.

Three hand-crafted cases:
  (a) a single-candidate case that should resolve via Noul.
  (b) a multi-candidate phone-number self-correction, modeled on the real transcript in
      `output/transcript_short/metadata/001_noise_neighbor_complaint.json` (the caller says
      "555...wait, it's 555-3212...uh, 555-4321") -- confirms the Choice call favors the
      corrected final value, 555-4321.
  (c) a forced failure (invalid API key), confirming the error-as-data path raises rather than
      swallowing the failure.

Also fires a handful of concurrent calls through one shared resolver/client to demonstrate the
shared `AsyncTypeSafeClient` is safe to use from multiple tasks at once.
"""

import asyncio

from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from jev_live_transcription import secrets
from jev_live_transcription.jev_pipeline import JevFieldResolver, JevResolutionError

CALL_SINGLE = "smoke-single-candidate"
CALL_CORRECTION = "smoke-phone-self-correction"
CALL_FAILURE = "smoke-forced-failure"
CALL_CONCURRENT_PREFIX = "smoke-concurrent"


async def case_single_candidate(resolver: JevFieldResolver) -> None:
    print("\n=== case (a): single-candidate Noul confirmation ===")
    result = await resolver.resolve_field(
        call_id=CALL_SINGLE,
        field_name="caller_name",
        candidates=["Lindsey Perkins"],
        context_window="Agent: Can I get your name please? Caller: Yeah, it's Lindsey Perkins.",
    )
    assert result is not None, "expected a Noul resolution, got None"
    assert result.question_type == "noul", f"expected noul, got {result.question_type}"
    assert result.candidate == "Lindsey Perkins"
    print(
        f"candidate={result.candidate!r} confidence={result.confidence:.3f} "
        f"is_committed={result.is_committed}"
    )


async def case_self_correction(resolver: JevFieldResolver) -> None:
    print("\n=== case (b): phone-number self-correction (Choice) ===")
    first = await resolver.resolve_field(
        call_id=CALL_CORRECTION,
        field_name="phone_number",
        candidates=["555-3212"],
        context_window="Caller: My number is 555...wait, it's 555-3212...",
    )
    print(f"after first mention: {first}")
    assert first is not None and first.question_type == "noul"

    second = await resolver.resolve_field(
        call_id=CALL_CORRECTION,
        field_name="phone_number",
        candidates=["555-3212", "555-4321"],
        context_window=(
            "Caller: My number is 555...wait, it's 555-3212...uh, 555-4321. "
            "Sorry, let me say that again, it's 555-4321."
        ),
    )
    assert second is not None, "expected a Choice resolution, got None"
    assert second.question_type == "choice", f"expected choice, got {second.question_type}"
    assert set(second.distinct_candidates) == {"555-3212", "555-4321"}
    print(
        f"after correction: candidate={second.candidate!r} confidence={second.confidence:.3f} "
        f"is_committed={second.is_committed}"
    )
    if second.candidate == "555-4321":
        print("PASS: jev favored the corrected number, 555-4321")
    else:
        print(f"FAIL: expected jev to favor 555-4321, got {second.candidate!r}")


async def case_forced_failure() -> None:
    print("\n=== case (c): forced failure (invalid API key) is not swallowed ===")
    bad_client = AsyncTypeSafeClient(
        api_key="sk-invalid-smoke-test-key", retry=RetryPolicy(max_retries=0)
    )
    resolver = JevFieldResolver(client=bad_client)
    try:
        result = await resolver.resolve_field(
            call_id=CALL_FAILURE,
            field_name="caller_name",
            candidates=["Someone"],
            context_window="Caller: My name is Someone.",
        )
    except JevResolutionError as exc:
        print(f"PASS: got expected JevResolutionError: {exc}")
    else:
        raise AssertionError(f"expected a JevResolutionError, got a result instead: {result!r}")
    finally:
        await bad_client.aclose()


async def case_concurrent_calls(resolver: JevFieldResolver) -> None:
    print("\n=== bonus: concurrent calls through one shared client ===")
    results = await asyncio.gather(
        *(
            resolver.resolve_field(
                call_id=f"{CALL_CONCURRENT_PREFIX}-{i}",
                field_name="unit_number",
                candidates=[f"B-{200 + i}"],
                context_window=f"Caller: I'm in unit B-{200 + i}.",
            )
            for i in range(4)
        )
    )
    assert all(r is not None for r in results)
    print(f"PASS: {len(results)} concurrent calls all completed")


async def main() -> None:
    secrets.load_typesafe_key()
    resolver = JevFieldResolver()
    try:
        await case_single_candidate(resolver)
        await case_self_correction(resolver)
        await case_forced_failure()
        await case_concurrent_calls(resolver)
    finally:
        await resolver.aclose()
    print("\nAll smoke cases completed.")


if __name__ == "__main__":
    asyncio.run(main())
