import asyncio
import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock

import httpx2
import openai

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jev_live_transcription import config, llm_baseline  # noqa: E402


def _fake_response(
    *,
    response_id: str,
    fields: dict[str, tuple[str | None, float]],
    input_tokens: int,
    cached_tokens: int,
    output_tokens: int,
    status: str = "completed",
    error: object | None = None,
) -> types.SimpleNamespace:
    body = {name: {"value": value, "confidence": conf} for name, (value, conf) in fields.items()}
    usage = types.SimpleNamespace(
        input_tokens=input_tokens,
        input_tokens_details=types.SimpleNamespace(cached_tokens=cached_tokens),
        output_tokens=output_tokens,
    )
    return types.SimpleNamespace(
        id=response_id,
        status=status,
        error=error,
        output_text=json.dumps(body),
        usage=usage,
    )


def _all_fields(value: tuple[str | None, float] = (None, 0.0)) -> dict[str, tuple[str | None, float]]:
    return {name: value for name in llm_baseline.FIELDS}


def _install_fake_client(monkeypatch, create_mock: AsyncMock) -> None:
    fake_client = types.SimpleNamespace(responses=types.SimpleNamespace(create=create_mock))
    monkeypatch.setattr(llm_baseline, "_get_client", lambda: fake_client)


def setup_function(_fn) -> None:
    # Chain state is process-global keyed by call_id; clear it so tests don't
    # leak state into each other via a shared call_id.
    llm_baseline._chain_states.clear()


def test_first_call_sends_full_transcript_with_no_previous_response_id(monkeypatch):
    fields = _all_fields()
    fields["caller_name"] = ("Jane Doe", 0.9)
    create_mock = AsyncMock(
        return_value=_fake_response(
            response_id="resp-1",
            fields=fields,
            input_tokens=500,
            cached_tokens=0,
            output_tokens=50,
        )
    )
    _install_fake_client(monkeypatch, create_mock)

    result = asyncio.run(llm_baseline.extract("call-1", "Agent: hello\nCaller: hi, I'm Jane"))

    sent_kwargs = create_mock.call_args.kwargs
    assert sent_kwargs["input"] == "Agent: hello\nCaller: hi, I'm Jane"
    assert "previous_response_id" not in sent_kwargs
    assert result["fields"]["caller_name"] == "Jane Doe"
    assert result["confidences"]["caller_name"] == 0.9
    assert result["is_committed"]["caller_name"] is True  # 0.9 >= JEV_COMMIT_THRESHOLD
    assert result["response_id"] == "resp-1"


def test_second_call_sends_only_delta_and_chains_from_prior_response_id(monkeypatch):
    create_mock = AsyncMock(
        side_effect=[
            _fake_response(
                response_id="resp-1", fields=_all_fields(), input_tokens=100, cached_tokens=0, output_tokens=10
            ),
            _fake_response(
                response_id="resp-2", fields=_all_fields(), input_tokens=200, cached_tokens=80, output_tokens=10
            ),
        ]
    )
    _install_fake_client(monkeypatch, create_mock)

    asyncio.run(llm_baseline.extract("call-1", "Agent: hello"))
    asyncio.run(llm_baseline.extract("call-1", "Agent: hello\nCaller: hi there"))

    second_kwargs = create_mock.call_args.kwargs
    assert second_kwargs["input"] == "\nCaller: hi there"
    assert second_kwargs["previous_response_id"] == "resp-1"


def test_failed_call_does_not_advance_chain_state_and_next_success_sends_everything_since_last_success(
    monkeypatch,
):
    request = httpx2.Request("POST", "https://api.openai.com/v1/responses")
    transient_error = openai.RateLimitError(
        "rate limited", response=httpx2.Response(429, request=request), body=None
    )
    create_mock = AsyncMock(
        side_effect=[
            _fake_response(
                response_id="resp-1", fields=_all_fields(), input_tokens=100, cached_tokens=0, output_tokens=10
            ),
            transient_error,
            _fake_response(
                response_id="resp-2", fields=_all_fields(), input_tokens=300, cached_tokens=90, output_tokens=10
            ),
        ]
    )
    _install_fake_client(monkeypatch, create_mock)

    asyncio.run(llm_baseline.extract("call-1", "Agent: hello"))

    try:
        asyncio.run(llm_baseline.extract("call-1", "Agent: hello\nCaller: one"))
        assert False, "expected the transient error to propagate"
    except openai.RateLimitError:
        pass

    # A third call, with even more transcript appended, must still chain from
    # resp-1 (the last *successful* response) and include everything since
    # then — the failed attempt's content plus what's new — not just the
    # delta since the failed attempt.
    asyncio.run(llm_baseline.extract("call-1", "Agent: hello\nCaller: one\nCaller: two"))

    third_kwargs = create_mock.call_args.kwargs
    assert third_kwargs["previous_response_id"] == "resp-1"
    assert third_kwargs["input"] == "\nCaller: one\nCaller: two"


def test_permanent_error_raises_permanent_llm_error_and_is_not_silently_swallowed(monkeypatch):
    request = httpx2.Request("POST", "https://api.openai.com/v1/responses")
    permanent_error = openai.NotFoundError(
        "model not found", response=httpx2.Response(404, request=request), body=None
    )
    create_mock = AsyncMock(side_effect=[permanent_error])
    _install_fake_client(monkeypatch, create_mock)

    try:
        asyncio.run(llm_baseline.extract("call-1", "Agent: hello"))
        assert False, "expected PermanentLLMError"
    except llm_baseline.PermanentLLMError as exc:
        assert "404" in str(exc)


def test_incomplete_status_raises_runtime_error(monkeypatch):
    create_mock = AsyncMock(
        return_value=_fake_response(
            response_id="resp-1",
            fields=_all_fields(),
            input_tokens=100,
            cached_tokens=0,
            output_tokens=10,
            status="incomplete",
        )
    )
    _install_fake_client(monkeypatch, create_mock)

    try:
        asyncio.run(llm_baseline.extract("call-1", "Agent: hello"))
        assert False, "expected a RuntimeError for a non-completed response"
    except RuntimeError as exc:
        assert "incomplete" in str(exc)


def test_reset_call_clears_chain_state(monkeypatch):
    create_mock = AsyncMock(
        return_value=_fake_response(
            response_id="resp-1", fields=_all_fields(), input_tokens=100, cached_tokens=0, output_tokens=10
        )
    )
    _install_fake_client(monkeypatch, create_mock)

    asyncio.run(llm_baseline.extract("call-1", "Agent: hello"))
    assert "call-1" in llm_baseline._chain_states

    llm_baseline.reset_call("call-1")
    assert "call-1" not in llm_baseline._chain_states


def test_separate_call_ids_do_not_share_chain_state(monkeypatch):
    create_mock = AsyncMock(
        side_effect=[
            _fake_response(
                response_id="resp-a1", fields=_all_fields(), input_tokens=50, cached_tokens=0, output_tokens=5
            ),
            _fake_response(
                response_id="resp-b1", fields=_all_fields(), input_tokens=60, cached_tokens=0, output_tokens=5
            ),
        ]
    )
    _install_fake_client(monkeypatch, create_mock)

    asyncio.run(llm_baseline.extract("call-a", "Agent: hello A"))
    asyncio.run(llm_baseline.extract("call-b", "Agent: hello B"))

    first_kwargs, second_kwargs = (c.kwargs for c in create_mock.call_args_list)
    assert first_kwargs["input"] == "Agent: hello A"
    assert "previous_response_id" not in first_kwargs
    assert second_kwargs["input"] == "Agent: hello B"
    assert "previous_response_id" not in second_kwargs


# --- cost formula ---


def test_estimate_cost_uses_cached_discount_not_flat_input_rate():
    pricing = config.PRICING_PER_MILLION_TOKENS[llm_baseline.MODEL_NAME]
    price_in = pricing["input"] / 1_000_000
    price_out = pricing["output"] / 1_000_000

    input_tokens = 10_000
    cached_tokens = 4_000
    output_tokens = 500

    expected = (
        (input_tokens - cached_tokens) * price_in
        + cached_tokens * price_in * config.PROMPT_CACHE_DISCOUNT
        + output_tokens * price_out
    )
    actual = llm_baseline._estimate_cost_usd(input_tokens, cached_tokens, output_tokens)
    assert actual == expected

    # Sanity check the formula actually discounts cached tokens rather than
    # billing all input tokens at the flat rate (the mistake this ticket's
    # correction exists to prevent).
    flat_rate_cost = input_tokens * price_in + output_tokens * price_out
    assert actual < flat_rate_cost


def test_estimate_cost_zero_cached_tokens_matches_flat_input_rate():
    pricing = config.PRICING_PER_MILLION_TOKENS[llm_baseline.MODEL_NAME]
    price_in = pricing["input"] / 1_000_000
    price_out = pricing["output"] / 1_000_000

    actual = llm_baseline._estimate_cost_usd(1000, 0, 100)
    expected = 1000 * price_in + 100 * price_out
    assert actual == expected


def test_response_format_schema_covers_all_eleven_fields_and_is_strict():
    schema = llm_baseline._RESPONSE_FORMAT["schema"]
    assert llm_baseline._RESPONSE_FORMAT["strict"] is True
    assert set(schema["properties"]) == set(llm_baseline.FIELDS)
    assert set(schema["required"]) == set(llm_baseline.FIELDS)
    assert schema["additionalProperties"] is False
    for field_schema in schema["properties"].values():
        assert set(field_schema["properties"]) == {"value", "confidence"}
        assert field_schema["additionalProperties"] is False
