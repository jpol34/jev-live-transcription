from jev_live_transcription import config


def test_call_concurrency_derived_from_ram():
    assert config.CALL_CONCURRENCY == (config.RAM_GB - 4) // 4
    assert config.CALL_CONCURRENCY == 3


def test_gliner_concurrency_matches_batch_engine_capacity():
    assert config.GLINER_CONCURRENCY == config.GLINER_BATCH_MAX_SIZE


def test_pricing_table_has_both_models():
    assert "gpt-5.1" in config.PRICING_PER_MILLION_TOKENS
    assert "jev" in config.PRICING_PER_MILLION_TOKENS
    for entry in config.PRICING_PER_MILLION_TOKENS.values():
        assert "input" in entry and "output" in entry
