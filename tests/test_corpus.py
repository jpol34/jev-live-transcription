import json

from jev_live_transcription import corpus


def _write_call(tmp_path, call_id: int):
    transcripts_dir = tmp_path / "transcripts"
    metadata_dir = tmp_path / "metadata"
    transcripts_dir.mkdir(exist_ok=True)
    metadata_dir.mkdir(exist_ok=True)

    stem = f"{call_id:03d}_test_subtype"
    (transcripts_dir / f"{stem}.txt").write_text(
        f"# Call {call_id:03d} — resident / test_subtype\n"
        "# edge_case=False  target_minutes=3.0  disclosure_style=cooperative\n"
        "\n"
        "Agent: Thank you for calling. How can I help?\n"
        "Caller: My sink is leaking.\n",
        encoding="utf-8",
    )
    (metadata_dir / f"{stem}.json").write_text(
        json.dumps(
            {
                "scenario": {"id": call_id, "category": "resident", "subtype": "test_subtype"},
                "metadata": {"caller_name": "Jamie Test", "unit_number": "A-101"},
            }
        ),
        encoding="utf-8",
    )
    return transcripts_dir, metadata_dir


def test_load_all_parses_scenario_ground_truth_and_turns(tmp_path):
    transcripts_dir, metadata_dir = _write_call(tmp_path, 1)

    calls = corpus.load_all(transcripts_dir=transcripts_dir, metadata_dir=metadata_dir)

    assert set(calls.keys()) == {1}
    call = calls[1]
    assert call["scenario"]["subtype"] == "test_subtype"
    assert call["ground_truth"]["caller_name"] == "Jamie Test"
    assert call["transcript_turns"] == [
        {"speaker": "Agent", "text": "Thank you for calling. How can I help?"},
        {"speaker": "Caller", "text": "My sink is leaking."},
    ]


def test_load_all_returns_one_entry_per_call(tmp_path):
    for call_id in range(1, 4):
        transcripts_dir, metadata_dir = _write_call(tmp_path, call_id)

    calls = corpus.load_all(transcripts_dir=transcripts_dir, metadata_dir=metadata_dir)

    assert len(calls) == 3
