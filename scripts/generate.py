"""Generate 100 synthetic property-management call transcripts concurrently.

Pulls OPENAI_API_KEY from Strongbox at runtime (never printed), fans out
requests to the OpenAI API under a concurrency semaphore, and writes each
transcript + its structured metadata to disk as soon as it completes.
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI
from scenarios import COMPANY, PROPERTY, build_scenarios

ROOT = Path(__file__).resolve().parent.parent
OUT_TRANSCRIPTS = ROOT / "output" / "transcripts"
OUT_METADATA = ROOT / "output" / "metadata"
MANIFEST_PATH = ROOT / "output" / "manifest.jsonl"
LOG_PATH = ROOT / "output" / "generation_log.txt"

MODEL = os.environ.get("TRANSCRIPT_MODEL", "gpt-4o")
CONCURRENCY = int(os.environ.get("TRANSCRIPT_CONCURRENCY", "8"))
MAX_RETRIES = 4


def load_openai_key() -> None:
    if os.environ.get("OPENAI_API_KEY"):
        return
    result = subprocess.run(
        [
            "pwsh", "-NoProfile", "-Command",
            "$WarningPreference = 'SilentlyContinue'; "
            "Import-Module Strongbox -WarningAction SilentlyContinue; "
            "Get-Secret -Name 'OPENAI_API_KEY' -Vault Strongbox -AsPlainText",
        ],
        capture_output=True,
        text=True,
    )
    # Defensive: only take the last non-blank line, in case any module output
    # still leaks onto stdout ahead of the secret.
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    key = lines[-1].strip() if lines else ""
    if result.returncode != 0 or not key:
        raise RuntimeError(
            "Could not retrieve OPENAI_API_KEY from Strongbox: "
            f"rc={result.returncode} stderr={result.stderr.strip()[:200]}"
        )
    os.environ["OPENAI_API_KEY"] = key


SYSTEM_PROMPT = f"""You generate realistic synthetic phone call transcripts between a caller \
and a call-center agent for "{COMPANY}", which manages "{PROPERTY}" and other communities. \
These transcripts are used to test/train a call-handling AI, so they must feel like real, \
messy, natural phone calls: filler words, interruptions, hold moments, the agent verifying \
info, occasional mishearing/spelling-out, background noise mentions, etc. Not scripted or \
overly clean. Vary caller names, ages, tones, and speech patterns across calls.

Return ONLY a JSON object with this exact shape:
{{
  "transcript": [ {{"speaker": "Agent" | "Caller", "text": "..."}}, ... ],
  "metadata": {{
    "caller_name": string|null,
    "email": string|null,
    "phone_number": string|null,
    "unit_number": string|null,
    "amenities_requested": [string]|null,
    "pet_info": string|null,
    "permission_to_enter": "yes"|"no"|"call_first"|null,
    "work_order_issue": string|null,
    "move_in_date": string|null,
    "price_quoted": string|null,
    "budget_amount": string|null,
    "outcome": string,
    "fields_disclosed": [string],
    "fields_withheld_or_unknown": [string],
    "estimated_call_seconds": integer
  }}
}}
Only populate metadata fields that are actually relevant to this call; leave others null. \
"fields_disclosed" lists which of the relevant fields the caller actually gave during the call. \
"fields_withheld_or_unknown" lists relevant fields that came up but were refused, unknown, or \
deferred ("I'll email that over," "I don't have that with me," etc.) — real calls don't collect \
every field cleanly, so use this to reflect that.
"""


def build_user_prompt(scenario: dict) -> str:
    fields = ", ".join(scenario["relevant_fields"])
    edge_note = (
        "This is an EDGE CASE — make it notably harder or more unusual than a routine call "
        "(escalation, confusion, distress, unusual request, policy conflict, etc.)."
        if scenario["edge_case"] else
        "This is a routine, representative call — still natural, not templated."
    )
    target_words = round(scenario["target_minutes"] * 150)
    return f"""Category: {scenario['category']} call
Call type: {scenario['subtype'].replace('_', ' ')}
Target length: approximately {scenario['target_minutes']} minutes of real-time conversation. \
This means the combined dialogue text (both speakers) MUST be approximately {target_words} words \
— not fewer. Pace the call accordingly: add natural back-and-forth (hold moments, verifying \
spelling, small talk, repeated questions, the agent restating info) rather than compressing the \
call into a short exchange. A call this length realistically has {max(8, round(target_words / 25))}+ \
speaker turns. Do not wrap up early — reaching the target word count matters as much as covering \
the scenario.
Caller disclosure style: {scenario['disclosure_style']} — let this shape HOW information comes \
out (freely, only when asked, reluctantly, confused/incomplete, or with a correction after a \
misheard detail), not whether the call happens at all.
Potentially relevant intake fields for this call: {fields}. Only include the ones that would \
naturally come up for this specific call — do not force every field in.
{edge_note}
"""


async def generate_one(client: AsyncOpenAI, scenario: dict, sem: asyncio.Semaphore) -> dict:
    async with sem:
        last_err = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    response_format={"type": "json_object"},
                    max_tokens=4000,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_prompt(scenario)},
                    ],
                    temperature=1.0,
                )
                data = json.loads(resp.choices[0].message.content)
                return {"ok": True, "scenario": scenario, "data": data}
            except Exception as e:  # noqa: BLE001
                last_err = e
                await asyncio.sleep(min(2 ** attempt, 20))
        return {"ok": False, "scenario": scenario, "error": str(last_err)}


def render_transcript_text(scenario: dict, data: dict) -> str:
    lines = [
        f"# Call {scenario['id']:03d} — {scenario['category']} / {scenario['subtype']}",
        f"# edge_case={scenario['edge_case']}  target_minutes={scenario['target_minutes']}  "
        f"disclosure_style={scenario['disclosure_style']}",
        "",
    ]
    for turn in data.get("transcript", []):
        speaker = turn.get("speaker", "?")
        text = turn.get("text", "")
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines) + "\n"


async def main():
    load_openai_key()
    OUT_TRANSCRIPTS.mkdir(parents=True, exist_ok=True)
    OUT_METADATA.mkdir(parents=True, exist_ok=True)

    scenarios = build_scenarios()
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(CONCURRENCY)

    tasks = [asyncio.create_task(generate_one(client, s, sem)) for s in scenarios]

    completed = 0
    failed = []
    total_words = 0
    manifest_lines = []
    log_lines = [f"Started at {time.strftime('%Y-%m-%d %H:%M:%S')} model={MODEL} concurrency={CONCURRENCY}"]

    for coro in asyncio.as_completed(tasks):
        result = await coro
        scenario = result["scenario"]
        sid = scenario["id"]
        if not result["ok"]:
            failed.append((sid, result["error"]))
            print(f"[{sid:03d}] FAILED: {result['error'][:150]}")
            continue

        data = result["data"]
        text = render_transcript_text(scenario, data)
        fname_base = f"{sid:03d}_{scenario['subtype']}"

        (OUT_TRANSCRIPTS / f"{fname_base}.txt").write_text(text, encoding="utf-8")
        meta_record = {"scenario": scenario, "metadata": data.get("metadata", {})}
        (OUT_METADATA / f"{fname_base}.json").write_text(
            json.dumps(meta_record, indent=2), encoding="utf-8"
        )

        word_count = sum(len(t.get("text", "").split()) for t in data.get("transcript", []))
        total_words += word_count
        manifest_lines.append(json.dumps({
            "id": sid,
            "category": scenario["category"],
            "subtype": scenario["subtype"],
            "edge_case": scenario["edge_case"],
            "target_minutes": scenario["target_minutes"],
            "word_count": word_count,
            "estimated_call_seconds": data.get("metadata", {}).get("estimated_call_seconds"),
        }))

        completed += 1
        print(f"[{sid:03d}] done ({completed}/{len(scenarios)}) — {scenario['subtype']}, {word_count} words")

    MANIFEST_PATH.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    avg_words = total_words / completed if completed else 0
    summary = (
        f"\nCompleted: {completed}/{len(scenarios)}  Failed: {len(failed)}\n"
        f"Average word count: {avg_words:.0f}\n"
    )
    if failed:
        summary += "Failed scenarios:\n" + "\n".join(f"  {sid}: {err[:200]}" for sid, err in failed)
    print(summary)
    log_lines.append(summary)
    LOG_PATH.write_text("\n".join(log_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
