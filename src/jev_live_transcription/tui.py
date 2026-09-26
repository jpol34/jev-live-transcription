"""Terminal UI for watching one call replay live, tick by tick.

Shows the `gliner_jev` pipeline's current committed value and confidence for all 11 fields, and
the `llm` (GPT-5.1) pipeline's alongside it when `--enable-llm-baseline` is passed, refreshing
every tick. Built entirely on
`pipeline_core.run_call(pacer_mode="realtime")` via its `on_tick` callback -- this module does not
reimplement any tick-pacing or transcript-replay logic itself.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from . import config, corpus, llm_baseline, pipeline_core, secrets

FIELDS: tuple[str, ...] = llm_baseline.FIELDS

DEFAULT_CAPTURE_DIR = corpus.ROOT / "output" / "tui_captures"

_PIPELINE_COLUMN_TITLES = {
    pipeline_core.GLINER_JEV_PIPELINE: "GLiNER + jev",
    pipeline_core.LLM_PIPELINE: "GPT-5.1 (llm)",
}


def _format_cell(value_confidence: tuple[str, float] | None) -> str:
    """Render one pipeline's committed value/confidence for one field as a table cell.

    `value` is extracted from live caller speech, so it may itself contain characters (`[`, `]`)
    that Rich would otherwise interpret as markup -- `escape()` keeps it literal.
    """
    if value_confidence is None:
        return "[dim]—[/dim]"
    value, confidence = value_confidence
    return f"{escape(value)}  [dim]({confidence:.2f})[/dim]"


class _TuiState:
    """Latest per-tick snapshot to render. Mutated from the `on_tick` callback each tick, and
    once more directly (`done`) when the replay finishes."""

    def __init__(self, call_id: int, scenario: dict) -> None:
        self.call_id = call_id
        self.scenario = scenario
        self.tick_number = 0
        self.total_ticks = 0
        self.committed: dict[tuple[str, str], tuple[str, float]] = {}
        self.done = False

    def update(self, tick_number: int, total_ticks: int, committed: dict) -> None:
        self.tick_number = tick_number
        self.total_ticks = total_ticks
        self.committed = committed

    def _header_panel(self) -> Panel:
        elapsed_seconds = self.tick_number * config.TICK_SECONDS
        edge_case_suffix = "  [yellow][edge case][/yellow]" if self.scenario.get("edge_case") else ""
        status = "[green]finished[/green]" if self.done else "[cyan]running[/cyan]"
        body = (
            f"Call {self.call_id} — {self.scenario['category']}/{self.scenario['subtype']}"
            f"{edge_case_suffix}\n"
            f"tick {self.tick_number}/{self.total_ticks}  "
            f"({elapsed_seconds:.0f}s simulated)  {status}"
        )
        return Panel(body, title="jev-live-transcription", border_style="cyan")

    def _fields_table(self) -> Table:
        table = Table(expand=True)
        table.add_column("Field", ratio=1)
        for pipeline in (pipeline_core.GLINER_JEV_PIPELINE, pipeline_core.LLM_PIPELINE):
            table.add_column(_PIPELINE_COLUMN_TITLES[pipeline], ratio=2, overflow="fold")
        for field_name in FIELDS:
            table.add_row(
                field_name,
                _format_cell(self.committed.get((pipeline_core.GLINER_JEV_PIPELINE, field_name))),
                _format_cell(self.committed.get((pipeline_core.LLM_PIPELINE, field_name))),
            )
        return table

    def render(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(self._header_panel(), name="header", size=4),
            Layout(Panel(self._fields_table(), title="Committed field values"), name="body"),
        )
        return layout


async def _run_async(call_id: int, db_path: Path, enable_llm_baseline: bool) -> None:
    calls = corpus.load_all()
    if call_id not in calls:
        raise SystemExit(
            f"call_id {call_id} not found in corpus (see output/metadata/ for available ids)"
        )

    if enable_llm_baseline:
        secrets.load_openai_key()
    secrets.load_typesafe_key()

    db_path.parent.mkdir(parents=True, exist_ok=True)

    call_data = calls[call_id]
    console = Console()
    state = _TuiState(call_id=call_id, scenario=call_data["scenario"])

    def on_tick(tick_number: int, total_ticks: int, snapshot: str, committed: dict) -> None:
        state.update(tick_number, total_ticks, committed)
        live.update(state.render())

    with Live(state.render(), console=console, refresh_per_second=4, screen=False) as live:
        await pipeline_core.run_call(
            call_id,
            db_path,
            pacer_mode="realtime",
            calls=calls,
            on_tick=on_tick,
            enable_llm_baseline=enable_llm_baseline,
        )
        state.done = True
        live.update(state.render())

    console.print(f"\nCall {call_id} finished. Capture DB written to {db_path}")


def run(call_id: int, db_path: Path | str | None = None, enable_llm_baseline: bool = False) -> None:
    """Entry point for `jlt tui <call_id>`: replay `call_id` live and render both pipelines.

    `enable_llm_baseline` defaults to `False` -- the GPT-5.1 comparison column stays empty unless
    explicitly requested, since enabling it calls the real OpenAI API and costs real money.
    """
    resolved_db_path = Path(db_path) if db_path is not None else DEFAULT_CAPTURE_DIR / f"call_{call_id}.db"
    asyncio.run(_run_async(call_id, resolved_db_path, enable_llm_baseline))
