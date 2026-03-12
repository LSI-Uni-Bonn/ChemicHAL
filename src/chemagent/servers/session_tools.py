"""chemagent.servers.session_tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MCP server instance, session logger singleton, ``_register`` decorator, and
session/utility MCP tool functions.

Imported by ``chemagent_mcp.py``:
    from chemagent.servers.session_tools import mcp, session_logger, _register, ...

Singletons
----------
mcp            — the FastMCP server instance
session_logger — the active SessionLogger (also used by get_session_logger())
_register      — decorator that wraps a function with logging and adds it as a tool

Functions
---------
log_thought        — record agent reasoning / planning in the session log
generate_report    — write a Markdown summary of the current session to disk
generate_pdf_report — write a clean PDF with agent narrative (thoughts) and plots
start_new_session  — start a fresh logging session, ending the current one
"""

from __future__ import annotations

import functools
import json as _json
import sys
import time
from datetime import datetime as _dt, timezone as _tz
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from chemagent.logging import SessionLogger


# ===========================================================================
# MCP server instance + session logger singleton + _register decorator
# ===========================================================================

mcp = FastMCP("chemagent")

_log_dir = Path(__file__).resolve().parents[3] / "data" / "logs"
session_logger = SessionLogger(_log_dir)


def _register(fn):
    """Wrap *fn* with call/result logging and register it as an MCP tool."""
    @functools.wraps(fn)
    def logged_fn(*args, **kwargs):
        call_id   = session_logger.start_call(fn.__name__, kwargs)
        t_start   = time.perf_counter()
        try:
            result      = fn(*args, **kwargs)
            duration_ms = (time.perf_counter() - t_start) * 1000
            session_logger.end_call(call_id, result=result, duration_ms=duration_ms)
            # Copy artifacts asynchronously so the MCP response is not delayed
            import threading
            threading.Thread(
                target=session_logger.copy_artifacts_from_result,
                args=(result,),
                daemon=True,
            ).start()
            return result
        except Exception as exc:
            duration_ms = (time.perf_counter() - t_start) * 1000
            session_logger.end_call(call_id, error=exc, duration_ms=duration_ms)
            raise
    mcp.add_tool(logged_fn)
    return logged_fn


# ===========================================================================
# log_thought
# ===========================================================================

def log_thought(
    thought: str,
    step: Optional[str] = None,
) -> dict[str, str]:
    """Record a reasoning or planning step in the session log.

    Call this to capture chain-of-thought, observations, or decisions in the
    session log. This is the only way the LLM's reasoning reaches the log.

    Args:
        thought: Reasoning, plan, observation, or decision text.
        step: Optional phase label ("plan", "observation", "decision", "summary").

    Returns:
        {"logged": "ok", "session_id": <id>}
    """
    session_logger.log_thought(thought, step=step)
    return {"logged": "ok", "session_id": session_logger.session_id}


# ===========================================================================
# Shared log-parsing helper
# ===========================================================================

def _parse_session_log(
    log_file: Path,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Parse a JSON-lines session log into (calls, thoughts, artifacts)."""
    events: list[dict] = []
    if log_file.exists():
        for raw in log_file.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                events.append(_json.loads(raw))
            except _json.JSONDecodeError:
                pass

    starts:    dict[str, dict] = {}
    calls:     list[dict] = []
    thoughts:  list[dict] = []
    artifacts: list[dict] = []

    for ev in events:
        etype = ev.get("type", "")
        if etype == "call_start":
            starts[ev["call_id"]] = ev
        elif etype == "call_end":
            cid   = ev.get("call_id")
            start = starts.pop(cid, {})
            calls.append({
                "call_id":     cid,
                "tool":        start.get("tool", "?"),
                "args":        start.get("args", {}),
                "status":      ev.get("status", "?"),
                "duration_ms": ev.get("duration_ms", 0),
                "result":      ev.get("result"),
                "error":       ev.get("error"),
                "timestamp":   start.get("timestamp", ev.get("timestamp", "")),
            })
        elif etype == "llm_thought":
            thoughts.append(ev)
        elif etype == "artifact_saved":
            artifacts.append(ev)

    return calls, thoughts, artifacts


def _parse_chat_events(log_file: Path) -> list[dict]:
    """Return session events as a time-ordered list of chat items.

    Each item is a dict with ``"type"`` of either ``"thought"`` or
    ``"tool_call"``, plus type-specific fields.
    """
    events: list[dict] = []
    if log_file.exists():
        for raw in log_file.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                events.append(_json.loads(raw))
            except _json.JSONDecodeError:
                pass

    starts: dict[str, dict] = {}
    chat:   list[dict]      = []

    for ev in events:
        etype = ev.get("type", "")
        if etype == "call_start":
            starts[ev["call_id"]] = ev
        elif etype == "call_end":
            cid   = ev.get("call_id")
            start = starts.pop(cid, {})
            chat.append({
                "type":        "tool_call",
                "timestamp":   start.get("timestamp", ev.get("timestamp", "")),
                "tool":        start.get("tool", "?"),
                "args":        start.get("args", {}),
                "status":      ev.get("status", "?"),
                "duration_ms": ev.get("duration_ms", 0),
                "result":      ev.get("result"),
                "error":       ev.get("error"),
            })
        elif etype == "llm_thought":
            chat.append({
                "type":      "thought",
                "timestamp": ev.get("timestamp", ""),
                "step":      ev.get("step") or "thought",
                "thought":   ev.get("thought", ""),
            })

    chat.sort(key=lambda x: x["timestamp"])
    return chat


# Keys to exclude when building a one-line result summary
_RESULT_SKIP = frozenset({
    "next_step", "smiles_sample", "indices", "hyperparameters_searched",
    "confusion_matrix", "class_labels", "columns", "session_dir", "log_file",
    "per_class_metrics", "label_stats", "warnings",
})


def _brief_result(result: Any) -> str:
    """Return a short one-line summary of a tool result value."""
    if result is None:
        return ""
    if not isinstance(result, (dict, list)):
        return str(result).split("\n")[0][:200]
    if isinstance(result, list):
        for item in result:           # first dict element, if any
            if isinstance(item, dict):
                return _brief_result(item)
        return ""
    # message / text field takes priority
    msg = result.get("message") or result.get("text")
    if msg and isinstance(msg, str):
        return msg.split("\n")[0][:200]
    # check_training completed — surface key metrics
    if result.get("status") == "completed" and isinstance(result.get("result"), dict):
        r  = result["result"]
        te = r.get("test_evaluation", {}).get("overall_metrics", {})
        parts = []
        if "cv_best_score" in r:
            parts.append(f"CV BA={r['cv_best_score']:.3f}")
        if "BA"       in te: parts.append(f"Test BA={te['BA']:.3f}")
        if "MCC"      in te: parts.append(f"MCC={te['MCC']:.3f}")
        if "Accuracy" in te: parts.append(f"Acc={te['Accuracy']:.3f}")
        if parts:
            return " · ".join(parts)
    # fallback: simple scalar fields
    parts = []
    for k, v in result.items():
        if k in _RESULT_SKIP or isinstance(v, (dict, list)):
            continue
        s = str(v)
        if len(s) <= 80:
            parts.append(f"{k}={s}")
        if len(parts) >= 5:
            break
    return " · ".join(parts)


def _render_thought(pdf: Any, ev: dict, W: float) -> None:
    """Render a thought event as a chat-UI thinking block in the PDF."""
    from fpdf.enums import XPos, YPos

    step = (ev.get("step") or "thought").upper()
    ts   = (ev.get("timestamp") or "")[:19].replace("T", " ")
    text = ev.get("thought", "")

    # Header
    pdf.set_fill_color(219, 234, 254)
    pdf.set_draw_color(147, 197, 253)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(30, 64, 175)
    pdf.cell(W, 6, f"  [ {step} ]  {ts}", border="TLR", fill=True,
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    # Body
    pdf.set_fill_color(239, 246, 255)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(30, 30, 80)
    pdf.multi_cell(W, 5, f"  {text}", border="BLR", fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_draw_color(0, 0, 0)
    pdf.ln(3)


def _render_tool_call(pdf: Any, ev: dict, W: float) -> None:
    """Render a tool call event as a chat-UI tool card in the PDF."""
    from fpdf.enums import XPos, YPos

    tool        = ev.get("tool", "?")
    args        = ev.get("args", {})
    status      = ev.get("status", "?")
    result      = ev.get("result")
    error       = ev.get("error")
    duration_ms = ev.get("duration_ms", 0)
    ts          = (ev.get("timestamp") or "")[:19].replace("T", " ")
    ok          = status == "success"

    # ── Tool call header ───────────────────────────────────────────────────
    pdf.set_fill_color(30, 64, 175)
    pdf.set_draw_color(30, 64, 175)
    pdf.set_font("Helvetica", "B", 8.5)
    pdf.set_text_color(255, 255, 255)
    status_text = "OK" if ok else "ERR"
    pdf.cell(W, 7,
             f"  >> {tool}   [{status_text}]  {duration_ms:.0f} ms   {ts}",
             fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)

    # ── Arguments ─────────────────────────────────────────────────────────
    if args:
        pdf.set_fill_color(235, 239, 252)
        pdf.set_draw_color(180, 195, 230)
        pdf.set_font("Helvetica", "B", 7.5)
        pdf.set_text_color(60, 60, 120)
        pdf.cell(W, 5, "  Arguments", border="LR", fill=True,
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Courier", "", 7.5)
        pdf.set_text_color(40, 40, 80)
        for k, v in args.items():
            v_str = _json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else str(v)
            pdf.set_fill_color(245, 247, 255)
            pdf.cell(W, 5, f"  {k}: {v_str}"[:95], border="LR", fill=True,
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(0, 0, 0)

    # ── Result ────────────────────────────────────────────────────────────
    if ok and result is not None:
        brief: dict
        if isinstance(result, dict):
            brief = {
                k: v for k, v in result.items()
                if isinstance(v, (str, int, float, bool)) and k not in ("next_step",)
            }
        else:
            brief = {"result": str(result)}
        if brief:
            pdf.set_fill_color(220, 252, 231)
            pdf.set_draw_color(134, 239, 172)
            pdf.set_font("Helvetica", "B", 7.5)
            pdf.set_text_color(20, 100, 50)
            pdf.cell(W, 5, "  Result", border="LR", fill=True,
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("Courier", "", 7.5)
            pdf.set_text_color(20, 60, 30)
            for k, v in brief.items():
                pdf.set_fill_color(240, 253, 244)
                pdf.cell(W, 5, f"  {k}: {v}"[:95], border="LR", fill=True,
                         new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ── Error ─────────────────────────────────────────────────────────────
    elif not ok and error:
        pdf.set_fill_color(254, 226, 226)
        pdf.set_draw_color(252, 165, 165)
        pdf.set_font("Helvetica", "B", 7.5)
        pdf.set_text_color(153, 27, 27)
        pdf.cell(W, 5, "  Error", border="LR", fill=True,
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(120, 20, 20)
        pdf.set_fill_color(255, 241, 241)
        pdf.multi_cell(W, 5, f"  {str(error)[:200]}", border="LR", fill=True)

    # ── Bottom border ─────────────────────────────────────────────────────
    pdf.set_draw_color(180, 195, 230)
    pdf.cell(W, 1, "", border="B", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(0, 0, 0)
    pdf.ln(3)


# ===========================================================================
# generate_report
# ===========================================================================

def generate_report(
    title: Optional[str] = None,
) -> dict[str, Any]:
    """Write a chat-style Markdown report of the current session to disk.

    Renders the session in chronological order: agent thoughts (logged via
    ``log_thought``) are shown as their full narrative text; tool calls are
    shown as compact one-line summaries with key result fields. The output
    intentionally mirrors the chat-UI format so the report reads like the
    original conversation.

    Writes ``report_<timestamp>.md`` inside ``<session_dir>/reports/``.

    Args:
        title: Optional headline. Defaults to "Session Report: <session_id>".

    Returns:
        {"report_path": <absolute path>, "session_id": <id>,
         "n_thoughts": <int>, "n_tool_calls": <int>}
    """
    log_file    = session_logger.log_file
    session_id  = session_logger.session_id
    session_dir = session_logger.session_dir

    chat_events = _parse_chat_events(log_file)
    now_str     = _dt.now(_tz.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    headline    = title or f"Session Report: {session_id}"

    lines: list[str] = [
        f"# {headline}",
        "",
        f"*Generated: {now_str} · Session: `{session_id}`*",
        "",
        "---",
        "",
    ]

    n_thoughts = 0
    n_calls    = 0

    for ev in chat_events:
        if ev["type"] == "thought":
            step = (ev.get("step") or "thought").capitalize()
            ts   = ev["timestamp"][:19].replace("T", " ")
            text = ev.get("thought", "")
            lines += [
                f"### {step} <sub>{ts}</sub>",
                "",
                text,
                "",
                "---",
                "",
            ]
            n_thoughts += 1

        elif ev["type"] == "tool_call":
            tool = ev["tool"]
            if tool == "log_thought":
                continue                      # already captured as a thought above
            ok      = ev["status"] == "success"
            ms      = ev["duration_ms"]
            icon    = "✅" if ok else "❌"
            summary = _brief_result(ev.get("result") if ok else ev.get("error"))
            line    = f"> 🔧 `{tool}` {icon} · {ms:.0f} ms"
            if summary:
                line += f"  \n> _{summary}_"
            lines += [line, ""]
            n_calls += 1

    markdown = "\n".join(lines)

    reports_dir = session_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts_file     = _dt.now(_tz.utc).strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"report_{ts_file}.md"
    report_path.write_text(markdown, encoding="utf-8")

    return {
        "report_path":  str(report_path),
        "session_id":   session_id,
        "n_thoughts":   n_thoughts,
        "n_tool_calls": n_calls,
    }


# ===========================================================================
# generate_pdf_report
# ===========================================================================

def generate_pdf_report(
    title: Optional[str] = None,
) -> dict[str, Any]:
    """Generate a clean PDF report with agent narrative and plot images.

    Renders the agent's reasoning (thoughts logged via ``log_thought``) as
    clean readable text — no tool call cards or technical details. Plots from
    the session's ``plots/`` directory are appended one per page. The result
    reads like a human-written analysis report.

    Writes ``report_<timestamp>.pdf`` to ``<session_dir>/reports/``.

    Args:
        title: Optional report headline. Defaults to "Session Report: <session_id>".

    Returns:
        {"pdf_report_path": <absolute path>, "session_id": <id>, "n_plots_embedded": <int>}
    """
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    session_id  = session_logger.session_id
    session_dir = session_logger.session_dir
    log_file    = session_logger.log_file
    headline    = title or f"Session Report: {session_id}"

    chat_events = _parse_chat_events(log_file)
    thoughts    = [ev for ev in chat_events if ev["type"] == "thought"]
    plots_dir   = session_dir / "plots"
    pngs        = sorted(plots_dir.glob("*.png")) if plots_dir.exists() else []

    W = 180  # usable page width in mm (A4: 210 − 2×15 margins)

    class _PDF(FPDF):
        def header(self):
            self.set_fill_color(30, 64, 175)
            self.rect(0, 0, 210, 16, "F")
            self.set_font("Helvetica", "B", 9)
            self.set_text_color(255, 255, 255)
            self.set_xy(15, 4)
            self.cell(155, 8, headline[:90], new_x=XPos.RIGHT, new_y=YPos.TOP)
            self.set_font("Helvetica", "", 8)
            self.cell(25, 8, f"Page {self.page_no()}", align="R")
            self.set_text_color(0, 0, 0)

        def footer(self):
            self.set_y(-14)
            self.set_font("Helvetica", "I", 7)
            self.set_text_color(140, 140, 140)
            self.cell(0, 5, "Generated by chemagent - AI Agent for Compound Selectivity Prediction", align="C")
            self.set_text_color(0, 0, 0)

    pdf = _PDF()
    pdf.set_margins(left=15, top=25, right=15)
    pdf.set_auto_page_break(auto=True, margin=20)

    # ── Narrative section — one paragraph per thought ──────────────────────
    if thoughts:
        pdf.add_page()
        for ev in thoughts:
            text = (ev.get("thought") or "").strip()
            if not text:
                continue
            # Render each thought as plain body text, separated by a thin rule
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(20, 20, 20)
            pdf.multi_cell(W, 6, text)
            # Subtle separator line
            pdf.set_draw_color(200, 200, 200)
            pdf.set_line_width(0.2)
            pdf.line(pdf.l_margin, pdf.get_y() + 2, pdf.l_margin + W, pdf.get_y() + 2)
            pdf.set_line_width(0.2)
            pdf.set_draw_color(0, 0, 0)
            pdf.ln(6)

    # ── Plots section ──────────────────────────────────────────────────────
    n_plots = 0
    for png in pngs:
        pdf.add_page()
        pdf.image(str(png), x=15, w=W)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(W, 5, png.stem, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        pdf.set_text_color(0, 0, 0)
        n_plots += 1

    # Fallback: always produce at least one page
    if not thoughts and n_plots == 0:
        pdf.add_page()
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(W, 10, "No content found for this session.",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        pdf.set_text_color(0, 0, 0)

    reports_dir = session_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts_file  = _dt.now(_tz.utc).strftime("%Y%m%d_%H%M%S")
    pdf_path = reports_dir / f"report_{ts_file}.pdf"
    pdf.output(str(pdf_path))

    return {
        "pdf_report_path":  str(pdf_path),
        "session_id":       session_id,
        "n_plots_embedded": n_plots,
    }


# ===========================================================================
# start_new_session
# ===========================================================================

def start_new_session() -> dict[str, str]:
    """Start a fresh logging session, ending the current one immediately.

    Use this at the beginning of a new chat or experiment to ensure
    artifacts and logs are not mixed with a previous session.
    Without calling this, sessions are automatically continued as long as
    the last activity was within the session timeout window (default 60 min).

    Returns:
        {"new_session_id": <id>, "session_dir": <path>}
    """
    # Must use the module-level singleton directly — _get_session_logger() may
    # return a different object when chemagent_mcp.py runs as __main__, so
    # calling force_new_session() on it would not affect the session_logger
    # that _register closes over.
    new_id = session_logger.force_new_session()
    return {
        "new_session_id": new_id,
        "session_dir":    str(session_logger.session_dir),
    }
