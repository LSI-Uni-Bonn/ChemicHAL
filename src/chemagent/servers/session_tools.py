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
export_chat_html         — export full chat as self-contained HTML with embedded figures
set_chat_scope     — bind session logging to a chat/window identifier
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



# MCP server instance + session logger singleton + _register decorator

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


# log_thought
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


def log_answer(
    answer: str,
    role: Optional[str] = None,
) -> dict[str, str]:
    """Record an assistant/LLM answer in the session log via the SessionLogger.

    Returns a small confirmation dict so this can be used as an MCP tool.
    """
    session_logger.log_answer(answer, role=role)
    return {"logged": "ok", "session_id": session_logger.session_id}


def _parse_chat_events(log_file: Path) -> list[dict]:
    """Return session events as a time-ordered list of chat items.

    Each item is a dict with ``"type"`` of either ``"thought``,
    ``"answer"`` or ``"tool_call"`, plus type-specific fields.
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
            cid   = ev.get("call_id") or ""
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
        elif etype == "llm_answer":
            chat.append({
                "type":      "answer",
                "timestamp": ev.get("timestamp", ""),
                "role":      ev.get("role") or "assistant",
                "answer":    ev.get("answer", ""),
            })

    chat.sort(key=lambda x: x["timestamp"])
    return chat


def _extract_image_paths(result: Any) -> list[str]:
    """Recursively collect file paths pointing to image files inside a result."""
    _IMG_EXTS = {".png", ".svg", ".jpg", ".jpeg"}
    found: list[str] = []
    if isinstance(result, dict):
        for v in result.values():
            if isinstance(v, str) and Path(v).suffix.lower() in _IMG_EXTS:
                found.append(v)
            elif isinstance(v, (dict, list)):
                found.extend(_extract_image_paths(v))
    elif isinstance(result, list):
        for item in result:
            found.extend(_extract_image_paths(item))
    return found


def _b64_data_uri(path: str) -> str | None:
    """Return a base64 data-URI for the image at *path*, or None if missing."""
    import base64
    p = Path(path)
    if not p.exists():
        return None
    ext = p.suffix.lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "svg": "svg+xml"}.get(ext, "png")
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{data}"


_HTML_STYLE = """\
<style>
  :root { --blue:#1e40af; --blue-lt:#dbeafe; --blue-pale:#eff6ff;
          --green:#15803d; --green-lt:#dcfce7; --red:#991b1b; --red-lt:#fee2e2;
          --gray:#6b7280; --border:#e5e7eb; --bg:#f9fafb; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: var(--bg);
         color: #111827; padding: 24px; max-width: 900px; margin: auto; }
  h1 { font-size: 1.4rem; color: var(--blue); margin-bottom: 4px; }
  .meta { font-size: .78rem; color: var(--gray); margin-bottom: 20px; }
  .event { margin-bottom: 14px; }
  /* Thought */
  .thought { background: var(--blue-pale); border: 1px solid #93c5fd;
             border-radius: 8px; overflow: hidden; }
  .thought-hdr { background: var(--blue-lt); padding: 5px 10px;
                 font-size: .75rem; font-weight: 700; color: var(--blue);
                 display: flex; justify-content: space-between; cursor: pointer;
                 user-select: none; }
  .thought-body { padding: 8px 12px; font-size: .85rem; color: #1e3a8a;
                  white-space: pre-wrap; }
  /* Answer */
  .answer { background: #fff; border: 1px solid var(--border);
            border-radius: 8px; padding: 10px 14px; font-size: .9rem;
            line-height: 1.55; white-space: pre-wrap; }
  .answer-hdr { font-size: .72rem; font-weight: 700; color: var(--gray);
                margin-bottom: 4px; }
  /* Tool call */
  .tool { border-radius: 8px; overflow: hidden;
          border: 1px solid #b4c3e6; }
  .tool-hdr { background: var(--blue); color: #fff; padding: 6px 10px;
              font-size: .8rem; font-weight: 700; display: flex;
              justify-content: space-between; cursor: pointer; user-select: none; }
  .tool-hdr .ok   { color: #86efac; }
  .tool-hdr .err  { color: #fca5a5; }
  .tool-section { padding: 5px 10px; font-size: .78rem; border-top: 1px solid #d1d5db; }
  .tool-section-lbl { font-weight: 700; color: #4b5563; margin-bottom: 2px; }
  .tool-args   { background: #f0f3ff; }
  .tool-result { background: var(--green-lt); color: var(--green); }
  .tool-error  { background: var(--red-lt);   color: var(--red); }
  .kv { font-family: monospace; font-size: .76rem; padding: 1px 0; }
  /* Figures */
  .figures { margin-top: 8px; }
  .fig { margin-top: 8px; text-align: center; }
  .fig img { max-width: 100%; border: 1px solid var(--border);
             border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,.1); }
  .fig-label { font-size: .72rem; color: var(--gray); margin-top: 3px; }
  /* Collapse */
  .collapsible-body { overflow: hidden; transition: max-height .25s ease; }
  .collapsed .collapsible-body { max-height: 0 !important; }
</style>
<script>
  function toggle(el) {
    el.closest('.collapsible').classList.toggle('collapsed');
  }
</script>
"""


def export_chat_html(
    title: Optional[str] = None,
    collapse_tool_calls: bool = True,
    collapse_thoughts: bool = False,
) -> dict[str, Any]:
    """Export the current session as a self-contained HTML chat with embedded figures.

    Renders the session in chronological order — thoughts, tool calls (with
    arguments and result summaries), and assistant answers — as a styled HTML
    page.  Any figures produced during the session are detected from tool-call
    results and embedded directly as base64 images so the file is fully
    self-contained and viewable offline.

    Writes ``chat_<timestamp>.html`` inside ``<session_dir>/reports/``.

    Args:
        title:               Optional page headline. Defaults to "Chat Export: <session_id>".
        collapse_tool_calls: Render tool-call cards collapsed by default (click to expand).
                             Set to False to show all arguments/results expanded.
        collapse_thoughts:   Render thought blocks collapsed by default.

    Returns:
        {"html_path": <absolute path>, "session_id": <id>,
         "n_thoughts": <int>, "n_tool_calls": <int>, "n_figures": <int>}
    """
    log_file    = session_logger.log_file
    session_id  = session_logger.session_id
    session_dir = session_logger.session_dir

    chat_events = _parse_chat_events(log_file)
    now_str     = _dt.now(_tz.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    headline    = title or f"Chat Export: {session_id}"

    n_thoughts = 0
    n_calls    = 0
    n_figs     = 0

    blocks: list[str] = []

    for ev in chat_events:
        etype = ev["type"]
        ts    = ev["timestamp"][:19].replace("T", " ")

        if etype == "thought":
            step = (ev.get("step") or "thought").upper()
            text = ev.get("thought", "").strip()
            if not text:
                continue
            cls  = "collapsible collapsed" if collapse_thoughts else "collapsible"
            blocks.append(
                f'<div class="event thought {cls}">'
                f'<div class="thought-hdr" onclick="toggle(this)">'
                f'<span>&#128161; {step}</span><span>{ts}</span></div>'
                f'<div class="thought-body collapsible-body" style="max-height:2000px">'
                f'{_html_escape(text)}</div></div>'
            )
            n_thoughts += 1

        elif etype == "answer":
            role = (ev.get("role") or "assistant").capitalize()
            text = ev.get("answer", "").strip()
            if not text:
                continue
            blocks.append(
                f'<div class="event answer">'
                f'<div class="answer-hdr">{_html_escape(role)} &bull; {ts}</div>'
                f'{_html_escape(text)}</div>'
            )

        elif etype == "tool_call":
            tool   = ev["tool"]
            if tool in ("log_thought", "log_answer"):
                continue
            ok     = ev["status"] == "success"
            ms     = ev["duration_ms"]
            args   = ev.get("args", {})
            result = ev.get("result")
            error  = ev.get("error")

            status_cls  = "ok" if ok else "err"
            status_text = "OK" if ok else "ERR"
            cls         = "collapsible collapsed" if collapse_tool_calls else "collapsible"

            # Build inner sections
            sections: list[str] = []

            # Arguments section
            if args:
                arg_rows = "".join(
                    f'<div class="kv">'
                    f'<b>{_html_escape(str(k))}:</b> '
                    f'{_html_escape(_json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v))[:200]}'
                    f'</div>'
                    for k, v in args.items()
                )
                sections.append(
                    f'<div class="tool-section tool-args collapsible-body" style="max-height:2000px">'
                    f'<div class="tool-section-lbl">Arguments</div>{arg_rows}</div>'
                )

            # Result section
            if ok and result is not None:
                if isinstance(result, dict):
                    brief = {
                        k: v for k, v in result.items()
                        if isinstance(v, (str, int, float, bool)) and k not in ("next_step",)
                    }
                else:
                    brief = {"result": str(result)}
                if brief:
                    res_rows = "".join(
                        f'<div class="kv">'
                        f'<b>{_html_escape(str(k))}:</b> {_html_escape(str(v))[:200]}'
                        f'</div>'
                        for k, v in brief.items()
                    )
                    sections.append(
                        f'<div class="tool-section tool-result collapsible-body" style="max-height:2000px">'
                        f'<div class="tool-section-lbl">Result</div>{res_rows}</div>'
                    )

            # Error section
            elif not ok and error:
                sections.append(
                    f'<div class="tool-section tool-error collapsible-body" style="max-height:2000px">'
                    f'<div class="tool-section-lbl">Error</div>'
                    f'<div class="kv">{_html_escape(str(error)[:400])}</div></div>'
                )

            # Figures: embed any image paths found in the result
            img_paths = _extract_image_paths(result) if result else []
            fig_html  = ""
            for img_path in img_paths:
                uri = _b64_data_uri(img_path)
                if uri:
                    label = Path(img_path).name
                    fig_html += (
                        f'<div class="fig">'
                        f'<img src="{uri}" alt="{_html_escape(label)}">'
                        f'<div class="fig-label">{_html_escape(label)}</div></div>'
                    )
                    n_figs += 1

            if fig_html:
                sections.append(f'<div class="figures">{fig_html}</div>')

            inner = "".join(sections)
            blocks.append(
                f'<div class="event tool {cls}">'
                f'<div class="tool-hdr" onclick="toggle(this)">'
                f'<span>&#128296; {_html_escape(tool)}</span>'
                f'<span><span class="{status_cls}">[{status_text}]</span>'
                f' &nbsp;{ms:.0f}&thinsp;ms &nbsp; {ts}</span></div>'
                f'{inner}</div>'
            )
            n_calls += 1

    body = "\n".join(blocks) or '<p style="color:#9ca3af;font-style:italic">No events found.</p>'

    html = (
        f'<!DOCTYPE html><html lang="en"><head>'
        f'<meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{_html_escape(headline)}</title>'
        f'{_HTML_STYLE}</head><body>'
        f'<h1>{_html_escape(headline)}</h1>'
        f'<p class="meta">Generated: {now_str} &bull; Session: <code>{session_id}</code> &bull; '
        f'{n_thoughts} thoughts &bull; {n_calls} tool calls &bull; {n_figs} figures</p>'
        f'{body}'
        f'</body></html>'
    )

    reports_dir = session_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts_file   = _dt.now(_tz.utc).strftime("%Y%m%d_%H%M%S")
    html_path = reports_dir / f"chat_{ts_file}.html"
    html_path.write_text(html, encoding="utf-8")

    return {
        "html_path":    str(html_path),
        "session_id":   session_id,
        "n_thoughts":   n_thoughts,
        "n_tool_calls": n_calls,
        "n_figures":    n_figs,
    }


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )


# set_chat_scope
def set_chat_scope(
    chat_scope_id: str,
    start_new_session_on_change: bool = True,
) -> dict[str, Any]:
    """Bind logging to a chat/window identifier.

    Pass a stable chat identifier (e.g. host chat UUID) at the beginning of
    each chat window. If the scope differs from the active one, this can
    optionally start a fresh session directory so artifacts are chat-scoped.

    Args:
        chat_scope_id: Stable non-empty chat/window identifier.
        start_new_session_on_change:
            When True (default), a changed scope forces a fresh session.

    Returns:
        {
            "session_id": <id>,
            "chat_scope_id": <scope>,
            "previous_chat_scope_id": <scope|None>,
            "changed": <bool>,
            "started_new_session": <bool>
        }
    """
    return session_logger.set_chat_scope(
        chat_scope_id=chat_scope_id,
        start_new_session_on_change=start_new_session_on_change,
    )


# start_new_session
def start_new_session(chat_scope_id: Optional[str] = None) -> dict[str, str | None]:
    """Start a fresh logging session, ending the current one immediately.

    Use this at the beginning of a new chat or experiment to ensure
    artifacts and logs are not mixed with a previous session.
    Optional ``chat_scope_id`` can be passed to bind the new session to a
    host-provided chat/window identity.

    Returns:
        {"new_session_id": <id>, "session_dir": <path>, "chat_scope_id": <scope|None>}
    """
    # Must use the module-level singleton directly — _get_session_logger() may
    # return a different object when chemagent_mcp.py runs as __main__, so
    # calling force_new_session() on it would not affect the session_logger
    # that _register closes over.
    new_id = session_logger.force_new_session(chat_scope_id=chat_scope_id)
    return {
        "new_session_id": new_id,
        "session_dir":    str(session_logger.session_dir),
        "chat_scope_id":  session_logger.chat_scope_id,
    }
