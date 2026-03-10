"""chemagent.session_utils
~~~~~~~~~~~~~~~~~~~~~~~~~
Shared path and session-logger helpers used by multiple sub-packages
(e.g. ``chemagent.explainability``, ``chemagent.servers``).

Public API
----------
* ``WORKSPACE_ROOT``        — absolute Path to the repository root
* ``resolve_path(p)``       — resolve a relative path against WORKSPACE_ROOT
* ``get_session_logger()``  — return the active :class:`SessionLogger` instance,
                              reusing the MCP server's singleton when available
                              and creating a standalone fallback otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chemagent.logging import SessionLogger

# Four levels up from src/chemagent/session_utils.py → repo root
WORKSPACE_ROOT: Path = Path(__file__).resolve().parents[2]

_standalone_logger: "SessionLogger | None" = None


def resolve_path(p: str) -> str:
    """Resolve *p* against :data:`WORKSPACE_ROOT` when it is a relative path.

    Creates any missing parent directories so callers can write immediately.
    """
    path = Path(p)
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def get_session_logger() -> "SessionLogger":
    """Return the active :class:`SessionLogger`.

    When the MCP server is running, reuses its singleton instance so all
    artefacts land in the same session directory.  Falls back to a
    standalone instance (e.g. when called from a notebook or test).
    """
    global _standalone_logger
    for mod_name in (
        "chemagent_mcp",
        "servers.chemagent_mcp",
        "src.chemagent.servers.chemagent_mcp",
    ):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "session_logger"):
            return mod.session_logger  # type: ignore[return-value]

    if _standalone_logger is None:
        from chemagent.logging import SessionLogger
        _standalone_logger = SessionLogger(WORKSPACE_ROOT / "data" / "logs")
    return _standalone_logger
