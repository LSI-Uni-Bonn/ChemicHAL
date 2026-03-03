"""session_logger.py — thread-safe logger for MCP tool calls.

One session = one server process lifetime = one subdirectory:

    data/logs/
      session_<YYYYMMDD_HHMMSS>_<short_uuid>/
        session_<id>.txt      ← JSON-lines call log
        datasets/             ← CSV copy of every loaded dataset
        splits/               ← .pkl copy of every saved split
        models/               ← .pkl copy of every saved model
        plots/                ← PNG copy of every generated plot
        results/              ← prediction + metrics .pkl files

Log file location (inside the session subdirectory):
    session_<id>.txt

Each line is a self-contained JSON object:

    {
      "session_id":   "20260302_134501_a3f9",
      "call_id":      "c1",
      "timestamp":    "2026-03-02T13:45:01.234Z",
      "tool":         "load_dataset",
      "args":         {"file_path": "data/datasets/chembl_...", "label_col": "class_label"},
      "status":       "success",          # or "error"
      "duration_ms":  42.7,
      "result":       {"dataset_id": "chembl_...", "n_samples": 1277, ...},
      # on error instead of result:
      "error":        "FileNotFoundError: ..."
    }

Large array arguments (list[list[float]] with > MAX_ARRAY_ROWS rows) are
replaced by a shape summary to keep the log files readable:
    {"__array_shape__": [1277, 2048], "__dtype__": "float"}

Usage (in chemagent_mcp.py):
    from chemagent.logging.session_logger import SessionLogger

    logger = SessionLogger(workspace_root / "data" / "logs")
    # session_dir = logger.session_dir  # e.g. data/logs/session_20260302_134501_a3f9/
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

# ── tuneable constants ──────────────────────────────────────────────────────
MAX_ARRAY_ROWS = 20   # lists with more rows → shape summary

# Result dict keys whose values are file paths → copied into the session dir.
_PATH_KEY_SUBDIR: dict[str, str] = {
    "model_path":    "models",
    "results_path":  "results",
    "metrics_path":  "results",
}
# For "saved_to" the subfolder is determined by file extension at runtime.
_EXT_SUBDIR: dict[str, str] = {
    ".pkl": "splits",
    ".png": "plots",
    ".svg": "plots",
    ".pdf": "plots",
}
# ───────────────────────────────────────────────────────────────────────────


def _get_git_username() -> str:
    """Return a filesystem-safe git user.name, falling back to the OS user.

    Priority:
    1. ``git config user.name``  (reads the global/local git config)
    2. ``GIT_AUTHOR_NAME`` env var
    3. ``USERNAME`` / ``USER`` OS env var
    4. ``"unknown"`` as final fallback

    The value is lowercased and any character that is not a letter, digit,
    hyphen, or dot is replaced with ``_`` so it is always safe in a path.
    """
    import os

    candidates: list[str] = []

    # 1. git config
    try:
        out = subprocess.check_output(
            ["git", "config", "user.name"],
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        name = out.decode().strip()
        if name:
            candidates.append(name)
    except Exception:  # noqa: BLE001
        pass

    # 2-3. env vars
    for var in ("GIT_AUTHOR_NAME", "USERNAME", "USER"):
        val = os.environ.get(var, "").strip()
        if val:
            candidates.append(val)

    raw = candidates[0] if candidates else "unknown"
    # Sanitise: lowercase, collapse unsafe chars to underscore
    safe = re.sub(r"[^\w.-]", "_", raw).strip("_").lower()
    return safe or "unknown"


def _summarise_value(value: Any, depth: int = 0) -> Any:
    """Replace any large numeric array with a compact shape summary.

    Recurses into dicts up to depth 2 so nested structures are also cleaned.
    """
    if depth > 2:
        return value

    # list[list[float]] — typical features argument
    if isinstance(value, list) and value and isinstance(value[0], list):
        if len(value) > MAX_ARRAY_ROWS:
            inner_len = len(value[0]) if value[0] else 0
            return {"__array_shape__": [len(value), inner_len], "__dtype__": "float"}
        return [[round(x, 4) if isinstance(x, float) else x for x in row] for row in value]

    # flat list that is very long
    if isinstance(value, list) and len(value) > 200:
        return {"__list_len__": len(value), "__sample__": value[:3]}

    # recurse into dict
    if isinstance(value, dict):
        return {k: _summarise_value(v, depth + 1) for k, v in value.items()}

    return value


def _summarise_args(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {k: _summarise_value(v) for k, v in kwargs.items()}


def _summarise_result(result: Any) -> Any:
    if isinstance(result, dict):
        return {k: _summarise_value(v) for k, v in result.items()}
    return result


class SessionLogger:
    """Write one session subdirectory per server process lifetime.

    The subdirectory is created inside *log_dir* with the name
    ``session_<username>_<YYYYMMDD_HHMMSS>_<6-char-uuid>``.  It contains the call log
    (``session_<id>.txt``) and sub-folders for each artifact category
    (``datasets/``, ``splits/``, ``models/``, ``plots/``, ``results/``).

    Thread-safe: all writes and copies are serialised through a Lock.

    Parameters
    ----------
    log_dir:
        Root directory for all session subdirectories.  Created automatically.
    """

    def __init__(self, log_dir: Path | str) -> None:
        import uuid as _uuid

        log_root = Path(log_dir)
        log_root.mkdir(parents=True, exist_ok=True)

        ts              = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_id        = _uuid.uuid4().hex[:6]
        self.username   = _get_git_username()
        self.session_id = f"{self.username}_{ts}_{short_id}"

        # Session subdirectory — everything for this session lives here
        self._session_dir = log_root / f"session_{self.session_id}"
        self._session_dir.mkdir(parents=True, exist_ok=True)

        # Pre-create artifact subdirectories
        for subdir in ("datasets", "splits", "models", "plots", "results"):
            (self._session_dir / subdir).mkdir(exist_ok=True)

        self._log_file = self._session_dir / f"session_{self.session_id}.txt"
        self._lock     = threading.Lock()
        self._counter  = 0

        self._write({
            "session_id":  self.session_id,
            "type":        "session_open",
            "timestamp":   self._now(),
            "username":    self.username,
            "session_dir": str(self._session_dir),
            "log_file":    str(self._log_file),
        })

    # ── properties ──────────────────────────────────────────────────────────

    @property
    def session_dir(self) -> Path:
        """Root of this session's artifact directory."""
        return self._session_dir

    # ── logging API ─────────────────────────────────────────────────────────

    def start_call(self, tool_name: str, kwargs: dict[str, Any]) -> str:
        """Record the start of a tool call; return a unique *call_id*."""
        with self._lock:
            self._counter += 1
            call_id = f"c{self._counter}"

        self._write({
            "session_id": self.session_id,
            "type":       "call_start",
            "call_id":    call_id,
            "timestamp":  self._now(),
            "tool":       tool_name,
            "args":       _summarise_args(kwargs),
        })
        return call_id

    def end_call(
        self,
        call_id:     str,
        result:      Any   = None,
        error:       Exception | None = None,
        duration_ms: float = 0.0,
    ) -> None:
        """Record the result (or error) of a completed tool call."""
        entry: dict[str, Any] = {
            "session_id":  self.session_id,
            "type":        "call_end",
            "call_id":     call_id,
            "timestamp":   self._now(),
            "duration_ms": round(duration_ms, 2),
        }
        if error is not None:
            entry["status"] = "error"
            entry["error"]  = f"{type(error).__name__}: {error}"
        else:
            entry["status"] = "success"
            entry["result"] = _summarise_result(result)

        self._write(entry)

    def log_event(self, event_type: str, **fields: Any) -> None:
        """Write a free-form event (e.g. server restart, background job state)."""
        self._write({
            "session_id": self.session_id,
            "type":       event_type,
            "timestamp":  self._now(),
            **fields,
        })

    @property
    def log_file(self) -> Path:
        """Absolute path to the current session log file."""
        return self._log_file

    # ── artifact API ────────────────────────────────────────────────────────

    def copy_artifact(self, src_path: str | Path, subdir: str) -> Path | None:
        """Copy *src_path* into ``<session_dir>/<subdir>/``.

        Returns the destination path, or None if the source does not exist.
        Safe to call from any thread.
        """
        src = Path(src_path)
        if not src.exists():
            return None
        dest_dir = self._session_dir / subdir
        dest_dir.mkdir(exist_ok=True)
        dest = dest_dir / src.name
        with self._lock:
            shutil.copy2(src, dest)
        self._write({
            "session_id": self.session_id,
            "type":       "artifact_saved",
            "timestamp":  self._now(),
            "category":   subdir,
            "source":     str(src),
            "dest":       str(dest),
        })
        return dest

    def copy_artifacts_from_result(self, result: Any) -> None:
        """Scan a tool result dict and copy any recognised file paths.

        Handles:
        - ``model_path``                          → models/
        - ``results_path`` / ``metrics_path``     → results/
        - ``saved_to`` with ``.pkl`` extension    → splits/
        - ``saved_to`` with ``.png/.svg/.pdf``    → plots/
        - Nested ``result`` sub-dict              (e.g. from get_training_result)
        """
        if not isinstance(result, dict):
            return
        for key, subdir in _PATH_KEY_SUBDIR.items():
            val = result.get(key)
            if isinstance(val, str) and val:
                self.copy_artifact(val, subdir)
        # "saved_to" — route by extension
        saved_to = result.get("saved_to")
        if isinstance(saved_to, str) and saved_to:
            ext    = Path(saved_to).suffix.lower()
            subdir = _EXT_SUBDIR.get(ext)
            if subdir:
                self.copy_artifact(saved_to, subdir)
        # Recurse into nested result (e.g. get_training_result wraps the dict)
        nested = result.get("result")
        if isinstance(nested, dict):
            self.copy_artifacts_from_result(nested)

    def save_dataframe(self, df: "pd.DataFrame", name: str) -> Path:
        """Save *df* as a CSV inside ``<session_dir>/datasets/``.

        Parameters
        ----------
        df:   The pandas DataFrame to save.
        name: Stem used for the filename (``<name>.csv``).
        """
        dest_dir = self._session_dir / "datasets"
        dest_dir.mkdir(exist_ok=True)
        dest = dest_dir / f"{name}.csv"
        with self._lock:
            df.to_csv(dest, index=False)
        self._write({
            "session_id": self.session_id,
            "type":       "artifact_saved",
            "timestamp":  self._now(),
            "category":   "datasets",
            "dest":       str(dest),
            "n_rows":     len(df),
            "n_cols":     len(df.columns),
        })
        return dest

    # ── internal ────────────────────────────────────────────────────────────

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    def _write(self, entry: dict[str, Any]) -> None:
        line = json.dumps(entry, default=str)
        with self._lock:
            with self._log_file.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
