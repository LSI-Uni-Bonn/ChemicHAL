from __future__ import annotations

import json
from pathlib import Path

from src.chemagent.logging.session_logger import SessionLogger


def _read_marker(log_dir: Path) -> dict:
    marker_path = log_dir / ".current_session.json"
    return json.loads(marker_path.read_text(encoding="utf-8"))


def _close_logger(logger: SessionLogger) -> None:
    try:
        logger._log_fh.close()  # noqa: SLF001 - test-only cleanup helper
    except Exception:  # noqa: BLE001
        pass


def test_reuses_same_session_for_matching_chat_scope(tmp_path: Path):
    log_dir = tmp_path / "logs"

    logger1 = SessionLogger(log_dir=log_dir, chat_scope_id="chat-A")
    sid1 = logger1.session_id
    _close_logger(logger1)

    logger2 = SessionLogger(log_dir=log_dir, chat_scope_id="chat-A")
    sid2 = logger2.session_id
    _close_logger(logger2)

    assert sid2 == sid1


def test_starts_new_session_for_different_chat_scope(tmp_path: Path):
    log_dir = tmp_path / "logs"

    logger1 = SessionLogger(log_dir=log_dir, chat_scope_id="chat-A")
    sid1 = logger1.session_id
    _close_logger(logger1)

    logger2 = SessionLogger(log_dir=log_dir, chat_scope_id="chat-B")
    sid2 = logger2.session_id
    _close_logger(logger2)

    assert sid2 != sid1


def test_legacy_marker_without_chat_scope_still_resumes_when_unset(tmp_path: Path):
    log_dir = tmp_path / "logs"

    logger1 = SessionLogger(log_dir=log_dir)
    sid1 = logger1.session_id
    _close_logger(logger1)

    marker = _read_marker(log_dir)
    marker.pop("chat_scope_id", None)
    marker_path = log_dir / ".current_session.json"
    marker_path.write_text(json.dumps(marker, indent=2), encoding="utf-8")

    logger2 = SessionLogger(log_dir=log_dir)
    sid2 = logger2.session_id
    _close_logger(logger2)

    assert sid2 == sid1


def test_force_new_session_updates_chat_scope_marker(tmp_path: Path):
    log_dir = tmp_path / "logs"

    logger = SessionLogger(log_dir=log_dir, chat_scope_id="chat-A")
    sid1 = logger.session_id

    sid2 = logger.force_new_session(chat_scope_id="chat-B")
    marker = _read_marker(log_dir)
    _close_logger(logger)

    assert sid2 != sid1
    assert marker["session_id"] == sid2
    assert marker["chat_scope_id"] == "chat-B"


def test_set_chat_scope_rotates_session_when_requested(tmp_path: Path):
    log_dir = tmp_path / "logs"

    logger = SessionLogger(log_dir=log_dir, chat_scope_id="chat-A")
    sid1 = logger.session_id

    result = logger.set_chat_scope("chat-B", start_new_session_on_change=True)
    marker = _read_marker(log_dir)
    _close_logger(logger)

    assert result["changed"] is True
    assert result["started_new_session"] is True
    assert result["session_id"] != sid1
    assert marker["chat_scope_id"] == "chat-B"
