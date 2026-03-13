"""
chemagent.plots.display
~~~~~~~~~~~~~~~~~~~~~~~~
Utility for displaying saved plot images inline in MCP-compatible chat UIs.
"""

from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import Image

from chemagent.session_utils import WORKSPACE_ROOT


def show_plot(plot_path: str) -> list:
    """Display a saved plot image directly in the chat UI.

    Reads the PNG file at *plot_path* and returns it as an inline image
    so it renders immediately in MCP-compatible chat interfaces (e.g.
    LM Studio, Claude Desktop).

    Typical workflow:
        paths = plot_classification_results(predictions_path)
        show_plot(paths["confusion_matrix"])   # renders in chat
        show_plot(paths["roc_curve"])           # renders in chat

    Args:
        plot_path: Absolute or workspace-relative path to a PNG file
                   previously created by plot_classification_results() or
                   plot_regression_results().

    Returns:
        A list of [{"plot": path}, Image] so the image renders inline alongside
        a text confirmation in MCP-compatible chat UIs.
    """
    p = Path(plot_path)
    if not p.exists():
        p = WORKSPACE_ROOT / plot_path
    if not p.exists():
        raise FileNotFoundError(f"Plot file not found: {plot_path}")
    return [{"plot": str(p)}, Image(path=str(p))]
