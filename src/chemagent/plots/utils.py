"""
chemagent.plots.utils — shared helpers for all plot modules.

Sets a consistent seaborn-based visual theme and exposes a colour palette
and a ``save_figure()`` helper used across every plot script::

    from chemagent.plots.utils import set_theme, PALETTE, SNS_PALETTE, save_figure
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns


# Colour palette
#: Named hex colours for explicit element colouring (lines, highlights, etc.)
PALETTE: dict[str, str] = {
    "primary":   "#2E86AB",   # blue
    "secondary": "#E84855",   # red
    "neutral":   "#3D3D3D",   # dark grey
    "light":     "#A8DADC",   # light teal
    "accent":    "#F4A261",   # orange
}

#: Ordered list passed directly to seaborn ``palette=`` arguments.
#: Follows the convention: class / group 0, 1, 2 …
SNS_PALETTE: list[str] = [
    "#2E86AB",  # blue
    "#E84855",  # red
    "#F4A261",  # orange
    "#3BB273",  # green
    "#9B5DE5",  # purple
    "#F9C74F",  # yellow
]

#: Default sequential colourmap used for heatmaps and confusion matrices.
CMAP_SEQ = "Blues"


# Global seaborn theme
def set_theme(
    font_scale: float = 1.1,
    style: Literal["white", "dark", "whitegrid", "darkgrid", "ticks"] = "whitegrid",
) -> None:
    """Apply a clean, publication-ready seaborn theme.

    Wraps :func:`seaborn.set_theme` with sensible defaults and additional
    rcParam overrides.

    Args:
    font_scale:
        Seaborn font scale factor (default 1.1).
    style:
        Seaborn style name: ``"whitegrid"``, ``"ticks"``, ``"white"``, etc.
    """
    sns.set_theme(
        style=style,
        font_scale=font_scale,
        rc={
            "figure.dpi":        150,
            "figure.facecolor":  "white",
            "axes.spines.top":   False,
            "axes.spines.right": False,
            "lines.linewidth":   1.8,
            "legend.framealpha": 0.85,
        },
    )



def save_figure(
    fig: Figure,
    save_path: Optional[str | Path],
    dpi: int = 300,
    tight: bool = True,
) -> None:
    """Save *fig* to *save_path* if a path is supplied.

    Args:
    fig:
        The :class:`matplotlib.figure.Figure` to save.
    save_path:
        File path (str or :class:`pathlib.Path`).  ``None`` is a no-op.
    dpi:
        Resolution in dots per inch.
    tight:
        Whether to call ``tight_layout()`` before saving.
    """
    if save_path is None:
        return
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)  # prevent figure accumulation in long-running servers
