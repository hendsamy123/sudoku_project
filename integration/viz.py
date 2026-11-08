import numpy as np


def draw_grid(ax, values, cell_colors=None, title="", fontsize=11, bold=None, muted=()):
    """values: list[81] of digits/0 or strings; cell_colors: dict cell->color."""
    ax.set_xlim(0, 9); ax.set_ylim(9, 0); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for i in range(81):
        r, c = divmod(i, 9)
        if cell_colors and i in cell_colors:
            ax.add_patch(__import__("matplotlib").patches.Rectangle((c, r), 1, 1, color=cell_colors[i], alpha=0.55, lw=0))
        v = values[i]
        if v not in (0, "", None):
            if i in muted:
                ax.text(c + .5, r + .5, str(v), ha="center", va="center", fontsize=fontsize - 4, color="0.55")
            else:
                ax.text(c + .5, r + .5, str(v), ha="center", va="center", fontsize=fontsize,
                        fontweight="bold" if (bold is None or i in bold) else "normal")
    for k in range(10):
        lw = 2.2 if k % 3 == 0 else 0.5
        ax.plot([0, 9], [k, k], "k", lw=lw); ax.plot([k, k], [0, 9], "k", lw=lw)
    ax.set_title(title, fontsize=10)
