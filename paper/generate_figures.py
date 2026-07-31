import matplotlib


matplotlib.use("Agg")
import os  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402


OUT = os.path.join(os.path.dirname(__file__), "figures")

# ── Palette ──────────────────────────────────────────────────────────────────
C_MISTRAL = "#2a78d6"
C_GEMMA = "#eb6834"
C_QWEN = "#1baf7a"
C_TIMEOUT = "#e34948"
C_EMPTY = "#c3c2b7"
C_UNRESOLVED = "#b0afa8"
C_TOOL_TERM = "#4a3aa7"
C_TOOL_EDIT = "#eda100"
C_TOOL_OTHER = "#b0afa8"
C_RESOLVED_BOX = "#1baf7a"
C_UNRESOLVED_BOX = "#b0afa8"

MODELS = ["Mistral\nSmall 24B", "Gemma 3\n27B", "Qwen 3.6\n27B"]
MODELS_SHORT = ["Mistral 24B", "Gemma 27B", "Qwen 3.6 27B"]
MODEL_COLORS = [C_MISTRAL, C_GEMMA, C_QWEN]

# ── Shared style ─────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "medium",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#898781",
        "xtick.color": "#898781",
        "ytick.color": "#898781",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "grid.linewidth": 0.4,
        "grid.color": "#e1e0d9",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    }
)

COL_WIDTH = 3.4  # single-column inches (IEEE/ACM)
TEXT_WIDTH = 7.0  # full text width


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Instance Outcomes
# ═══════════════════════════════════════════════════════════════════════════════
def fig_outcomes():
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 1.8))

    resolved = [10, 8, 51]
    unresolved = [54, 84, 36]
    empty = [1, 15, 0]
    timedout = [35, 3, 13]

    y = np.arange(3)
    bar_h = 0.55

    lefts = np.zeros(3)
    categories = [
        (resolved, MODEL_COLORS, "Resolved"),
        (unresolved, [C_UNRESOLVED] * 3, "Unresolved"),
        (empty, [C_EMPTY] * 3, "Empty patch"),
        (timedout, [C_TIMEOUT] * 3, "Timed out"),
    ]

    legend_handles = []
    for vals, colors, label in categories:
        for i in range(3):
            if vals[i] == 0:
                continue
            ax.barh(
                y[i],
                vals[i],
                height=bar_h,
                left=lefts[i],
                color=colors[i],
                edgecolor="white",
                linewidth=0.8,
            )
            if vals[i] >= 8:
                ax.text(
                    lefts[i] + vals[i] / 2,
                    y[i],
                    str(vals[i]),
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                    color="white" if label != "Empty patch" else "#52514e",
                )
        legend_handles.append(
            Rectangle(
                (0, 0),
                1,
                1,
                fc=colors[0] if label != "Resolved" else "#898781",
                ec="none",
                label=label,
            )
        )
        lefts = lefts + np.array(vals, dtype=float)

    # For the resolved legend, use a gradient-ish approach — just use gray
    legend_handles[0] = Rectangle(
        (0, 0), 1, 1, fc="#52514e", ec="none", label="Resolved"
    )

    ax.set_yticks(y)
    ax.set_yticklabels(MODELS_SHORT, fontsize=8)
    ax.set_xlabel("Instances (of 100 attempted)")
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=4,
        frameon=False,
        fontsize=7,
    )
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.grid(axis="x", alpha=0.5)

    fig.savefig(os.path.join(OUT, "fig_outcomes.pdf"))
    plt.close(fig)
    print("  fig_outcomes.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Energy Comparison
# ═══════════════════════════════════════════════════════════════════════════════
def fig_energy():
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(TEXT_WIDTH, 2.0), gridspec_kw={"width_ratios": [1.2, 1]}
    )

    x = np.arange(3)
    w = 0.5

    # Total energy
    total = [4211, 1654, 12150]
    bars1 = ax1.bar(x, total, w, color=MODEL_COLORS, edgecolor="white", linewidth=0.8)
    for b, v in zip(bars1, total):
        ax1.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 200,
            f"{v:,}",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
        )
    ax1.set_ylabel("Wh")
    ax1.set_title("Total GPU Energy")
    ax1.set_xticks(x)
    ax1.set_xticklabels(MODELS_SHORT, fontsize=7)
    ax1.set_ylim(0, 14000)
    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k")
    )
    ax1.grid(axis="y", alpha=0.5)

    # Energy per resolved
    per_res = [421, 207, 238]
    bars2 = ax2.bar(x, per_res, w, color=MODEL_COLORS, edgecolor="white", linewidth=0.8)
    for b, v in zip(bars2, per_res):
        ax2.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 8,
            str(v),
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
        )
    ax2.set_ylabel("Wh")
    ax2.set_title("Energy per Resolved Instance")
    ax2.set_xticks(x)
    ax2.set_xticklabels(MODELS_SHORT, fontsize=7)
    ax2.set_ylim(0, 500)
    ax2.grid(axis="y", alpha=0.5)

    fig.tight_layout(w_pad=3)
    fig.savefig(os.path.join(OUT, "fig_energy.pdf"))
    plt.close(fig)
    print("  fig_energy.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Venn Diagram
# ═══════════════════════════════════════════════════════════════════════════════
def fig_venn():
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.6))

    # Three equal circles, standard Venn placement
    r = 0.38
    cx_m, cy_m = -0.22, 0.12  # Mistral — upper left
    cx_g, cy_g = 0.22, 0.12  # Gemma — upper right
    cx_q, cy_q = 0.00, -0.18  # Qwen — lower center

    for cx, cy, color, label, ly in [
        (cx_m, cy_m, C_MISTRAL, "Mistral 24B", cy_m + r + 0.08),
        (cx_g, cy_g, C_GEMMA, "Gemma 27B", cy_g + r + 0.08),
        (cx_q, cy_q, C_QWEN, "Qwen 3.6 27B", cy_q - r - 0.06),
    ]:
        circ = Circle(
            (cx, cy), r, fc=color, alpha=0.18, ec=color, linewidth=1.5, linestyle="-"
        )
        ax.add_patch(circ)
        ax.text(
            cx,
            ly,
            label,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="medium",
            color=color,
        )

    # Region labels — (x, y, text, fontsize, color)
    regions = [
        # Only Mistral
        (cx_m - 0.18, cy_m + 0.05, "0", 8, "#898781"),
        # Only Gemma
        (cx_g + 0.18, cy_g + 0.05, "1", 9, "#0b0b0b"),
        # Only Qwen
        (cx_q, cy_q - 0.22, "39", 12, "#0b0b0b"),
        # M ∩ G (not Q) — top between M and G
        (0.0, cy_m + 0.18, "0", 8, "#898781"),
        # M ∩ Q (not G) — left between M and Q
        (cx_m + 0.08, cy_m - 0.22, "5", 10, "#0b0b0b"),
        # G ∩ Q (not M) — right between G and Q
        (cx_g - 0.08, cy_g - 0.22, "2", 9, "#0b0b0b"),
        # All three — center
        (0.0, 0.0, "5", 11, "#0b0b0b"),
    ]
    for x, y, text, fs, color in regions:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=fs,
            fontweight="bold",
            color=color,
        )

    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(-0.72, 0.65)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(
        0.5,
        -0.02,
        "52 unique bugs resolved; 39 by Qwen alone",
        transform=ax.transAxes,
        ha="center",
        fontsize=7,
        color="#898781",
    )

    fig.savefig(os.path.join(OUT, "fig_venn.pdf"))
    plt.close(fig)
    print("  fig_venn.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Tool Usage
# ═══════════════════════════════════════════════════════════════════════════════
def fig_tools():
    fig, ax = plt.subplots(figsize=(COL_WIDTH, 1.6))

    terminal = [29.1, 48.5, 55.4]
    file_edit = [68.2, 41.6, 36.4]
    other = [2.7, 9.9, 8.2]

    y = np.arange(3)
    bar_h = 0.5

    ax.barh(
        y,
        terminal,
        bar_h,
        label="Terminal",
        color=C_TOOL_TERM,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.barh(
        y,
        file_edit,
        bar_h,
        left=terminal,
        label="File editor",
        color=C_TOOL_EDIT,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.barh(
        y,
        other,
        bar_h,
        left=[t + f for t, f in zip(terminal, file_edit)],
        label="Other",
        color=C_TOOL_OTHER,
        edgecolor="white",
        linewidth=0.8,
    )

    for i in range(3):
        ax.text(
            terminal[i] / 2,
            y[i],
            f"{terminal[i]:.0f}%",
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            color="white",
        )
        ax.text(
            terminal[i] + file_edit[i] / 2,
            y[i],
            f"{file_edit[i]:.0f}%",
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            color="#0b0b0b",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(MODELS_SHORT, fontsize=8)
    ax.set_xlabel("Share of tool calls (%)")
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.3),
        ncol=3,
        frameon=False,
        fontsize=7,
    )
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))

    fig.savefig(os.path.join(OUT, "fig_tools.pdf"))
    plt.close(fig)
    print("  fig_tools.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Duration Box Plots
# ═══════════════════════════════════════════════════════════════════════════════
def fig_duration():
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.2))

    resolved_data = [
        {"min": 39, "q1": 52, "med": 59.5, "q3": 115, "max": 149},
        {"min": 27, "q1": 42, "med": 42, "q3": 91, "max": 323},
        {"min": 65, "q1": 139, "med": 219, "q3": 416, "max": 969},
    ]
    unresolved_data = [
        {"min": 44, "q1": 74, "med": 102, "q3": 173, "max": 788},
        {"min": 31, "q1": 66, "med": 111, "q3": 197, "max": 643},
        {"min": 69, "q1": 213, "med": 425, "q3": 636, "max": 1610},
    ]

    positions_r = [0.8, 2.8, 4.8]
    positions_u = [1.3, 3.3, 5.3]

    def draw_box(pos, d, color, alpha):
        bp = ax.bxp(
            [
                {
                    "med": d["med"],
                    "q1": d["q1"],
                    "q3": d["q3"],
                    "whislo": d["min"],
                    "whishi": d["max"],
                    "fliers": [],
                }
            ],
            positions=[pos],
            widths=0.35,
            patch_artist=True,
            manage_ticks=False,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(alpha)
            patch.set_edgecolor(color)
            patch.set_linewidth(1.2)
        for element in ["whiskers", "caps"]:
            for line in bp[element]:
                line.set_color(color)
                line.set_linewidth(1)
        for line in bp["medians"]:
            line.set_color(color)
            line.set_linewidth(2)

    for i in range(3):
        draw_box(positions_r[i], resolved_data[i], C_RESOLVED_BOX, 0.3)
        draw_box(positions_u[i], unresolved_data[i], C_UNRESOLVED_BOX, 0.3)

    ax.set_xticks([1.05, 3.05, 5.05])
    ax.set_xticklabels(MODELS_SHORT, fontsize=8)
    ax.set_ylabel("Duration (seconds)")
    ax.set_xlim(0.2, 5.9)
    ax.grid(axis="y", alpha=0.5)

    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(
                facecolor=C_RESOLVED_BOX,
                alpha=0.4,
                edgecolor=C_RESOLVED_BOX,
                label="Resolved",
            ),
            Patch(
                facecolor=C_UNRESOLVED_BOX,
                alpha=0.4,
                edgecolor=C_UNRESOLVED_BOX,
                label="Unresolved",
            ),
        ],
        loc="upper left",
        frameon=False,
        fontsize=7,
    )

    fig.savefig(os.path.join(OUT, "fig_duration.pdf"))
    plt.close(fig)
    print("  fig_duration.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("Generating figures...")
    fig_outcomes()
    fig_energy()
    fig_venn()
    fig_tools()
    fig_duration()
    print("Done.")
