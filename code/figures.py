from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]


def load_values(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected one JSON object")
    return value


DEFAULT_ANALYSIS = ROOT / "data" / "analysis"
DEFAULT_FIGURE_DIR = ROOT / "data" / "figures"

FIGURE_SYSTEMS = (
    ("sonnet5", "Sonnet 5"),
    ("terra", "GPT-5.6 T."),
    ("gemini", "Gemini 3.6 F."),
    ("glm", "GLM-5.2"),
)
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
FONT_REGULAR = FontProperties(fname=str(FONT_DIR / "DejaVuSans.ttf"))
FONT_BOLD = FontProperties(fname=str(FONT_DIR / "DejaVuSans-Bold.ttf"))
PDF_METADATA = {
    "Title": "Research Agents Feed Doubts, Not Beliefs",
    "Author": "Siddharth Vohra; Min Xu",
    "Subject": "REALM 2026",
    "Keywords": "",
    "Creator": "code/figures.py",
    "Producer": "Matplotlib",
    "CreationDate": datetime(2026, 8, 2, tzinfo=timezone.utc),
    "ModDate": datetime(2026, 8, 2, tzinfo=timezone.utc),
}


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "path.simplify": False,
            "svg.hashsalt": "framedfacts-realm-2026",
        }
    )


def save(fig, output: Path) -> None:
    """Write a figure at exactly one column wide, with nothing spilling out.

    A tight bounding box would crop each figure to a different width, and
    ``\\includegraphics[width=\\columnwidth]`` would then rescale each one by a
    different factor, so the point sizes set in code would not be the point
    sizes a reader sees. Saving the canvas verbatim keeps them equal. The
    trade is that an overwide label is silently clipped instead of expanding the
    canvas, so check for it here and fail the build rather than ship it.
    """

    width, height = fig.get_size_inches()
    if abs(width - COLUMN_IN) > 0.005:
        raise ValueError(
            f"{output.name}: figure is {width:.3f}in wide, expected {COLUMN_IN:.3f}in "
            "(one column). LaTeX would rescale it and shrink every label."
        )
    fig.canvas.draw()
    spill = fig.get_tightbbox(fig.canvas.get_renderer())
    margin = 0.004
    if (
        spill.x0 < -margin
        or spill.y0 < -margin
        or spill.x1 > width + margin
        or spill.y1 > height + margin
    ):
        raise ValueError(
            f"{output.name}: content runs off the canvas "
            f"(x {spill.x0:.3f}..{spill.x1:.3f}, y {spill.y0:.3f}..{spill.y1:.3f} "
            f"against {width:.3f} x {height:.3f}). Shorten a label or widen a margin."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", metadata=PDF_METADATA)
    plt.close(fig)


ACCENT = "#B2182B"
BASE = "#2166AC"
MUTED = "#777777"
NEUTRAL_INK = "#444444"

# Colour alone cannot carry the contrast identity in print or for colour-blind
# readers, so each contrast also owns a marker shape.
CONTRAST_ROWS = (
    ("believer_vs_skeptic", "Believer - skeptic", BASE, "D"),
    ("skeptic_vs_neutral", "Skeptic - neutral", ACCENT, "s"),
    ("believer_vs_neutral", "Believer - neutral", NEUTRAL_INK, "o"),
)

# One ACL column is 219.086 TeX points. Figures are built at exactly that width
# so LaTeX scales them by one, which makes these point sizes the point sizes a
# reader actually sees. Nothing goes below 7pt.
COLUMN_IN = 219.08614 / 72.27
FIG_TEXT = 7.0
FIG_AXIS = 7.4


def finite(value: object, context: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{context}: value is not a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{context}: value is not finite")
    return number


def padded_limits(lows: list[float], highs: list[float]) -> tuple[float, float]:
    """Fit the axis to the data with headroom, never a fixed 0-to-100 frame."""

    low, high = min(lows), max(highs)
    span = high - low
    pad = max(span * 0.12, 0.02)
    return low - pad, high + pad


def make_forest(analysis_dir: Path, output: Path) -> None:
    values = load_values(analysis_dir / "values.json")
    contrasts = values["decomposition"]["contrasts"]
    entries = []
    for key, label, color, marker in CONTRAST_ROWS:
        contrast = contrasts[key]
        entries.append(("pooled", label, contrast["overall"], color, marker))
        for system, system_label in FIGURE_SYSTEMS:
            entries.append(
                ("system", system_label, contrast["by_system"][system], color, marker)
            )
    configure()
    height = 0.110 * len(entries) + 0.37
    fig, ax = plt.subplots(figsize=(COLUMN_IN, height), facecolor="white")
    positions = list(range(len(entries)))[::-1]
    lows, highs = [], []
    for position, (kind, label, summary, color, marker) in zip(positions, entries):
        estimate = finite(summary["mean_delta"], f"forest:{label}")
        low = finite(summary["ci_low"], f"forest:{label}:low")
        high = finite(summary["ci_high"], f"forest:{label}:high")
        lows.append(low)
        highs.append(high)
        ax.plot(
            [low, high],
            [position, position],
            color=color,
            linewidth=1.5 if kind == "pooled" else 0.9,
            solid_capstyle="butt",
            zorder=3,
        )
        ax.plot(
            [estimate],
            [position],
            marker=marker,
            markersize=4.2 if kind == "pooled" else 3.2,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            zorder=4,
        )
    ax.axvline(0.0, color="#333333", linewidth=0.8, zorder=2)
    ax.set_ylim(-0.7, len(entries) - 0.3)
    ax.set_xlim(*padded_limits(lows, highs))
    ax.set_yticks(positions)
    ax.set_yticklabels(
        [label if kind == "pooled" else f"   {label}" for kind, label, *_ in entries]
    )
    ax.set_xlabel(
        "Ordinal shift (scale points)",
        fontproperties=FONT_REGULAR,
        fontsize=FIG_AXIS,
    )
    ax.grid(axis="x", color="#E2E2E2", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.tick_params(axis="x", length=2.5, width=0.6, color="#555555")
    for label_object in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label_object.set_fontproperties(FONT_REGULAR)
        label_object.set_fontsize(FIG_TEXT)
    for tick, (kind, *_rest) in zip(ax.get_yticklabels(), entries):
        if kind == "pooled":
            tick.set_fontproperties(FONT_BOLD)
            tick.set_fontsize(FIG_TEXT)
    fig.subplots_adjust(left=0.35, right=0.985, top=0.985, bottom=0.18)
    save(fig, output)


def make_mechanism(analysis_dir: Path, output: Path) -> None:
    values = load_values(analysis_dir / "values.json")
    direction = values["decomposition"]["query_direction"]
    stance = values.get("followups", {}).get("evidence_stance", {}).get("retrieved", {})
    framings = ("neutral", "believer", "skeptic")
    configure()
    # Stacked, not side by side: two panels at half a column leave less than
    # 0.4in per category, so the framing names overprint each other.
    fig, axes = plt.subplots(
        2, 1, figsize=(COLUMN_IN, 2.85), facecolor="white", sharex=True
    )

    left = axes[0]
    nets, lows, highs = [], [], []
    for framing in framings:
        entry = direction[framing]
        nets.append(finite(entry["net"], f"net:{framing}") * 100.0)
        lows.append(finite(entry["net_ci_low"], f"net_low:{framing}") * 100.0)
        highs.append(finite(entry["net_ci_high"], f"net_high:{framing}") * 100.0)
    positions = list(range(len(framings)))
    colors = [MUTED, BASE, ACCENT]
    for position, net, low, high, color in zip(positions, nets, lows, highs, colors):
        # Caps, not bare line ends: at column width an uncapped whisker reads as
        # part of the gridline.
        left.errorbar(
            [position], [net], yerr=[[net - low], [high - net]],
            fmt="o", markersize=4.6, color=color, ecolor=color,
            elinewidth=1.4, capsize=3.5, capthick=1.4,
            markeredgecolor="white", markeredgewidth=0.5, zorder=4,
        )
        # Above the whisker, not beside the marker: a near-zero value printed
        # beside its marker lands on the zero rule.
        left.text(
            position, high + 0.8, f"{net:+.1f}", fontsize=FIG_TEXT, ha="center",
            fontproperties=FONT_REGULAR, va="bottom", color="#222222",
        )
    left.axhline(0.0, color="#333333", linewidth=0.8, zorder=2)
    left.set_xlim(-0.4, len(framings) - 0.6)
    low_limit, high_limit = padded_limits(lows, highs)
    left.set_ylim(low_limit, high_limit + 2.2)
    left.set_ylabel(
        "Net affirming\nqueries (pts)", fontproperties=FONT_REGULAR, fontsize=FIG_TEXT
    )
    style_panel(left)

    right = axes[1]
    if stance:
        order = ("supports_claim", "neutral_or_unrelated", "opposes_claim")
        palette = {"supports_claim": BASE, "neutral_or_unrelated": "#CCCCCC", "opposes_claim": ACCENT}
        legend = {
            "supports_claim": "Supports",
            "neutral_or_unrelated": "Neutral",
            "opposes_claim": "Opposes",
        }
        bottoms = [0.0] * len(framings)
        for label in order:
            shares = [
                finite(stance[framing][label]["share"], f"stance:{framing}:{label}") * 100.0
                for framing in framings
            ]
            right.bar(
                positions, shares, bottom=bottoms, width=0.42,
                color=palette[label], edgecolor="white", linewidth=0.5,
                label=legend[label], zorder=3,
            )
            for position, share, bottom in zip(positions, shares, bottoms):
                if share >= 8.0:
                    right.text(
                        position, bottom + share / 2, f"{share:.0f}",
                        ha="center", va="center", fontsize=FIG_TEXT,
                        fontproperties=FONT_REGULAR,
                        color="#FFFFFF" if label != "neutral_or_unrelated" else "#333333",
                        zorder=5,
                    )
            bottoms = [bottom + share for bottom, share in zip(bottoms, shares)]
        right.set_ylim(0, 100)
        right.set_xticks(positions, [framing.title() for framing in framings])
        right.set_ylabel(
            "Retrieved\nresults (%)", fontproperties=FONT_REGULAR, fontsize=FIG_TEXT
        )
        legend_font = FONT_REGULAR.copy()
        legend_font.set_size(FIG_TEXT)
        # Figure-level legend: anchored to a panel it would overhang the canvas,
        # since three keys are wider than the plotting area.
        fig.legend(
            *right.get_legend_handles_labels(),
            loc="upper center", frameon=False, ncol=3, prop=legend_font,
            handlelength=0.7, handletextpad=0.4, columnspacing=0.9,
            borderaxespad=0.0, bbox_to_anchor=(0.5, 0.997),
        )
        style_panel(right)
    else:
        right.axis("off")
    fig.subplots_adjust(left=0.19, right=0.985, top=0.9, bottom=0.075, hspace=0.16)
    save(fig, output)


def style_panel(ax) -> None:
    ax.grid(axis="y", color="#E2E2E2", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.tick_params(axis="y", length=2.5, width=0.6, color="#555555")
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontproperties(FONT_REGULAR)
        label.set_fontsize(FIG_TEXT)


def make_channels(analysis_dir: Path, output: Path) -> None:
    values = load_values(analysis_dir / "values.json")
    cross = values.get("followups", {}).get("cross_replay", {})
    replay_pair = values["decomposition"]["replay_ordinal"].get("believer_vs_skeptic") or {}
    # Top block is the crossed design: the two channels below the end-to-end row
    # add up to it, so they carry the marker that says they belong together.
    # Below the rule is context, a replay that removes the search tool entirely.
    rows = (
        ("End to end\n(own evidence)", cross.get("end_to_end_matched"), NEUTRAL_INK, "o"),
        ("Synthesis channel\n(framing)", cross.get("synthesis_channel"), BASE, "D"),
        ("Evidence channel\n(retrieval)", cross.get("evidence_channel"), ACCENT, "D"),
        ("Neutral evidence\n(no-tool bound)", replay_pair.get("fixed"), NEUTRAL_INK, "o"),
    )
    entries = [row for row in rows if (row[1] or {}).get("n")]
    if len(entries) < len(rows):
        missing = [row[0].replace("\n", " ") for row in rows if not (row[1] or {}).get("n")]
        raise ValueError(f"channel figure is missing measured arms: {missing}")
    configure()
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.05), facecolor="white")
    positions = list(range(len(entries)))[::-1]
    lows, highs = [], []
    for position, (label, summary, color, marker) in zip(positions, entries):
        estimate = finite(summary["mean_delta"], f"channel:{label}")
        low = finite(summary["ci_low"], f"channel:{label}:low")
        high = finite(summary["ci_high"], f"channel:{label}:high")
        lows.append(low)
        highs.append(high)
        ax.plot([low, high], [position, position], color=color, linewidth=1.4, zorder=3)
        ax.plot(
            [estimate], [position], marker=marker, markersize=4.2, color=color,
            markeredgecolor="white", markeredgewidth=0.5, zorder=4,
        )
        ax.text(
            estimate, position + 0.2, f"{estimate:+.2f}", ha="center",
            fontsize=FIG_TEXT, fontproperties=FONT_REGULAR, color="#222222",
        )
    # Separator between the decomposition block and the reference row.
    ax.axhline(0.5, color="#DDDDDD", linewidth=0.6, zorder=1)
    ax.axvline(0.0, color="#333333", linewidth=0.8, zorder=2)
    ax.set_ylim(-0.55, len(entries) - 0.2)
    ax.set_xlim(*padded_limits(lows, highs))
    ax.set_yticks(positions, [label for label, *_rest in entries])
    ax.set_xlabel(
        "Believer minus skeptic (scale points)",
        fontproperties=FONT_REGULAR,
        fontsize=FIG_AXIS,
    )
    ax.grid(axis="x", color="#E2E2E2", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.tick_params(axis="x", length=2.5, width=0.6, color="#555555")
    for label_object in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label_object.set_fontproperties(FONT_REGULAR)
        label_object.set_fontsize(FIG_TEXT)
    fig.subplots_adjust(left=0.365, right=0.985, top=0.97, bottom=0.185)
    save(fig, output)


def make_figures(analysis_dir: Path, output_dir: Path) -> list[Path]:
    outputs = [
        output_dir / "fig_forest.pdf",
        output_dir / "fig_mechanism.pdf",
        output_dir / "fig_channels.pdf",
    ]
    make_forest(analysis_dir, outputs[0])
    make_mechanism(analysis_dir, outputs[1])
    make_channels(analysis_dir, outputs[2])
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the three figures from the analysis tables.")
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    args = parser.parse_args()
    for path in make_figures(args.analysis, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
