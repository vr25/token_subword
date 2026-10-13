"""
Graphical abstract: a crossover/interaction plot showing the paper's entire finding
in one glance. Two schematic lines (processing cost, unitless/illustrative -- real
effect sizes are in Figures 1/3-4) cross between word-initial and word-final
position: human reading cost goes down, LLM tokenization cost goes up, using the
paper's flagship example pair (fvken/enfvk) annotated at each endpoint.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#2a78d6"   # human
ORANGE = "#eb6834"  # LLM
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d9d8d3"


def main():
    fig, ax = plt.subplots(figsize=(3.3, 2.5), dpi=300)

    # vertical layout: y-axis = position (top: word-initial, bottom: word-final),
    # x-axis = schematic processing cost (left: low, right: high)
    y = [1, 0]
    human_x = [0.82, 0.18]   # high cost at initial (top), low at final (bottom)
    llm_x = [0.18, 0.82]     # low cost at initial (top), high at final (bottom)

    ax.plot(human_x, y, color=BLUE, linewidth=3, marker="o", markersize=8,
            solid_capstyle="round", zorder=3, label="Human reading\n(established)")
    ax.plot(llm_x, y, color=ORANGE, linewidth=3, marker="o", markersize=8,
            solid_capstyle="round", zorder=3, label="LLM tokenization\n(this paper)")

    # endpoint annotations: each sits directly above/below its own marker, outside
    # the y in [0,1] band the lines occupy, so nothing crosses a line.
    ax.annotate("\"fvken\"\nhard to read", xy=(human_x[0], 1), xytext=(human_x[0], 1.68),
                fontsize=7, color=BLUE, ha="center", va="top", fontweight="bold", linespacing=1.3)
    ax.annotate("\"enfvk\"\neasy to read", xy=(human_x[1], 0), xytext=(human_x[1], -0.76),
                fontsize=7, color=BLUE, ha="center", va="bottom", fontweight="bold", linespacing=1.3)
    ax.annotate("fv|ken\n(2 tok.)", xy=(llm_x[0], 1), xytext=(llm_x[0], 1.68),
                fontsize=7, color=ORANGE, ha="center", va="top", fontweight="bold", linespacing=1.3)
    ax.annotate("en|fv|k\n(3 tok.)", xy=(llm_x[1], 0), xytext=(llm_x[1], -0.76),
                fontsize=7, color=ORANGE, ha="center", va="bottom", fontweight="bold", linespacing=1.3)

    ax.set_ylim(-1.05, 1.95)
    ax.set_xlim(-0.05, 1.05)
    ax.set_yticks(y)
    ax.set_yticklabels(["Word-initial\nposition", "Word-final\nposition"], fontsize=9.5, color=INK)
    ax.set_xticks([0.18, 0.82])
    ax.set_xticklabels(["low", "high"], fontsize=9, color=MUTED)
    ax.set_xlabel("Processing cost (schematic)", fontsize=9, color=INK)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK)

    ax.text(0.5, 0.5, "INVERTED", ha="center", va="center", fontsize=13, color=INK,
            fontweight="bold", transform=ax.transAxes, rotation=0,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=INK, linewidth=1))

    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncols=2, columnspacing=0.8)

    fig.tight_layout()
    out = "/Users/vipularawte/token_exp/paper/figures/fig0_graphical_abstract.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
