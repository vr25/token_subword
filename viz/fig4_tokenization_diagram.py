"""
Figure: colorful visual diagram of how the SAME pair of pseudowords gets split
into tokens differently depending on position, for a couple of representative
tokenizers. Each unique character substring gets a consistent color across rows,
so the reader can visually track which pieces merge/split differently.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

INK = "#0b0b0b"
MUTED = "#52514e"
SURFACE = "#fcfcfb"

# Validated categorical palette (dataviz skill reference palette, light mode)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]

# (tokenizer_label, word_initial_pieces, word_final_pieces)
EXAMPLES = [
    ("GPT-4o/GPT-5 (o200k_base)", ["fv", "ken"], ["en", "fv", "k"]),
    ("Gemma-2 (same pair, no difference here)", ["fv", "ken"], ["enf", "vk"]),
]


def color_for_piece(piece, color_map):
    if piece not in color_map:
        color_map[piece] = PALETTE[len(color_map) % len(PALETTE)]
    return color_map[piece]


def draw_row(ax, y, pieces, color_map, label, char_width=0.34, gap=0.03):
    x = 0.0
    for piece in pieces:
        w = len(piece) * char_width
        color = color_for_piece(piece, color_map)
        rect = mpatches.FancyBboxPatch((x, y), w, 0.34, boxstyle="round,pad=0.012,rounding_size=0.035",
                                        linewidth=0, facecolor=color)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + 0.17, piece, ha="center", va="center", fontsize=7.5,
                color="white", fontweight="bold", family="monospace")
        x += w + gap
    ax.text(-0.08, y + 0.17, label, ha="right", va="center", fontsize=7.5, color=INK, fontweight="bold")
    ax.text(x + 0.08, y + 0.17, f"({len(pieces)} tokens)", ha="left", va="center",
             fontsize=7.5, color=INK, fontweight="bold")
    return x


def main():
    n = len(EXAMPLES)
    row_h = 0.62
    title_gap = 0.55
    block_h = row_h * 2 + title_gap
    fig, ax = plt.subplots(figsize=(3.3, 0.5 * n + 0.6), dpi=300)
    color_map = {}

    for i, (tok_name, wi_pieces, wf_pieces) in enumerate(EXAMPLES):
        y0 = (n - 1 - i) * block_h
        ax.text(0.85, y0 + row_h * 2 + 0.04, tok_name, fontsize=7, color=INK, fontweight="bold", ha="left")
        draw_row(ax, y0 + 0.62, wi_pieces, color_map, "init.")
        draw_row(ax, y0 + 0.0, wf_pieces, color_map, "final")

    ax.set_xlim(-0.85, 3.35)
    ax.set_ylim(-0.15, n * block_h + 0.15)
    ax.axis("off")
    fig.tight_layout()
    out = "/Users/vipularawte/token_exp/paper/figures/fig4_tokenization_diagram.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
