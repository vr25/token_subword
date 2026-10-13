"""
Figure 3: real-word validation. Two panels: surprisal vs initial-biphone rarity,
surprisal vs final-biphone rarity (pooled across all 10 models' isolated-word
surprisal, z-scored per model before pooling so models with different absolute
surprisal scales don't distort the pooled scatter). Flat regression line
illustrates the null (no significant slope after controlling for length -- shown
via the annotated partial-regression p-value from the real mixed model).
"""
import csv
from collections import defaultdict
import statistics
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#6b6a66"
GRID = "#d9d8d3"

PATH = "/Users/vipularawte/token_exp/results/real_word_surprisal.csv"
OUT_PATH = "/Users/vipularawte/token_exp/paper/figures/fig3_realword.pdf"


def zscore_by_model(rows, col):
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(float(r[col]))
    stats = {m: (statistics.mean(v), statistics.stdev(v)) for m, v in by_model.items()}
    out = []
    for r in rows:
        mean, sd = stats[r["model"]]
        out.append((float(r[col]) - mean) / sd)
    return out


def main():
    with open(PATH) as f:
        rows = list(csv.DictReader(f))

    z_surprisal = zscore_by_model(rows, "surprisal_bits_isolated")
    initial_lf = [float(r["initial_biphone_log_freq"]) for r in rows]
    final_lf = [float(r["final_biphone_log_freq"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(3.4, 2.2), dpi=300, sharey=True)

    for ax, x, label, color in [
        (axes[0], initial_lf, "Initial-biphone\nlog-frequency", BLUE),
        (axes[1], final_lf, "Final-biphone\nlog-frequency", ORANGE),
    ]:
        x = np.array(x)
        y = np.array(z_surprisal)
        ax.scatter(x, y, s=6, color=color, alpha=0.35, edgecolors="none")
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, slope * xs + intercept, color=INK, linewidth=1.5)
        ax.set_xlabel(label, fontsize=8, color=INK)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(GRID)
        ax.tick_params(colors=INK, labelsize=7.5)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Surprisal (z-scored)", fontsize=8, color=INK)
    fig.suptitle("Real words ($n$=200): no reliable slope after length control\n"
                 "(pooled model: initial $p$=0.62, final $p$=0.12)",
                 fontsize=7.5, color=MUTED, y=1.01)

    fig.tight_layout()
    fig.savefig(OUT_PATH, bbox_inches="tight")
    fig.savefig(OUT_PATH.replace(".pdf", ".png"), bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
