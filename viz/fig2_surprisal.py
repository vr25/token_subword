"""
Figure 2: mean surprisal (bits) by condition, grouped by model, as two small
multiples (isolated word vs in-sentence-context) sharing one y-axis scale unit
(bits) but plotted as two separate panels since the underlying context differs.
"""
import csv
from collections import defaultdict
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#6b6a66"
GRID = "#d9d8d3"

SUR_PATH = "/Users/vipularawte/token_exp/results/surprisal.csv"
SUMMARY_PATH = "/Users/vipularawte/token_exp/results/surprisal_summary.csv"
OUT_PATH_A = "/Users/vipularawte/token_exp/paper/figures/fig2a_isolated.pdf"
OUT_PATH_B = "/Users/vipularawte/token_exp/paper/figures/fig2b_context.pdf"

DISPLAY_NAMES = {
    "gpt2": "GPT-2",
    "gpt2-medium": "GPT-2-M",
    "gpt2-large": "GPT-2-L",
    "EleutherAI/pythia-160m": "Pythia-160M",
    "EleutherAI/pythia-410m": "Pythia-410M",
    "EleutherAI/pythia-1.4b": "Pythia-1.4B",
    "EleutherAI/pythia-2.8b": "Pythia-2.8B",
    "mistralai/Mistral-7B-v0.3": "Mistral-7B",
    "meta-llama/Meta-Llama-3-8B": "Llama-3-8B",
    "google/gemma-2-2b": "Gemma-2",
}
ORDER = ["gpt2", "gpt2-medium", "gpt2-large", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
         "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b", "mistralai/Mistral-7B-v0.3",
         "meta-llama/Meta-Llama-3-8B", "google/gemma-2-2b"]


def sem(vals):
    return statistics.stdev(vals) / (len(vals) ** 0.5)


def stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def panel(ax, metric_col, summary_rows, data, title):
    x = range(len(ORDER))
    width = 0.34
    means_i = [statistics.mean(data[m]["word_initial"][metric_col]) for m in ORDER]
    means_f = [statistics.mean(data[m]["word_final"][metric_col]) for m in ORDER]
    sem_i = [sem(data[m]["word_initial"][metric_col]) for m in ORDER]
    sem_f = [sem(data[m]["word_final"][metric_col]) for m in ORDER]

    ax.bar([i - width/2 - 0.01 for i in x], means_i, width, yerr=sem_i, color=BLUE,
           capsize=2, error_kw={"elinewidth": 1, "ecolor": INK, "alpha": 0.7},
           label="Word-initial violation")
    ax.bar([i + width/2 + 0.01 for i in x], means_f, width, yerr=sem_f, color=ORANGE,
           capsize=2, hatch="///", error_kw={"elinewidth": 1, "ecolor": INK, "alpha": 0.7},
           label="Word-final violation")

    max_top = 0
    for i, m in enumerate(ORDER):
        key = (m, metric_col)
        p = summary_rows[key]
        top = max(means_i[i] + sem_i[i], means_f[i] + sem_f[i]) + 0.6
        max_top = max(max_top, top)
        s = stars(p)
        weight = "bold" if s != "n.s." else "normal"
        ax.text(i, top, s, ha="center", va="bottom", fontsize=7, color=MUTED, fontweight=weight)

    ax.set_ylim(0, max_top * 1.35)
    ax.set_xticks(list(x))
    ax.set_xticklabels([DISPLAY_NAMES[m] for m in ORDER], rotation=45, ha="right", fontsize=7, color=INK)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=7)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def main():
    with open(SUR_PATH) as f:
        rows = list(csv.DictReader(f))
    with open(SUMMARY_PATH) as f:
        summary = list(csv.DictReader(f))

    data = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        for col in ["surprisal_bits_isolated", "surprisal_bits_in_context"]:
            data[r["model"]][r["condition"]].setdefault(col, []).append(float(r[col]))

    summary_p = {}
    for metric_col in ["surprisal_bits_isolated", "surprisal_bits_in_context"]:
        sub = [r for r in summary if r["metric"] == metric_col]
        _, p_fdr, _, _ = multipletests([float(r["paired_t_p"]) for r in sub], method="fdr_bh")
        for r, p in zip(sub, p_fdr):
            summary_p[(r["model"], metric_col)] = p

    for out_path, metric_col, title in [
        (OUT_PATH_A, "surprisal_bits_isolated", "Isolated pseudoword"),
        (OUT_PATH_B, "surprisal_bits_in_context", "In sentence context"),
    ]:
        fig, ax = plt.subplots(figsize=(3.4, 2.0), dpi=300)
        panel(ax, metric_col, summary_p, data, title)
        ax.set_ylabel("Mean surprisal (bits)", fontsize=9, color=INK)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, frameon=False, fontsize=5.5, loc="upper center", ncols=2,
                  bbox_to_anchor=(0.5, 1.02), columnspacing=0.8, handletextpad=0.4)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        fig.savefig(out_path.replace(".pdf", ".png"), bbox_inches="tight")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
