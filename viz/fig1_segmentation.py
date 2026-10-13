"""
Figure 1: mean token count by condition (word-initial vs word-final), grouped by
tokenizer. Print-friendly (hatching + color redundancy for grayscale printing),
CVD-safe palette (blue/orange, validated dataviz skill default slots 1-2).
"""
import csv
from collections import defaultdict
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from statsmodels.stats.multitest import multipletests

BLUE = "#2a78d6"    # word-initial
ORANGE = "#eb6834"  # word-final
INK = "#0b0b0b"
MUTED = "#6b6a66"
GRID = "#d9d8d3"

SEG_PATH = "/Users/vipularawte/token_exp/results/segmentation.csv"
SUMMARY_PATH = "/Users/vipularawte/token_exp/results/segmentation_summary.csv"
OUT_PATH = "/Users/vipularawte/token_exp/paper/figures/fig1_segmentation.pdf"

DISPLAY_NAMES = {
    "tiktoken:gpt2": "GPT-2",
    "tiktoken:cl100k_base": "GPT-4/3.5",
    "tiktoken:o200k_base": "GPT-4o/GPT-5",
    "hf:bert-base-uncased": "BERT",
    "hf:t5-base": "T5",
    "hf:mistral-7b-v0.3": "Mistral-7B",
    "hf:qwen2-7b": "Qwen2-7B",
    "hf:llama-3-8b": "Llama-3-8B",
    "hf:gemma-2-2b": "Gemma-2",
}
ORDER = ["tiktoken:o200k_base", "hf:bert-base-uncased", "hf:gemma-2-2b", "hf:t5-base",
         "hf:mistral-7b-v0.3", "tiktoken:gpt2", "tiktoken:cl100k_base", "hf:qwen2-7b", "hf:llama-3-8b"]


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


def main():
    with open(SEG_PATH) as f:
        rows = list(csv.DictReader(f))
    with open(SUMMARY_PATH) as f:
        summary_rows = list(csv.DictReader(f))
    summary = {r["tokenizer"]: r for r in summary_rows}
    tokenizers_ = [r["tokenizer"] for r in summary_rows]
    _, p_fdr, _, _ = multipletests([float(r["paired_t_p"]) for r in summary_rows], method="fdr_bh")
    fdr_lookup = dict(zip(tokenizers_, p_fdr))

    data = defaultdict(lambda: defaultdict(list))
    for r in rows:
        data[r["tokenizer"]][r["condition"]].append(int(r["n_tokens"]))

    fig, ax = plt.subplots(figsize=(3.4, 2.1), dpi=300)
    x = range(len(ORDER))
    width = 0.34

    means_i = [statistics.mean(data[t]["word_initial"]) for t in ORDER]
    means_f = [statistics.mean(data[t]["word_final"]) for t in ORDER]
    sem_i = [sem(data[t]["word_initial"]) for t in ORDER]
    sem_f = [sem(data[t]["word_final"]) for t in ORDER]

    bars_i = ax.bar([i - width/2 - 0.01 for i in x], means_i, width, yerr=sem_i,
                    color=BLUE, label="Word-initial violation", capsize=2,
                    error_kw={"elinewidth": 1, "ecolor": INK, "alpha": 0.7})
    bars_f = ax.bar([i + width/2 + 0.01 for i in x], means_f, width, yerr=sem_f,
                    color=ORANGE, label="Word-final violation", capsize=2, hatch="///",
                    error_kw={"elinewidth": 1, "ecolor": INK, "alpha": 0.7})

    max_top = 0
    for i, t in enumerate(ORDER):
        p = fdr_lookup[t]
        top = max(means_i[i] + sem_i[i], means_f[i] + sem_f[i]) + 0.25
        max_top = max(max_top, top)
        s = stars(p)
        weight = "bold" if s != "n.s." else "normal"
        ax.text(i, top, s, ha="center", va="bottom", fontsize=7, color=MUTED, fontweight=weight)

    ax.set_xticks(list(x))
    ax.set_xticklabels([DISPLAY_NAMES[t] for t in ORDER], rotation=45, ha="right", fontsize=7.5, color=INK)
    ax.set_ylabel("Mean token count", fontsize=9, color=INK)
    ax.set_ylim(0, max_top * 1.35)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.tick_params(axis="y", labelsize=8)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(frameon=False, fontsize=5.5, loc="upper center", ncols=2,
              bbox_to_anchor=(0.5, 1.14), columnspacing=0.8, handletextpad=0.4)

    fig.tight_layout()
    fig.savefig(OUT_PATH, bbox_inches="tight")
    fig.savefig(OUT_PATH.replace(".pdf", ".png"), bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
