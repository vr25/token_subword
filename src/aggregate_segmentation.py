"""
Aggregates results/segmentation.csv across all 50 pairs (not just the 2 spot-checked
examples) per tokenizer: mean token count by condition, paired difference, sign test,
Wilcoxon signed-rank, paired t-test. Reports whatever direction/significance actually
comes out -- no assumed conclusion.
"""
import csv
from collections import defaultdict
import statistics
from scipy import stats as sstats


def cohens_d_paired(diffs):
    """Standardized effect size for paired data: mean(diff) / sd(diff)."""
    if len(diffs) < 2 or statistics.stdev(diffs) == 0:
        return float("nan")
    return statistics.mean(diffs) / statistics.stdev(diffs)

IN_PATH = "/Users/vipularawte/token_exp/results/segmentation.csv"
OUT_PATH = "/Users/vipularawte/token_exp/results/segmentation_summary.csv"


def main():
    with open(IN_PATH) as f:
        rows = list(csv.DictReader(f))

    by_tok_pair = defaultdict(dict)  # (tokenizer) -> pair_id -> {condition: n_tokens}
    for r in rows:
        by_tok_pair[r["tokenizer"]][r["pair_id"]] = by_tok_pair[r["tokenizer"]].get(r["pair_id"], {})
        by_tok_pair[r["tokenizer"]][r["pair_id"]][r["condition"]] = int(r["n_tokens"])

    summary_rows = []
    print(f"{'tokenizer':22s} {'mean_init':>10s} {'mean_final':>11s} {'mean_diff(i-f)':>15s} "
          f"{'n_lower':>8s} {'n_equal':>8s} {'n_higher':>9s} {'wilcoxon_p':>11s} {'paired_t_p':>11s} {'cohens_d':>9s}")
    for tok_name, pairs in by_tok_pair.items():
        init_vals, final_vals = [], []
        for pid, d in pairs.items():
            if "word_initial" in d and "word_final" in d:
                init_vals.append(d["word_initial"])
                final_vals.append(d["word_final"])
        diffs = [i - f for i, f in zip(init_vals, final_vals)]
        n_lower = sum(1 for d in diffs if d < 0)   # initial has FEWER tokens than final
        n_equal = sum(1 for d in diffs if d == 0)
        n_higher = sum(1 for d in diffs if d > 0)  # initial has MORE tokens than final

        mean_init = statistics.mean(init_vals)
        mean_final = statistics.mean(final_vals)
        mean_diff = statistics.mean(diffs)

        nonzero_diffs = [d for d in diffs if d != 0]
        if len(nonzero_diffs) >= 1:
            try:
                w_stat, w_p = sstats.wilcoxon(init_vals, final_vals)
            except ValueError:
                w_p = float("nan")
        else:
            w_p = float("nan")
        t_stat, t_p = sstats.ttest_rel(init_vals, final_vals)
        d = cohens_d_paired(diffs)

        print(f"{tok_name:22s} {mean_init:10.3f} {mean_final:11.3f} {mean_diff:15.3f} "
              f"{n_lower:8d} {n_equal:8d} {n_higher:9d} {w_p:11.4g} {t_p:11.4g} {d:9.3f}")

        summary_rows.append({
            "tokenizer": tok_name, "n_pairs": len(diffs),
            "mean_tokens_word_initial": mean_init, "mean_tokens_word_final": mean_final,
            "mean_diff_initial_minus_final": mean_diff,
            "n_pairs_initial_fewer": n_lower, "n_pairs_equal": n_equal, "n_pairs_initial_more": n_higher,
            "wilcoxon_p": w_p, "paired_t_p": t_p, "cohens_d": d,
        })

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nWrote summary to {OUT_PATH}")


if __name__ == "__main__":
    main()
