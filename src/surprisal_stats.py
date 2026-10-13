"""
Step 5 (surprisal half): paired stats on results/surprisal.csv, per model, both isolated
and in-sentence-context surprisal. Same honesty rule as segmentation stats: report
whatever direction/significance comes out.
"""
import csv
from collections import defaultdict
import statistics
from scipy import stats as sstats


def cohens_d_paired(diffs):
    if len(diffs) < 2 or statistics.stdev(diffs) == 0:
        return float("nan")
    return statistics.mean(diffs) / statistics.stdev(diffs)

IN_PATH = "/Users/vipularawte/token_exp/results/surprisal.csv"
OUT_PATH = "/Users/vipularawte/token_exp/results/surprisal_summary.csv"


def run_metric(rows, metric_col):
    by_model_pair = defaultdict(dict)
    for r in rows:
        by_model_pair[r["model"]][r["pair_id"]] = by_model_pair[r["model"]].get(r["pair_id"], {})
        by_model_pair[r["model"]][r["pair_id"]][r["condition"]] = float(r[metric_col])

    out = []
    for model, pairs in by_model_pair.items():
        init_vals, final_vals = [], []
        for pid, d in pairs.items():
            if "word_initial" in d and "word_final" in d:
                init_vals.append(d["word_initial"])
                final_vals.append(d["word_final"])
        if len(init_vals) < 2:
            continue
        mean_init = statistics.mean(init_vals)
        mean_final = statistics.mean(final_vals)
        mean_diff = mean_init - mean_final
        n_lower = sum(1 for i, f in zip(init_vals, final_vals) if i < f)
        n_equal = sum(1 for i, f in zip(init_vals, final_vals) if abs(i - f) < 1e-9)
        n_higher = sum(1 for i, f in zip(init_vals, final_vals) if i > f)
        try:
            w_stat, w_p = sstats.wilcoxon(init_vals, final_vals)
        except ValueError:
            w_p = float("nan")
        t_stat, t_p = sstats.ttest_rel(init_vals, final_vals)
        diffs = [i - f for i, f in zip(init_vals, final_vals)]
        d = cohens_d_paired(diffs)
        out.append({
            "model": model, "metric": metric_col, "n_pairs": len(init_vals),
            "mean_word_initial": mean_init, "mean_word_final": mean_final, "mean_diff": mean_diff,
            "n_initial_lower": n_lower, "n_equal": n_equal, "n_initial_higher": n_higher,
            "wilcoxon_p": w_p, "paired_t_p": t_p, "cohens_d": d,
        })
    return out


def main():
    with open(IN_PATH) as f:
        rows = list(csv.DictReader(f))

    all_out = []
    for metric in ["surprisal_bits_isolated", "surprisal_bits_in_context"]:
        results = run_metric(rows, metric)
        all_out.extend(results)
        print(f"\n=== {metric} ===")
        print(f"{'model':28s} {'mean_init':>10s} {'mean_final':>11s} {'mean_diff':>10s} "
              f"{'n_lower':>8s} {'n_eq':>5s} {'n_higher':>9s} {'wilcoxon_p':>11s} {'t_p':>10s} {'cohens_d':>9s}")
        for r in results:
            print(f"{r['model']:28s} {r['mean_word_initial']:10.2f} {r['mean_word_final']:11.2f} "
                  f"{r['mean_diff']:10.2f} {r['n_initial_lower']:8d} {r['n_equal']:5d} "
                  f"{r['n_initial_higher']:9d} {r['wilcoxon_p']:11.4g} {r['paired_t_p']:10.4g} {r['cohens_d']:9.3f}")

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_out[0].keys()))
        writer.writeheader()
        writer.writerows(all_out)
    print(f"\nWrote {len(all_out)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
