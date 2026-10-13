"""
Three robustness checks on the main pseudoword results (segmentation.csv, surprisal.csv):
1. FDR (Benjamini-Hochberg) correction on the per-tokenizer/per-model p-values already
   reported in segmentation_summary.csv / surprisal_summary.csv.
2. Condition x category interaction test: is the effect concentrated in 1-2 of the 5
   phonetic categories, or broadly consistent? (mirrors the existing condition x
   tokenizer/model interaction test, just with category as the interacting factor)
3. Leave-one-cluster-out: refit the primary pooled mixed-effects model 15 times, each
   excluding one of the 15 clusters, confirm the condition effect survives every drop.
"""
import warnings
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy import stats as sstats

warnings.filterwarnings("ignore")

SEG_PATH = "/Users/vipularawte/token_exp/results/segmentation.csv"
SUR_PATH = "/Users/vipularawte/token_exp/results/surprisal.csv"
SEG_SUMMARY = "/Users/vipularawte/token_exp/results/segmentation_summary.csv"
SUR_SUMMARY = "/Users/vipularawte/token_exp/results/surprisal_summary.csv"
OUT_PATH = "/Users/vipularawte/token_exp/results/robustness_checks_summary.txt"

lines = []


def section(title):
    lines.append(f"\n{'='*10} {title} {'='*10}")
    print(f"\n{'='*10} {title} {'='*10}")


def emit(s):
    lines.append(s)
    print(s)


# ---------- 1. FDR correction ----------
def fdr_correction():
    section("1. FDR (Benjamini-Hochberg) correction on per-tokenizer/per-model p-values")

    seg = pd.read_csv(SEG_SUMMARY)
    rej, p_adj, _, _ = multipletests(seg["paired_t_p"], method="fdr_bh")
    seg["paired_t_p_fdr"] = p_adj
    seg["significant_after_fdr"] = rej
    emit("\nSegmentation (9 tokenizers), paired t-test p-values, FDR-corrected:")
    emit(seg[["tokenizer", "paired_t_p", "paired_t_p_fdr", "significant_after_fdr"]].to_string(index=False))

    sur = pd.read_csv(SUR_SUMMARY)
    for metric in ["surprisal_bits_isolated", "surprisal_bits_in_context"]:
        sub = sur[sur["metric"] == metric].copy()
        rej, p_adj, _, _ = multipletests(sub["paired_t_p"], method="fdr_bh")
        sub["paired_t_p_fdr"] = p_adj
        sub["significant_after_fdr"] = rej
        emit(f"\nSurprisal ({metric}), 10 models, FDR-corrected:")
        emit(sub[["model", "paired_t_p", "paired_t_p_fdr", "significant_after_fdr"]].to_string(index=False))


# ---------- 2. Condition x category interaction ----------
def category_interaction():
    section("2. Condition x category interaction (is the effect concentrated in 1-2 of 5 categories?)")

    for path, response, group_col, label in [
        (SEG_PATH, "n_tokens", "tokenizer", "Segmentation"),
        (SUR_PATH, "surprisal_bits_isolated", "model", "Surprisal (isolated)"),
        (SUR_PATH, "surprisal_bits_in_context", "model", "Surprisal (in-context)"),
    ]:
        df = pd.read_csv(path)
        df["condition"] = pd.Categorical(df["condition"], categories=["word_final", "word_initial"])
        df["cluster"] = df["cluster"].astype(str)
        df["category"] = df["category"].astype(str)

        null_formula = f"{response} ~ condition + {group_col} + category"
        full_formula = f"{response} ~ condition * category + {group_col}"

        null_model = smf.mixedlm(null_formula, df, groups=df["cluster"]).fit(reml=False)
        full_model = smf.mixedlm(full_formula, df, groups=df["cluster"]).fit(reml=False)

        lr_stat = 2 * (full_model.llf - null_model.llf)
        df_diff = full_model.df_modelwc - null_model.df_modelwc
        p_value = sstats.chi2.sf(lr_stat, df_diff) if df_diff > 0 else float("nan")

        emit(f"\n{label}: condition x category LR test: stat={lr_stat:.3f}, df={df_diff}, p={p_value:.4g}")
        if p_value < 0.05:
            emit("  -> effect size DOES differ significantly across the 5 categories.")
        else:
            emit("  -> no significant evidence the effect is concentrated in specific categories (broadly consistent).")

        # descriptive: mean diff per category (pooled across tokenizer/model), for context
        cat_means = df.groupby(["category", "condition"])[response].mean().unstack()
        cat_means["diff(initial-final)"] = cat_means["word_initial"] - cat_means["word_final"]
        emit(cat_means.to_string())


# ---------- 3. Leave-one-cluster-out ----------
def leave_one_cluster_out():
    section("3. Leave-one-cluster-out robustness (primary pooled model, 15 refits)")

    for path, response, group_col, label in [
        (SEG_PATH, "n_tokens", "tokenizer", "Segmentation"),
        (SUR_PATH, "surprisal_bits_isolated", "model", "Surprisal (isolated)"),
        (SUR_PATH, "surprisal_bits_in_context", "model", "Surprisal (in-context)"),
    ]:
        df = pd.read_csv(path)
        df["condition"] = pd.Categorical(df["condition"], categories=["word_final", "word_initial"])
        df["cluster"] = df["cluster"].astype(str)
        df["pair_id"] = df["pair_id"].astype(str)
        clusters = sorted(df["cluster"].unique())

        emit(f"\n{label}: dropping each of {len(clusters)} clusters in turn, refitting "
             f"'{response} ~ condition + {group_col}', random=(1|cluster)+(1|pair nested):")
        emit(f"{'dropped_cluster':16s} {'condition_coef':>14s} {'p_value':>10s}")
        coefs, ps = [], []
        for c in clusters:
            sub = df[df["cluster"] != c]
            vc = {"pair": "0 + C(pair_id)"}
            model = smf.mixedlm(f"{response} ~ condition + {group_col}", sub, groups=sub["cluster"], vc_formula=vc).fit(reml=True)
            coef = model.params["condition[T.word_initial]"]
            p = model.pvalues["condition[T.word_initial]"]
            coefs.append(coef); ps.append(p)
            emit(f"{c:16s} {coef:14.4f} {p:10.4g}")
        max_p = max(ps)
        emit(f"  -> max p-value across all 15 leave-one-out refits: {max_p:.4g} "
             f"({'still significant everywhere' if max_p < 0.05 else 'NOT significant in at least one refit'})")


def main():
    fdr_correction()
    category_interaction()
    leave_one_cluster_out()
    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n\nWrote full summary to {OUT_PATH}")


if __name__ == "__main__":
    main()
