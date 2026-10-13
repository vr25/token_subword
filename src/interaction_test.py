"""
Formal test of whether the condition (word_initial vs word_final) effect size differs
significantly across tokenizers/models -- i.e. is the effect uniform, or heterogeneous
as the per-model breakdown suggested (e.g. Mistral-7B's in-context effect ~0 vs smaller
models' larger effect)? Likelihood-ratio test: null model (condition + group, no
interaction) vs full model (condition * group, with interaction), both fit by ML (not
REML, required for valid LR tests comparing fixed effects) with the same random-effects
structure (cluster + pair-nested-in-cluster).
"""
import warnings
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as sstats

warnings.filterwarnings("ignore")

SEG_PATH = "/Users/vipularawte/token_exp/results/segmentation.csv"
SUR_PATH = "/Users/vipularawte/token_exp/results/surprisal.csv"
OUT_PATH = "/Users/vipularawte/token_exp/results/interaction_test_summary.txt"


def lr_test(df, response, group_col, label, out_lines):
    df = df.copy()
    df["condition"] = pd.Categorical(df["condition"], categories=["word_final", "word_initial"])
    df["cluster"] = df["cluster"].astype(str)
    df["pair_id"] = df["pair_id"].astype(str)
    vc = {"pair": "0 + C(pair_id)"}

    null_formula = f"{response} ~ condition + {group_col}"
    full_formula = f"{response} ~ condition * {group_col}"

    null_model = smf.mixedlm(null_formula, df, groups=df["cluster"], vc_formula=vc).fit(reml=False)
    full_model = smf.mixedlm(full_formula, df, groups=df["cluster"], vc_formula=vc).fit(reml=False)

    lr_stat = 2 * (full_model.llf - null_model.llf)
    df_diff = full_model.df_modelwc - null_model.df_modelwc
    p_value = sstats.chi2.sf(lr_stat, df_diff) if df_diff > 0 else float("nan")

    out_lines.append(f"\n=== Interaction test: {label} ===")
    out_lines.append(f"null:  {null_formula}")
    out_lines.append(f"full:  {full_formula}")
    out_lines.append(f"LR statistic = {lr_stat:.3f}, df = {df_diff}, p = {p_value:.4g}")
    if p_value < 0.05:
        out_lines.append("  -> effect size DOES differ significantly across groups (heterogeneous).")
    else:
        out_lines.append("  -> no significant evidence the effect size differs across groups (data consistent with a uniform effect).")

    interaction_terms = [i for i in full_model.params.index if ":" in i]
    out_lines.append("  per-group interaction coefficients (relative to reference group):")
    for t in interaction_terms:
        out_lines.append(f"    {t}: coef={full_model.params[t]:.4f}  p={full_model.pvalues[t]:.4g}")
    return lr_stat, df_diff, p_value


def main():
    out_lines = []
    seg = pd.read_csv(SEG_PATH)
    lr_test(seg, "n_tokens", "tokenizer", "Segmentation: condition x tokenizer", out_lines)

    sur = pd.read_csv(SUR_PATH)
    lr_test(sur, "surprisal_bits_isolated", "model", "Surprisal isolated: condition x model", out_lines)
    lr_test(sur, "surprisal_bits_in_context", "model", "Surprisal in-context: condition x model", out_lines)

    text = "\n".join(out_lines)
    print(text)
    with open(OUT_PATH, "w") as f:
        f.write(text + "\n")
    print(f"\n\nWrote summary to {OUT_PATH}")


if __name__ == "__main__":
    main()
