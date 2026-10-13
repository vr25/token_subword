"""
Regression: does real model surprisal on real words depend on corpus-derived positional
phonotactic rarity (initial/final biphone log-frequency), controlling for length? Same
question as real_word_regression.py but for true surprisal instead of token count.
"""
import warnings
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

warnings.filterwarnings("ignore")

PATH = "/Users/vipularawte/token_exp/results/real_word_surprisal.csv"
OUT_PATH = "/Users/vipularawte/token_exp/results/real_word_surprisal_regression_summary.txt"


def main():
    df = pd.read_csv(PATH)
    lines = []

    lines.append("=== Per-model OLS: surprisal_bits_isolated ~ initial_biphone_log_freq + final_biphone_log_freq + n_phones ===")
    lines.append(f"{'model':28s} {'coef_initial':>13s} {'p_initial':>10s} {'coef_final':>11s} {'p_final':>9s}")
    for model_name, sub in df.groupby("model"):
        X = sm.add_constant(sub[["initial_biphone_log_freq", "final_biphone_log_freq", "n_phones"]])
        y = sub["surprisal_bits_isolated"]
        m = sm.OLS(y, X).fit()
        ci = m.params["initial_biphone_log_freq"]; pi = m.pvalues["initial_biphone_log_freq"]
        cf = m.params["final_biphone_log_freq"]; pf = m.pvalues["final_biphone_log_freq"]
        lines.append(f"{model_name:28s} {ci:13.4f} {pi:10.4g} {cf:11.4f} {pf:9.4g}")

    lines.append("\n=== Pooled mixed model: surprisal_bits_isolated ~ initial + final + n_phones + model, random intercept per word ===")
    df["word"] = df["word"].astype(str)
    formula = "surprisal_bits_isolated ~ initial_biphone_log_freq + final_biphone_log_freq + n_phones + model"
    mixed = smf.mixedlm(formula, df, groups=df["word"]).fit(reml=True)
    for term in ["initial_biphone_log_freq", "final_biphone_log_freq", "n_phones"]:
        lines.append(f"  {term}: coef={mixed.params[term]:.4f}  se={mixed.bse[term]:.4f}  p={mixed.pvalues[term]:.4g}")

    text = "\n".join(lines)
    print(text)
    with open(OUT_PATH, "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
