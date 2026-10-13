"""
Regression test for the real-word validation: does token count depend on real,
corpus-derived positional phonotactic rarity (initial_biphone_log_freq,
final_biphone_log_freq), and if so, in which direction? Predicts, per the main
pseudoword finding, that LOWER initial_biphone_log_freq (rarer word-initial biphone,
more phonotactically marked start) predicts FEWER tokens (negative coefficient), while
LOWER final_biphone_log_freq predicts MORE tokens (positive coefficient) -- but reports
whatever actually comes out, per tokenizer and pooled.
"""
import warnings
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

warnings.filterwarnings("ignore")

SEG_PATH = "/Users/vipularawte/token_exp/results/real_word_segmentation.csv"
OUT_PATH = "/Users/vipularawte/token_exp/results/real_word_regression_summary.txt"


def main():
    df = pd.read_csv(SEG_PATH)
    lines = []

    lines.append("=== Per-tokenizer OLS: n_tokens ~ initial_biphone_log_freq + final_biphone_log_freq + n_phones ===")
    lines.append(f"{'tokenizer':22s} {'coef_initial':>13s} {'p_initial':>10s} {'coef_final':>11s} {'p_final':>9s}")
    for tok_name, sub in df.groupby("tokenizer"):
        X = sm.add_constant(sub[["initial_biphone_log_freq", "final_biphone_log_freq", "n_phones"]])
        y = sub["n_tokens"]
        model = sm.OLS(y, X).fit()
        ci = model.params["initial_biphone_log_freq"]
        pi = model.pvalues["initial_biphone_log_freq"]
        cf = model.params["final_biphone_log_freq"]
        pf = model.pvalues["final_biphone_log_freq"]
        lines.append(f"{tok_name:22s} {ci:13.4f} {pi:10.4g} {cf:11.4f} {pf:9.4g}")

    lines.append("\n=== Pooled mixed model: n_tokens ~ initial_biphone_log_freq + final_biphone_log_freq + n_phones + tokenizer, random intercept per word ===")
    df["word"] = df["word"].astype(str)
    formula = "n_tokens ~ initial_biphone_log_freq + final_biphone_log_freq + n_phones + tokenizer"
    mixed = smf.mixedlm(formula, df, groups=df["word"]).fit(reml=True)
    for term in ["initial_biphone_log_freq", "final_biphone_log_freq", "n_phones"]:
        lines.append(f"  {term}: coef={mixed.params[term]:.4f}  se={mixed.bse[term]:.4f}  p={mixed.pvalues[term]:.4g}")

    text = "\n".join(lines)
    print(text)
    with open(OUT_PATH, "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
