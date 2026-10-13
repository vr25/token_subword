"""
Step 5: full mixed-effects analysis across both datasets (segmentation.csv, surprisal.csv),
accounting for non-independence: only 15 unique clusters underlie the 50 pairs, and each
pair/cluster is measured repeatedly across tokenizers/models. Random effects: cluster
(grouping factor) with pair_id nested within cluster as a variance component. Fixed
effects: condition (word_initial vs word_final, the effect of interest) + tokenizer/model
(to control for baseline differences between tokenizers/models). Reports whatever the
fixed-effect coefficient/p-value for condition actually is, per dataset/metric -- no
assumed direction.
"""
import csv
import warnings
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

SEG_PATH = "/Users/vipularawte/token_exp/results/segmentation.csv"
SUR_PATH = "/Users/vipularawte/token_exp/results/surprisal.csv"
OUT_PATH = "/Users/vipularawte/token_exp/results/mixed_effects_summary.txt"


def fit_and_report(df, response, fixed_extra, label, out_lines):
    df = df.copy()
    df["condition"] = pd.Categorical(df["condition"], categories=["word_final", "word_initial"])
    df["cluster"] = df["cluster"].astype(str)
    df["pair_id"] = df["pair_id"].astype(str)

    formula = f"{response} ~ condition + {fixed_extra}"
    vc = {"pair": "0 + C(pair_id)"}
    model = smf.mixedlm(formula, df, groups=df["cluster"], vc_formula=vc)
    result = model.fit(reml=True)

    cond_rows = [i for i in result.params.index if i.startswith("condition")]
    out_lines.append(f"\n=== {label} ===")
    out_lines.append(f"formula: {formula}  |  random: (1|cluster) + (1|pair_id nested in cluster)")
    out_lines.append(f"n_obs={int(result.nobs)}")
    for r in cond_rows:
        coef = result.params[r]
        p = result.pvalues[r]
        se = result.bse[r]
        out_lines.append(f"  {r}: coef={coef:.4f}  se={se:.4f}  p={p:.4g}")
    out_lines.append(f"  (interpretation: coef is the fixed effect of being word_initial vs word_final,")
    out_lines.append(f"   holding tokenizer/model fixed, after accounting for cluster/pair random effects)")
    return result


def main():
    out_lines = []

    seg = pd.read_csv(SEG_PATH)
    fit_and_report(seg, "n_tokens", "tokenizer", "Segmentation: n_tokens ~ condition + tokenizer", out_lines)

    sur = pd.read_csv(SUR_PATH)
    fit_and_report(sur, "surprisal_bits_isolated", "model", "Surprisal (isolated word)", out_lines)
    fit_and_report(sur, "surprisal_bits_in_context", "model", "Surprisal (in sentence context)", out_lines)

    text = "\n".join(out_lines)
    print(text)
    with open(OUT_PATH, "w") as f:
        f.write(text + "\n")
    print(f"\n\nWrote summary to {OUT_PATH}")


if __name__ == "__main__":
    main()
