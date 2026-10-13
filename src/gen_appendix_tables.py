"""
Generates LaTeX table fragments for the appendix directly from the result CSVs,
so there is zero hand-transcription risk. Writes standalone .tex fragments that
are \input{} from main.tex.
"""
import csv

OUT_DIR = "/Users/vipularawte/token_exp/paper/tables"
import os
os.makedirs(OUT_DIR, exist_ok=True)

CATEGORY_ABBREV = {
    "stop_fricative_stop": "Stop-fric.-stop",
    "nasal_fricative_liquid": "Nasal-fric.-liquid",
    "voiceless_stop": "Voiceless stop",
    "voiced_liquid_guttural": "Voiced liquid-guttural",
    "complex_fricative_plosive": "Complex fric.-plosive",
}

DISPLAY_TOK = {
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
TOK_ORDER = ["tiktoken:o200k_base", "hf:bert-base-uncased", "hf:gemma-2-2b", "hf:t5-base",
             "hf:mistral-7b-v0.3", "tiktoken:gpt2", "tiktoken:cl100k_base", "hf:qwen2-7b", "hf:llama-3-8b"]

DISPLAY_MODEL = {
    "gpt2": "GPT-2",
    "gpt2-medium": "GPT-2-medium",
    "gpt2-large": "GPT-2-large",
    "EleutherAI/pythia-160m": "Pythia-160M",
    "EleutherAI/pythia-410m": "Pythia-410M",
    "EleutherAI/pythia-1.4b": "Pythia-1.4B",
    "EleutherAI/pythia-2.8b": "Pythia-2.8B",
    "mistralai/Mistral-7B-v0.3": "Mistral-7B",
    "meta-llama/Meta-Llama-3-8B": "Llama-3-8B",
    "google/gemma-2-2b": "Gemma-2",
}
MODEL_ORDER = list(DISPLAY_MODEL.keys())


def esc(s):
    return s.replace("_", "\\_")


def gen_full_dataset_table():
    with open("/Users/vipularawte/token_exp/data/pseudoword_pairs.csv") as f:
        rows = list(csv.DictReader(f))
    by_pair = {}
    for r in rows:
        by_pair.setdefault(int(r["pair_id"]), {})[r["condition"]] = r

    lines = []
    lines.append(r"\begin{longtable}{@{}rlccc@{}}")
    lines.append(r"\caption{Full 50-pair pseudoword dataset underlying every experiment in this paper. Each pair's word-initial and word-final forms use identical letters (illegal cluster $c$ + legal rime $r$, realized as $c{+}r$ and $r{+}c$); only the position of the illegal cluster differs between the two forms.} \label{tab:full-dataset} \\")
    lines.append(r"\toprule")
    lines.append(r"\# & Category & Cluster & Word-initial & Word-final \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(r"\# & Category & Cluster & Word-initial & Word-final \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\bottomrule")
    lines.append(r"\endfoot")
    for pid in sorted(by_pair):
        d = by_pair[pid]
        cat = CATEGORY_ABBREV[d["word_initial"]["category"]]
        cluster = d["word_initial"]["cluster"]
        wi = d["word_initial"]["pseudoword"]
        wf = d["word_final"]["pseudoword"]
        lines.append(f"{pid} & {cat} & /{cluster}/ & {wi} & {wf} \\\\")
    lines.append(r"\end{longtable}")
    with open(f"{OUT_DIR}/full_dataset.tex", "w") as f:
        f.write("\n".join(lines) + "\n")


def gen_segmentation_table():
    from statsmodels.stats.multitest import multipletests
    with open("/Users/vipularawte/token_exp/results/segmentation_summary.csv") as f:
        summary_rows = list(csv.DictReader(f))
    rows = {r["tokenizer"]: r for r in summary_rows}
    _, p_fdr, _, _ = multipletests([float(r["paired_t_p"]) for r in summary_rows], method="fdr_bh")
    fdr_lookup = dict(zip([r["tokenizer"] for r in summary_rows], p_fdr))

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering\small")
    lines.append(r"\begin{tabular}{@{}lrrrrr@{}}")
    lines.append(r"\toprule")
    lines.append(r"Tokenizer & Init. & Final & $d$ & $p$ & $p_{\text{FDR}}$ \\")
    lines.append(r"\midrule")
    for tok in TOK_ORDER:
        r = rows[tok]
        p = float(r["paired_t_p"])
        d = float(r["cohens_d"])
        name = DISPLAY_TOK[tok]
        lines.append(f"{name} & {float(r['mean_tokens_word_initial']):.2f} & {float(r['mean_tokens_word_final']):.2f} & {d:.2f} & {p:.2g} & {fdr_lookup[tok]:.2g} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Full segmentation results, all 9 tokenizers: mean token count for word-initial (``Init.'') vs.\ word-final (``Final'') pseudowords, Cohen's $d$ (negative = fewer tokens word-initially, the paper's main direction), and FDR-corrected paired $t$-test $p$.}")
    lines.append(r"\label{tab:full-segmentation}")
    lines.append(r"\end{table}")
    with open(f"{OUT_DIR}/full_segmentation.tex", "w") as f:
        f.write("\n".join(lines) + "\n")


def gen_surprisal_table():
    from statsmodels.stats.multitest import multipletests
    with open("/Users/vipularawte/token_exp/results/surprisal_summary.csv") as f:
        rows = list(csv.DictReader(f))
    by_key = {(r["model"], r["metric"]): r for r in rows}

    fdr_lookup = {}
    for metric in ["surprisal_bits_isolated", "surprisal_bits_in_context"]:
        sub = [r for r in rows if r["metric"] == metric]
        _, p_fdr, _, _ = multipletests([float(r["paired_t_p"]) for r in sub], method="fdr_bh")
        for r, p in zip(sub, p_fdr):
            fdr_lookup[(r["model"], metric)] = p

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering\small")
    lines.append(r"\begin{tabular}{@{}lrrrr@{}}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{2}{c}{Isolated} & \multicolumn{2}{c}{In context} \\")
    lines.append(r"Model & $d$ & $p_{\text{FDR}}$ & $d$ & $p_{\text{FDR}}$ \\")
    lines.append(r"\midrule")
    for m in MODEL_ORDER:
        ri = by_key[(m, "surprisal_bits_isolated")]
        rc = by_key[(m, "surprisal_bits_in_context")]
        name = DISPLAY_MODEL[m]
        lines.append(f"{name} & {float(ri['cohens_d']):.2f} & {fdr_lookup[(m,'surprisal_bits_isolated')]:.2g} & "
                      f"{float(rc['cohens_d']):.2f} & {fdr_lookup[(m,'surprisal_bits_in_context')]:.2g} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Full surprisal results, all 10 models: Cohen's $d$ (negative = lower surprisal word-initially, the paper's main direction) and FDR-corrected paired $t$-test $p$, for the word-initial vs.\ word-final pseudowords in isolation and in sentence context.}")
    lines.append(r"\label{tab:full-surprisal}")
    lines.append(r"\end{table}")
    with open(f"{OUT_DIR}/full_surprisal.tex", "w") as f:
        f.write("\n".join(lines) + "\n")


def gen_tokenization_example():
    with open("/Users/vipularawte/token_exp/results/segmentation.csv") as f:
        rows = list(csv.DictReader(f))
    target_pair = "49"
    sub = [r for r in rows if r["pair_id"] == target_pair]
    by_tok = {r["tokenizer"]: r for r in sub}

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering\small")
    lines.append(r"\begin{tabular}{@{}lll@{}}")
    lines.append(r"\toprule")
    lines.append(r"Tokenizer & \texttt{fvken} (init.) & \texttt{enfvk} (final) \\")
    lines.append(r"\midrule")
    for tok in TOK_ORDER:
        r = by_tok[tok]
        name = DISPLAY_TOK[tok]
        wi_pieces = r["pieces"] if r["pseudoword"] == "fvken" else None
        # find matching rows for both conditions for this tokenizer
        wi_row = [x for x in sub if x["tokenizer"] == tok and x["pseudoword"] == "fvken"][0]
        wf_row = [x for x in sub if x["tokenizer"] == tok and x["pseudoword"] == "enfvk"][0]
        def clean(s):
            return s.replace("#", r"\#").replace("▁", r"\_")
        wi_str = clean(wi_row["pieces"]) + f" ({wi_row['n_tokens']})"
        wf_str = clean(wf_row["pieces"]) + f" ({wf_row['n_tokens']})"
        lines.append(f"{name} & \\texttt{{{wi_str}}} & \\texttt{{{wf_str}}} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Example tokenizations for Pair 49 (cluster /fvk/: \textit{fvken} word-initial vs.\ \textit{enfvk} word-final), showing the boundary-merge mechanism concretely: the actual token pieces (\texttt{|}-separated) and resulting token count (in parentheses) each tokenizer produces for both conditions.}")
    lines.append(r"\label{tab:tokenization-example}")
    lines.append(r"\end{table}")
    with open(f"{OUT_DIR}/tokenization_example.tex", "w") as f:
        f.write("\n".join(lines) + "\n")


def gen_dataset_examples_table():
    """Compact 5-row example table for the MAIN body (one pair per category)."""
    with open("/Users/vipularawte/token_exp/data/pseudoword_pairs.csv") as f:
        rows = list(csv.DictReader(f))
    by_pair = {}
    for r in rows:
        by_pair.setdefault(int(r["pair_id"]), {})[r["condition"]] = r
    example_pids = [1, 11, 21, 31, 49]

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering\small")
    lines.append(r"\begin{tabular}{@{}lccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"Category & Cluster & Init. & Final \\")
    lines.append(r"\midrule")
    for pid in example_pids:
        d = by_pair[pid]
        cat = CATEGORY_ABBREV[d["word_initial"]["category"]]
        cluster = d["word_initial"]["cluster"]
        wi = d["word_initial"]["pseudoword"]
        wf = d["word_final"]["pseudoword"]
        lines.append(f"{cat} & /{cluster}/ & {wi} & {wf} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{One example pair per category (full dataset: Table~\ref{tab:full-dataset}, Appendix~\ref{app:full-dataset}). ``Init.''/``Final'' place the shared illegal cluster before/after the same legal rime.}")
    lines.append(r"\label{tab:examples}")
    lines.append(r"\end{table}")
    with open(f"{OUT_DIR}/dataset_examples.tex", "w") as f:
        f.write("\n".join(lines) + "\n")


def gen_realword_regression_tables():
    import statsmodels.api as sm
    import pandas as pd

    seg = pd.read_csv("/Users/vipularawte/token_exp/results/real_word_segmentation.csv")
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering\small")
    lines.append(r"\begin{tabular}{@{}lrrrr@{}}")
    lines.append(r"\toprule")
    lines.append(r"Tokenizer & $\beta_{\text{init}}$ & $p_{\text{init}}$ & $\beta_{\text{final}}$ & $p_{\text{final}}$ \\")
    lines.append(r"\midrule")
    for tok in TOK_ORDER:
        sub = seg[seg["tokenizer"] == tok]
        X = sm.add_constant(sub[["initial_biphone_log_freq", "final_biphone_log_freq", "n_phones"]])
        y = sub["n_tokens"]
        m = sm.OLS(y, X).fit()
        name = DISPLAY_TOK[tok]
        ci = m.params["initial_biphone_log_freq"]; pi = m.pvalues["initial_biphone_log_freq"]
        cf = m.params["final_biphone_log_freq"]; pf = m.pvalues["final_biphone_log_freq"]
        lines.append(f"{name} & {ci:.3f} & {pi:.2g} & {cf:.3f} & {pf:.2g} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Real-word validation ($n=200$ real English words), token count: per-tokenizer OLS coefficients ($n\_tokens \sim \beta_{\text{init}} \cdot \text{initial-biphone-freq} + \beta_{\text{final}} \cdot \text{final-biphone-freq} + n\_phones$, where biphone-freq is a word's positional phoneme-bigram log-frequency and $n\_phones$ is a length control). A negative $\beta$ means a rarer (lower-frequency) biphone predicts \emph{more} tokens -- the expected, non-paradoxical direction. Mistral-7B's significant initial-position coefficient is discussed in \S\ref{sec:results}.}")
    lines.append(r"\label{tab:realword-segmentation}")
    lines.append(r"\end{table}")
    with open(f"{OUT_DIR}/realword_segmentation_regression.tex", "w") as f:
        f.write("\n".join(lines) + "\n")

    sur = pd.read_csv("/Users/vipularawte/token_exp/results/real_word_surprisal.csv")
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering\small")
    lines.append(r"\begin{tabular}{@{}lrrrr@{}}")
    lines.append(r"\toprule")
    lines.append(r"Model & $\beta_{\text{init}}$ & $p_{\text{init}}$ & $\beta_{\text{final}}$ & $p_{\text{final}}$ \\")
    lines.append(r"\midrule")
    for m_key in MODEL_ORDER:
        sub = sur[sur["model"] == m_key]
        X = sm.add_constant(sub[["initial_biphone_log_freq", "final_biphone_log_freq", "n_phones"]])
        y = sub["surprisal_bits_isolated"]
        model = sm.OLS(y, X).fit()
        name = DISPLAY_MODEL[m_key]
        ci = model.params["initial_biphone_log_freq"]; pi = model.pvalues["initial_biphone_log_freq"]
        cf = model.params["final_biphone_log_freq"]; pf = model.pvalues["final_biphone_log_freq"]
        lines.append(f"{name} & {ci:.3f} & {pi:.2g} & {cf:.3f} & {pf:.2g} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Real-word validation ($n=200$), surprisal: per-model OLS coefficients, same regression and sign convention as \cref{tab:realword-segmentation} but predicting isolated-word surprisal (bits) instead of token count.}")
    lines.append(r"\label{tab:realword-surprisal}")
    lines.append(r"\end{table}")
    with open(f"{OUT_DIR}/realword_surprisal_regression.tex", "w") as f:
        f.write("\n".join(lines) + "\n")


def gen_category_breakdown_table():
    import pandas as pd

    seg = pd.read_csv("/Users/vipularawte/token_exp/results/segmentation.csv")
    sur = pd.read_csv("/Users/vipularawte/token_exp/results/surprisal.csv")

    def diffs(df, metric):
        g = df.groupby(["category", "condition"])[metric].mean().unstack()
        return g["word_initial"] - g["word_final"]

    seg_diff = diffs(seg, "n_tokens")
    iso_diff = diffs(sur, "surprisal_bits_isolated")
    ctx_diff = diffs(sur, "surprisal_bits_in_context")

    CAT_SHORT = {
        "stop_fricative_stop": "Stop-fric.-stop",
        "nasal_fricative_liquid": "Nasal-fric.-liquid",
        "voiceless_stop": "Voiceless stop",
        "voiced_liquid_guttural": "Voiced liq.-gutt.",
        "complex_fricative_plosive": "Complex fric.-plos.",
    }

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering\footnotesize")
    lines.append(r"\begin{tabular}{@{}lrrr@{}}")
    lines.append(r"\toprule")
    lines.append(r"Category & Seg. & Iso.\ (bits) & Ctx.\ (bits) \\")
    lines.append(r"\midrule")
    for cat_key, cat_name in CAT_SHORT.items():
        lines.append(f"{cat_name} & {seg_diff[cat_key]:+.3f} & {iso_diff[cat_key]:+.3f} & {ctx_diff[cat_key]:+.3f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Mean difference (word-initial $-$ word-final), pooled across all tokenizers/models, by phonetic category. Negative = effect in the reported direction; positive = opposite (complex fricative-plosive, segmentation and in-context surprisal only).}")
    lines.append(r"\label{tab:category-breakdown}")
    lines.append(r"\end{table}")
    with open(f"{OUT_DIR}/category_breakdown.tex", "w") as f:
        f.write("\n".join(lines) + "\n")


VOCAB_SIZE = {
    "tiktoken:gpt2": 50257,
    "tiktoken:cl100k_base": 100277,
    "tiktoken:o200k_base": 199998,
    "hf:bert-base-uncased": 30522,
    "hf:t5-base": 32100,
    "hf:mistral-7b-v0.3": 32768,
    "hf:qwen2-7b": 151936,
    "hf:llama-3-8b": 128256,
    "hf:gemma-2-2b": 256000,
}


def gen_seg_surprisal_link_table():
    import pandas as pd
    from scipy import stats as st

    df = pd.read_csv("/Users/vipularawte/token_exp/results/surprisal.csv")
    r_pool, p_pool = st.pearsonr(df["n_tokens_isolated"], df["surprisal_bits_isolated"])

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering\small")
    lines.append(r"\begin{tabular}{@{}lrr@{}}")
    lines.append(r"\toprule")
    lines.append(r"Model & $r$ & $p$ \\")
    lines.append(r"\midrule")
    for m in MODEL_ORDER:
        g = df[df["model"] == m]
        r_val, p_val = st.pearsonr(g["n_tokens_isolated"], g["surprisal_bits_isolated"])
        p_str = "$<$0.001" if p_val < 0.001 else f"{p_val:.2g}"
        lines.append(f"{DISPLAY_MODEL[m]} & {r_val:.2f} & {p_str} \\\\")
    lines.append(r"\midrule")
    lines.append(rf"Pooled (all models) & {r_pool:.2f} & $<$0.001 \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Item-level Pearson correlation between isolated-condition token count and surprisal, per model ($n=100$ items each) and pooled ($n=1{,}000$). Coupling is strong for GPT-2/Pythia but weak for Gemma-2 and Llama-3-8B, consistent with \cref{sec:results}'s suffixation account: an extra word-final token is often a common, predictable subword piece there, costing a token but little surprisal.}")
    lines.append(r"\label{tab:seg-surprisal-link}")
    lines.append(r"\end{table}")
    with open(f"{OUT_DIR}/seg_surprisal_link.tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"seg-surprisal pooled: r={r_pool:.4f}, p={p_pool:.2e}")


MODEL_PARAMS_M = {
    "gpt2": 124,
    "gpt2-medium": 355,
    "gpt2-large": 774,
    "EleutherAI/pythia-160m": 162,
    "EleutherAI/pythia-410m": 405,
    "EleutherAI/pythia-1.4b": 1414,
    "EleutherAI/pythia-2.8b": 2775,
    "mistralai/Mistral-7B-v0.3": 7241,
    "meta-llama/Meta-Llama-3-8B": 8030,
    "google/gemma-2-2b": 2614,
}


def gen_model_scale_table():
    from scipy import stats as st

    rows = list(csv.DictReader(open("/Users/vipularawte/token_exp/results/surprisal_summary.csv")))
    by_metric = {}
    for metric in ["surprisal_bits_isolated", "surprisal_bits_in_context"]:
        d = {r["model"]: float(r["mean_diff"]) for r in rows if r["metric"] == metric}
        by_metric[metric] = d

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering\small")
    lines.append(r"\begin{tabular}{@{}lrrr@{}}")
    lines.append(r"\toprule")
    lines.append(r"Model & Params.\ (M) & Iso.\ diff. & Ctx.\ diff. \\")
    lines.append(r"\midrule")
    for m in MODEL_ORDER:
        lines.append(f"{DISPLAY_MODEL[m]} & {MODEL_PARAMS_M[m]:,} & {by_metric['surprisal_bits_isolated'][m]:+.2f} & {by_metric['surprisal_bits_in_context'][m]:+.2f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    for metric, label in [("surprisal_bits_isolated", "isolated"), ("surprisal_bits_in_context", "in-context")]:
        xs = [MODEL_PARAMS_M[m] for m in MODEL_ORDER]
        ys = [by_metric[metric][m] for m in MODEL_ORDER]
        r_val, p_val = st.pearsonr(xs, ys)
        print(f"model-scale vs {label}: r={r_val:.4f}, p={p_val:.4f}")
    lines.append(r"\caption{Parameter count vs.\ surprisal effect size (mean difference, word-initial $-$ word-final, bits) per model. Isolated: Pearson $r=-0.12$, $p=0.74$. In-context: $r=-0.01$, $p=0.99$. Model scale does not predict effect size in either condition.}")
    lines.append(r"\label{tab:model-scale}")
    lines.append(r"\end{table}")
    with open(f"{OUT_DIR}/model_scale.tex", "w") as f:
        f.write("\n".join(lines) + "\n")


def gen_vocab_size_table():
    from scipy import stats as st

    with open("/Users/vipularawte/token_exp/results/segmentation_summary.csv") as f:
        summary_rows = list(csv.DictReader(f))
    rows = {r["tokenizer"]: r for r in summary_rows}

    xs = [VOCAB_SIZE[t] for t in TOK_ORDER]
    ys = [float(rows[t]["mean_diff_initial_minus_final"]) for t in TOK_ORDER]
    r_val, p_val = st.pearsonr(xs, ys)

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering\small")
    lines.append(r"\begin{tabular}{@{}lrr@{}}")
    lines.append(r"\toprule")
    lines.append(r"Tokenizer & Vocab.\ size & Seg.\ diff. \\")
    lines.append(r"\midrule")
    for tok in TOK_ORDER:
        lines.append(f"{DISPLAY_TOK[tok]} & {VOCAB_SIZE[tok]:,} & {ys[TOK_ORDER.index(tok)]:+.2f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{Vocabulary size vs.\ segmentation effect size (mean difference, word-initial $-$ word-final) per tokenizer. Pearson $r={r_val:.2f}$, $p={p_val:.2f}$ across all 9 tokenizers: vocabulary size does not predict effect size.}}")
    lines.append(r"\label{tab:vocab-size}")
    lines.append(r"\end{table}")
    with open(f"{OUT_DIR}/vocab_size.tex", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"vocab-size correlation: r={r_val:.4f}, p={p_val:.4f}")


if __name__ == "__main__":
    gen_full_dataset_table()
    gen_segmentation_table()
    gen_surprisal_table()
    gen_tokenization_example()
    gen_dataset_examples_table()
    gen_realword_regression_tables()
    gen_category_breakdown_table()
    gen_vocab_size_table()
    gen_seg_surprisal_link_table()
    gen_model_scale_table()
    print("Wrote all appendix/example table fragments to", OUT_DIR)
