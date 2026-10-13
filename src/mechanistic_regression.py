"""
Step 3 of the plan: mechanistic account for *why* token count differs by position.

Key fact used here (an identity, not an assumption): cluster and rime are the same two
substrings in both conditions of a pair, just concatenated in opposite order. So
naive_min_tokens := tokens(cluster alone) + tokens(rime alone) is IDENTICAL between the
word_initial and word_final forms of a given pair -- it only depends on the two
substrings, not their order. Therefore any measured difference in actual token count
between the two conditions must come entirely from whether the tokenizer's BPE/vocab
happens to merge across the cluster-then-rime boundary or the rime-then-cluster
boundary more efficiently. This lets us decompose the effect exactly rather than
speculate about "frequency" in the abstract.

savings(condition) := naive_min_tokens - actual_n_tokens(condition)
  -> savings > 0 means the tokenizer merged across that specific boundary, "saving" tokens
     relative to just concatenating the two pieces' own tokenizations.

Identity to verify: mean_diff(initial - final) == savings(final) - savings(initial)
"""
import csv
import sys
import warnings
from collections import defaultdict
import statistics

warnings.filterwarnings("ignore")

PAIRS_PATH = "/Users/vipularawte/token_exp/data/pseudoword_pairs.csv"
OUT_PATH = "/Users/vipularawte/token_exp/results/mechanistic.csv"


def get_tokenizers():
    import tiktoken
    from transformers import AutoTokenizer
    toks = {}
    for name in ["gpt2", "cl100k_base", "o200k_base"]:
        enc = tiktoken.get_encoding(name)
        toks[f"tiktoken:{name}"] = ("tiktoken", enc)
    hf_candidates = {
        "hf:bert-base-uncased": "bert-base-uncased",
        "hf:t5-base": "google-t5/t5-base",
        "hf:mistral-7b-v0.3": "mistralai/Mistral-7B-v0.3",
        "hf:qwen2-7b": "Qwen/Qwen2-7B",
        "hf:llama-3-8b": "meta-llama/Meta-Llama-3-8B",
        "hf:gemma-2-2b": "google/gemma-2-2b",
    }
    for key, repo in hf_candidates.items():
        try:
            toks[key] = ("hf", AutoTokenizer.from_pretrained(repo))
        except Exception as e:
            print(f"[skip] {key}: {e}", file=sys.stderr)
    return toks


def n_tokens(kind, tok, s):
    if kind == "tiktoken":
        return len(tok.encode(s))
    return len(tok.encode(s, add_special_tokens=False))


def main():
    with open(PAIRS_PATH) as f:
        pair_rows = list(csv.DictReader(f))
    by_pair = defaultdict(dict)
    for r in pair_rows:
        by_pair[r["pair_id"]][r["condition"]] = r
        by_pair[r["pair_id"]]["cluster"] = r["cluster"]
        by_pair[r["pair_id"]]["rime"] = r["rime"]

    tokenizers = get_tokenizers()
    out_rows = []
    per_tok_savings_init = defaultdict(list)
    per_tok_savings_final = defaultdict(list)

    for pair_id, d in by_pair.items():
        cluster, rime = d["cluster"], d["rime"]
        wi = d["word_initial"]["pseudoword"]
        wf = d["word_final"]["pseudoword"]
        for tok_name, (kind, tok) in tokenizers.items():
            n_cluster = n_tokens(kind, tok, cluster)
            n_rime = n_tokens(kind, tok, rime)
            naive_min = n_cluster + n_rime
            n_wi = n_tokens(kind, tok, wi)
            n_wf = n_tokens(kind, tok, wf)
            savings_init = naive_min - n_wi
            savings_final = naive_min - n_wf
            out_rows.append({
                "pair_id": pair_id, "tokenizer": tok_name, "cluster": cluster, "rime": rime,
                "n_cluster_alone": n_cluster, "n_rime_alone": n_rime, "naive_min_tokens": naive_min,
                "n_tokens_word_initial": n_wi, "n_tokens_word_final": n_wf,
                "savings_word_initial": savings_init, "savings_word_final": savings_final,
            })
            per_tok_savings_init[tok_name].append(savings_init)
            per_tok_savings_final[tok_name].append(savings_final)

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {len(out_rows)} rows to {OUT_PATH}\n")

    print(f"{'tokenizer':22s} {'mean_savings_init':>18s} {'mean_savings_final':>19s} "
          f"{'boundary_bias(final-init)':>26s}")
    for tok_name in tokenizers:
        si = statistics.mean(per_tok_savings_init[tok_name])
        sf = statistics.mean(per_tok_savings_final[tok_name])
        print(f"{tok_name:22s} {si:18.3f} {sf:19.3f} {sf - si:26.3f}")

    # Verify the identity against the earlier segmentation_summary.csv (mean_diff column)
    print("\nIdentity check (mean_diff should equal savings_final - savings_init exactly, per tokenizer):")
    summary_path = "/Users/vipularawte/token_exp/results/segmentation_summary.csv"
    with open(summary_path) as f:
        seg_summary = {r["tokenizer"]: float(r["mean_diff_initial_minus_final"]) for r in csv.DictReader(f)}
    for tok_name in tokenizers:
        si = statistics.mean(per_tok_savings_init[tok_name])
        sf = statistics.mean(per_tok_savings_final[tok_name])
        predicted = sf - si
        actual = seg_summary.get(tok_name)
        if actual is not None:
            match = "OK" if abs(predicted - actual) < 1e-9 else "MISMATCH"
            print(f"  {tok_name:22s} predicted={predicted:.3f} actual={actual:.3f} [{match}]")


if __name__ == "__main__":
    main()
