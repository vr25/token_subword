"""
Step 2 of the plan: real tokenizer segmentation only (no model weights, no logprobs).
For every pseudoword in data/pseudoword_pairs.csv, records the actual token count and
token pieces produced by each tokenizer, across multiple algorithm families:
  - BPE:              gpt2, cl100k_base (GPT-4/3.5), o200k_base (GPT-4o/GPT-5 family),
                       Mistral-7B, Qwen2, Llama-3 (gated, skipped until HF login)
  - WordPiece:         BERT
  - SentencePiece-uni: T5, Gemma-2 (gated, skipped until HF login)

No claims are assumed in advance -- this script just measures and writes results/segmentation.csv.
Tokenizers that fail to load (e.g. gated repos without HF auth) are skipped with a clear warning,
not silently faked.
"""
import csv
import sys
import warnings

warnings.filterwarnings("ignore")

DATA_PATH = "/Users/vipularawte/token_exp/data/pseudoword_pairs.csv"
OUT_PATH = "/Users/vipularawte/token_exp/results/segmentation.csv"


def load_items():
    with open(DATA_PATH) as f:
        return list(csv.DictReader(f))


def get_tiktoken_tokenizers():
    import tiktoken
    out = {}
    for name in ["gpt2", "cl100k_base", "o200k_base"]:
        try:
            enc = tiktoken.get_encoding(name)
            out[f"tiktoken:{name}"] = ("tiktoken", enc)
        except Exception as e:
            print(f"[skip] tiktoken {name}: {e}", file=sys.stderr)
    return out


def get_hf_tokenizers():
    from transformers import AutoTokenizer
    candidates = {
        "hf:bert-base-uncased": "bert-base-uncased",
        "hf:t5-base": "google-t5/t5-base",
        "hf:mistral-7b-v0.3": "mistralai/Mistral-7B-v0.3",
        "hf:qwen2-7b": "Qwen/Qwen2-7B",
        "hf:llama-3-8b": "meta-llama/Meta-Llama-3-8B",
        "hf:gemma-2-2b": "google/gemma-2-2b",
    }
    out = {}
    for key, repo in candidates.items():
        try:
            tok = AutoTokenizer.from_pretrained(repo)
            out[key] = ("hf", tok)
        except Exception as e:
            print(f"[skip] {key} ({repo}): {type(e).__name__}: {str(e)[:150]}", file=sys.stderr)
    return out


def tokenize(kind, tok, word):
    if kind == "tiktoken":
        ids = tok.encode(word)
        pieces = [tok.decode([i]) for i in ids]
        return len(ids), pieces
    else:  # hf fast tokenizer
        ids = tok.encode(word, add_special_tokens=False)
        pieces = tok.convert_ids_to_tokens(ids)
        return len(ids), pieces


def main():
    items = load_items()
    tokenizers = {}
    tokenizers.update(get_tiktoken_tokenizers())
    tokenizers.update(get_hf_tokenizers())

    print(f"\nLoaded {len(tokenizers)} tokenizers: {list(tokenizers.keys())}\n", file=sys.stderr)

    rows = []
    for item in items:
        word = item["pseudoword"]
        for tok_name, (kind, tok) in tokenizers.items():
            n_tokens, pieces = tokenize(kind, tok, word)
            rows.append({
                "pair_id": item["pair_id"],
                "category": item["category"],
                "cluster": item["cluster"],
                "condition": item["condition"],
                "pseudoword": word,
                "tokenizer": tok_name,
                "n_tokens": n_tokens,
                "pieces": "|".join(pieces),
            })

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pair_id", "category", "cluster", "condition", "pseudoword",
            "tokenizer", "n_tokens", "pieces"
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT_PATH}", file=sys.stderr)

    # Spot-check the two disputed examples right here, in real output, not from memory.
    print("\n--- Spot check: disputed examples from the earlier chat claims ---", file=sys.stderr)
    for r in rows:
        if r["pseudoword"] in ("pvkon", "onpvk", "fvken", "enfvk"):
            print(f"{r['tokenizer']:20s} {r['pseudoword']:8s} n_tokens={r['n_tokens']:<3} pieces={r['pieces']}", file=sys.stderr)


if __name__ == "__main__":
    main()
