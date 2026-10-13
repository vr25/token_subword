"""
Runs the same surprisal measurement as surprisal_study.py, but only for the two models
that were gated/unavailable in the first pass (Llama-3-8B, Gemma-2-9b), and APPENDS to
the existing results/surprisal.csv rather than overwriting it.
"""
import csv
import sys

sys.path.insert(0, "/Users/vipularawte/token_exp/src")
from surprisal_study import load_model, total_surprisal_bits, PAIRS_PATH, DEVICE

OUT_PATH = "/Users/vipularawte/token_exp/results/surprisal.csv"
EXTRA_MODELS = ["meta-llama/Meta-Llama-3-8B"]


def main():
    with open(PAIRS_PATH) as f:
        items = list(csv.DictReader(f))
    with open(OUT_PATH) as f:
        existing_rows = list(csv.DictReader(f))
    fieldnames = list(existing_rows[0].keys())

    new_rows = []
    for model_name in EXTRA_MODELS:
        print(f"\n=== Loading {model_name} on {DEVICE} ===", file=sys.stderr)
        try:
            tok, model = load_model(model_name)
        except Exception as e:
            print(f"[skip] {model_name}: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
            continue
        for item in items:
            word = item["pseudoword"]
            bits_isolated, n_isolated = total_surprisal_bits(tok, model, word)
            sentence = item["carrier_sentence"]
            prefix = sentence.split(word)[0] if word in sentence else ""
            bits_context, n_context = total_surprisal_bits(tok, model, word, bos_context=prefix)
            new_rows.append({
                "pair_id": item["pair_id"], "category": item["category"], "cluster": item["cluster"],
                "condition": item["condition"], "pseudoword": word, "model": model_name,
                "surprisal_bits_isolated": bits_isolated, "n_tokens_isolated": n_isolated,
                "surprisal_bits_in_context": bits_context, "n_tokens_in_context": n_context,
            })
        print(f"Done: {model_name} ({len(items)} items)", file=sys.stderr)
        del model

    if new_rows:
        with open(OUT_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_rows)
            writer.writerows(new_rows)
        print(f"\nAppended {len(new_rows)} rows. Total now {len(existing_rows) + len(new_rows)}.", file=sys.stderr)
    else:
        print("\nNo new rows produced.", file=sys.stderr)


if __name__ == "__main__":
    main()
