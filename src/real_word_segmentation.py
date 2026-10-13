"""
Tokenizer segmentation for the real-word validation sample: same tokenizers as the main
pseudoword study, applied to data/real_words.csv (200 genuine English words spanning a
continuous range of initial/final positional-biphone rarity).
"""
import csv
import sys
import warnings

warnings.filterwarnings("ignore")

from segmentation_study import get_tiktoken_tokenizers, get_hf_tokenizers, tokenize

DATA_PATH = "/Users/vipularawte/token_exp/data/real_words.csv"
OUT_PATH = "/Users/vipularawte/token_exp/results/real_word_segmentation.csv"


def main():
    with open(DATA_PATH) as f:
        items = list(csv.DictReader(f))

    tokenizers = {}
    tokenizers.update(get_tiktoken_tokenizers())
    tokenizers.update(get_hf_tokenizers())
    print(f"Loaded {len(tokenizers)} tokenizers", file=sys.stderr)

    rows = []
    for item in items:
        word = item["word"]
        for tok_name, (kind, tok) in tokenizers.items():
            n_tokens, pieces = tokenize(kind, tok, word)
            rows.append({
                "word": word,
                "initial_biphone_log_freq": item["initial_biphone_log_freq"],
                "final_biphone_log_freq": item["final_biphone_log_freq"],
                "n_phones": item["n_phones"],
                "tokenizer": tok_name,
                "n_tokens": n_tokens,
                "pieces": "|".join(pieces),
            })

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
