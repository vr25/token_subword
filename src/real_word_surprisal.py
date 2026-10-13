"""
Real log-surprisal for the real-word validation sample (isolated word only -- no carrier
sentences exist for this dataset), across the same open-weight models used in the main
pseudoword study (excluding Gemma-2-9b, never fully obtained; Gemma-2-2b used instead,
same tokenizer family).
"""
import csv
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, "/Users/vipularawte/token_exp/src")
from surprisal_study import load_model, total_surprisal_bits, DEVICE

DATA_PATH = "/Users/vipularawte/token_exp/data/real_words.csv"
OUT_PATH = "/Users/vipularawte/token_exp/results/real_word_surprisal.csv"

MODELS = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "mistralai/Mistral-7B-v0.3",
    "meta-llama/Meta-Llama-3-8B",
    "google/gemma-2-2b",
]


def main():
    with open(DATA_PATH) as f:
        items = list(csv.DictReader(f))

    rows = []
    for model_name in MODELS:
        try:
            print(f"\n=== Loading {model_name} on {DEVICE} ===", file=sys.stderr)
            tok, model = load_model(model_name)
        except Exception as e:
            print(f"[skip] {model_name}: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
            continue

        for item in items:
            word = item["word"]
            bits, n_scored = total_surprisal_bits(tok, model, word)
            rows.append({
                "word": word,
                "initial_biphone_log_freq": item["initial_biphone_log_freq"],
                "final_biphone_log_freq": item["final_biphone_log_freq"],
                "n_phones": item["n_phones"],
                "model": model_name,
                "surprisal_bits_isolated": bits,
                "n_tokens_isolated": n_scored,
            })
        del model
        print(f"Done: {model_name} ({len(items)} items)", file=sys.stderr)

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
