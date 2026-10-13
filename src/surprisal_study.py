"""
Step 4 of the plan: real log-surprisal from open-weight models (actual forward passes,
not tokenizer segmentation proxies). Computes total surprisal (sum of -log2 P(token_i |
context)) for each pseudoword, both in isolation and inside its carrier sentence.

Models attempted, in increasing cost order (skip gracefully if unavailable/gated):
  gpt2, gpt2-medium, gpt2-large, gpt2-xl   (ungated, small/free)
  EleutherAI/pythia-160m .. pythia-2.8b    (ungated, scaling suite)
  mistralai/Mistral-7B-v0.3                (ungated, larger)
  meta-llama/Meta-Llama-3-8B               (gated -- needs HF login)
  google/gemma-2-9b                        (gated -- needs HF login)
"""
import csv
import sys
import warnings
import math

warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PAIRS_PATH = "/Users/vipularawte/token_exp/data/pseudoword_pairs.csv"
OUT_PATH = "/Users/vipularawte/token_exp/results/surprisal.csv"

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
    "google/gemma-2-9b",
]

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(name):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16 if DEVICE == "mps" else torch.float32)
    model.to(DEVICE)
    model.eval()
    return tok, model


@torch.no_grad()
def total_surprisal_bits(tok, model, text, bos_context=""):
    """Sum of -log2 P(token_i | preceding context) for all tokens of `text`,
    optionally preceded by `bos_context` (e.g. a carrier-sentence prefix) which is
    included for conditioning but excluded from the summed surprisal."""
    full = bos_context + text
    ids_full = tok(full, return_tensors="pt").input_ids.to(DEVICE)
    ids_ctx = tok(bos_context, return_tensors="pt").input_ids.to(DEVICE) if bos_context else None
    n_ctx = ids_ctx.shape[1] if ids_ctx is not None else 0

    out = model(ids_full)
    logits = out.logits[0]  # (seq_len, vocab)
    log_probs = torch.log_softmax(logits.float(), dim=-1)

    total_bits = 0.0
    n_tokens_scored = 0
    seq_len = ids_full.shape[1]
    start = max(n_ctx, 1)  # never score position 0 (no preceding context to condition on)
    for i in range(start, seq_len):
        target_id = ids_full[0, i].item()
        lp = log_probs[i - 1, target_id].item()  # predicts token i from position i-1
        total_bits += -lp / math.log(2)
        n_tokens_scored += 1
    return total_bits, n_tokens_scored


def main():
    with open(PAIRS_PATH) as f:
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
            word = item["pseudoword"]
            bits_isolated, n_isolated = total_surprisal_bits(tok, model, word)
            # sentence-context: score the word as it appears inside its carrier sentence
            sentence = item["carrier_sentence"]
            prefix = sentence.split(word)[0] if word in sentence else ""
            bits_context, n_context = total_surprisal_bits(tok, model, word, bos_context=prefix)

            rows.append({
                "pair_id": item["pair_id"], "category": item["category"], "cluster": item["cluster"],
                "condition": item["condition"], "pseudoword": word, "model": model_name,
                "surprisal_bits_isolated": bits_isolated, "n_tokens_isolated": n_isolated,
                "surprisal_bits_in_context": bits_context, "n_tokens_in_context": n_context,
            })
        del model
        if DEVICE == "mps":
            torch.mps.empty_cache()
        print(f"Done: {model_name} ({len(items)} items)", file=sys.stderr)

    if not rows:
        print("No models loaded successfully -- nothing to write.", file=sys.stderr)
        return

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
