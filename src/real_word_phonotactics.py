"""
Real-word validation study (reframed centerpiece per COLING theme fit): computes a real,
corpus-derived phonotactic probability proxy (positional biphone frequency, the same
construct Vitevitch & Luce use, computed here on the CMU Pronouncing Dictionary -- a
standard public phonemic corpus) for real English words, then tests whether the SAME
directional relationship found for constructed pseudowords (word-initial markedness ->
fewer tokens/lower surprisal; word-final markedness -> more tokens/higher surprisal)
generalizes to naturally occurring words with continuous (not binary) markedness.

This is a genuinely different, complementary design from the main pseudoword study:
continuous phonotactic rarity in real words, not categorical illegal/legal clusters in
invented ones. Whatever direction it shows is reported -- no assumed outcome.
"""
import csv
import re
import math
from collections import Counter
import nltk

nltk.download("cmudict", quiet=True)
from nltk.corpus import cmudict

OUT_PATH = "/Users/vipularawte/token_exp/data/real_words.csv"

STRESS_RE = re.compile(r"\d")


def strip_stress(phones):
    return [STRESS_RE.sub("", p) for p in phones]


def main():
    d = cmudict.dict()

    with open("/usr/share/dict/words", encoding="latin-1") as f:
        # keep only entries stored in lowercase in the source dictionary -- excludes
        # proper nouns/names (e.g. "Johan", "Shirley" appear capitalized and are dropped)
        real_words = set(line.strip() for line in f if line.strip() and line.strip().islower())

    # one pronunciation per word, alphabetic-only, reasonable length (match pseudoword scale),
    # AND verified as a genuine dictionary word (cmudict alone includes many proper nouns/names
    # that are not real English words, e.g. "ahab", "erany" -- checked and excluded here)
    entries = {}
    for word, prons in d.items():
        if not word.isalpha():
            continue
        if not (4 <= len(word) <= 9):
            continue
        if word not in real_words:
            continue
        phones = strip_stress(prons[0])
        if len(phones) < 4:
            continue
        entries[word] = phones

    print(f"Usable dictionary entries: {len(entries)}")

    initial_biphone_counts = Counter()
    final_biphone_counts = Counter()
    for word, phones in entries.items():
        initial_biphone_counts[tuple(phones[:2])] += 1
        final_biphone_counts[tuple(phones[-2:])] += 1

    n_words = len(entries)
    rows = []
    for word, phones in entries.items():
        ib = tuple(phones[:2])
        fb = tuple(phones[-2:])
        ib_freq = initial_biphone_counts[ib]
        fb_freq = final_biphone_counts[fb]
        rows.append({
            "word": word,
            "phones": " ".join(phones),
            "initial_biphone": "-".join(ib),
            "final_biphone": "-".join(fb),
            "initial_biphone_freq": ib_freq,
            "final_biphone_freq": fb_freq,
            "initial_biphone_log_freq": math.log(ib_freq),
            "final_biphone_log_freq": math.log(fb_freq),
            "n_phones": len(phones),
        })

    # Sample: take a stratified spread across the full range of initial/final biphone
    # frequency (not just extremes) for a proper continuous regression, capped for
    # tractable surprisal compute cost.
    rows.sort(key=lambda r: r["initial_biphone_log_freq"])
    n_target = 200
    step = max(1, len(rows) // n_target)
    sample = rows[::step][:n_target]

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sample[0].keys()))
        writer.writeheader()
        writer.writerows(sample)
    print(f"Wrote {len(sample)} real-word rows to {OUT_PATH}")
    print(f"initial_biphone_log_freq range: {min(r['initial_biphone_log_freq'] for r in sample):.2f} "
          f"to {max(r['initial_biphone_log_freq'] for r in sample):.2f}")
    print(f"final_biphone_log_freq range: {min(r['final_biphone_log_freq'] for r in sample):.2f} "
          f"to {max(r['final_biphone_log_freq'] for r in sample):.2f}")


if __name__ == "__main__":
    main()
