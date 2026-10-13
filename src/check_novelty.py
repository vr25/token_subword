"""
Empirically verifies the "zero pretraining frequency" / non-word claim in Methods,
rather than just asserting it: checks all 100 pseudowords (50 pairs x 2 conditions)
against a real English dictionary (macOS /usr/share/dict/words, Webster's Second,
~236k entries) for exact matches (case-insensitive).
"""
import csv

PAIRS_PATH = "/Users/vipularawte/token_exp/data/pseudoword_pairs.csv"
DICT_PATH = "/usr/share/dict/words"
OUT_PATH = "/Users/vipularawte/token_exp/results/novelty_check.txt"


def main():
    with open(DICT_PATH, encoding="latin-1") as f:
        dictionary = set(line.strip().lower() for line in f if line.strip())

    with open(PAIRS_PATH) as f:
        items = list(csv.DictReader(f))

    lines = []
    collisions = []
    for item in items:
        w = item["pseudoword"].lower()
        if w in dictionary:
            collisions.append(item)

    # Also check for duplicate pseudowords across the dataset (uniqueness sanity check)
    words = [item["pseudoword"] for item in items]
    dupes = set(w for w in words if words.count(w) > 1)

    lines.append(f"Checked {len(items)} pseudowords against {len(dictionary)}-entry English dictionary "
                 f"({DICT_PATH}).")
    lines.append(f"Real-word collisions found: {len(collisions)}")
    for c in collisions:
        lines.append(f"  COLLISION: pair {c['pair_id']} ({c['condition']}): '{c['pseudoword']}'")
    lines.append(f"Duplicate pseudowords within dataset: {len(dupes)} -- {sorted(dupes) if dupes else 'none'}")
    if not collisions and not dupes:
        lines.append("PASS: all 100 pseudowords are non-words (no dictionary matches) and all unique.")

    text = "\n".join(lines)
    print(text)
    with open(OUT_PATH, "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
