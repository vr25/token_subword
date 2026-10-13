"""
Builds data/pseudoword_pairs.csv from first principles: each item is defined by its
illegal cluster + rime + carrier sentence template, and the initial/final pseudowords
are *derived* (cluster+rime, rime+cluster) rather than transcribed by hand. This is the
onset-rime swap geometry, applied programmatically so a transcription slip (like the
original Pair 1) can't happen silently.
"""
import csv

# (pair_id, category, cluster, rime, carrier_template)
# carrier_template has a single {w} placeholder for the target pseudoword.
ITEMS = [
    (1, "stop_fricative_stop", "bzv", "im", "The artisan crafted the raw {w}."),
    (2, "stop_fricative_stop", "pvk", "on", "The chef prepared a rare {w}."),
    (3, "stop_fricative_stop", "dbv", "ax", "He dropped the heavy {w}."),
    (4, "stop_fricative_stop", "bzv", "ek", "The child pointed at the {w}."),
    (5, "stop_fricative_stop", "pvk", "is", "They painted the old {w}."),
    (6, "stop_fricative_stop", "dbv", "om", "She stepped over the {w}."),
    (7, "stop_fricative_stop", "bzv", "un", "We noticed a sudden {w}."),
    (8, "stop_fricative_stop", "pvk", "et", "The machine produced a {w}."),
    (9, "stop_fricative_stop", "dbv", "il", "The storm destroyed the {w}."),
    (10, "stop_fricative_stop", "bzv", "og", "He found a broken {w}."),

    (11, "nasal_fricative_liquid", "mzv", "al", "The uniform was a bright {w}."),
    (12, "nasal_fricative_liquid", "nvz", "it", "The guide pointed out the {w}."),
    (13, "nasal_fricative_liquid", "mvr", "ux", "She cleaned the dusty {w}."),
    (14, "nasal_fricative_liquid", "mzv", "en", "They discovered a deep {w}."),
    (15, "nasal_fricative_liquid", "nvz", "op", "The liquid filled the {w}."),
    (16, "nasal_fricative_liquid", "mvr", "ig", "He carried the fragile {w}."),
    (17, "nasal_fricative_liquid", "mzv", "od", "We looked at the ancient {w}."),
    (18, "nasal_fricative_liquid", "nvz", "um", "The doctor examined the {w}."),
    (19, "nasal_fricative_liquid", "mvr", "af", "The map showed a tiny {w}."),
    (20, "nasal_fricative_liquid", "mzv", "yk", "She wore a beautiful {w}."),

    (21, "voiceless_stop", "ptk", "am", "He tripped over a loose {w}."),
    (22, "voiceless_stop", "ktp", "in", "The box contained a heavy {w}."),
    (23, "voiceless_stop", "tkp", "os", "The hunter tracked the {w}."),
    (24, "voiceless_stop", "ptk", "ed", "The document was a brief {w}."),
    (25, "voiceless_stop", "ktp", "ug", "She drank from the clean {w}."),
    (26, "voiceless_stop", "tkp", "en", "They threw away the old {w}."),
    (27, "voiceless_stop", "ptk", "if", "The surface was a smooth {w}."),
    (28, "voiceless_stop", "ktp", "ol", "He looked through the {w}."),
    (29, "voiceless_stop", "tkp", "ub", "The sound came from the {w}."),
    (30, "voiceless_stop", "ptk", "ax", "She mended the torn {w}."),

    (31, "voiced_liquid_guttural", "gzn", "if", "The label read as {w}."),
    (32, "voiced_liquid_guttural", "vzm", "ok", "The texture felt like {w}."),
    (33, "voiced_liquid_guttural", "lbz", "at", "The creature made a {w}."),
    (34, "voiced_liquid_guttural", "gzn", "ep", "The artist drew a simple {w}."),
    (35, "voiced_liquid_guttural", "vzm", "id", "He stepped onto the {w}."),
    (36, "voiced_liquid_guttural", "lbz", "un", "The container held the {w}."),
    (37, "voiced_liquid_guttural", "gzn", "ov", "We entered a quiet {w}."),
    (38, "voiced_liquid_guttural", "vzm", "ag", "The recipe required a {w}."),
    (39, "voiced_liquid_guttural", "lbz", "ek", "She found a shiny {w}."),
    (40, "voiced_liquid_guttural", "gzn", "ul", "The signal showed a {w}."),

    (41, "complex_fricative_plosive", "sfp", "om", "The worker fixed the {w}."),
    (42, "complex_fricative_plosive", "zvd", "it", "The device gave a loud {w}."),
    (43, "complex_fricative_plosive", "fvk", "un", "They harvested the ripe {w}."),
    (44, "complex_fricative_plosive", "sfp", "ex", "The mirror reflected a {w}."),
    (45, "complex_fricative_plosive", "zvd", "ag", "The driver turned the {w}."),
    (46, "complex_fricative_plosive", "fvk", "is", "He cleaned out the wet {w}."),
    (47, "complex_fricative_plosive", "sfp", "ob", "The report outlined the {w}."),
    (48, "complex_fricative_plosive", "zvd", "um", "She sat near the warm {w}."),
    (49, "complex_fricative_plosive", "fvk", "en", "They reached a rocky {w}."),
    (50, "complex_fricative_plosive", "sfp", "il", "The pattern displayed a {w}."),
]

assert len(ITEMS) == 50, f"expected 50 items, got {len(ITEMS)}"


def build_rows():
    rows = []
    for pair_id, category, cluster, rime, template in ITEMS:
        initial_word = cluster + rime
        final_word = rime + cluster
        rows.append({
            "pair_id": pair_id,
            "category": category,
            "cluster": cluster,
            "rime": rime,
            "condition": "word_initial",
            "pseudoword": initial_word,
            "carrier_sentence": template.format(w=initial_word),
        })
        rows.append({
            "pair_id": pair_id,
            "category": category,
            "cluster": cluster,
            "rime": rime,
            "condition": "word_final",
            "pseudoword": final_word,
            "carrier_sentence": template.format(w=final_word),
        })
    return rows


def validate(rows):
    by_pair = {}
    for r in rows:
        by_pair.setdefault(r["pair_id"], {})[r["condition"]] = r["pseudoword"]
    for pair_id, conds in by_pair.items():
        wi, wf = conds["word_initial"], conds["word_final"]
        assert sorted(wi) == sorted(wf), f"pair {pair_id}: letters differ between conditions ({wi} vs {wf})"
        assert len(wi) == len(wf), f"pair {pair_id}: length mismatch"
    # spot-check the two examples disputed earlier in the session
    p2 = by_pair[2]
    assert p2["word_initial"] == "pvkon" and p2["word_final"] == "onpvk"
    p49 = by_pair[49]
    assert p49["word_initial"] == "fvken" and p49["word_final"] == "enfvk"
    p1 = by_pair[1]
    assert p1["word_initial"] == "bzvim" and p1["word_final"] == "imbzv", "Pair 1 fix did not apply"


def main():
    rows = build_rows()
    validate(rows)
    out_path = "/Users/vipularawte/token_exp/data/pseudoword_pairs.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "pair_id", "category", "cluster", "rime", "condition", "pseudoword", "carrier_sentence"
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows ({len(rows)//2} pairs) to {out_path}")
    print("Validation passed: letters match within pair, Pair 1 fix applied, Pair 2/49 match expected forms.")


if __name__ == "__main__":
    main()
