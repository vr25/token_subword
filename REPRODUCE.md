# Reproducing this project

Written 2026-09-27 so the pipeline can be re-run from scratch without Claude Code.
All scripts use hardcoded absolute paths (`/Users/vipularawte/token_exp/...`), so run
everything on this machine, from any cwd, with `src/` reachable on `sys.path`
(scripts that import from `segmentation_study`/`surprisal_study` do
`sys.path.insert(0, "/Users/vipularawte/token_exp/src")` themselves, but run `python
src/<script>.py` from the repo so relative imports between sibling scripts resolve).

## Dependencies

pandas, numpy, scipy, statsmodels, torch, transformers, tiktoken, nltk (+ `cmudict`
corpus via `nltk.download("cmudict")`), matplotlib. Paper build uses `tectonic`
(LaTeX). Some models in `surprisal_study.py`/`surprisal_study_extra.py`
(Llama-3-8B, Gemma-2) are HF-gated — need `huggingface-cli login` first, and the
scripts skip them gracefully (with a stderr warning) if auth is missing.

## Pipeline order

1. **Build the dataset**
   `python src/build_dataset.py` → `data/pseudoword_pairs.csv` (50 pairs, onset-rime
   swap geometry, derived programmatically from cluster+rime+carrier-template triples
   in the script, not hand-transcribed).

2. **Tokenizer segmentation (main pseudoword set)**
   `python src/segmentation_study.py` → `results/segmentation.csv`
   (no model weights needed — tiktoken + HF tokenizers only).

3. **Real-word validation set + its segmentation**
   `data/real_words.csv` already exists (200 real English words). Then:
   `python src/real_word_segmentation.py` → `results/real_word_segmentation.csv`.

4. **Surprisal (real forward passes)**
   `python src/surprisal_study.py` → `results/surprisal.csv`
   (gpt2 family, pythia suite, Mistral-7B-v0.3; gated models skipped).
   `python src/surprisal_study_extra.py` — appends Llama-3-8B/Gemma-2 rows to the
   same `results/surprisal.csv` once HF-gated access is available.
   `python src/real_word_surprisal.py` → `results/real_word_surprisal.csv`
   (isolated-word surprisal only, no carrier sentences for this set; excludes
   Gemma-2-9b, uses Gemma-2-2b instead).

5. **Novelty check** (verifies pseudowords aren't real words)
   `python src/check_novelty.py` → `results/novelty_check.txt`
   (needs `/usr/share/dict/words`, macOS-only path).

6. **Stats / regressions** (run in any order once steps 1-4 are done):
   - `python src/aggregate_segmentation.py` → segmentation summary (paired tests per
     tokenizer, reads `results/segmentation.csv`)
   - `python src/surprisal_stats.py` → `results/surprisal_summary.csv` (paired tests
     per model, isolated + in-context)
   - `python src/mechanistic_regression.py` → `results/mechanistic.csv` (decomposes
     token-count difference into per-boundary BPE merge savings)
   - `python src/mixed_effects_stats.py` → `results/mixed_effects_summary.txt`
     (cluster + pair-nested-in-cluster random effects, across both datasets)
   - `python src/interaction_test.py` → `results/interaction_test_summary.txt`
     (LR test: does the condition effect size vary by tokenizer/model?)
   - `python src/real_word_phonotactics.py` — computes CMU-dict-derived positional
     biphone frequency for `data/real_words.csv` (needs `nltk.download("cmudict")`
     first)
   - `python src/real_word_regression.py` → `results/real_word_regression_summary.txt`
   - `python src/real_word_surprisal_regression.py` →
     `results/real_word_surprisal_regression_summary.txt`
   - `python src/robustness_checks.py` → `results/robustness_checks_summary.txt`
     (FDR correction, condition×category interaction, leave-one-cluster-out)

7. **Appendix tables**
   `python src/gen_appendix_tables.py` → writes all `.tex` fragments into
   `paper/tables/` from the result CSVs (zero hand-transcription).

8. **Figures** (each is standalone, reads from `results/`/`data/`, writes to
   `paper/figures/*.pdf` + `.png`):
   - `python viz/fig0_graphical_abstract.py` — schematic crossover plot (illustrative,
     not real effect sizes)
   - `python viz/fig1_segmentation.py` — compact mean token count by condition × tokenizer
   - `python viz/fig2_surprisal.py` — mean surprisal by condition × model, isolated vs
     in-context panels
   - `python viz/fig3_realword.py` — real-word surprisal vs biphone rarity, pooled
     z-scored across models, with initial/final panels side by side
   - `python viz/fig4_tokenization_diagram.py` — visual token-split diagram for one
     representative pair

9. **Compile the paper**
   `bash paper/build.sh` (wraps `tectonic main.tex`, reports `main.pdf` success/fail).

## Notes for later you
- All figure/table scripts are idempotent and safe to re-run individually once their
  upstream CSV exists — you don't need to redo steps 2-4 (the slow ones) just to
  regenerate a figure, only if the underlying data changed.
- `data/pseudoword_pairs.csv` and `data/real_words.csv` are already committed data —
  step 1 only needs to be rerun if you're regenerating the dataset itself.
- No visualizations were generated in this Claude Code session per your request —
  use the `viz/*.py` scripts above directly when you're ready.
