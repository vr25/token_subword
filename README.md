# Tokenization and positional phonotactic asymmetry

Research code, datasets, and saved results for measuring tokenizer segmentation
and language-model surprisal in constructed pseudowords and real English words.
The manuscript and all existing paper assets are excluded.

## Contents

- `src/`: dataset construction, tokenization, surprisal, statistics, and table-generation scripts.
- `viz/`: figure-generation scripts (code only).
- `data/`: 50 pseudoword pairs (100 strings) and 200 real English words.
- `results/`: saved CSV measurements and text analysis summaries, without run logs.
- `REPRODUCE.md`: the original detailed pipeline instructions.
- `requirements.txt`: Python dependencies, not a version-pinned environment.

## Setup and portability

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import nltk; nltk.download('cmudict')"
```

The scripts are preserved as originally used and contain the absolute project
path `/Users/vipularawte/token_exp`. Before running on another machine, replace
that prefix in `src/*.py` and `viz/*.py` with the absolute path to your checkout.
Run scripts from the project root.

`check_novelty.py` and `real_word_phonotactics.py` also use the macOS dictionary
at `/usr/share/dict/words`. Supply the equivalent dictionary path if needed;
changing the wordlist may change regenerated data. The supplied datasets and
results can be analyzed without rebuilding the datasets or downloading model
weights.

Some Hugging Face models require approved access and authentication. Surprisal
inference uses Apple MPS when available, otherwise CPU, and larger models require
substantial memory. Model weights and credentials are not included.

For example, after updating paths:

```bash
python src/aggregate_segmentation.py
python src/surprisal_stats.py
python src/real_word_regression.py
```

See `REPRODUCE.md` for the remaining pipeline. Its manuscript compilation step
does not apply to this code-only distribution. To run visualization scripts,
create `paper/figures/` locally first; table-generation code creates
`paper/tables/` itself. These are generated output directories, not included
paper files.

Unzip this archive and upload the extracted contents to GitHub so that the code
and data can be browsed directly.
