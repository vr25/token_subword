# Tokenization and positional phonotactic asymmetry

Research code, datasets, and saved results for measuring tokenizer segmentation
and language-model surprisal in constructed pseudowords and real English words.

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