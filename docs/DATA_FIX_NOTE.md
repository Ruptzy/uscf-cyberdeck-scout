# Data Fix Note — Parser Rating Regex Bug

## Symptom
During exploratory data analysis we observed that `rating_diff`
correlated **negatively** (r = −0.13) with `target_binary`.  This is
mathematically impossible for a well-formed chess dataset — higher-
rated players should win more, not less.

Even more concretely, our Elo zero-shot baseline on the held-out test
set scored **AUC 0.37** — *worse than random* — when calibrated to a
correctly-coded rating_diff.

## Root cause
In `src/parser/msa_parser.py`, the `parse_crosstable` function captured
the player's pre-game rating with this regex:

```python
rating_match = re.search(r'(?:R:)?\s*(\d{3,4})', id_rating_str)
```

The string it operates on looks like:
```
14821464 / R: 2589   ->2597
```
The regex's `(?:R:)?` group is **optional** and `\s*` is **non-greedy
about anchoring**, so the engine matched the **first** 3-4 digit run —
which is the first 4 digits of the 8-digit USCF ID (`1482`), not the
actual rating (`2589`).

Result: every parsed game stored the *first four digits of the player's
USCF ID* in `player_pre_rating` and `opponent_pre_rating`.

## Fix
Anchor the regex on the literal `R:` token so the rating is always
captured from the field that explicitly labels it:

```python
rating_match = re.search(r'R:\s*(\d{3,4}P?\d*)', id_rating_str)
```

We also preserve the optional provisional suffix (e.g. `1450P12`) for
the downstream cleaner to strip.

## Verification
- Before fix: `rating_diff` corr with win = −0.13
- After fix:  `rating_diff` corr with win = **+0.49**
- Before fix: Elo zero-shot test AUC = 0.37 (worse than random)
- After fix:  Elo zero-shot test AUC = **0.825**
- Before fix: Tuned RF test AUC = 0.674
- After fix:  Tuned RF test AUC = **0.823**

## Re-running

```bash
# 1. Re-parse the entire local HTML cache (no network calls):
python scripts/18_reparse_from_cache.py

# 2. Re-run cleaning, EDA, recency engineering, k-fold CV, and GenAI:
python scripts/03_clean_and_engineer.py
python scripts/04_eda_and_benchmark_prep.py
python scripts/12_add_recency_features.py
python scripts/16_kfold_cv_tuning.py
python scripts/17_genai_comparison.py
```

## Takeaway
Always sanity-check feature correlations against domain knowledge
before trusting a model.  A regex bug in the data-collection layer
made the strongest known predictor in chess look worse than random,
which we would never have caught from accuracy alone — the original
ML model still got ~0.59 accuracy on the held-out test set, which
*looks* fine if you don't know what good looks like.
