# Results Summary — USCF Opponent Scouting System

## Final headline numbers

All metrics on the **same 1,278-row chronological held-out test set**
(July 2025 → Oct 2025). Tuning done with `TimeSeriesSplit(n_splits=5)`
+ `GridSearchCV` inside the 70% training pool.

| Model                                | CV AUC | Val AUC | **Test AUC** | Test F1 | Test Acc | Test Prec | Test Rec |
| :----------------------------------- | -----: | ------: | -----------: | ------: | -------: | --------: | -------: |
| Logistic Regression (benchmark)      | 0.7924 |  0.8187 |    **0.8246** | 0.7254  | 0.7340   | 0.7138    | 0.7373   |
| Random Forest (tuned)                | 0.7869 |  0.8169 |    **0.8234** | 0.7364  | 0.7379   | 0.7069    | 0.7685   |
| Gradient Boosting (tuned)            | 0.7871 |  0.8159 |    **0.8248** | 0.7367  | 0.7410   | 0.7145    | 0.7603   |

Per-fold AUC std ≤ 0.015 for all three → results are stable.

## Best hyperparameters

| Model              | Best params (from 5-fold time-series GridSearch)                                            |
| :----------------- | :------------------------------------------------------------------------------------------- |
| Logistic Regression | `C=0.1`, `penalty=l2`, `solver=lbfgs`                                                       |
| Random Forest       | `n_estimators=200`, `max_depth=8`, `min_samples_split=2`, `min_samples_leaf=4`              |
| Gradient Boosting   | `n_estimators=100`, `learning_rate=0.05`, `max_depth=3`                                     |

## Feature importance (Random Forest)

| Rank | Feature                                  | Importance |
| :--: | :--------------------------------------- | ---------: |
|  1   | `rating_diff`                            |      0.550 |
|  2   | `opponent_pre_rating`                    |      0.174 |
|  3   | `player_pre_rating`                      |      0.092 |
|  4   | `player_recent_avg_opponent_rating_90d`  |      0.048 |
|  5   | `player_recent_win_rate_90d`             |      0.045 |
|  6   | `player_games_last_365d`                 |      0.035 |
|  7   | `player_games_last_90d`                  |      0.027 |
|  8   | `player_games_last_30d`                  |      0.018 |

Recency features contribute ~13% of total tree importance, validating
the Glicko / "rust" hypothesis from the literature review.

## Extra-credit GenAI head-to-head

| Approach                                                   | Test AUC | Test F1 | Test Acc |
| :--------------------------------------------------------- | -------: | ------: | -------: |
| Elo zero-shot formula (what an LLM would derive)           |   0.8250 |  0.7429 |   0.7097 |
| Tuned Random Forest (full ML pipeline)                     |   0.8234 |  0.7364 |   0.7379 |
| Tuned Gradient Boosting (full ML pipeline)                 |   0.8248 |  0.7367 |   0.7410 |

Conclusion: the 70-year-old Elo formula and a modern tuned ML pipeline
are statistically indistinguishable on this task. The ML pipeline wins
on threshold-sensitive metrics (Accuracy, Precision) thanks to learned
non-linear interactions, while Elo wins on Recall by being more
aggressive about predicting wins for higher-rated players.

## Reproducing

```bash
# Repair the data (one-time, after parser bug fix):
python scripts/18_reparse_from_cache.py

# Full pipeline:
python scripts/03_clean_and_engineer.py      # cleaning -> features_v1.csv
python scripts/04_eda_and_benchmark_prep.py  # EDA plots + correlation matrix
python scripts/12_add_recency_features.py    # rolling features -> features_v3_recency.csv
python scripts/16_kfold_cv_tuning.py         # TimeSeriesSplit + GridSearchCV for 3 models
python scripts/17_genai_comparison.py        # extra-credit GenAI baseline
```
