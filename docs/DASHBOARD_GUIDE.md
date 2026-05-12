# USCF Opponent Scouting — Dashboard & Pipeline Guide

This guide explains how to run the full pipeline, launch the dashboard,
and capture screenshots for the final presentation.

---

## 1. One-time setup

```bash
cd "C:/Users/harol/Desktop/Machine Learning Final Project"
pip install streamlit plotly
```

The rest of the stack (`pandas`, `numpy`, `scikit-learn`, `beautifulsoup4`,
`requests`, `pyyaml`) is already in `requirements.txt`.

---

## 2. Run the full data pipeline

If you want to regenerate everything from the cached HTML (no
re-scraping):

```bash
python scripts/18_reparse_from_cache.py        # rebuild raw CSVs with fixed parser
python scripts/03_clean_and_engineer.py        # -> features_v1.csv
python scripts/04_eda_and_benchmark_prep.py    # EDA charts -> outputs/eda/
python scripts/12_add_recency_features.py      # -> features_v3_recency.csv
python scripts/16_kfold_cv_tuning.py           # TimeSeriesSplit + GridSearchCV
python scripts/17_genai_comparison.py          # Elo zero-shot baseline (+ optional Claude)
python scripts/19_build_player_profiles.py     # -> player_profiles.csv + event_*.csv
python scripts/20_score_underrated_potential.py # -> player_scouting_scores.csv
python scripts/21_generate_matchup_report.py --player <ID> --opponent <ID>
```

The scouting layer (`19` → `20` → `21`) reads from the model layer's
outputs but does **not** modify them. Steps `19`–`21` are safe to
re-run independently.

---

## 3. Launch the dashboard

```bash
streamlit run app.py
```

Default URL: <http://localhost:8501>

(If port 8501 is busy, use `--server.port 8765` or any free port.)

---

## 4. Dashboard tour (use these for the PowerPoint screenshots)

| Tab | What to capture |
| :--- | :--- |
| **Project Overview** | 4-metric strip + data-quality story. This is the headline slide. |
| **Model Results** | Headline comparison table with the green-gradient `test_auc` column + the box plot of per-fold validation AUC. |
| **Feature Intelligence** | Feature documentation table (filtered) + the horizontal feature-importance bar chart. |
| **Player Lookup** | Pick a top-rated player (e.g. `13215196` rated 2647). The radar chart of 6 subscores + the "Why this player might be dangerous" bullets are the most demo-worthy. |
| **Matchup Prediction** | Pick `15244943` (player) vs `13215196` (opponent), Time Control = Regular. Hit **Predict matchup**. Showcases the model–vs–Elo disagreement, the two side-by-side scouting cards, and the plain-English outlook. |

**Recommended PowerPoint screenshot set (one per slide):**

1. Project Overview — top metric strip + intro text.
2. The parser-bug story panel.
3. Model Results — headline table.
4. Model Results — per-fold box plot.
5. Feature Intelligence — feature-importance bar chart.
6. Player Lookup — radar chart + highlights for a "watchlist" player.
7. Matchup Prediction — full layout showing both cards and outlook text.

---

## 5. Where things live

```
data/
  raw/
    html_cache/                   # 4,292 cached USCF HTML pages
    tables/raw_games.csv          # post-fix parser output (13,305 rows)
    tables/raw_events.csv
    tables/raw_players.csv
  processed/
    features_v1.csv               # cleaned game-level
    features_v3_recency.csv       # cleaned + recency features (model input)
    event_player_scores.csv       # NEW: 104k (event, player) rows
    event_field_stats.csv         # NEW: 2,215 events with field metrics
    player_profiles.csv           # NEW: 179 players × ~60 scouting features
    player_scouting_scores.csv    # NEW: per-player 0-100 underrated score

outputs/
  eda/                            # EDA charts
  models/
    kfold_cv_metrics.csv          # model comparison
    kfold_cv_per_fold.csv
    kfold_cv_feature_importances.csv
    genai_comparison_metrics.csv  # Elo zero-shot vs ML
    genai_predictions.csv
    best_gb_model.pkl             # NEW: persisted GBM for the dashboard
  frozen/                         # snapshot from before the scouting layer

docs/
  LITERATURE_REVIEW.md
  PRESENTATION_OUTLINE.md         # rubric-aligned slide outline
  RESULTS_SUMMARY.md
  GENAI_COMPARISON.md
  DATA_FIX_NOTE.md                # the parser-bug story
  DASHBOARD_GUIDE.md              # (this file)

scripts/
  01–11        # raw collection + early modeling experiments
  12_add_recency_features.py
  13–14        # earlier RF / GBM with recency (pre-k-fold)
  15_batch_scale_collect.py
  16_kfold_cv_tuning.py           # final tuned models
  17_genai_comparison.py
  18_reparse_from_cache.py        # parser-bug repair
  19_build_player_profiles.py     # NEW
  20_score_underrated_potential.py # NEW
  21_generate_matchup_report.py    # NEW (CLI + dashboard backend)

app.py                            # NEW: Streamlit dashboard
```

---

## 6. Demo script (verbatim, for a 5-min walkthrough)

> "This is a USCF Opponent Scouting System. The data comes from
> public USCF tournament crosstables — 13,000 raw games scraped and
> cached, 8,500 clean games after a serious parser bug we caught and
> fixed. The fix moved test AUC from 0.67 to 0.82.
>
> The ML model — a tuned Gradient Boosting classifier — gets test AUC
> 0.825, which is essentially identical to the 70-year-old Elo formula.
> That's the *honest* story: ML matches Elo on this task. So we built
> a second layer on top of the model: a 0-to-100 Underrated Potential
> score that aggregates each player's activity, recent form,
> rating trend, schedule strength, and upset history.
>
> [Click Player Lookup tab.] Here's a top-rated player from our
> dataset. The radar chart shows their six scouting subscores.
> The 'Why this player might be dangerous' bullets are auto-generated
> from the strongest signals — won 2 events in 90 days, gained +126
> rating, 61% score rate against stronger fields.
>
> [Click Matchup Prediction tab.] Now if I'm playing this player,
> the system blends the model probability with the Elo baseline,
> shows me the rating gap, and gives me a side-by-side scouting card
> for both of us. The plain-English outlook tells me whether this is
> a coin flip, an upset opportunity, or a routine win — and warns me
> if the opponent's underrated score is high.
>
> The product value isn't predicting the winner — Elo does that.
> The product value is *explaining what kind of opponent this is*."

---

## 7. Known limitations & honest disclaimers (also in the dashboard)

| Limitation | Why | Mitigation |
| :--- | :--- | :--- |
| `rating_change_*` are proxies | `player_post_rating` is `Unknown` in raw CSV — never parsed | Reconstruct from chronological `pre_rating` values |
| No travel distance / event GPS | USCF MSA doesn't expose event coordinates | Stub fields as `None`; `inferred_home_region` derived from player's most common state |
| Most opponents are not scouted | The scraper started from 1 seed and only collected focal-player histories | Use cold-start flags + median imputation when opponent isn't in `player_profiles.csv` |
| 130 of 179 players flagged "Insufficient data" | Many "focal" players have very few recent games in the snapshot | Dashboard warns user; only ~30 players have full-confidence scouting |
| Underrated score is rule-based, not learned | No supervised target for "underrated-ness" without post-rating data | Trivially upgradable to a regressor predicting `rating_change_90d` once post-ratings are parsed |
