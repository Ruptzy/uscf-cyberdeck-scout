# USCF Cyberdeck Scout

> Opponent scouting + game-outcome prediction built on a custom-scraped
> USCF tournament dataset. Cyberdeck/HUD-styled Streamlit dashboard.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

## What it does

- **Predicts** the probability that a focal USCF player wins a tournament
  game against a given opponent, comparing a tuned Gradient Boosting
  classifier against the closed-form Elo baseline.
- **Scouts** any player in the dataset with a 0–100 *Underrated Potential*
  score broken into six interpretable subscores
  (upset history, recent form, rating momentum, schedule strength,
  activity, volatility).
- **Explains** the matchup in plain English: predicted P(win) + Elo
  delta + side-by-side dossier cards + "why this player might be
  dangerous" bullets.

Built on a **novel, non-Kaggle dataset** of 13,305 raw games / 8,514
clean rows scraped from public USCF MSA tournament crosstables.

## Numbers (held-out 1,278-game test set)

| Model | CV AUC | Test AUC | Test F1 | Test Acc |
| :--- | ---: | ---: | ---: | ---: |
| Logistic Regression (benchmark) | 0.7924 | 0.8246 | 0.7254 | 0.7340 |
| Random Forest                   | 0.7869 | 0.8234 | 0.7364 | 0.7379 |
| Gradient Boosting               | 0.7871 | 0.8248 | 0.7367 | 0.7410 |
| Elo zero-shot baseline          | —      | 0.8250 | 0.7429 | 0.7097 |

Honest framing: **ML matches Elo on this task** (a 70-year-old domain
baseline is shockingly hard to beat). Product value lives in the
scouting layer that explains *what kind of opponent* you're facing.

## Quick start (local)

```bash
pip install -r requirements-dashboard.txt
streamlit run app.py
```

App opens at <http://localhost:8501>.

## Quick start (Streamlit Community Cloud)

1. Fork or push this repo to your GitHub.
2. Go to <https://share.streamlit.io> → **New app**.
3. Point at this repo, set:
   - **Main file path**: `app.py`
   - **Python version**: 3.11
4. Click Deploy.

Full deployment instructions including alternative hosts (Render,
Hugging Face Spaces, Railway) live in [DEPLOYMENT.md](DEPLOYMENT.md).

## Dashboard pages

| # | Page | What's on it |
| :- | :--- | :--- |
| 01 | **COMMAND CENTER** | Mission brief, dataset metrics, data-quality alert teaser. |
| 02 | **MODEL INTEL** | Comparison table (3 ML models + Elo), per-fold box plot. |
| 03 | **FEATURE VECTORS** | Plain-English feature docs + Random Forest importance bar chart. |
| 04 | **PLAYER DOSSIER** | Per-player scouting card — activity / form / upset / momentum / travel. |
| 05 | **UNDERRATED PROTOCOL** | 0–100 gauge, 6 subscores, "Why this player might be dangerous" bullets. |
| 06 | **MATCHUP SIM** | Model P(win) vs Elo + side-by-side dossier cards + plain-English outlook. |
| 07 | **DATA REPAIR LOG** | The parser-bug story: how a regex error made the strongest chess predictor look worse than random. |

## Architecture

```
data/processed/                  # the only data the deployed app needs
  features_v3_recency.csv        # cleaned game-level (~2 MB)
  player_profiles.csv            # one row per scouted player
  player_scouting_scores.csv     # 0-100 underrated scores + subscores
  event_metadata.csv             # 3,020 events × location/geo
  player_travel_features.csv     # haversine travel distances
outputs/models/
  kfold_cv_metrics.csv           # for the Model Intel tab
  kfold_cv_feature_importances.csv
  genai_comparison_metrics.csv   # Elo zero-shot baseline
  best_gb_model.pkl              # persisted Gradient Boosting classifier
app.py                           # 7-page Streamlit dashboard
scripts/19_build_player_profiles.py     # rebuilds player_profiles.csv
scripts/20_score_underrated_potential.py # rebuilds scouting scores
scripts/21_generate_matchup_report.py    # CLI report + dashboard backend
scripts/22_extract_event_geography.py    # event metadata + offline geocoding
```

The deployed app never scrapes USCF live — everything reads from the
committed CSVs and the pickled model.

## Honest caveats

- **`rating_change_*` are proxies** (reconstructed from chronological
  pre-rating values). True post-rating-based momentum requires
  parsing `player_post_rating`, which is a future TODO.
- **Travel distances are approximate** — USCF MSA does not expose
  event GPS coordinates, so we geocode at city or state centroid
  level. Each player's travel-distance card shows the confidence.
- **Crosstable order ≠ official tiebreak/prize order.** US Chess
  sorts crosstables by score group then post-event rating. The
  dashboard always uses placement-safe language ("top crosstable
  score", "approximate top-5 finish") rather than "official 2nd
  place" or "podium".
- **130 of 179 players are flagged "Insufficient data"** because the
  seed-based crawl yielded many opponents with few games. Full-
  confidence scouting is available for ~30 players; the dashboard
  shows small-sample warnings everywhere this matters.

## License & use

Educational / portfolio project. Data is scraped from public USCF
MSA pages and cached locally per the §2-second polite-delay policy
during collection. The deployed app is read-only against precomputed
CSVs and **never** scrapes USCF at runtime.
