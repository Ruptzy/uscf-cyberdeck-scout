# USCF Cyberdeck Scout — Submission Bundle

**Student:** Harold &nbsp;·&nbsp; **Project:** USCF Cyberdeck Scout
**Course:** ML Final Project &nbsp;·&nbsp; **Date:** May 2026
**Live Dashboard:** *(paste Streamlit Community Cloud URL here once deployed)*
**Repo:** <https://github.com/Ruptzy/uscf-cyberdeck-scout>

---

## Headline deliverables

| File | What it is |
| :--- | :--- |
| `USCF_Cyberdeck_Scout_Presentation.pptx` | 13-slide deck, ~10 min talk |
| `Presentation_Notes.docx` | Printable quick-reference card for the live talk |
| `USCF_Cyberdeck_Scout_SUBMISSION.zip` | Full bundle of code, data, and writeups (~0.53 MB) |
| The live Streamlit dashboard | 7-page interactive app — gets demoed at the end of the presentation |

---

## Rubric map — every section accounted for

| Rubric item (20% each) | Evidence in this bundle | Status |
| :--- | :--- | :---: |
| **1. Novel dataset (not Kaggle) + cleaning + EDA + feature redundancy** | `README.md`, `docs/DATA_FIX_NOTE.md`, `scripts/03_clean_and_engineer.py`, `scripts/04_eda_and_benchmark_prep.py`. Dataset scraped from USCF MSA (13,305 → 8,514 games). Parser-bug catch & repair documented. EDA charts + redundancy audit (`rating_diff` = `player_pre − opp_pre`). | ✅ |
| **2. Literature review** | `docs/LITERATURE_REVIEW.md` — 15 references covering Elo, Glicko, TrueSkill, DeepChess, Maharaj, momentum literature. | ✅ |
| **3. Benchmark model + correct metrics** | Logistic Regression benchmark with AUC, F1, Accuracy, Precision, Recall — `scripts/16_kfold_cv_tuning.py`, `outputs/models/kfold_cv_metrics.csv`. Balanced classes (48.8 / 51.2) so accuracy is informative; AUC + F1 reported as guards. | ✅ |
| **4. ML model + train/val/test + k-fold CV** | Random Forest + Gradient Boosting tuned via `GridSearchCV` + `TimeSeriesSplit(5)` on chronological 70 / 15 / 15 split. Metrics on all three splits in `kfold_cv_metrics.csv`. Per-fold stability < 0.015 AUC std-dev. | ✅ |
| **5. Final presentation (10–15 slides, 10 min, stakeholder framing)** | `USCF_Cyberdeck_Scout_Presentation.pptx` — 13 slides, cinematic + clean. `Presentation_Notes.docx` for live talk. Live dashboard demo follows the deck. | ✅ |
| **Extra credit (+10%) — Generative-AI end-to-end comparison** | `scripts/24_claude_inline_predictions.py` runs Claude Opus 4.7 over 60 sampled test rows. Per-row reasoning traces in `outputs/models/claude_inline_predictions.csv`. Results in `genai_comparison_metrics.csv`. Writeup in `docs/GENAI_COMPARISON.md`. | ✅ |

**Total locked in: 110 / 110.**

---

## Key numbers (memorize for Q&A)

| Model | CV AUC | Test AUC | F1 | Accuracy |
| :--- | ---: | ---: | ---: | ---: |
| Logistic Regression (benchmark) | 0.7924 | 0.8246 | 0.7254 | 0.7340 |
| Random Forest | 0.7869 | 0.8234 | 0.7364 | 0.7379 |
| Gradient Boosting | 0.7871 | **0.8248** | 0.7367 | 0.7410 |
| Elo zero-shot baseline | — | 0.8250 | 0.7429 | 0.7097 |
| Claude Opus 4.7 (LLM, n=60) | — | 0.7239 | **0.7463** | 0.7167 |

Held-out test set: **1,278 games** the model never saw. Per-fold AUC standard deviation ≤ **0.015**.

Feature importance: Rating ≈ **55%**, recency / activity ≈ **13%**, time-control + cold-start ≈ remainder.

---

## The data-quality saga (slide 4)

- During EDA, `rating_diff` correlated **−0.13** with winning (impossible for chess).
- Root cause: unanchored regex captured the first 4 digits of the 8-digit USCF ID as the "rating."
- Fix: anchored regex on `R:` token + re-parsed 4,292 cached HTML pages (no re-scraping).
- After fix: correlation **+0.49**, Elo AUC **0.37 → 0.825**, Random Forest AUC **0.67 → 0.82**.

---

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Browser opens at <http://localhost:8501>. See `DEPLOYMENT.md` for online hosting (Streamlit Community Cloud, Render, Hugging Face Spaces).

---

## What's in this bundle

```
docs/
  USCF_Cyberdeck_Scout_Presentation.pptx   ← THE DECK
  Presentation_Notes.docx                  ← print this for the talk
  USCF_Cyberdeck_Scout_SUBMISSION.zip      ← one-file submission
  LITERATURE_REVIEW.md
  RESULTS_SUMMARY.md
  GENAI_COMPARISON.md
  DATA_FIX_NOTE.md
  DASHBOARD_GUIDE.md
  PRESENTATION_OUTLINE.md
  SUBMISSION_README.md                     ← this file
app.py                                     ← Streamlit dashboard (7 pages)
requirements.txt + requirements-dashboard.txt + runtime.txt
scripts/
  03–04, 12, 16–22, 24                     ← data, model, scouting, geography, GenAI
src/parser/msa_parser.py                   ← fixed parser
data/processed/
  features_v3_recency.csv                  ← model input
  player_profiles.csv                      ← scouting layer
  player_scouting_scores.csv               ← 0–100 underrated score per player
  event_metadata.csv                       ← event geography
  player_travel_features.csv               ← travel distance per player
outputs/models/
  kfold_cv_*.csv                           ← model metrics
  genai_comparison_metrics.csv             ← Elo + Claude vs ML
  claude_inline_predictions.csv            ← 60 per-row LLM reasoning traces
  best_gb_model.pkl                        ← persisted final model
```
