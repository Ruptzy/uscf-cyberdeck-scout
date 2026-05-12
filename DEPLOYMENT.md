# Deployment Guide — USCF Cyberdeck Scout

This dashboard is designed to deploy to any free Python web host with
minimal config. It loads **only** precomputed CSVs and a pickled model
— it **never** scrapes USCF at runtime.

## 1. Local run

```bash
pip install -r requirements-dashboard.txt
streamlit run app.py
```

App opens at <http://localhost:8501>.

## 2. What must be committed for deployment

```
app.py
requirements-dashboard.txt
runtime.txt                              # optional — pins Python (3.11+ recommended)
scripts/21_generate_matchup_report.py    # imported lazily by app.py
src/parser/msa_parser.py                 # fixed parser, imported indirectly
src/__init__.py                          # package marker
src/parser/__init__.py                   # package marker
data/processed/
  features_v3_recency.csv                # ~2 MB — training data for live model retrain
  player_profiles.csv                    # ~100 KB — scouting features
  player_scouting_scores.csv             # ~30 KB — underrated potential scores
outputs/models/
  kfold_cv_metrics.csv
  kfold_cv_per_fold.csv
  kfold_cv_feature_importances.csv
  genai_comparison_metrics.csv
  best_gb_model.pkl                      # ~100 KB pickled GBM
docs/                                    # optional, referenced from dashboard captions
```

Total deploy payload: **~3 MB**. Comfortably fits any free tier.

## 3. What must NOT be committed

```
data/raw/                                # 4,292 HTML files (~150+ MB)
data/raw/tables/                         # raw scraper CSVs (regenerable)
outputs/eda/                             # PNG charts (regenerable)
outputs/frozen/                          # local snapshot
__pycache__/, .ipynb_checkpoints/, .DS_Store
*.env, *.envrc, .streamlit/secrets.toml  # secrets (none required, but excluded for safety)
```

A suggested `.gitignore`:

```
__pycache__/
*.pyc
data/raw/
outputs/eda/
outputs/frozen/
.env
.envrc
.streamlit/secrets.toml
```

## 4. Deployment targets

### Option A — Streamlit Community Cloud  (recommended)

1. Push the repo to GitHub.
2. Visit <https://share.streamlit.io> → **New app** → point at the repo.
3. Set:
   - **Main file path**: `app.py`
   - **Python version**: 3.11
   - **Requirements file**: `requirements-dashboard.txt`
4. Click Deploy. First boot takes 2-3 minutes (installs deps + caches data).

No environment variables needed. No secrets.

### Option B — Hugging Face Spaces

1. Create a new Space → SDK = **Streamlit**.
2. Upload the deploy payload from §2.
3. Spaces auto-detects `requirements-dashboard.txt` if it's renamed to
   `requirements.txt`, or set the file in `README.md` YAML front matter:

   ```yaml
   ---
   app_file: app.py
   pinned: false
   ---
   ```

### Option C — Render

1. New → **Web Service** → connect GitHub repo.
2. Build command:
   ```
   pip install -r requirements-dashboard.txt
   ```
3. Start command:
   ```
   streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```
4. Plan: free tier is fine. Disk: ephemeral is fine — all data is in-repo.

### Option D — Railway

Similar to Render. Use:

```
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

as the start command.

## 5. Optional: `runtime.txt`

Pin Python for hosts that respect it (Render, HF Spaces):

```
python-3.11
```

## 6. Sanity checks before deploy

Run these locally to confirm the deploy bundle is self-sufficient:

```bash
# 1. Regenerate the model from features (in case best_gb_model.pkl is stale)
python scripts/21_generate_matchup_report.py --player 15244943 --opponent 13215196 --retrain

# 2. Boot streamlit and probe
streamlit run app.py --server.port 8765 &
sleep 5 && curl -fsS http://localhost:8765/_stcore/health   # -> "ok"
```

## 7. What the deployed app will NOT do

* Will **not** scrape USCF MSA (the scraper config is disabled in the dashboard runtime).
* Will **not** retrain the model on every page load (uses cached pickle, retrains only if missing).
* Will **not** require any API key (the optional Claude API path in
  `scripts/17_genai_comparison.py` is dev-only, not invoked from `app.py`).
* Will **not** write to disk except for the pickled model file (auto-created on first run).

## 8. Performance notes

* All CSV loads use `@st.cache_data`. Subsequent page navigations are
  instantaneous.
* The Gradient Boosting model is loaded once via `@st.cache_resource`.
* Plotly figures are rendered client-side — no server CPU cost after
  first paint.
* Cold-start time on Streamlit Community Cloud is ~10–20 s (mostly
  installing scikit-learn).

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
| :--- | :--- | :--- |
| App boots but shows "kfold_cv_metrics.csv missing" | `outputs/models/` not committed | Add CSVs from §2 to the repo |
| Matchup tab errors on Predict | `best_gb_model.pkl` missing AND `features_v3_recency.csv` missing | Commit at least one — the matchup module retrains lazily if only the CSV is present |
| Sidebar font renders as default sans-serif | Google Fonts blocked by network | Cosmetic only — app remains functional |
| Plotly charts render dark on dark (unreadable) | Browser dark-mode override | Streamlit theme defaults to dark; force light mode in `.streamlit/config.toml` if needed |

## 10. Quick `.streamlit/config.toml` (optional)

```toml
[theme]
base = "dark"
primaryColor = "#7AF7FF"
backgroundColor = "#120812"
secondaryBackgroundColor = "#1A0B14"
textColor = "#D8FFFF"
font = "monospace"
```

Commit this if you want the Streamlit-Cloud dark theme to match the
in-app Cyberdeck styling without relying solely on injected CSS.
