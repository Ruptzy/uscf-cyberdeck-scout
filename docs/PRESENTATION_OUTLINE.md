# Presentation Outline — USCF Opponent Scouting System
*13 slides, ~10 minutes, senior-stakeholder framing*

---

## Slide 1 — Title & Hook
**Title.** *"Can we predict who wins a chess game before it's played?"*

**Sub-title.** *A supervised ML system trained on 13,305 scraped USCF
tournament games.*

**Hook (verbal).** "Every chess federation in the world has been doing
this since 1970 with a single equation — Arpad Elo's. We asked: can a
modern ML stack with player momentum features actually beat it? The
answer surprised us."

---

## Slide 2 — Why this matters
- **For players & coaches.** Pre-game scouting: identify opponents
  whose recent form differs from their rating.
- **For tournament organisers.** Better pairing and seeding decisions.
- **For betting / fantasy markets.** Calibrated win probabilities for a
  $1B+ chess content economy (Lichess, Chess.com, Twitch).
- **For the ML community.** A clean case study in *whether engineered
  domain features beat a strong closed-form baseline.*

---

## Slide 3 — The data: novel, scraped, our own
- **Source.** USCF MSA member-detail and crosstable pages.
- **Collection.** Custom Python scraper (BeautifulSoup + requests),
  HTML cache for reproducibility, 2-second polite delay.
- **Volume.** 4,292 cached pages → **13,305 raw game rows** → **8,514
  modeling rows** after cleaning.
- **Coverage.** 179 focal players × 4,141 distinct opponents across
  3,020 tournament events from 2001-03 to 2025-10.
- **Why novel?** USCF data is not on Kaggle and not in standard chess
  ML benchmarks (which use Lichess / FICS dumps).

---

## Slide 4 — Data cleaning & the bug we caught
> *Real-world data is messier than it looks.*

- During EDA we noticed `rating_diff` was **negatively** correlated with
  winning (r = −0.13).  That's mathematically impossible if the rating
  field is correct.
- Root cause: the original parser regex `\d{3,4}` greedy-matched the
  first 4 digits of the 8-digit USCF ID rather than the rating after
  the literal `R:` token.
- Fix: anchor the regex on `R:` and re-parse from the cached HTML.
  After the fix, `rating_diff` correlates **+0.49** with winning — the
  expected chess signal.
- **Lesson for stakeholders.** Always sanity-check your feature
  correlations against domain intuition before believing your model.

---

## Slide 5 — EDA highlights
*[Embed: `outputs/eda/04_rating_diff_hist.png` + `07_rating_diff_by_result.png`]*

- **Win/Draw/Loss split:** 48.8% / 19.9% / 31.3% (well-balanced for
  binary classification: 48.8% Win vs 51.2% Not-Win).
- **Rating range:** 227 → 2,861 (children's USCF ratings to GM level).
- **`rating_diff` is unimodal, roughly symmetric** around zero with
  long tails — Elo theory's normality assumption holds reasonably well.
- **Redundancy audit.** `rating_diff = player_pre − opponent_pre`
  exactly, so we never use all three in a linear model. For trees, we
  keep both views since splits benefit from level + difference.

---

## Slide 6 — Literature review (one slide, three bullets)
*Full review in `docs/LITERATURE_REVIEW.md`*
- **Elo (1978).** Closed-form logistic on rating difference. Still the
  industry baseline 50 years later.
- **Glickman 1995–2012 / TrueSkill 2007.** Add uncertainty (RD) and
  volatility around each rating. Our recency features approximate this.
- **Sports analytics 2018–2023.** Miller–Sanjurjo restored "hot hand";
  Künn et al. quantified chess fatigue; Maharaj et al. showed temporal
  trajectories beat static ratings on individual game forecasting.

---

## Slide 7 — Benchmark & metrics choice
- **Task framing.** Binary classification: did the focal player win?
  (Draws ⇒ 0.)
- **Class balance.** 48.8% / 51.2%, so accuracy is informative — but
  we report **ROC-AUC + F1 + Precision + Recall** to be safe.
- **Benchmark model.** Logistic Regression on `rating_diff` +
  `time_control` one-hot.
- **Why this benchmark?** It is literally what FIDE has used for
  decades. If our complex models can't beat it, the complexity isn't
  earning its keep.

---

## Slide 8 — Feature engineering: recency & momentum
- We added five **strictly leakage-controlled** rolling features:
  `games_last_30d`, `games_last_90d`, `games_last_365d`,
  `recent_avg_opponent_rating_90d`, `recent_win_rate_90d`.
- Rule: for every game on date *t*, only games on dates *strictly < t*
  for that same focal player feed the rolling stats.
- Cold-start cases (45% of rows have zero games in the prior 30 days)
  are explicitly flagged with binary indicators instead of being silently
  imputed.

---

## Slide 9 — K-fold cross-validation methodology
> *Time-respecting CV, not vanilla KFold.*
- Chronological 70 / 15 / 15 train / val / test split (2001-03 → 2025-10).
- Inside the 70% training pool we run `TimeSeriesSplit` with 5 folds —
  each fold's validation block sits **strictly after** its training
  block, mirroring deployment.
- `GridSearchCV` over the model-specific grids; refit on the full
  training set with the best params; report metrics on the untouched
  validation and test sets.
- Per-fold AUC std is **≤ 0.015** for all three models → results are
  stable, not lucky.

---

## Slide 10 — Results (the headline table)

| Model                | CV AUC | Val AUC | **Test AUC** | Test F1 | Test Acc |
| :------------------- | -----: | ------: | -----------: | ------: | -------: |
| Logistic (benchmark) | 0.792  |   0.819 |    **0.825** |   0.725 |    0.734 |
| Random Forest        | 0.787  |   0.817 |    **0.823** |   0.736 |    0.738 |
| Gradient Boosting    | 0.787  |   0.816 |    **0.825** |   0.737 |    0.741 |

- All three models converge on ~0.82–0.83 test AUC.
- Tree ensembles win **F1 by ~1 pt** thanks to non-linear rating ×
  recency interactions, but **AUC is essentially tied**.
- Feature importances on RF: `rating_diff` 55%, ratings level 26%,
  recency features 13%, time-control dummies 6%.

---

## Slide 11 — Surprising finding: complexity ≈ baseline
- The benchmark **logistic regression matches or beats** the tuned
  ensembles on AUC, only losing 1 pt of F1.
- *Why?* Once we use `rating_diff`, the prediction problem is almost
  one-dimensional — the Elo formula is already a logistic of that
  single feature. There's not much non-linearity for trees to exploit.
- **Stakeholder takeaway.** When a 70-year-old closed-form model is
  already near-optimal, the path forward is *more data* and *richer
  features* (openings, colour, prior head-to-head), not bigger models.

---

## Slide 12 — Extra credit: Generative AI head-to-head

| Approach                              |   AUC | F1    |
| :------------------------------------ | ----: | ----- |
| Elo zero-shot formula (what an LLM would derive)             | 0.825 | 0.743 |
| Tuned Random Forest (our pipeline)                            | 0.823 | 0.736 |
| Tuned Gradient Boosting (our pipeline)                        | 0.825 | 0.737 |

- Evaluated on the **same** 1,278-row chronological held-out test set.
- A general-purpose AI that simply applies the Elo formula is
  **statistically indistinguishable** from our purpose-built ML
  pipeline on this dataset.
- The ML pipeline does win on **F1 calibration at the 0.5 threshold**
  for certain rating bands — but not on overall ranking quality (AUC).
- Optional path: `python scripts/17_genai_comparison.py --api-sample 200`
  to additionally probe Claude Opus 4.7 when an API key is available.

---

## Slide 13 — Conclusions & next steps
**What we proved.**
1. Pre-game USCF outcome prediction is solidly **above** chance:
   AUC 0.825 / Accuracy 74%.
2. **Recency features add real but modest signal** (~10–13% of tree
   importance) — confirming the Glicko / Linnemer–Visser intuition.
3. **Elo is brutally hard to beat** on its home turf. Modern ML
   matches, doesn't dominate.
4. **Generative AI ≈ specialised ML** on this task, because the task
   reduces to a one-equation problem that LLMs encode natively.

**What we'd do next with 3 more months.**
- Pull move-by-move PGNs and add opening-repertoire features.
- Per-time-control models (separate Blitz / Quick / Regular).
- Bayesian rating drift (Glicko-2) instead of frozen pre-game ratings.
- Calibration plots & decision-curve analysis for downstream betting use.

**One-line take-away.** *"With careful feature engineering and honest
time-series CV, we built a system that matches what FIDE has used since
1970 — and learned exactly why that's hard to beat."*
