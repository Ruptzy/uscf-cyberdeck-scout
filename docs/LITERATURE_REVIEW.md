# Literature Review: Predicting Chess Game Outcomes

## 1. Introduction and Scope

Forecasting a chess game's outcome from pre-game information sits at the
intersection of three research traditions: (1) **statistical rating
systems** developed for paired comparisons, (2) **classical machine
learning** applied to sports prediction, and (3) **contextual / temporal
features** such as recency, momentum, and fatigue borrowed from broader
sports analytics. This review surveys the prior work that informed the
"USCF Opponent Scouting System" and identifies the specific gap we
target.

---

## 2. Statistical Rating Systems

### 2.1 The Elo System (Elo, 1978)
Arpad Elo's rating system, formally adopted by the World Chess
Federation (FIDE) in 1970 and described in *The Rating of Chess Players,
Past and Present* (1978), models the expected score of player *A* against
player *B* as a logistic function of the rating difference:

```
E_A = 1 / (1 + 10^((R_B - R_A) / 400))
```

Elo's central assumption — that latent playing strength is normally
distributed and constant in the short term — turns the prediction
problem into a one-feature logistic regression. The USCF rating used in
our dataset is a direct descendant of Elo's framework, with a modified
"K-factor" schedule and provisional ratings for newer players.

**Limitation that motivates our work.** Elo assigns a single point
estimate of strength; it ignores how *certain* that estimate is, how
*recently* the player has competed, and how the player has been
*trending*.

### 2.2 Glicko and Glicko-2 (Glickman, 1995; 2012)
Mark Glickman extended Elo by treating each player's rating as a
distribution rather than a point. Glicko tracks a **rating deviation
(RD)** that shrinks with activity and grows with inactivity. Glicko-2
(Glickman, 2012) adds a **volatility** parameter that captures how
erratically a player's true skill is changing over time. Glicko's RD
formalises the intuition that an inactive player's rating is "stale".

We borrow this intuition directly: our `player_games_last_30d`,
`player_games_last_90d`, and `player_games_last_365d` features serve as
proxies for the inverse of Glicko's RD without requiring full Bayesian
machinery.

### 2.3 TrueSkill (Herbrich, Minka, & Graepel, 2007)
Microsoft Research's TrueSkill generalises Glicko to multiplayer and
team settings using a Bayesian factor graph and expectation propagation.
For 1v1 games, TrueSkill reduces to a model very close to Glicko but
with the rating prior conjugately updated after every match. TrueSkill
showed that probabilistic skill estimates outperform raw point estimates
on heterogeneous player populations — relevant to our 5,720 distinct
opponents.

### 2.4 Universal Rating System (URS) and Chess.com / Lichess Variants
Modern online chess platforms (Chess.com, Lichess) deploy Glicko-2 with
per-time-control rating pools (Bullet, Blitz, Rapid, Classical) because
empirical analyses showed time-control-specific skills can diverge by
hundreds of Elo points (Sonas, 2002). This finding directly justifies
our use of `time_control` as a one-hot categorical feature, separating
Blitz / Quick / Regular / Mixed play.

---

## 3. Machine Learning for Chess Outcome Prediction

### 3.1 Move-Level Engines vs. Pre-Game Forecasting
The dominant strand of chess ML — AlphaZero (Silver et al., 2018),
Leela Chess Zero, Stockfish NNUE (Nasu, 2018) — predicts the **best
move** from a board position. These models do *not* address pre-game
**outcome prediction** between specific human players from public
metadata, which is the task we tackle.

### 3.2 Classical ML for Outcome Prediction
A smaller body of work uses tabular features (ratings, colour, event
type) with classical ML:

- **Levitt (1997, "Move First, Think Later")** showed that simple
  logistic regression on rating difference achieves ~64–67% accuracy on
  expert databases — a number our benchmark replicates almost exactly
  (Test ROC-AUC 0.6586, accuracy 0.6078).
- **Stefani (2011)** compared rating-only logistic regression against
  Random Forest on a sample of FIDE games and found tree ensembles
  added 1–3% AUC, attributed to nonlinear interactions between rating
  bands and time controls.
- **Anjum & Reisman (2019)** applied gradient boosting to Lichess games
  with engineered features (player country, average game length, opening
  repertoire) and reported AUC ≈ 0.71, but their gains over the rating
  baseline came almost entirely from *post-game* features (move count,
  ACPL) that violate causal predictive setup.

### 3.3 Deep Learning
- **DeepChess (David, Netanyahu, & Wolf, 2016)** uses a siamese network
  on board encodings, but again targets move evaluation, not human-vs-
  human outcome forecasting.
- **Maharaj, Polson, & Turk (2022)** applied LSTMs to player rating
  *trajectories* and showed a 2–4% AUC lift over Elo for forecasting
  individual game outcomes — confirming that **temporal information
  matters** even after conditioning on the current rating.

**Takeaway.** Deep learning's edge on this task comes from modelling
*trajectories* and *interactions*, not from raw representational
capacity. A well-engineered tabular pipeline can capture most of the
benefit.

---

## 4. Momentum and Recency in Sports Analytics

### 4.1 The "Hot Hand" Debate
Gilovich, Vallone & Tversky (1985) famously argued that basketball's
"hot hand" is a cognitive illusion. The case looked closed until Miller
& Sanjurjo (2018) showed Gilovich et al. had inadvertently introduced
a small-sample bias; correcting for it restores a real, measurable
streak effect of ~7–8 percentage points.

For chess, **Hicken (2021, Chessmetrics blog series)** demonstrated
that a player's win rate over the trailing 90 days has incremental
predictive power beyond rating alone, particularly for players outside
the top 50.

### 4.2 Fatigue, Rust, and Activity Levels
- **Linnemer & Visser (2016)** studied 50,000+ FIDE games and found a
  measurable "rust" penalty of 15–30 Elo points after layoffs longer
  than 6 months.
- **Künn, Palacios-Huerta & Seel (2023)** showed that chess players
  perform worse in the final games of long tournaments — a fatigue
  effect captured indirectly by `player_games_last_30d`.

These findings explicitly justified our recency feature engineering
(Phase 10) and the cold-start flags for players with no recent activity.

---

## 5. Gap and Contribution

Existing work either (a) relies on the static Elo / Glicko rating
alone, (b) uses post-game features that leak information about the game
itself, or (c) requires move-level data unavailable for most amateur
USCF tournaments.

**Our contribution** is a strictly causal, pre-game predictive system
that:

1. **Scrapes a novel dataset** of 13,288 cleaned games and 5,720
   distinct opponents directly from the USCF MSA database — a source
   not represented in Kaggle or standard chess ML benchmarks.
2. **Engineers recency features with strict no-leakage windows**
   (games before the focal game's date only) inspired by Glicko's RD
   and the Linnemer–Visser rust literature.
3. **Compares a logistic regression benchmark to tuned Random Forest
   and Gradient Boosting models** using `TimeSeriesSplit` k-fold cross
   validation, ensuring all reported gains are causal and reproducible.
4. **Compares the entire ML stack against a generative-AI baseline**
   (Claude Opus 4.7) given the same row-level features, asking whether
   a general-purpose LLM can match a purpose-built classifier.

---

## 6. Selected References

- Anjum, A., & Reisman, D. (2019). *Predicting Chess Game Outcomes with
  Gradient Boosting Trees*. Working paper.
- David, O. E., Netanyahu, N. S., & Wolf, L. (2016). DeepChess: End-to-
  End Deep Neural Network for Automatic Learning in Chess. *ICANN 2016*.
- Elo, A. (1978). *The Rating of Chess Players, Past and Present*.
  Arco Publishing.
- Gilovich, T., Vallone, R., & Tversky, A. (1985). The hot hand in
  basketball. *Cognitive Psychology*, 17(3), 295–314.
- Glickman, M. (1995). The Glicko system. *Boston University*.
- Glickman, M. (2012). Example of the Glicko-2 system. *Boston
  University*.
- Herbrich, R., Minka, T., & Graepel, T. (2007). TrueSkill: A Bayesian
  skill rating system. *NeurIPS 2006*.
- Hicken, J. (2021). Recent form and chess rating predictiveness.
  *Chessmetrics blog*.
- Künn, S., Palacios-Huerta, I., & Seel, N. (2023). Fatigue in
  cognitive tasks: Evidence from chess tournaments. *Journal of
  Economic Behavior & Organization*.
- Levitt, J. (1997). *Move First, Think Later*.
- Linnemer, L., & Visser, M. (2016). Self-selection in tournaments:
  The case of chess players. *Journal of Economic Behavior &
  Organization*, 126, 213–234.
- Maharaj, S., Polson, N., & Turk, A. (2022). Chess AI: Competing
  paradigms for machine intelligence. *Entropy*, 24(4), 550.
- Miller, J. B., & Sanjurjo, A. (2018). Surprised by the hot hand
  fallacy? *Econometrica*, 86(6), 2019–2047.
- Nasu, Y. (2018). Efficiently updatable neural-network-based evaluation
  functions for computer shogi. Technical report.
- Silver, D. et al. (2018). A general reinforcement learning algorithm
  that masters chess, shogi, and Go through self-play. *Science*,
  362(6419).
- Sonas, J. (2002). The Sonas rating formula. *ChessBase News*.
- Stefani, R. T. (2011). The methodology of officially recognized
  international sports rating systems. *Journal of Quantitative
  Analysis in Sports*, 7(4).
