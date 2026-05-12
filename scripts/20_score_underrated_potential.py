"""
Phase 16: Underrated Potential Scoring (Scouting Layer)
=======================================================
Takes `data/processed/player_profiles.csv` and emits a 0-100
"Underrated Potential" score per player, with six interpretable
sub-scores and plain-English labels.

Design principles
-----------------
* Rule-based, no ML — every signal is auditable.
* Each sub-score is **damped by a sample-size factor** so small-sample
  flukes can't drive the headline number.
* If the dataset is too thin for a player, we return "Insufficient data"
  instead of a numeric score.
* Language is portfolio-safe: the primary label is "Underrated
  Potential."  We expose a playful `smurf_label` separately for tone.
"""

import os

import numpy as np
import pandas as pd


# Component weights — sum to 100
W_UPSET, W_FORM, W_MOMENT, W_SCHED, W_ACTIVE, W_VOL = 25, 20, 20, 15, 10, 10


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _sigmoid01(x, midpoint, slope):
    """Sigmoid mapping a value to [0,1]; midpoint maps to 0.5."""
    return 1.0 / (1.0 + np.exp(-slope * (x - midpoint)))


def _sample_size_multiplier(n_games_90d, n_career):
    """Damp scores when the underlying sample is thin.
    1.0 = full credit, 0.0 = essentially no signal."""
    factor_recent = min(1.0, n_games_90d / 10.0) if n_games_90d is not None else 0.0
    factor_career = min(1.0, n_career / 20.0) if n_career is not None else 0.0
    return 0.5 * factor_recent + 0.5 * factor_career


def upset_component(row) -> tuple[float, str]:
    n_higher = (row.get("wins_vs_100_plus_higher", 0) or 0) \
        + ((row.get("draws_vs_200_plus_higher", 0) or 0))
    if pd.isna(row.get("score_rate_vs_higher_rated")) or n_higher == 0:
        return 0.0, "Insufficient data vs stronger opponents"
    sr = float(row["score_rate_vs_higher_rated"])
    raw = _sigmoid01(sr, midpoint=0.35, slope=10.0)
    score = W_UPSET * raw
    if sr >= 0.50:
        label = "Dangerous against higher-rated players"
    elif sr >= 0.35:
        label = "Competitive against stronger fields"
    else:
        label = "Struggles against stronger opponents"
    return float(score), label


def form_component(row) -> tuple[float, str]:
    if pd.isna(row.get("recent_win_rate_90d")):
        return 0.0, "No recent games"
    wr = float(row["recent_win_rate_90d"])
    delta = float(row.get("form_delta_90d") or 0.0)
    # base = how strong is recent absolute form?
    base = _clamp((wr - 0.4) / 0.4)  # 0.4->0, 0.8->1
    boost = _clamp((delta + 0.05) / 0.2)  # delta -0.05 -> 0, +0.15 -> 1
    raw = _clamp(0.6 * base + 0.4 * boost)
    score = W_FORM * raw
    if wr >= 0.65 and delta >= 0.05:
        label = "Hot streak"
    elif wr >= 0.55:
        label = "In form"
    elif wr <= 0.40:
        label = "Cold form"
    else:
        label = "Stable form"
    return float(score), label


def momentum_component(row) -> tuple[float, str]:
    """Rating-trend proxy (post-ratings not yet parsed — flagged)."""
    change_90 = row.get("rating_change_90d_proxy")
    change_180 = row.get("rating_change_180d_proxy")
    slope = row.get("rating_slope_180d_proxy")
    sigs = [v for v in (change_90, change_180) if v is not None and not pd.isna(v)]
    if not sigs:
        return 0.0, "No rating-trend data"
    # Score off both the recent delta and the slope direction.
    delta = float(np.mean(sigs))  # avg of available 90d/180d change
    # Map +100 pts in 90-180 days -> 1.0; -100 -> 0.0
    raw_delta = _clamp((delta + 100) / 200)
    raw_slope = 0.5
    if slope is not None and not pd.isna(slope):
        # slope is rating units per day.  +0.5/day = ~+45 in 90d -> ~1.0
        raw_slope = _clamp((float(slope) + 0.3) / 0.8)
    raw = 0.6 * raw_delta + 0.4 * raw_slope
    score = W_MOMENT * raw
    if delta >= 75:
        label = "Fast-rising"
    elif delta >= 25:
        label = "Trending up"
    elif delta <= -50:
        label = "Declining"
    else:
        label = "Stable"
    return float(score), label


def schedule_component(row) -> tuple[float, str]:
    pct_higher = row.get("pct_games_vs_higher_rated_90d")
    avg_diff = row.get("avg_rating_diff_recent_90d")
    if pd.isna(pct_higher):
        return 0.0, "No recent games"
    raw_pct = _clamp(float(pct_higher) / 0.7)  # 70%+ of games against higher = full credit
    raw_diff = 0.5
    if not pd.isna(avg_diff):
        # Avg playing against someone 150+ higher on avg -> full credit
        raw_diff = _clamp((-float(avg_diff)) / 150.0 + 0.5, 0.0, 1.0)
    raw = 0.6 * raw_pct + 0.4 * raw_diff
    score = W_SCHED * raw
    if pct_higher >= 0.6:
        label = "Battle-tested"
    elif pct_higher >= 0.35:
        label = "Mixed field"
    else:
        label = "Mostly plays weaker fields"
    return float(score), label


def activity_component(row) -> tuple[float, str]:
    g90 = row.get("games_last_90d", 0) or 0
    days_since = row.get("days_since_last_game")
    if days_since is not None and not pd.isna(days_since) and days_since > 365:
        return 0.0, "Long inactive"
    raw = _clamp(float(g90) / 15.0)  # 15+ games in 90d = full credit
    score = W_ACTIVE * raw
    if g90 >= 15:
        label = "Very active"
    elif g90 >= 5:
        label = "Active"
    elif g90 >= 1:
        label = "Light activity"
    else:
        label = "Rust risk"
    return float(score), label


def volatility_component(row) -> tuple[float, str]:
    """Volatility carries underrated signal: a player with high
    result-volatility and a positive form_delta is more likely a
    breakout candidate than someone who is flat-bad."""
    rv = row.get("result_volatility")
    delta = row.get("form_delta_90d") or 0.0
    if pd.isna(rv):
        return 0.0, "Insufficient game history"
    rv = float(rv)
    # High volatility + positive delta = high score.
    raw_v = _clamp(rv / 0.5)
    if delta < 0:
        raw_v *= 0.4  # volatility without upward delta is just messy
    score = W_VOL * raw_v
    if rv >= 0.40 and delta >= 0.05:
        label = "Volatile — boom potential"
    elif rv >= 0.40:
        label = "Volatile"
    elif rv >= 0.25:
        label = "Mid consistency"
    else:
        label = "Consistent"
    return float(score), label


def bucket_label(score: float) -> str:
    if score < 30:
        return "Rating looks accurate"
    if score < 55:
        return "Normal — no strong signal"
    if score < 70:
        return "Watchlist — possibly underrated"
    if score < 85:
        return "Strong underrated signal"
    return "Very likely playing above rating"


def smurf_label(score: float) -> str:
    if score < 30: return "🟢 Honest rating"
    if score < 55: return "⚪ Looks fair"
    if score < 70: return "🟡 Possibly sandbagging"
    if score < 85: return "🟠 Likely underrated"
    return "🔴 Smurf alert"


def score_player(row) -> dict:
    if row.get("small_sample_warning"):
        return {
            "player_id": row["player_id"],
            "underrated_score": None,
            "bucket_label": "Insufficient data",
            "smurf_label": "⚪ Insufficient data",
            "upset_score": 0.0, "form_score": 0.0, "momentum_score": 0.0,
            "schedule_score": 0.0, "activity_score": 0.0, "volatility_score": 0.0,
            "upset_label": "N/A", "form_label": "N/A", "momentum_label": "N/A",
            "schedule_label": "N/A", "activity_label": "N/A", "volatility_label": "N/A",
            "sample_size_multiplier": 0.0,
            "highlight_signals": "Needs more recent games before scouting score is meaningful.",
        }

    upset, upset_lab = upset_component(row)
    form, form_lab = form_component(row)
    moment, moment_lab = momentum_component(row)
    sched, sched_lab = schedule_component(row)
    active, active_lab = activity_component(row)
    vol, vol_lab = volatility_component(row)

    mult = _sample_size_multiplier(row.get("games_last_90d", 0) or 0,
                                   row.get("career_n_games", 0) or 0)
    total_raw = upset + form + moment + sched + active + vol
    total = total_raw * mult

    # Build "Why this player might be dangerous" — top contributing signals
    contribs = sorted(
        [("Upset", upset, upset_lab),
         ("Form", form, form_lab),
         ("Momentum", moment, moment_lab),
         ("Schedule", sched, sched_lab),
         ("Activity", active, active_lab),
         ("Volatility", vol, vol_lab)],
        key=lambda t: t[1], reverse=True,
    )
    bullets = []
    # Numeric specifics — wording reflects that USCF crosstables aren't true
    # prize standings; we say "top crosstable score" / "approximate top-5".
    if (row.get("events_won_last_90d") or 0) >= 1:
        bullets.append(
            f"Posted top crosstable score in {int(row['events_won_last_90d'])} "
            f"event(s) over the last 90 days."
        )
    if (row.get("top_5_finishes_last_365d") or 0) >= 3:
        bullets.append(
            f"Approximate top-5 finish (by displayed crosstable order) in "
            f"{int(row['top_5_finishes_last_365d'])} events over the last 365 days."
        )
    if row.get("rating_change_180d_proxy") and row["rating_change_180d_proxy"] >= 50:
        bullets.append(f"Gained +{int(row['rating_change_180d_proxy'])} rating points over the last 180 days (proxy).")
    if row.get("score_rate_vs_higher_rated") and row["score_rate_vs_higher_rated"] >= 0.5:
        bullets.append(
            f"Scored {row['score_rate_vs_higher_rated']*100:.0f}% against opponents rated 100+ points higher."
        )
    if row.get("best_time_control_by_score_rate"):
        bullets.append(f"Performs best in {row['best_time_control_by_score_rate']} events.")
    if (row.get("games_last_90d") or 0) >= 15:
        bullets.append(f"Very active — {int(row['games_last_90d'])} games in the last 90 days.")
    # Fill from top components if we don't have enough specifics
    for name, val, lab in contribs:
        if len(bullets) >= 5:
            break
        if val >= 5:
            bullets.append(f"{name}: {lab}.")
    if not bullets:
        bullets.append("No standout scouting signals.")
    bullets = bullets[:5]

    return {
        "player_id": row["player_id"],
        "underrated_score": round(float(total), 1),
        "raw_score_pre_damping": round(float(total_raw), 1),
        "sample_size_multiplier": round(mult, 2),
        "bucket_label": bucket_label(total),
        "smurf_label": smurf_label(total),
        "upset_score": round(upset, 1), "upset_label": upset_lab,
        "form_score": round(form, 1), "form_label": form_lab,
        "momentum_score": round(moment, 1), "momentum_label": moment_lab,
        "schedule_score": round(sched, 1), "schedule_label": sched_lab,
        "activity_score": round(active, 1), "activity_label": active_lab,
        "volatility_score": round(vol, 1), "volatility_label": vol_lab,
        "highlight_signals": " ".join(f"• {b}" for b in bullets),
    }


def main():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    in_path = os.path.join(base, "data", "processed", "player_profiles.csv")
    out_path = os.path.join(base, "data", "processed", "player_scouting_scores.csv")
    if not os.path.exists(in_path):
        print(f"Missing {in_path} — run scripts/19_build_player_profiles.py first.")
        return

    print("=" * 60)
    print("PHASE 16: UNDERRATED POTENTIAL SCORING")
    print("=" * 60)

    df = pd.read_csv(in_path)
    rows = [score_player(r) for _, r in df.iterrows()]
    out = pd.DataFrame(rows)

    # Reorder columns
    cols = ["player_id", "underrated_score", "bucket_label", "smurf_label",
            "upset_score", "form_score", "momentum_score",
            "schedule_score", "activity_score", "volatility_score",
            "upset_label", "form_label", "momentum_label",
            "schedule_label", "activity_label", "volatility_label",
            "sample_size_multiplier", "raw_score_pre_damping",
            "highlight_signals"]
    out = out[[c for c in cols if c in out.columns]]
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out)} players)")

    # Top 10 by score (drop smurf_label column from console print — Windows cp1252
    # console can't render the emoji; the CSV keeps it).
    top = out.dropna(subset=["underrated_score"]).sort_values("underrated_score", ascending=False).head(10)
    print("\n--- Top 10 underrated-potential players ---")
    print(top[["player_id", "underrated_score", "bucket_label"]].to_string(index=False))

    print("\n--- Bucket distribution ---")
    print(out["bucket_label"].value_counts().to_string())
    print("=" * 60)


if __name__ == "__main__":
    main()
