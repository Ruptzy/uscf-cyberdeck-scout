"""
Phase 17: Matchup Prediction & Report (Product Layer)
=====================================================
Two entry points:

* `python scripts/21_generate_matchup_report.py --player <ID> --opponent <ID>`
  Prints a plain-English matchup report and writes JSON/Markdown copies.

* `from scripts._matchup import predict_matchup, build_matchup_report`
  Programmatic API used by `app.py`.

Implementation notes
--------------------
* We fit the best Gradient Boosting hyperparameters (chosen by k-fold
  CV in phase 12) on the FULL train+val pool and persist the model to
  `outputs/models/best_gb_model.pkl` so the dashboard never has to
  retrain on every page load.  The held-out test split is preserved
  for honesty — we never refit including test rows.
* `predict_matchup` accepts any (player_id, opponent_id) pair: if the
  player is in our profiles, we use their real recency stats; if not,
  the row is imputed with the training-set medians and `missing_*`
  flags are set so the model knows to discount recency.
"""

import argparse
import json
import os
import sys
import pickle
import textwrap

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer


BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE, "data", "processed", "features_v3_recency.csv")
PROFILES_PATH = os.path.join(BASE, "data", "processed", "player_profiles.csv")
SCORES_PATH = os.path.join(BASE, "data", "processed", "player_scouting_scores.csv")
MODEL_PATH = os.path.join(BASE, "outputs", "models", "best_gb_model.pkl")

NUM_FEATURES = [
    "player_pre_rating",
    "opponent_pre_rating",
    "rating_diff",
    "player_games_last_30d",
    "player_games_last_90d",
    "player_games_last_365d",
    "player_recent_avg_opponent_rating_90d",
    "player_recent_win_rate_90d",
    "missing_recent_avg_opp_90d",
    "missing_recent_win_rate_90d",
]
TC_OPTIONS = ["Blitz", "Quick", "Regular", "Unknown"]


def _prepare_training_frame():
    df = pd.read_csv(DATA_PATH)
    df["event_end_date"] = pd.to_datetime(df["event_end_date"], errors="coerce")
    df = df.sort_values(by=["event_end_date", "game_id"]).reset_index(drop=True)
    df["missing_recent_avg_opp_90d"] = df["player_recent_avg_opponent_rating_90d"].isna().astype(int)
    df["missing_recent_win_rate_90d"] = df["player_recent_win_rate_90d"].isna().astype(int)
    return df


def _one_hot_tc(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df, columns=["time_control"], drop_first=True)


def train_and_persist(force: bool = False):
    """Train the chosen GBM on train+val (85% chronological pool) and
    save it.  Skip if a fresh model already exists unless force=True."""
    if not force and os.path.exists(MODEL_PATH):
        return load_model()

    print(f"[train] (re)training best GBM and saving to {MODEL_PATH}")
    df = _prepare_training_frame()
    n = len(df)
    val_end = int(n * 0.85)
    train_pool = df.iloc[:val_end].copy()

    X_pool = _one_hot_tc(train_pool[NUM_FEATURES + ["time_control"]])
    y_pool = train_pool["target_binary"].astype(int)

    imputer = SimpleImputer(strategy="median")
    impute_cols = ["player_recent_avg_opponent_rating_90d", "player_recent_win_rate_90d"]
    X_pool[impute_cols] = imputer.fit_transform(X_pool[impute_cols])

    # Use the best params from the k-fold CV phase
    model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42,
    )
    model.fit(X_pool, y_pool)

    bundle = {
        "model": model,
        "imputer": imputer,
        "impute_cols": impute_cols,
        "feature_columns": X_pool.columns.tolist(),
        "num_features": NUM_FEATURES,
    }
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    return bundle


def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _profile_for(player_id: str | int):
    """Return the player_profiles row for an ID, or None if absent."""
    if not os.path.exists(PROFILES_PATH):
        return None
    df = pd.read_csv(PROFILES_PATH, dtype={"player_id": str})
    sub = df[df["player_id"] == str(player_id)]
    if sub.empty:
        return None
    return sub.iloc[0].to_dict()


def _scouting_for(player_id: str | int):
    if not os.path.exists(SCORES_PATH):
        return None
    df = pd.read_csv(SCORES_PATH, dtype={"player_id": str})
    sub = df[df["player_id"] == str(player_id)]
    if sub.empty:
        return None
    return sub.iloc[0].to_dict()


def predict_matchup(
    player_id: str | int,
    opponent_id: str | int,
    time_control: str = "Regular",
    player_rating_override: float | None = None,
    opponent_rating_override: float | None = None,
) -> dict:
    """Return a dict with win probability + Elo baseline + feature row.

    Handles three sources of rating, in order of preference:
    1. Explicit override passed by the caller.
    2. `current_rating` from `player_profiles.csv`.
    3. NaN — model will use imputed median (only reasonable for
       opponents not present in our scouted population).
    """
    bundle = train_and_persist(force=False)
    model = bundle["model"]
    imputer = bundle["imputer"]
    feature_columns = bundle["feature_columns"]

    p_prof = _profile_for(player_id) or {}
    o_prof = _profile_for(opponent_id) or {}

    player_rating = player_rating_override or p_prof.get("current_rating")
    opponent_rating = opponent_rating_override or o_prof.get("current_rating")
    if player_rating is None or opponent_rating is None:
        raise ValueError(
            "Need both player and opponent ratings.  Provide an override "
            "or use IDs present in player_profiles.csv."
        )

    rating_diff = float(player_rating) - float(opponent_rating)
    if time_control not in TC_OPTIONS:
        time_control = "Regular"

    # Pull recency from the player's profile if available
    def _get(k, default=np.nan):
        v = p_prof.get(k)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return default
        return v

    row = {
        "player_pre_rating": float(player_rating),
        "opponent_pre_rating": float(opponent_rating),
        "rating_diff": rating_diff,
        "player_games_last_30d": _get("games_last_30d", 0),
        "player_games_last_90d": _get("games_last_90d", 0),
        "player_games_last_365d": _get("games_last_365d", 0),
        "player_recent_avg_opponent_rating_90d": _get("avg_field_strength_last_365d", np.nan),
        "player_recent_win_rate_90d": _get("recent_win_rate_90d", np.nan),
    }
    row["missing_recent_avg_opp_90d"] = int(pd.isna(row["player_recent_avg_opponent_rating_90d"]))
    row["missing_recent_win_rate_90d"] = int(pd.isna(row["player_recent_win_rate_90d"]))
    row["time_control"] = time_control

    X = pd.DataFrame([row])
    X = _one_hot_tc(X)
    # Ensure all training columns exist
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_columns]
    # Impute recency cols
    X[bundle["impute_cols"]] = imputer.transform(X[bundle["impute_cols"]])

    p_win = float(model.predict_proba(X)[0, 1])

    # Elo baseline
    p_win_elo = 1.0 / (1.0 + 10.0 ** (-rating_diff / 400.0))

    return {
        "player_id": str(player_id),
        "opponent_id": str(opponent_id),
        "player_rating": float(player_rating),
        "opponent_rating": float(opponent_rating),
        "rating_diff": rating_diff,
        "time_control": time_control,
        "p_win_model": p_win,
        "p_win_elo": float(p_win_elo),
        "model_minus_elo": p_win - float(p_win_elo),
        "feature_row": row,
        "player_in_profiles": bool(p_prof),
        "opponent_in_profiles": bool(o_prof),
    }


def build_matchup_report(player_id, opponent_id, time_control="Regular",
                         player_rating_override=None, opponent_rating_override=None) -> dict:
    pred = predict_matchup(player_id, opponent_id, time_control,
                           player_rating_override, opponent_rating_override)
    p_prof = _profile_for(player_id) or {}
    o_prof = _profile_for(opponent_id) or {}
    p_score = _scouting_for(player_id) or {}
    o_score = _scouting_for(opponent_id) or {}

    def card(prof, score, label):
        if not prof:
            return {
                "label": label, "in_dataset": False,
                "summary": "Not in the scouted dataset — using rating only.",
            }
        return {
            "label": label,
            "in_dataset": True,
            "rating": prof.get("current_rating"),
            "career_games": prof.get("career_n_games"),
            "games_last_90d": prof.get("games_last_90d"),
            "recent_win_rate_90d": prof.get("recent_win_rate_90d"),
            "events_won_last_365d": prof.get("events_won_last_365d"),
            "top_5_finishes_last_365d": prof.get("top_5_finishes_last_365d"),
            "rating_change_180d_proxy": prof.get("rating_change_180d_proxy"),
            "best_time_control": prof.get("best_time_control_by_score_rate"),
            "home_region": prof.get("inferred_home_region"),
            "underrated_score": score.get("underrated_score"),
            "underrated_bucket": score.get("bucket_label"),
            "smurf_label": score.get("smurf_label"),
            "highlight_signals": score.get("highlight_signals"),
            "small_sample_warning": bool(prof.get("small_sample_warning")),
        }

    return {
        "prediction": {
            "p_win_model": pred["p_win_model"],
            "p_win_elo": pred["p_win_elo"],
            "rating_diff": pred["rating_diff"],
            "time_control": pred["time_control"],
            "model_vs_elo_disagreement_pp": (pred["p_win_model"] - pred["p_win_elo"]) * 100,
        },
        "player_card": card(p_prof, p_score, "Player (you)"),
        "opponent_card": card(o_prof, o_score, "Opponent"),
    }


def _pp(report: dict) -> str:
    """Plain-text rendering for CLI."""
    pred = report["prediction"]
    p = report["player_card"]
    o = report["opponent_card"]
    p_win = pred["p_win_model"] * 100
    p_elo = pred["p_win_elo"] * 100
    parts = []
    parts.append("=" * 64)
    parts.append("MATCHUP REPORT")
    parts.append("=" * 64)
    parts.append(f"Predicted win probability (model): {p_win:5.1f}%")
    parts.append(f"Predicted win probability (Elo)  : {p_elo:5.1f}%")
    parts.append(f"Rating diff (player - opponent)  : {pred['rating_diff']:+.0f}")
    parts.append(f"Time control                     : {pred['time_control']}")
    parts.append("")
    for card, key in [(p, "Player"), (o, "Opponent")]:
        parts.append("-" * 64)
        parts.append(f"{key} card — {card.get('label','')}")
        if not card.get("in_dataset"):
            parts.append("  (not in scouted dataset — using rating only)")
            continue
        parts.append(f"  Current rating       : {card.get('rating')}")
        parts.append(f"  Career games         : {card.get('career_games')}")
        parts.append(f"  Games last 90 days   : {card.get('games_last_90d')}")
        wr = card.get('recent_win_rate_90d')
        parts.append(f"  Recent 90d win rate  : {f'{wr*100:.0f}%' if pd.notna(wr) else 'n/a'}")
        parts.append(f"  Events won (365d)    : {card.get('events_won_last_365d')}")
        parts.append(f"  Top-5 finishes (365d): {card.get('top_5_finishes_last_365d')}")
        parts.append(f"  Rating change (180d) : {card.get('rating_change_180d_proxy')} (proxy)")
        parts.append(f"  Best time control    : {card.get('best_time_control')}")
        parts.append(f"  Home region          : {card.get('home_region')}")
        parts.append(f"  Underrated Potential : {card.get('underrated_score')} -> {card.get('underrated_bucket')}")
        sigs = card.get("highlight_signals")
        if sigs:
            parts.append("  Why this player is interesting:")
            for line in sigs.split("•"):
                if line.strip():
                    parts.append("    • " + line.strip())
        if card.get("small_sample_warning"):
            parts.append("  ⚠ Small-sample warning — scouting score may be unreliable.")
    parts.append("=" * 64)
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--time-control", default="Regular",
                        choices=TC_OPTIONS)
    parser.add_argument("--retrain", action="store_true",
                        help="Force re-training of the persisted GBM.")
    parser.add_argument("--player-rating", type=float, default=None)
    parser.add_argument("--opponent-rating", type=float, default=None)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    if args.retrain:
        train_and_persist(force=True)

    report = build_matchup_report(
        args.player, args.opponent, args.time_control,
        args.player_rating, args.opponent_rating,
    )

    try:
        print(_pp(report))
    except UnicodeEncodeError:
        # Windows cp1252 — strip any emoji and reprint
        print(_pp(report).encode("ascii", "ignore").decode("ascii"))

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
