"""
Phase 15: Build Player Profiles (Scouting Layer)
================================================
Produces three artifacts:

1. `data/processed/event_player_scores.csv` — one row per (event, player)
   with total_score, games_played, finish_rank, finish_percentile, state.
   Built by re-parsing the cached crosstable HTML in-place (NO network).
2. `data/processed/event_field_stats.csv` — one row per event with the
   field's avg / median / max rating, number of players, etc.
3. `data/processed/player_profiles.csv` — ONE ROW PER PLAYER with every
   scouting feature requested in the project brief.

This script reads from existing processed and cached data only.  It
does NOT re-scrape USCF.

Travel/geography features are stubbed because USCF crosstables do not
expose event coordinates — see TODO note at the bottom of this file.
True rating_change_* features require parsed `player_post_rating`,
which is currently `Unknown` in the raw CSV.  We reconstruct momentum
from chronological `player_pre_rating` values as a best-effort proxy
and flag this clearly.
"""

import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup


# ----------------------------------------------------------------------
# Stage 1 — rich event field parser
# ----------------------------------------------------------------------
def parse_event_field(html: str, event_id: str):
    """Extract every player's pre-rating, state, total score, and games
    played from the <pre> crosstable block.  Returns a list of dicts."""
    soup = BeautifulSoup(html, "html.parser")
    pre = soup.find("pre")
    if not pre:
        return []

    lines = pre.text.split("\n")
    rows = []
    current = None

    for line in lines:
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue

        if parts[0].isdigit():
            # First line of a player block: pair_num | name | total_pts | rounds... | last
            try:
                total = float(parts[2])
            except ValueError:
                total = None
            results = parts[3:-1]
            games_played = sum(1 for r in results if r.strip())
            current = {
                "pair_num": parts[0],
                "name": parts[1],
                "total_score": total,
                "games_played": games_played,
            }
        elif current is not None and len(parts) >= 2 and "/" in parts[1]:
            # Second line: state | USCF_ID / R: rating  ->post_rating | title
            state = parts[0] if parts[0] else None
            id_rating = parts[1]
            id_match = re.search(r"(\d{8})", id_rating)
            rating_match = re.search(r"R:\s*(\d{3,4})", id_rating)
            uscf_id = id_match.group(1) if id_match else None
            try:
                pre_rating = int(rating_match.group(1)) if rating_match else None
            except (TypeError, ValueError):
                pre_rating = None

            if uscf_id and current.get("total_score") is not None:
                rows.append({
                    "event_id": event_id,
                    "uscf_id": uscf_id,
                    "state": state,
                    "pre_rating": pre_rating,
                    "total_score": current["total_score"],
                    "games_played": current["games_played"],
                    "pair_num": current["pair_num"],
                })
            current = None

    return rows


def rich_event_pass(cache_dir: str, raw_games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk the cached XtblMain HTML once per (event, focal player) and
    build per-event-player score rows + per-event field-stat rows."""
    # We have multiple crosstable files per event (one per focal player)
    # but all of them should yield the same full participant list.  De-dup
    # on event_id by picking the first file we see.
    pairs = raw_games[["event_id", "player_id"]].drop_duplicates()
    seen_events = set()
    all_player_rows = []

    print(f"Walking {pairs['event_id'].nunique()} unique events ...")
    for _, row in pairs.iterrows():
        event_id = str(row["event_id"])
        if event_id in seen_events:
            continue
        cache_path = os.path.join(cache_dir, f"XtblMain.php_{event_id}.0-{row['player_id']}.html")
        if not os.path.exists(cache_path):
            continue
        with open(cache_path, "r", encoding="utf-8", errors="replace") as f:
            html = f.read()
        event_rows = parse_event_field(html, event_id)
        if event_rows:
            all_player_rows.extend(event_rows)
            seen_events.add(event_id)

    df = pd.DataFrame(all_player_rows)
    print(f"  parsed {len(df)} (event, player) score rows from {df['event_id'].nunique()} events")

    # ---- Event field stats ----
    df_rated = df.dropna(subset=["pre_rating"]).copy()
    df_rated["pre_rating"] = df_rated["pre_rating"].astype(int)
    stats = df_rated.groupby("event_id").agg(
        event_n_players=("uscf_id", "nunique"),
        event_avg_rating=("pre_rating", "mean"),
        event_median_rating=("pre_rating", "median"),
        event_max_rating=("pre_rating", "max"),
        event_min_rating=("pre_rating", "min"),
        event_field_score_std=("total_score", "std"),
    ).reset_index()

    # ---- Per-player finish rank + percentile within event ----
    df = df.merge(
        df.groupby("event_id")["uscf_id"].count().rename("event_n_players_total").reset_index(),
        on="event_id", how="left",
    )
    df["finish_rank"] = (
        df.groupby("event_id")["total_score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    # percentile: 1 - (rank-1) / (n_players-1)  => top finisher = 1.0
    df["finish_percentile"] = np.where(
        df["event_n_players_total"] > 1,
        1.0 - (df["finish_rank"] - 1) / (df["event_n_players_total"] - 1),
        1.0,
    )
    df["event_winner"] = (df["finish_rank"] == 1).astype(int)
    df["top3_finish"] = (df["finish_rank"] <= 3).astype(int)
    df["top5_finish"] = (df["finish_rank"] <= 5).astype(int)

    return df, stats


# ----------------------------------------------------------------------
# Stage 2 — player-level aggregation
# ----------------------------------------------------------------------
def safe_div(num, den):
    return float(num) / float(den) if den else 0.0


def build_profiles(
    games: pd.DataFrame,
    event_player: pd.DataFrame,
    event_stats: pd.DataFrame,
    reference_date: pd.Timestamp,
) -> pd.DataFrame:
    """Aggregate per (focal) player.  Each feature is computed at the
    REFERENCE_DATE (default = max event date in our data), giving a
    snapshot of 'where this player is right now'."""
    games = games.copy()
    games["event_end_date"] = pd.to_datetime(games["event_end_date"], errors="coerce")
    games = games.dropna(subset=["event_end_date"])
    games["days_ago"] = (reference_date - games["event_end_date"]).dt.days
    games["event_id"] = games["event_id"].astype(str)
    games["player_id"] = games["player_id"].astype(str)

    # Merge event stats and player finish info onto game rows
    ep = event_player.rename(columns={"uscf_id": "player_id"}).copy()
    ep["event_id"] = ep["event_id"].astype(str)
    ep["player_id"] = ep["player_id"].astype(str)
    event_stats = event_stats.copy()
    event_stats["event_id"] = event_stats["event_id"].astype(str)
    games = games.merge(
        ep[["event_id", "player_id", "state", "finish_rank", "finish_percentile",
            "event_winner", "top3_finish", "top5_finish", "event_n_players_total"]],
        on=["event_id", "player_id"], how="left",
    )
    games = games.merge(event_stats, on="event_id", how="left", suffixes=("", "_es"))

    profiles = []
    for pid, g in games.groupby("player_id"):
        latest_date = g["event_end_date"].max()
        days_since_last_game = int((reference_date - latest_date).days)
        days_since_last_event = days_since_last_game  # event = game's date

        career_n = len(g)

        # --- A. Activity ---
        g30 = g[g["days_ago"] <= 30]
        g90 = g[g["days_ago"] <= 90]
        g180 = g[g["days_ago"] <= 180]
        g365 = g[g["days_ago"] <= 365]

        events_last_90d = g90["event_id"].nunique()
        events_last_365d = g365["event_id"].nunique()

        # --- B. Recent form ---
        def score_rate(sub):
            w = (sub["result_raw"] == "W").sum()
            d = (sub["result_raw"] == "D").sum()
            n = len(sub)
            return (w + 0.5 * d) / n if n else None

        recent_win_rate_30d = (g30["target_binary"].mean() if len(g30) else None)
        recent_win_rate_90d = (g90["target_binary"].mean() if len(g90) else None)
        recent_score_rate_90d = score_rate(g90)
        career_win_rate = g["target_binary"].mean() if career_n else None
        form_delta_90d = (
            recent_win_rate_90d - career_win_rate
            if recent_win_rate_90d is not None and career_win_rate is not None
            else None
        )

        # --- C. Strength of schedule ---
        def pct(sub, mask):
            return sub[mask].shape[0] / len(sub) if len(sub) else None

        pct_vs_higher = pct(g90, g90["rating_diff"] < 0) if len(g90) else None
        pct_vs_100_plus_higher = pct(g90, g90["rating_diff"] <= -100) if len(g90) else None
        pct_vs_200_plus_higher = pct(g90, g90["rating_diff"] <= -200) if len(g90) else None
        avg_rating_diff_recent = g90["rating_diff"].mean() if len(g90) else None
        avg_abs_rating_diff_recent = g90["rating_diff"].abs().mean() if len(g90) else None

        # --- D. Upset metrics (career, more stable than 90d) ---
        higher_100 = g[g["rating_diff"] <= -100]
        higher_200 = g[g["rating_diff"] <= -200]
        lower_100 = g[g["rating_diff"] >= 100]
        wins_vs_100_plus_higher = int((higher_100["result_raw"] == "W").sum())
        wins_vs_200_plus_higher = int((higher_200["result_raw"] == "W").sum())
        draws_vs_200_plus_higher = int((higher_200["result_raw"] == "D").sum())
        losses_vs_100_plus_lower = int((lower_100["result_raw"] == "L").sum())
        score_rate_vs_higher_rated = score_rate(higher_100) if len(higher_100) else None
        upset_win_rate = safe_div(wins_vs_100_plus_higher, len(higher_100))
        # career upset rate within last 365d
        higher_100_90d = g90[g90["rating_diff"] <= -100]
        upset_rate_90d = safe_div((higher_100_90d["result_raw"] == "W").sum(), len(higher_100_90d))
        higher_100_365d = g365[g365["rating_diff"] <= -100]
        upset_rate_365d = safe_div((higher_100_365d["result_raw"] == "W").sum(), len(higher_100_365d))

        # --- E. Rating trend (proxy, from chronological pre_ratings) ---
        sorted_g = g.sort_values("event_end_date")
        ratings_series = sorted_g["player_pre_rating"].astype(float)
        current_rating = float(ratings_series.iloc[-1]) if career_n else None

        def rating_x_days_ago(d):
            past = sorted_g[sorted_g["days_ago"] >= d]
            return float(past["player_pre_rating"].iloc[-1]) if len(past) else None

        rating_30_ago  = rating_x_days_ago(30)
        rating_90_ago  = rating_x_days_ago(90)
        rating_180_ago = rating_x_days_ago(180)
        rating_365_ago = rating_x_days_ago(365)
        def rd(now, then): return float(now - then) if (now is not None and then is not None) else None
        rating_change_30d = rd(current_rating, rating_30_ago)
        rating_change_90d = rd(current_rating, rating_90_ago)
        rating_change_180d = rd(current_rating, rating_180_ago)
        rating_change_365d = rd(current_rating, rating_365_ago)

        # rating slope (per day) from last 180 days, linear fit
        recent_180 = sorted_g[sorted_g["days_ago"] <= 180]
        rating_slope_180d = None
        if len(recent_180) >= 3:
            x = (recent_180["event_end_date"] - recent_180["event_end_date"].min()).dt.days.values
            y = recent_180["player_pre_rating"].astype(float).values
            if x.max() > 0:
                rating_slope_180d = float(np.polyfit(x, y, 1)[0])  # rating units per day

        rating_peak_recent = float(g365["player_pre_rating"].max()) if len(g365) else current_rating
        distance_from_peak_rating = (
            float(rating_peak_recent - current_rating)
            if (rating_peak_recent is not None and current_rating is not None) else None
        )
        max_90d_rating_gain = None
        if len(g90) >= 2:
            r = g90.sort_values("event_end_date")["player_pre_rating"].astype(float).values
            # max running gain = max - min when min comes first ... use a simple cummax-rolling approach
            cummin = np.minimum.accumulate(r)
            max_90d_rating_gain = float((r - cummin).max())
        # rating_volatility = std of pre-rating over recent year (proxy)
        rating_volatility = float(g365["player_pre_rating"].std()) if len(g365) > 1 else None

        # --- F. Time-control profile ---
        tc_counts = g["time_control"].value_counts(normalize=True)
        pct_regular = float(tc_counts.get("Regular", 0))
        pct_quick   = float(tc_counts.get("Quick", 0))
        pct_blitz   = float(tc_counts.get("Blitz", 0))
        # best time control by score rate (career)
        best_tc, best_tc_score = None, -1.0
        for tc, sub in g.groupby("time_control"):
            if len(sub) < 5:
                continue
            sr = score_rate(sub)
            if sr is not None and sr > best_tc_score:
                best_tc_score, best_tc = sr, tc
        if pct_regular > 0.7:
            tc_specialist = "Regular specialist"
        elif pct_quick > 0.7:
            tc_specialist = "Quick specialist"
        elif pct_blitz > 0.7:
            tc_specialist = "Blitz specialist"
        else:
            tc_specialist = "Mixed-format player"

        # --- G. Event/tournament success ---
        events_uniq = g.dropna(subset=["finish_rank"]).drop_duplicates(subset=["event_id"])
        events_uniq_90d  = events_uniq[events_uniq["days_ago"] <= 90]
        events_uniq_365d = events_uniq[events_uniq["days_ago"] <= 365]
        events_won_last_90d  = int(events_uniq_90d["event_winner"].sum())
        events_won_last_365d = int(events_uniq_365d["event_winner"].sum())
        top_3_last_365d = int(events_uniq_365d["top3_finish"].sum())
        top_5_last_365d = int(events_uniq_365d["top5_finish"].sum())
        best_recent_finish_percentile = (
            float(events_uniq_365d["finish_percentile"].max())
            if len(events_uniq_365d) else None
        )
        avg_finish_percentile_365d = (
            float(events_uniq_365d["finish_percentile"].mean())
            if len(events_uniq_365d) else None
        )

        # --- H. Field strength ---
        avg_field_strength_last_365d = (
            float(events_uniq_365d["event_avg_rating"].mean())
            if "event_avg_rating" in events_uniq_365d and len(events_uniq_365d) else None
        )

        # --- I. Consistency / volatility ---
        # result_volatility = std of (result_score - expected_from_rating)
        expected = 1.0 / (1.0 + 10.0 ** (-g["rating_diff"].astype(float) / 400.0))
        actual = g["result_raw"].map({"W": 1.0, "D": 0.5, "L": 0.0})
        residuals = (actual - expected).dropna()
        result_volatility = float(residuals.std()) if len(residuals) > 1 else None

        # best win streak / worst losing streak (career)
        results_chrono = sorted_g["result_raw"].tolist()
        best_w_streak, worst_l_streak = 0, 0
        w_cur, l_cur = 0, 0
        for r in results_chrono:
            if r == "W":
                w_cur += 1; l_cur = 0
                best_w_streak = max(best_w_streak, w_cur)
            elif r == "L":
                l_cur += 1; w_cur = 0
                worst_l_streak = max(worst_l_streak, l_cur)
            else:
                w_cur = 0; l_cur = 0

        boom_bust_flag = bool(
            best_w_streak >= 5 and worst_l_streak >= 5 and result_volatility and result_volatility > 0.35
        )
        consistency_score = (
            float(1.0 - min(1.0, (result_volatility or 0) * 2)) if result_volatility is not None else None
        )

        # --- J. Travel / geography (STUB) ---
        # USCF crosstable does expose each player's home state.  We pick
        # the player's most common state across all their crosstable
        # rows as their inferred home region.  Distances and event GPS
        # are NOT in our data — flagged TODO.
        states = g["state"].dropna().astype(str)
        inferred_home_region = (
            states.value_counts().idxmax() if not states.empty else None
        )
        unique_states_played = int(states.nunique())

        # --- K. Cold-start / sample-size flags ---
        insufficient_recent_games_flag = bool(len(g90) < 10)
        insufficient_higher_rated_games_flag = bool(len(higher_100) < 5)
        small_sample_warning = bool(career_n < 20 or len(g90) < 5)

        profiles.append({
            "player_id": pid,
            "as_of_date": reference_date.date().isoformat(),
            # snapshot
            "current_rating": current_rating,
            "career_n_games": career_n,
            "career_win_rate": career_win_rate,
            "days_since_last_game": days_since_last_game,
            "days_since_last_event": days_since_last_event,
            # A. activity
            "games_last_30d": len(g30),
            "games_last_90d": len(g90),
            "games_last_180d": len(g180),
            "games_last_365d": len(g365),
            "events_last_90d": events_last_90d,
            "events_last_365d": events_last_365d,
            # B. recent form
            "recent_win_rate_30d": recent_win_rate_30d,
            "recent_win_rate_90d": recent_win_rate_90d,
            "recent_score_rate_90d": recent_score_rate_90d,
            "form_delta_90d": form_delta_90d,
            # C. strength of schedule
            "pct_games_vs_higher_rated_90d": pct_vs_higher,
            "pct_games_vs_100_plus_higher_90d": pct_vs_100_plus_higher,
            "pct_games_vs_200_plus_higher_90d": pct_vs_200_plus_higher,
            "avg_rating_diff_recent_90d": avg_rating_diff_recent,
            "avg_abs_rating_diff_recent_90d": avg_abs_rating_diff_recent,
            # D. upsets
            "wins_vs_100_plus_higher": wins_vs_100_plus_higher,
            "wins_vs_200_plus_higher": wins_vs_200_plus_higher,
            "draws_vs_200_plus_higher": draws_vs_200_plus_higher,
            "losses_vs_100_plus_lower": losses_vs_100_plus_lower,
            "score_rate_vs_higher_rated": score_rate_vs_higher_rated,
            "upset_win_rate": upset_win_rate,
            "upset_rate_90d": upset_rate_90d,
            "upset_rate_365d": upset_rate_365d,
            # E. rating trend (proxy — post-rating not yet parsed)
            "rating_change_30d_proxy": rating_change_30d,
            "rating_change_90d_proxy": rating_change_90d,
            "rating_change_180d_proxy": rating_change_180d,
            "rating_change_365d_proxy": rating_change_365d,
            "rating_slope_180d_proxy": rating_slope_180d,
            "rating_peak_recent_365d": rating_peak_recent,
            "distance_from_peak_rating": distance_from_peak_rating,
            "max_90d_rating_gain_proxy": max_90d_rating_gain,
            "rating_volatility_proxy": rating_volatility,
            # F. time-control profile
            "pct_regular_games": pct_regular,
            "pct_quick_games": pct_quick,
            "pct_blitz_games": pct_blitz,
            "best_time_control_by_score_rate": best_tc,
            "time_control_specialist_label": tc_specialist,
            # G. event success
            "events_won_last_90d": events_won_last_90d,
            "events_won_last_365d": events_won_last_365d,
            "top_3_finishes_last_365d": top_3_last_365d,
            "top_5_finishes_last_365d": top_5_last_365d,
            "best_recent_finish_percentile": best_recent_finish_percentile,
            "avg_finish_percentile_365d": avg_finish_percentile_365d,
            # H. field strength
            "avg_field_strength_last_365d": avg_field_strength_last_365d,
            # I. consistency
            "result_volatility": result_volatility,
            "best_win_streak": best_w_streak,
            "worst_losing_streak": worst_l_streak,
            "boom_bust_flag": boom_bust_flag,
            "consistency_score": consistency_score,
            # J. travel / geography (partial)
            "inferred_home_region": inferred_home_region,
            "unique_states_played": unique_states_played,
            "unique_event_locations": None,         # TODO: needs event GPS
            "avg_travel_distance_miles": None,      # TODO: needs event GPS
            "max_travel_distance_miles": None,      # TODO: needs event GPS
            "pct_events_outside_home_region": None, # TODO: needs event GPS
            "traveling_competitor_label": None,     # TODO: depends on above
            # K. cold-start
            "insufficient_recent_games_flag": insufficient_recent_games_flag,
            "insufficient_higher_rated_games_flag": insufficient_higher_rated_games_flag,
            "small_sample_warning": small_sample_warning,
        })

    return pd.DataFrame(profiles).sort_values("current_rating", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------
def main():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cache_dir = os.path.join(base, "data", "raw", "html_cache")
    proc_dir = os.path.join(base, "data", "processed")
    os.makedirs(proc_dir, exist_ok=True)

    raw_games_path = os.path.join(base, "data", "raw", "tables", "raw_games.csv")
    recency_path = os.path.join(proc_dir, "features_v3_recency.csv")

    print("=" * 60)
    print("PHASE 15: PLAYER PROFILES (SCOUTING LAYER)")
    print("=" * 60)

    raw_games = pd.read_csv(raw_games_path, dtype=str)
    games_clean = pd.read_csv(recency_path)
    games_clean["event_end_date"] = pd.to_datetime(games_clean["event_end_date"], errors="coerce")
    games_clean = games_clean.dropna(subset=["event_end_date"])

    # Stage 1
    print("\n[Stage 1/2] Re-parsing cached crosstables for event field data ...")
    event_player_df, event_stats_df = rich_event_pass(cache_dir, raw_games)
    event_player_path = os.path.join(proc_dir, "event_player_scores.csv")
    event_stats_path = os.path.join(proc_dir, "event_field_stats.csv")
    event_player_df.to_csv(event_player_path, index=False)
    event_stats_df.to_csv(event_stats_path, index=False)
    print(f"  wrote {event_player_path} ({len(event_player_df)} rows)")
    print(f"  wrote {event_stats_path}  ({len(event_stats_df)} rows)")

    # Stage 2
    print("\n[Stage 2/2] Aggregating per-player profiles ...")
    reference_date = games_clean["event_end_date"].max()
    print(f"  reference date (most recent in dataset): {reference_date.date()}")
    profiles = build_profiles(games_clean, event_player_df, event_stats_df, reference_date)
    out_path = os.path.join(proc_dir, "player_profiles.csv")
    profiles.to_csv(out_path, index=False)
    print(f"  wrote {out_path} ({len(profiles)} players)")

    # Quick sanity print
    print("\n--- Sample (top 5 by current rating) ---")
    cols_show = [
        "player_id", "current_rating", "career_n_games", "career_win_rate",
        "games_last_90d", "recent_win_rate_90d", "events_won_last_365d",
        "top_5_finishes_last_365d", "rating_change_90d_proxy", "inferred_home_region",
    ]
    print(profiles[cols_show].head(5).to_string(index=False))

    print("\n--- TODO (data gaps to flag in dashboard) ---")
    print("  * Travel distance: requires event GPS coordinates (USCF MSA does not expose).")
    print("  * True rating_change_* / rating_slope: requires post-rating per game; we use")
    print("    chronological pre-rating as a proxy. To unlock the true version, extend the")
    print("    crosstable parser to capture the post-rating column (currently 'Unknown').")
    print("=" * 60)


if __name__ == "__main__":
    main()
