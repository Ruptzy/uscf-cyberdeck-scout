"""
Phase 14: Re-parse raw games from cached HTML
=============================================
Diagnostic finding (logged in docs/DATA_FIX_NOTE.md): the original
crosstable parser captured the first 3-4 digits of each player's 8-digit
USCF ID as their "rating", because the rating regex
``(?:R:)?\\s*(\\d{3,4})`` was unanchored and the ID appears earlier in the
same string than the actual rating.

This script re-walks the local HTML cache (no network calls) and
rebuilds ``raw_games.csv`` using the corrected parser, without
re-scraping.

We also retro-fit ``end_date`` and ``normalized_time_control`` on
``raw_events.csv`` so the cleaning pipeline has trustworthy event
metadata.
"""

import logging
import os
import re
import sys

import pandas as pd
from bs4 import BeautifulSoup

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.parser.msa_parser import parse_crosstable, parse_tournament_history  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# USCF time-control heuristics (per USCF MSA rules).
# - Regular (RG): primary control >= 30 min per side
# - Quick     :  5 min <= base time < 30 min  ("dual" rated)
# - Blitz     :  <  5 min base
def classify_time_control(tc_text: str) -> str:
    if not tc_text:
        return "Unknown"
    t = tc_text.strip().lower()
    # Find every "G/<minutes>" or "d<seconds>" token; pick the largest minute count.
    minute_tokens = re.findall(r"g\s*/\s*(\d+)", t)
    if not minute_tokens:
        return "Unknown"
    mins = [int(x) for x in minute_tokens]
    base = max(mins)
    if base >= 30:
        return "Regular"
    if base >= 10:
        return "Quick"
    if base >= 5:
        # 5- and 10-min games can be Blitz or Quick depending on USCF rules;
        # USCF rates Blitz at 5-10 min so call this Blitz to be conservative.
        return "Blitz"
    return "Blitz"


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cache_dir = os.path.join(base_dir, "data", "raw", "html_cache")
    tables_dir = os.path.join(base_dir, "data", "raw", "tables")
    out_games = os.path.join(tables_dir, "raw_games.csv")
    out_events = os.path.join(tables_dir, "raw_events.csv")

    # Existing CSVs hold the mapping (event_id, player_id) we need to know
    # which crosstable rows to re-parse for which focal player.
    games_old = pd.read_csv(out_games, dtype=str)
    events_old = pd.read_csv(out_events, dtype=str)

    print("=" * 60)
    print("PHASE 14: RE-PARSE FROM CACHE (with fixed rating regex)")
    print("=" * 60)
    print(f"Cache dir: {cache_dir}")
    print(f"Old raw_games rows:  {len(games_old)}")
    print(f"Old raw_events rows: {len(events_old)}")

    # ---------------- Re-parse games ----------------
    pairs = games_old[["event_id", "player_id"]].drop_duplicates().reset_index(drop=True)
    print(f"Unique (event, player) crosstables to re-parse: {len(pairs)}")

    all_games = []
    tc_by_event = {}
    miss_html = 0
    miss_pre = 0

    for i, row in pairs.iterrows():
        event_id = str(row["event_id"])
        player_id = str(row["player_id"])
        cache_path = os.path.join(cache_dir, f"XtblMain.php_{event_id}.0-{player_id}.html")
        if not os.path.exists(cache_path):
            miss_html += 1
            continue
        with open(cache_path, "r", encoding="utf-8", errors="replace") as f:
            html = f.read()
        source_url = f"https://www.uschess.org/msa/XtblMain.php?{event_id}.0-{player_id}"
        tc_text, _tc_norm_old, games = parse_crosstable(html, source_url, cache_path, player_id, event_id)

        tc_norm = classify_time_control(tc_text)
        tc_by_event[event_id] = (tc_text, tc_norm)

        if not games:
            miss_pre += 1
        all_games.extend(games)

        if (i + 1) % 500 == 0:
            print(f"  [{i + 1}/{len(pairs)}] crosstables parsed; rows so far: {len(all_games)}")

    print(f"  Crosstables with no cached HTML: {miss_html}")
    print(f"  Crosstables that yielded zero games: {miss_pre}")
    print(f"  Total raw game rows recovered: {len(all_games)}")

    new_games = pd.DataFrame(all_games)
    new_games.to_csv(out_games, index=False)
    print(f"Wrote {out_games}")

    # ---------------- Re-parse tournament-history pages for dates ----------------
    unique_players = sorted({str(p) for p in pairs["player_id"].tolist()})
    print(f"\nRe-parsing tournament-history pages for {len(unique_players)} players ...")

    event_meta = {}  # event_id -> end_date
    for player_id in unique_players:
        cache_path = os.path.join(cache_dir, f"MbrDtlTnmtHst.php_{player_id}.html")
        if not os.path.exists(cache_path):
            continue
        with open(cache_path, "r", encoding="utf-8", errors="replace") as f:
            html = f.read()
        source_url = f"https://www.uschess.org/msa/MbrDtlTnmtHst.php?{player_id}"
        events = parse_tournament_history(html, source_url, cache_path)
        for ev in events:
            eid = str(ev["event_id"])
            if eid not in event_meta and ev.get("end_date") and ev["end_date"] != "Unknown":
                event_meta[eid] = ev["end_date"]

    # Rebuild events CSV
    new_event_rows = []
    for eid, (tc_text, tc_norm) in tc_by_event.items():
        new_event_rows.append({
            "event_id": eid,
            "event_name": "",
            "end_date": event_meta.get(eid, "Unknown"),
            "raw_time_control_text": tc_text,
            "normalized_time_control": tc_norm,
            "source_url": f"https://www.uschess.org/msa/XtblMain.php?{eid}",
            "source_html_file": "",
            "parse_status": "Success",
        })
    new_events = pd.DataFrame(new_event_rows)
    new_events.to_csv(out_events, index=False)
    print(f"Wrote {out_events} ({len(new_events)} events)")

    # ---------------- Sanity check ----------------
    print("\n--- Sanity check: ratings should now look real ---")
    sample = new_games[new_games["player_id"] == "12640800"].head(8)
    if not sample.empty:
        print(sample[["player_id", "opponent_id", "player_pre_rating", "opponent_pre_rating", "result"]].to_string())

    # Quick distribution print
    try:
        ratings = pd.to_numeric(new_games["player_pre_rating"].str.replace(r"P\d+$", "", regex=True), errors="coerce").dropna()
        print(f"\nplayer_pre_rating  min/median/max: {int(ratings.min())} / {int(ratings.median())} / {int(ratings.max())}")
        opp_ratings = pd.to_numeric(new_games["opponent_pre_rating"].str.replace(r"P\d+$", "", regex=True), errors="coerce").dropna()
        print(f"opponent_pre_rating min/median/max: {int(opp_ratings.min())} / {int(opp_ratings.median())} / {int(opp_ratings.max())}")
    except Exception as e:  # noqa: BLE001
        print(f"(rating sanity print failed: {e!r})")

    # Wipe stale processed CSVs so downstream phases must regenerate them
    for stale_name in [
        "features_v1.csv",
        "features_v2_expected_score.csv",
        "features_v3_recency.csv",
    ]:
        stale_path = os.path.join(base_dir, "data", "processed", stale_name)
        if os.path.exists(stale_path):
            os.remove(stale_path)
            print(f"Removed stale: {stale_path}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
