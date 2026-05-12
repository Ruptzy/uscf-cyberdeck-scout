import os
import sys
import yaml
import logging
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scraper.fetcher import Fetcher
from src.parser.msa_parser import parse_player_profile, parse_tournament_history, parse_crosstable
from src.data.writer import append_to_csv
from src.data.validator import validate_poc_games

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    # 1. Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    target_id = config['collection']['target_player_ids'][0]
    max_events = config['collection']['max_poc_events']
    
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    cache_dir = os.path.join(base_dir, config['project']['data_dir'], 'raw', 'html_cache')
    tables_dir = os.path.join(base_dir, config['project']['data_dir'], 'raw', 'tables')
    
    fetcher = Fetcher(
        cache_dir=cache_dir,
        delay_sec=config['collection']['delay_between_requests_sec'],
        user_agent=config['collection']['user_agent']
    )
    
    # 2. Fetch Player Profile
    prof_url = f"https://www.uschess.org/msa/MbrDtlMain.php?{target_id}"
    p_html, p_cache_path = fetcher.fetch(prof_url)
    if p_html:
        player_info = parse_player_profile(p_html, prof_url, p_cache_path, target_id)
        append_to_csv([player_info], os.path.join(tables_dir, 'raw_players.csv'))
    else:
        logger.error("Failed to fetch profile.")
    
    # 3. Fetch Tournament History
    hist_url = f"https://www.uschess.org/msa/MbrDtlTnmtHst.php?{target_id}"
    html, cache_path = fetcher.fetch(hist_url)
    
    if not html:
        logger.error("Failed to fetch history.")
        return
        
    events = parse_tournament_history(html, hist_url, cache_path)
    logger.info(f"Found {len(events)} total events in history.")
    
    # Limit for PoC
    events_to_process = events[:max_events]
    
    all_games = []
    tc_observed = set()
    
    # 4. Fetch Crosstables for top events
    for ev in events_to_process:
        event_id = ev['event_id']
        xtbl_url = f"https://www.uschess.org/msa/XtblMain.php?{event_id}.0-{target_id}"
        
        x_html, x_cache_path = fetcher.fetch(xtbl_url)
        if not x_html:
            continue
            
        tc_text, tc_norm, games = parse_crosstable(x_html, xtbl_url, x_cache_path, target_id, event_id)
        
        # Update event time control
        ev['raw_time_control_text'] = tc_text
        ev['normalized_time_control'] = tc_norm
        if tc_norm != "Unknown":
            tc_observed.add(tc_norm)
        
        all_games.extend(games)
        
    # 5. Write Raw Tables
    if events_to_process:
        append_to_csv(events_to_process, os.path.join(tables_dir, 'raw_events.csv'))
    if all_games:
        append_to_csv(all_games, os.path.join(tables_dir, 'raw_games.csv'))
    
    # 6. Validate
    games_df = pd.DataFrame(all_games) if all_games else pd.DataFrame()
    val_summary = {}
    if not games_df.empty:
        valid_df, val_summary = validate_poc_games(games_df)
    else:
        valid_df = pd.DataFrame()

    # 7. Checkpoint Evaluation
    total_valid = len(valid_df)
    total_raw = len(games_df) if not games_df.empty else 0
    duplicate_count = total_raw - len(games_df.drop_duplicates(subset=['game_id'])) if not games_df.empty else 0
    
    valid_ratio = total_valid / total_raw if total_raw > 0 else 0
    
    # Simple check for healthy cache
    cache_files = os.listdir(cache_dir) if os.path.exists(cache_dir) else []
    cache_looks_healthy = len(cache_files) > 0
    
    checkpoint_result = ""
    reason = ""
    
    if not cache_looks_healthy or total_raw == 0:
        checkpoint_result = "FAILURE"
        reason = "Cache is empty or script failed to fetch USCF tables. Likely Cloudflare block or 403."
    elif total_valid >= 25 and valid_ratio >= 0.95 and duplicate_count == 0:
        checkpoint_result = "SUCCESS"
        reason = "Collected 25+ valid rows with real IDs/Ratings, 95%+ pass rate, no duplicates."
    else:
        checkpoint_result = "PARTIAL SUCCESS"
        reason = f"Cache works, but parser yielded {total_valid} valid rows (Ratio: {valid_ratio:.0%}). Placeholders or missing fields exist."

    # 8. Summary Output
    print("\n" + "="*40)
    print("PROOF OF CONCEPT SUMMARY")
    print("="*40)
    print(f"Events Fetched: {len(events_to_process)}")
    print(f"Raw Games Parsed: {total_raw}")
    print(f"Usable Games After Validation: {total_valid}")
    print(f"Time Controls Observed: {', '.join(tc_observed) if tc_observed else 'None'}")
    print("\n--- Validation Drop Reasons ---")
    if val_summary:
        for k, v in val_summary.items():
            print(f"  {k}: {v}")
    else:
        print("  No games to validate.")
    print("\nCHECKPOINT RESULT:")
    print(f"[{checkpoint_result}] - {reason}")
    print("="*40)

if __name__ == "__main__":
    main()
