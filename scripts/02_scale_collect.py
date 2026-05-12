import os
import sys
import yaml
import logging
import pandas as pd
from collections import deque
from datetime import datetime, timedelta

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
        
    seeds = config['collection'].get('seed_player_ids', [])
    max_players = config['collection'].get('max_players', 5)
    max_events_total = config['collection'].get('max_events_total', 50)
    max_games_total = config['collection'].get('max_games_total', 500)
    resume_mode = config['collection'].get('resume_mode', True)
    history_years = config['collection'].get('history_years', 5)
    
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    cache_dir = os.path.join(base_dir, config['project']['data_dir'], 'raw', 'html_cache')
    tables_dir = os.path.join(base_dir, config['project']['data_dir'], 'raw', 'tables')
    
    fetcher = Fetcher(
        cache_dir=cache_dir,
        delay_sec=config['collection'].get('delay_between_requests_sec', 2.0),
        user_agent=config['collection'].get('user_agent', 'Mozilla/5.0')
    )
    
    # State tracking
    processed_players = set()
    processed_events = set()
    queued_players = set(str(s) for s in seeds)
    
    players_file = os.path.join(tables_dir, 'raw_players.csv')
    events_file = os.path.join(tables_dir, 'raw_events.csv')
    games_file = os.path.join(tables_dir, 'raw_games.csv')
    
    # 2. Resume Logic: Rebuild state from existing CSVs
    if resume_mode:
        logger.info("Resume mode active. Rebuilding state from CSVs...")
        if os.path.exists(players_file):
            df_p = pd.read_csv(players_file, dtype=str)
            if not df_p.empty and 'uscf_id' in df_p.columns:
                processed_players.update(df_p['uscf_id'].dropna().tolist())
                
        if os.path.exists(events_file):
            df_e = pd.read_csv(events_file, dtype=str)
            if not df_e.empty and 'event_id' in df_e.columns:
                processed_events.update(df_e['event_id'].dropna().tolist())
                
        if os.path.exists(games_file):
            df_g = pd.read_csv(games_file, dtype=str)
            if not df_g.empty and 'opponent_id' in df_g.columns:
                opps = df_g['opponent_id'].dropna().tolist()
                valid_opps = [o for o in opps if not str(o).startswith('OPP_') and str(o) != 'Unknown']
                queued_players.update(valid_opps)
                
    # Remove already processed players from the starting queue
    queue = deque([p for p in queued_players if p not in processed_players])
    
    logger.info(f"Starting Queue Size: {len(queue)} | Already Processed Players: {len(processed_players)}")
    
    total_valid_games_parsed_run = 0
    total_raw_games_parsed_run = 0
    events_fetched_run = 0
    players_processed_run = 0
    unique_opponents_discovered = set()
    
    stop_reason = "Queue Empty"
    
    cutoff_date = datetime.now() - timedelta(days=365 * history_years)
    
    # 3. Scale Collection Loop
    while queue:
        if players_processed_run >= max_players:
            stop_reason = "Max Players Reached"
            break
        if events_fetched_run >= max_events_total:
            stop_reason = "Max Events Reached"
            break
        if total_valid_games_parsed_run >= max_games_total:
            stop_reason = "Max Games Reached"
            break
            
        target_id = queue.popleft()
        
        if target_id in processed_players:
            continue
            
        logger.info(f"--- Processing Player: {target_id} ---")
        
        # A. Fetch Profile
        prof_url = f"https://www.uschess.org/msa/MbrDtlMain.php?{target_id}"
        p_html, p_cache_path = fetcher.fetch(prof_url)
        if p_html:
            player_info = parse_player_profile(p_html, prof_url, p_cache_path, target_id)
            append_to_csv([player_info], players_file)
            
        processed_players.add(target_id)
        players_processed_run += 1
        
        # B. Fetch History
        hist_url = f"https://www.uschess.org/msa/MbrDtlTnmtHst.php?{target_id}"
        html, cache_path = fetcher.fetch(hist_url)
        if not html:
            logger.warning(f"Failed to fetch history for {target_id}")
            continue
            
        events = parse_tournament_history(html, hist_url, cache_path)
        logger.info(f"  Found {len(events)} events in total history.")
        
        # C. Filter History by Date
        recent_events = []
        for ev in events:
            if ev['end_date'] != "Unknown":
                try:
                    ev_date = datetime.strptime(ev['end_date'], "%Y-%m-%d")
                    if ev_date >= cutoff_date:
                        recent_events.append(ev)
                except ValueError:
                    recent_events.append(ev) 
            else:
                recent_events.append(ev)
                
        logger.info(f"  Filtering to {len(recent_events)} events within the last {history_years} years.")
        
        # D. Fetch Crosstables
        for ev in recent_events:
            event_id = ev['event_id']
            
            # Guard against duplicating previously fetched events across different players
            if event_id in processed_events:
                continue
            
            if events_fetched_run >= max_events_total or total_valid_games_parsed_run >= max_games_total:
                break
                
            # Safely extract base ID to prevent ".0.0" malformed URLs
            base_event_id = event_id.split('.')[0]
            xtbl_url = f"https://www.uschess.org/msa/XtblMain.php?{base_event_id}.0-{target_id}"
            x_html, x_cache_path = fetcher.fetch(xtbl_url)
            if not x_html:
                continue
                
            tc_text, tc_norm, games = parse_crosstable(x_html, xtbl_url, x_cache_path, target_id, event_id)
            ev['raw_time_control_text'] = tc_text
            ev['normalized_time_control'] = tc_norm
            
            append_to_csv([ev], events_file)
            processed_events.add(event_id)
            events_fetched_run += 1
            
            if games:
                append_to_csv(games, games_file)
                total_raw_games_parsed_run += len(games)
                
                # Validate and Extract Opponents
                games_df = pd.DataFrame(games)
                valid_df, _ = validate_poc_games(games_df)
                
                total_valid_games_parsed_run += len(valid_df)
                
                for opp in games_df['opponent_id'].dropna().unique():
                    if not str(opp).startswith('OPP_') and str(opp) != 'Unknown':
                        unique_opponents_discovered.add(str(opp))
                        if opp not in processed_players and opp not in queue:
                            queue.append(opp)
                            
        logger.info(f"  Queue size: {len(queue)}")
        
    # 4. Final Summary
    print("\n" + "="*40)
    print("PHASE 2: SCALABLE COLLECTION SUMMARY")
    print("="*40)
    print(f"Players Processed (This Run): {players_processed_run}")
    print(f"Events Fetched (This Run): {events_fetched_run}")
    print(f"Raw Games Parsed (This Run): {total_raw_games_parsed_run}")
    print(f"Usable Games (This Run): {total_valid_games_parsed_run}")
    print(f"Unique Opponents Discovered (This Run): {len(unique_opponents_discovered)}")
    print(f"Queue Size Remaining: {len(queue)}")
    print("\n--- Project Globals ---")
    print(f"Total Unique Players Processed (All Time): {len(processed_players)}")
    print(f"Total Unique Events Processed (All Time): {len(processed_events)}")
    print("\nSTOPPING REASON:")
    print(f"[{stop_reason}]")
    print("="*40)

if __name__ == "__main__":
    main()
