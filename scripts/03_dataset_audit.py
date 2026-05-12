import os
import sys
import pandas as pd
import numpy as np
import yaml

def main():
    # Load config to get data_dir
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.yaml'))
    base_dir = os.path.dirname(config_path)
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        data_dir = config['project'].get('data_dir', 'data')
    except Exception:
        data_dir = 'data'
        
    tables_dir = os.path.join(base_dir, data_dir, 'raw', 'tables')
    games_file = os.path.join(tables_dir, 'raw_games.csv')
    events_file = os.path.join(tables_dir, 'raw_events.csv')
    
    if not os.path.exists(games_file):
        print(f"Error: Could not find raw_games.csv at {games_file}")
        sys.exit(1)
        
    df_games = pd.read_csv(games_file, dtype=str)
    
    print("\n" + "="*40)
    print("PHASE 2: DATASET AUDIT REPORT")
    print("="*40)
    
    # 1. Row counts & Diversity
    total_games = len(df_games)
    unique_players = df_games['player_id'].nunique()
    unique_opponents = df_games['opponent_id'].nunique()
    
    print(f"Total Raw Games: {total_games}")
    print(f"Unique Focal Players: {unique_players}")
    print(f"Unique Opponents: {unique_opponents}")
    
    # 2. Result distributions
    print("\n--- Result Distribution ---")
    if 'result' in df_games.columns:
        result_counts = df_games['result'].value_counts()
        result_props = df_games['result'].value_counts(normalize=True) * 100
        for res, count in result_counts.items():
            prop = result_props.get(res, 0)
            print(f"  {res}: {count} ({prop:.1f}%)")
    else:
        print("  Column 'result' missing.")
        
    # 3. Missing/Invalid Fields
    print("\n--- Missing or Invalid Fields ---")
    
    def is_invalid_opp(x):
        return pd.isna(x) or str(x) == 'Unknown' or str(x).startswith('OPP_')
        
    def is_invalid_rating(x):
        return pd.isna(x) or str(x) == 'Unknown'
        
    def is_invalid_result(x):
        return pd.isna(x) or str(x) not in ['W', 'L', 'D']
        
    missing_opp = df_games['opponent_id'].apply(is_invalid_opp).sum()
    missing_p_rating = df_games['player_pre_rating'].apply(is_invalid_rating).sum()
    missing_o_rating = df_games['opponent_pre_rating'].apply(is_invalid_rating).sum()
    missing_res = df_games['result'].apply(is_invalid_result).sum()
    
    print(f"  Invalid opponent_id: {missing_opp}")
    print(f"  Invalid player_pre_rating: {missing_p_rating}")
    print(f"  Invalid opponent_pre_rating: {missing_o_rating}")
    print(f"  Invalid result: {missing_res}")
    
    valid_mask = (
        ~df_games['opponent_id'].apply(is_invalid_opp) &
        ~df_games['player_pre_rating'].apply(is_invalid_rating) &
        ~df_games['opponent_pre_rating'].apply(is_invalid_rating) &
        ~df_games['result'].apply(is_invalid_result)
    )
    usable_games = valid_mask.sum()
    print(f"\nTotal Usable Games: {usable_games}")
    
    # 4. Rating Difference Summary
    print("\n--- Rating Difference (player - opponent) ---")
    # Clean provisional 'P' flags (e.g. 1500P4 -> 1500) for numeric calculation
    def clean_rating(x):
        try:
            return float(str(x).split('P')[0])
        except:
            return np.nan
            
    df_valid = df_games[valid_mask].copy()
    df_valid['p_rate_clean'] = df_valid['player_pre_rating'].apply(clean_rating)
    df_valid['o_rate_clean'] = df_valid['opponent_pre_rating'].apply(clean_rating)
    df_valid['rating_diff'] = df_valid['p_rate_clean'] - df_valid['o_rate_clean']
    
    if not df_valid['rating_diff'].isna().all():
        diff_stats = df_valid['rating_diff'].describe()
        print(f"  Mean: {diff_stats['mean']:.2f}")
        print(f"  Std:  {diff_stats['std']:.2f}")
        print(f"  Min:  {diff_stats['min']:.2f}")
        print(f"  Max:  {diff_stats['max']:.2f}")
    else:
        print("  Could not calculate rating differences.")
        
    # 5. Time Control Diversity
    print("\n--- Time Control Diversity ---")
    if os.path.exists(events_file):
        df_events = pd.read_csv(events_file, dtype=str)
        if 'normalized_time_control' in df_events.columns:
            tc_counts = df_events['normalized_time_control'].value_counts()
            tc_props = df_events['normalized_time_control'].value_counts(normalize=True) * 100
            for tc, count in tc_counts.items():
                prop = tc_props.get(tc, 0)
                print(f"  {tc}: {count} ({prop:.1f}%)")
        else:
            print("  Column 'normalized_time_control' missing in raw_events.")
    else:
        print("  raw_events.csv not found.")
        
    # 6. Readiness Verdict
    print("\n" + "="*40)
    print("READINESS VERDICT")
    print("="*40)
    
    reasons = []
    ready = True
    
    if usable_games < 1000:
        ready = False
        reasons.append(f"Insufficient usable games ({usable_games} < 1000).")
        
    if unique_players < 25:
        ready = False
        reasons.append(f"Insufficient focal player diversity ({unique_players} < 25).")
        
    if unique_opponents < 200:
        ready = False
        reasons.append(f"Low opponent diversity ({unique_opponents}).")
        
    if 'W' in result_counts and result_props.get('W', 0) > 85.0:
        ready = False
        reasons.append("Catastrophic class skew (>85% Wins).")
        
    if 'diff_stats' in locals() and pd.notna(diff_stats['std']) and diff_stats['std'] < 1.0:
        ready = False
        reasons.append("Rating difference variance is practically zero.")
        
    if ready:
        print(">>> READY FOR PHASE 3 <<<")
        print("Dataset meets all threshold requirements for cleaning, EDA, and modeling.")
    else:
        print(">>> NOT READY <<<")
        for r in reasons:
            print(f"- {r}")
        print("Consider increasing collection limits in config.yaml and resuming collection.")
        
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
