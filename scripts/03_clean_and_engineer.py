import os
import sys
import pandas as pd
import numpy as np
import yaml

def main():
    # 1. Configuration & Paths
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.yaml'))
    base_dir = os.path.dirname(config_path)
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        data_dir = config['project'].get('data_dir', 'data')
    except Exception:
        data_dir = 'data'
        
    tables_dir = os.path.join(base_dir, data_dir, 'raw', 'tables')
    processed_dir = os.path.join(base_dir, data_dir, 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    games_file = os.path.join(tables_dir, 'raw_games.csv')
    events_file = os.path.join(tables_dir, 'raw_events.csv')
    out_file = os.path.join(processed_dir, 'features_v1.csv')
    
    if not os.path.exists(games_file) or not os.path.exists(events_file):
        print("Error: Missing raw data files.")
        sys.exit(1)
        
    # 2. Load Data
    df_g = pd.read_csv(games_file, dtype=str)
    df_e = pd.read_csv(events_file, dtype=str)
    
    raw_rows_loaded = len(df_g)
    
    # 3. Clean Game Rows
    # Drop exact duplicates on game_id
    df_g = df_g.drop_duplicates(subset=['game_id'])
    duplicates_dropped = raw_rows_loaded - len(df_g)
    
    # Identify valid rows (No placeholders, no missing values)
    def is_invalid_opp(x):
        return pd.isna(x) or str(x) == 'Unknown' or str(x).startswith('OPP_')
        
    def is_invalid_rating(x):
        return pd.isna(x) or str(x) == 'Unknown'
        
    def is_invalid_result(x):
        return pd.isna(x) or str(x) not in ['W', 'L', 'D']
        
    valid_mask = (
        ~df_g['player_id'].isna() &
        ~df_g['opponent_id'].apply(is_invalid_opp) &
        ~df_g['player_pre_rating'].apply(is_invalid_rating) &
        ~df_g['opponent_pre_rating'].apply(is_invalid_rating) &
        ~df_g['result'].apply(is_invalid_result)
    )
    
    invalid_rows_dropped = len(df_g) - valid_mask.sum()
    df_clean = df_g[valid_mask].copy()
    
    # Clean provisional 'P' flags from ratings safely
    def strip_rating(x):
        try:
            return str(x).split('P')[0]
        except:
            return np.nan
            
    df_clean['player_pre_rating'] = pd.to_numeric(df_clean['player_pre_rating'].apply(strip_rating), errors='coerce')
    df_clean['opponent_pre_rating'] = pd.to_numeric(df_clean['opponent_pre_rating'].apply(strip_rating), errors='coerce')
    
    # Drop rows where rating parsing failed
    pre_rating_drops = len(df_clean)
    df_clean = df_clean.dropna(subset=['player_pre_rating', 'opponent_pre_rating'])
    invalid_ratings_dropped = pre_rating_drops - len(df_clean)
    
    df_clean['player_pre_rating'] = df_clean['player_pre_rating'].astype(int)
    df_clean['opponent_pre_rating'] = df_clean['opponent_pre_rating'].astype(int)
    
    # 4. Merge in Event-Level Information & Parse Dates safely
    df_e_subset = df_e[['event_id', 'end_date', 'normalized_time_control']].drop_duplicates()
    df_e_subset['end_date'] = pd.to_datetime(df_e_subset['end_date'], errors='coerce')
    
    df_merged = df_clean.merge(df_e_subset, on='event_id', how='left')
    
    # Drop rows where date parsing failed
    pre_date_drops = len(df_merged)
    df_merged = df_merged.dropna(subset=['end_date'])
    invalid_dates_dropped = pre_date_drops - len(df_merged)
    
    # 5. Create Core Modeling Features
    df_merged['rating_diff'] = df_merged['player_pre_rating'] - df_merged['opponent_pre_rating']
    
    # 6. Create Machine Learning Targets
    target_multi_map = {'W': 2, 'D': 1, 'L': 0}
    target_bin_map = {'W': 1, 'D': 0, 'L': 0}
    
    df_merged['target_multiclass'] = df_merged['result'].map(target_multi_map)
    df_merged['target_binary'] = df_merged['result'].map(target_bin_map)
    
    # 7. Select & Rename Final Columns
    final_cols = {
        'game_id': 'game_id',
        'event_id': 'event_id',
        'end_date': 'event_end_date',
        'player_id': 'player_id',
        'opponent_id': 'opponent_id',
        'player_pre_rating': 'player_pre_rating',
        'opponent_pre_rating': 'opponent_pre_rating',
        'rating_diff': 'rating_diff',
        'normalized_time_control': 'time_control',
        'result': 'result_raw',
        'target_multiclass': 'target_multiclass',
        'target_binary': 'target_binary'
    }
    
    df_final = df_merged[list(final_cols.keys())].rename(columns=final_cols)
    
    # Handle Time Control missingness and apply the "Mixed" logic rule
    df_final['time_control'] = df_final['time_control'].fillna('Unknown')
    tc_rule = "Kept 'Mixed' as a distinct, valid categorical time control instead of grouping it as Unknown."
    
    # 8. Output Final Dataset
    df_final.to_csv(out_file, index=False)
    
    # 9. Print Concise Summary
    print("\n" + "="*40)
    print("PHASE 3: CLEANING & ENGINEERING SUMMARY")
    print("="*40)
    print(f"Raw rows loaded: {raw_rows_loaded}")
    print(f"Duplicate rows dropped: {duplicates_dropped}")
    print(f"Structurally invalid rows dropped: {invalid_rows_dropped}")
    print(f"Invalid numeric ratings dropped: {invalid_ratings_dropped}")
    print(f"Invalid event dates dropped: {invalid_dates_dropped}")
    print(f"Final modeled row count: {len(df_final)}")
    print(f"\nUnique focal players: {df_final['player_id'].nunique()}")
    print(f"Unique opponents: {df_final['opponent_id'].nunique()}")
    
    print("\n--- Result Distribution ---")
    dist = df_final['result_raw'].value_counts(normalize=True) * 100
    for res, prop in dist.items():
        print(f"  {res}: {prop:.1f}%")
        
    print("\n--- Time Control Distribution ---")
    tc_dist = df_final['time_control'].value_counts(normalize=True) * 100
    for tc, prop in tc_dist.items():
        print(f"  {tc}: {prop:.1f}%")
        
    print("\n--- Time Control Rule Applied ---")
    print(f"  {tc_rule}")
    
    print(f"\n>>> Saved modeling dataset to: {out_file} <<<")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
