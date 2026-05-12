import os
import sys
import pandas as pd
import numpy as np

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    in_file = os.path.join(base_dir, 'data', 'processed', 'features_v1.csv')
    out_file = os.path.join(base_dir, 'data', 'processed', 'features_v3_recency.csv')
    
    if not os.path.exists(in_file):
        print(f"Error: Could not find {in_file}")
        sys.exit(1)
        
    # 1. Load Data
    df = pd.read_csv(in_file)
    
    print("="*50)
    print("PHASE 10: FEATURE ENGINEERING (RECENCY & MOMENTUM)")
    print("="*50)
    
    df['event_end_date'] = pd.to_datetime(df['event_end_date'], errors='coerce')
    
    # Sort chronologically to ensure time continuity
    df = df.sort_values(by=['player_id', 'event_end_date', 'game_id']).reset_index(drop=True)
    
    # Initialize new columns
    df['player_games_last_30d'] = 0
    df['player_games_last_90d'] = 0
    df['player_games_last_365d'] = 0
    df['player_recent_avg_opponent_rating_90d'] = np.nan
    df['player_recent_win_rate_90d'] = np.nan
    
    # 2. Iterate and calculate rolling features safely
    print("Calculating strict historical rolling features (this may take a moment)...")
    
    # Group by focal player to build individual histories
    for player_id, group in df.groupby('player_id'):
        for i, row in group.iterrows():
            current_date = row['event_end_date']
            if pd.isna(current_date):
                continue
                
            # STRICT LEAKAGE RULE: strictly earlier dates only
            # No same day events, no future events
            past_mask = (group['event_end_date'] < current_date)
            past_games = group[past_mask]
            
            if past_games.empty:
                continue
                
            # Calculate days back for each historical game
            days_diff = (current_date - past_games['event_end_date']).dt.days
            
            # Create masks for time windows
            mask_30d = days_diff <= 30
            mask_90d = days_diff <= 90
            mask_365d = days_diff <= 365
            
            # Apply Activity Counts
            df.at[i, 'player_games_last_30d'] = mask_30d.sum()
            df.at[i, 'player_games_last_90d'] = mask_90d.sum()
            df.at[i, 'player_games_last_365d'] = mask_365d.sum()
            
            # Apply Momentum / Strength features (using 90d window)
            recent_90d_games = past_games[mask_90d]
            if not recent_90d_games.empty:
                df.at[i, 'player_recent_avg_opponent_rating_90d'] = recent_90d_games['opponent_pre_rating'].mean()
                df.at[i, 'player_recent_win_rate_90d'] = recent_90d_games['target_binary'].mean()
                
    # 3. Output the new processed dataset
    df.to_csv(out_file, index=False)
    
    # 4. Summary
    print("\n--- Dataset Summary ---")
    print(f"Total rows: {len(df)}")
    print(f"New dataset saved to: {os.path.basename(out_file)}")
    
    new_cols = [
        'player_games_last_30d', 
        'player_games_last_90d', 
        'player_games_last_365d',
        'player_recent_avg_opponent_rating_90d',
        'player_recent_win_rate_90d'
    ]
    
    print("\n--- Feature Statistics ---")
    print(df[new_cols].describe().round(3).loc[['mean', 'std', 'min', 'max']])
    
    print("\n--- Cold Start (Zero Recent Games) ---")
    print(f"0 games in 30d:  {(df['player_games_last_30d'] == 0).mean()*100:.1f}%")
    print(f"0 games in 90d:  {(df['player_games_last_90d'] == 0).mean()*100:.1f}%")
    print(f"0 games in 365d: {(df['player_games_last_365d'] == 0).mean()*100:.1f}%")
    
    print("\n--- Feature Context Note ---")
    print("Why add recency and activity features?")
    print("- The 'Expected Score' feature failed because static USCF ratings are just snapshots; they don't capture momentum, 'rust', or recent growth trajectories.")
    print("- A player who has played 20 games in the last 90 days with a 75% win rate is demonstrably 'hotter' (and likely underrated) compared to a player with the exact same Elo who hasn't played a tournament in 3 years.")
    print("- These rolling features explicitly teach the model about player momentum and activity level, providing the dynamic context that pure static ratings lack, without introducing data leakage.")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()
