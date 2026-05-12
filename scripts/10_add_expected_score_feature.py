import os
import sys
import pandas as pd
import numpy as np

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    in_file = os.path.join(base_dir, 'data', 'processed', 'features_v1.csv')
    out_file = os.path.join(base_dir, 'data', 'processed', 'features_v2_expected_score.csv')
    
    if not os.path.exists(in_file):
        print(f"Error: Could not find {in_file}")
        sys.exit(1)
        
    # 1. Load Data
    df = pd.read_csv(in_file)
    
    print("="*50)
    print("PHASE 8: FEATURE ENGINEERING (EXPECTED SCORE)")
    print("="*50)
    
    # 2. Create Expected Score Feature
    # Using the standard chess Elo expected score formula:
    # E_player = 1 / (1 + 10^((R_opponent - R_player) / 400))
    # This is entirely leakage-safe as it relies strictly on pre-game ratings.
    
    df['expected_score_player'] = 1 / (1 + 10 ** ((df['opponent_pre_rating'] - df['player_pre_rating']) / 400))
    
    # 3. Output the new processed dataset
    df.to_csv(out_file, index=False)
    
    # 4. Print Summary Information
    print("\n--- Dataset Summary ---")
    print(f"Total rows: {len(df)}")
    print(f"New dataset saved to: {os.path.basename(out_file)}")
    
    print("\n--- Feature Statistics (expected_score_player) ---")
    print(f"  Min:  {df['expected_score_player'].min():.4f}")
    print(f"  Max:  {df['expected_score_player'].max():.4f}")
    print(f"  Mean: {df['expected_score_player'].mean():.4f}")
    
    print("\n--- Target Correlations ---")
    corr_binary = df['expected_score_player'].corr(df['target_binary'])
    corr_multi = df['expected_score_player'].corr(df['target_multiclass'])
    
    print(f"  Correlation with target_binary:     {corr_binary:.4f}")
    print(f"  Correlation with target_multiclass: {corr_multi:.4f}")
    
    print("\n--- Feature Context Note ---")
    print("Why add this feature?")
    print("- Logistic Regression models struggled to naturally optimize raw rating inputs into the correct mathematical shape.")
    print("- Tree-based models hit a hard accuracy ceiling on unseen test data, implying that simple subtraction (rating_diff) wasn't giving them enough context.")
    print("- By engineering the theoretical 'Elo Expected Score', we explicitly feed the algorithm a domain-specific, non-linear transformation. It maps raw rating scales into a true pre-game win probability bounded perfectly between 0.0 and 1.0, providing a much cleaner mathematical signal for our models to branch on.")
    print("="*50 + "\n")

if __name__ == '__main__':
    main()
