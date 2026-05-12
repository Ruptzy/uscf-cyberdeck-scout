import pandas as pd
import logging

logger = logging.getLogger(__name__)

def validate_poc_games(games_df):
    """Validates games for the proof-of-concept with strict usability checks. Color is considered optional."""
    total = len(games_df)
    if total == 0:
        return pd.DataFrame(), {}
        
    # Validation summary counts (Color is no longer tracked as a failure reason here)
    summary = {
        "missing_opponent_id": games_df['opponent_id'].apply(lambda x: pd.isna(x) or str(x) == 'Unknown' or str(x).startswith('OPP_')).sum(),
        "missing_player_pre_rating": games_df['player_pre_rating'].apply(lambda x: pd.isna(x) or str(x) == 'Unknown').sum(),
        "missing_opponent_pre_rating": games_df['opponent_pre_rating'].apply(lambda x: pd.isna(x) or str(x) == 'Unknown').sum(),
        "missing_result": games_df['result'].apply(lambda x: pd.isna(x) or str(x) not in ['W', 'L', 'D']).sum()
    }
    
    # Strictly reject placeholders (Color is permitted to be Unknown)
    valid_df = games_df.copy()
    valid_df = valid_df[~valid_df['opponent_id'].apply(lambda x: pd.isna(x) or str(x) == 'Unknown' or str(x).startswith('OPP_'))]
    valid_df = valid_df[~valid_df['player_pre_rating'].apply(lambda x: pd.isna(x) or str(x) == 'Unknown')]
    valid_df = valid_df[~valid_df['opponent_pre_rating'].apply(lambda x: pd.isna(x) or str(x) == 'Unknown')]
    valid_df = valid_df[valid_df['result'].isin(['W', 'L', 'D'])]
    
    # Drop duplicates
    valid_df = valid_df.drop_duplicates(subset=['game_id'])
    
    usable = len(valid_df)
    logger.info(f"Validation: {usable}/{total} games are perfectly usable.")
    return valid_df, summary
