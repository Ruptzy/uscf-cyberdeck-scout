import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def append_to_csv(data_dicts, file_path):
    """Appends a list of dictionaries to a CSV file safely."""
    if not data_dicts:
        return
        
    df = pd.DataFrame(data_dicts)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if not os.path.exists(file_path):
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, mode='a', header=False, index=False)
        
    logger.debug(f"Wrote {len(df)} rows to {file_path}")
