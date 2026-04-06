import pandas as pd
import os

def load_data(path: str) -> pd.DataFrame:
    """Load dataset (csv file) from 'path' in to a DataFrame"""

    if not os.path.exists(path):
        raise FileNotFoundError(f'Not found file in path: {path}')
    
    return pd.read_csv(path)