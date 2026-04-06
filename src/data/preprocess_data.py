import pandas as pd
import re

def preprocess_data(df: pd.DataFrame, target_col: str='burnout_level') -> pd.DataFrame:
    """
    Basic cleaning
    - trim col names
    - drop ID col
    - map category cols to numeric
    """

    df.columns = df.columns.str.strip() 

    cols_to_drop = [col for col in df.columns if re.search(r'ID', col, re.IGNORECASE)]
    df = df.drop(columns=cols_to_drop)

    num_col = df.select_dtypes(exclude='object').columns
    df[num_col] = df[num_col].fillna(0)

    if target_col in df.columns and df[target_col].dtype == "object":
        df[target_col] = df[target_col].str.strip().map({'Low': 0, 'Medium': 1, 'High': 2})

    return df