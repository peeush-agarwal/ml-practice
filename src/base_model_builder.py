from typing import Tuple
import pandas as pd

from sklearn.model_selection import train_test_split


def load_data(file_path:str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def split_data(df:pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(df, test_size=0.2, random_state=41)
