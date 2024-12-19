"""
Implement the parent class for BTC and Twitter to avoid redundancy in common functions.
"""

import pandas as pd
from tqdm import tqdm


class BTC:
    def __init__(self, file_path):
        self.data: pd.DataFrame = self._load_data(file_path)

    @staticmethod
    def _load_data(file_path) -> pd.DataFrame:
        """Load the data and set the index as the time stamp and make sure it is defined as a Data Time object in
        UTC format."""
        data = pd.read_parquet(file_path)
        data.index = pd.to_datetime(data.index, utc=True)
        return data


class Tweets:
    def __init__(self, file_path: str):
        self.data = self._load_data_in_chunk(file_path)

    @staticmethod
    def _load_data_in_chunk(file_path, chunk_size: int = 1000000) -> pd.DataFrame:
        """We load data in chunk just to give feedback to the user while loading data since it can be relatively long."""
        # Initialize an empty DataFrame to collect chunk data
        df = pd.DataFrame()

        # Read the file in chunks with tqdm to know the progression
        for chunk in tqdm(pd.read_csv(file_path, compression='zip', delimiter=';',
                                      chunksize=chunk_size, on_bad_lines='skip', low_memory=False,
                                      skiprows=0, lineterminator='\n'),
                          total=int(16889765 / chunk_size),
                          unit='chunks'):
            df = pd.concat([df, chunk])  # Append chunk to the DataFrame
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        return df.sort_values(by='timestamp', ascending=True).reset_index()
