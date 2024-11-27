"""
Python script used to define function allowing easy data analysis.
"""

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


class TweetsAnalyzer:
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
        return df

    def performance_distribution(self) -> pd.DataFrame:
        """Plot the distribution of the likes, replies and retweet column to assess the influence of a tweet"""
        # Set up the figure
        fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

        colors = ['blue', 'green', 'orange']
        metrics = ['likes', 'replies', 'retweets']

        for idx, (color, metric) in enumerate(zip(colors, metrics)):
            # Plot distributions
            sns.histplot(self.data[metric], ax=axes[idx], color=color)
            axes[idx].set_title(f'Distribution of {metric}')
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_yscale('log')  # Set y-axis to logarithmic scale

        # Adjust layout
        plt.tight_layout()
        plt.show()

        return pd.concat([self.data['likes'].describe(percentiles=[0.89, 0.99]),
                          self.data['replies'].describe(percentiles=[0.95, 0.99]),
                          self.data['retweets'].describe(percentiles=[0.88, 0.99])],
                         axis=1)

class BTCAnalyzer:
    def __init__(self, file_path: str):
        self.data = self.load_data(file_path)

    def load_data(self, file_path: str) -> pd.DataFrame:
        return pd.read_parquet(file_path)

    def plot_price(self):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(self.data.index, self.data['high'])
        ax.set_title('BTC Price over time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price in $')
        plt.show()

