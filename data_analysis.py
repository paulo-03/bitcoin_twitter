"""
Python script used to define function allowing easy data analysis.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import datetime


class TweetsAnalyzer:
    def __init__(self, file_path: str):
        self.data = self._load_data_in_chunk(file_path)
        self._basic_info()

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
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values(by='timestamp', ascending=True).reset_index()

    def _basic_info(self):
        """Print basic information at the initialization of the class TweetsAnalyzer"""
        print(f"{'#' * 10} Basic information {'#' * 10}\n"
              f"- Shape of dataset: {self.data.shape}\n"
              f"- Data starting from {self.data['timestamp'].min()} to {self.data['timestamp'].max()}\n"
              f"- Data Types of each column:\n{self.data.dtypes}")

    def performance_distribution(self) -> pd.DataFrame:
        """Plot the distribution of the likes, replies and retweet column to assess the influence of a tweet"""
        # Set up the figure
        fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=False)

        colors = ['blue', 'green', 'orange']
        metrics = ['likes', 'replies', 'retweets']

        for idx, (color, metric) in enumerate(zip(colors, metrics)):
            # Plot distributions
            sns.histplot(self.data[metric], ax=axes[idx], color=color,
                         bins=range(0, self.data[metric].max() + 100, 100))
            axes[idx].set_title(f'Distribution of {metric}')
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_yscale('log')  # Set y-axis to logarithmic scale

        # Adjust layout
        plt.tight_layout()
        plt.show()

        return pd.concat([self.data['likes'].describe(percentiles=np.linspace(start=0.9, stop=0.99, num=10)),
                          self.data['replies'].describe(percentiles=np.linspace(start=0.9, stop=0.99, num=10)),
                          self.data['retweets'].describe(percentiles=np.linspace(start=0.9, stop=0.99, num=10))],
                         axis=1)

    def pertinent(self, start_date: datetime, end_date: datetime):
        """Plot a graph to easily see the distribution of pertinent/meaningful/impactful tweets meaning having more
        than at least 10 likes/replies/retweets"""
        filtered_data = self.data[(self.data['timestamp'] >= start_date) & (self.data['timestamp'] < end_date)]
        thresholds = np.linspace(start=10, stop=500, num=50)
        tweet_number = []

        for threshold in tqdm(thresholds, desc="Computing the number of tweet per threshold of perf", unit="threshold"):
            tweet_number.append((filtered_data[['likes', 'replies', 'retweets']] > threshold).any(axis=1).sum())

        plt.figure(figsize=(15, 6))
        plt.plot(thresholds, tweet_number)

        plt.show()


class BTCAnalyzer:
    def __init__(self, file_path: str):
        self.data = self._load_data(file_path)
        self.data.index = pd.to_datetime(self.data.index).tz_localize('UTC')  # Make sure the time zone is UTC
        self._basic_info()

    @staticmethod
    def _load_data(file_path: str) -> pd.DataFrame:
        return pd.read_parquet(file_path)

    def _basic_info(self):
        """Print basic information at the initialization of the class BTCAnalyzer"""
        print(f"{'#' * 10} Basic information {'#' * 10}\n"
              f"- Shape of dataset: {self.data.shape}\n"
              f"- Average time granularity: {self.data.index.to_series().diff().mean()} (~1 min granularity)\n"
              f"- Data starting from {self.data.index.min()} to {self.data.index.max()}\n"
              f"- Data Types of each column:\n{self.data.dtypes}")

    def _compute_return(self):
        """Private function to compute returns of bitcoin instead of only having its price"""
        self.data['return'] = self.data['close'].pct_change() * 100

    def plot_price(self, start_date: datetime, end_date: datetime):
        """Plot the price evolution of the bitcoin and allows the user to select the time range he wants"""
        filtered_price = self.data[(self.data.index >= start_date) & (self.data.index < end_date)]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(filtered_price.index, filtered_price['close'])
        ax.set_title('BTC Price over time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price in $')
        plt.show()

    def plot_return(self, start_date: datetime, end_date: datetime):
        """Plot the return evolution of the bitcoin and allows the user to select the time range he wants"""
        self._compute_return()
        filtered_return = self.data[(self.data.index >= start_date) & (self.data.index < end_date)]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(filtered_return.index, filtered_return['return'])
        ax.set_title('BTC Return over time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Return in %')
        plt.show()
