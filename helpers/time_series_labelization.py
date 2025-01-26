"""
Python script used to define functions to labelize time series data.
"""

import numpy as np
import pandas as pd
from abc import abstractmethod, ABC

from helpers.classes import BTC, Tweets

# Time intersection of both of our datasets
START_TIME = pd.Timestamp('2017-08-17 05:00:00+00:00')
END_TIME = pd.Timestamp('2019-11-23 14:00:20+00:00')


class Labelization(ABC):
    @abstractmethod
    def labelize(self, threshold: float) -> np.array:
        """
        Transform the data into a time series of labels.

        :param threshold: The threshold to use for the labelization.
        :return: The labelized time series data.
        """
        pass  # abstract method. Define below for BTC and Twitter

    @abstractmethod
    def apply_granularity(self, data: pd.DataFrame, granularity: str) -> pd.DataFrame:
        """
        Apply the granularity to the data.

        :param data: The data to apply the granularity.
        :param granularity: The granularity to apply.
        :return: The data with the applied granularity.
        """
        pass  # abstract method. Define below for BTC and Twitter

    @staticmethod
    def label(rate, threshold) -> int:
        """Map rate to label. Only values bigger or equal to 0 are chosen to respect input requirement of our chosen
        Transfer Entropy method."""

        if rate > threshold:
            return 2  # BULLISH
        elif rate < -threshold:
            return 0  # BEARISH
        else:
            return 1  # NEUTRAL

    @staticmethod
    def print_time_series_info(data):
        total = data.shape[0]
        label_dist = "".join([f"\n\t{label}: {count} ({(count/total*100):.2f}%)" for label, count in data['label'].value_counts().items()])
        print(f"First timestamp: {data.index[0]}\n"
              f"Last timestamp: {data.index[-1]}\n"
              f"Number of elements: {len(data)}\n"
              f"Number of NaN: {data.isna().sum().sum()}\n"
              f"Label distribution: {label_dist}")


class TweetsLabelization(Tweets, Labelization):
    def __init__(self, clean_tweets_path: str):
        super().__init__(clean_tweets_path)
        self.data.set_index('timestamp', inplace=True)
        self.data = self.data.loc[START_TIME:END_TIME]

    def apply_granularity(self, data: pd.DataFrame, granularity: str = 'h') -> pd.DataFrame:
        return data.resample(granularity).agg({'weighted_sentiment': 'mean'}).fillna(0)

    def labelize(self, threshold: float = 0.33, granularity: str = 'h', verbose: bool = False) -> np.array:
        """
        Transform the data into a time series of labels using a weighted average of our sentiment, using the sentiment
        score provided by the LLM.

        :param threshold: Limit around 0 [-threshold,+threshold] to map sentiment scores between NEUTRAL and BEARISH/BULLISH.
        :param granularity: Time granularity to apply. Must be more than hour, since it is our reference granularity.
        :param verbose: boolean to decide if the program should display the time series info when function called.
        :return: The labelized time series data.
        """
        data_copy = self.data.copy()
        mapped_sentiment = {'BEARISH': -1, 'NEUTRAL': 0, 'BULLISH': 1}

        # Mapping from sentiment string to sentiment int class
        data_copy['mapped_sentiment'] = data_copy['sentiment'].map(mapped_sentiment).copy()

        # Weight the sentiment class with their sentiment score given by the LLM during data_processing step
        data_copy['weighted_sentiment'] = (data_copy['mapped_sentiment'] * data_copy['sentiment_score']).copy()

        # Aggregate all tweets in a single value representing the given granularity time step
        data_copy = self.apply_granularity(data_copy, granularity)

        # From aggregation score, come back to a final sentiment int class representing the average sentiment of the time step
        data_copy['label'] = data_copy['weighted_sentiment'].apply(lambda x: self.label(x, threshold))

        if verbose:
            print(f"{'#' * 10} Tweets Labelization {'#' * 10}")
            self.print_time_series_info(data_copy)

        return data_copy['label'].astype(int).to_numpy()


class BtcLabelization(BTC, Labelization):
    def __init__(self, clean_btc_path: str):
        super().__init__(clean_btc_path)
        self.data = self.data.loc[START_TIME:END_TIME]

    def apply_granularity(self, data: pd.DataFrame, granularity: str = 'h') -> pd.DataFrame:
        return data.resample(granularity).agg({'close': 'last'}).dropna()

    def labelize(self, threshold: float = 0.01, granularity: str = 'h', verbose: bool = False) -> np.array:
        """
        Transform the data into a time series of labels.

        :param threshold: The limit around 0 [-threshold,+threshold] to map returns between NEUTRAL and BEARISH/BULLISH.
        :param granularity: The time granularity to apply. Must be more than hour.
        :param verbose: boolean to decide if the program should display the time series info when function called.
        :return: The labelized time series data.
        """
        data_copy = self.data.copy()
        data_copy = self.apply_granularity(data_copy, granularity)
        self._compute_returns(data_copy)

        # Decide if btc is in bullish, bearish or neutral mode in function of its return
        data_copy['label'] = data_copy['returns'].apply(lambda x: self.label(x, threshold))

        if verbose:
            print(f"{'#' * 10} BTC Labelization {'#' * 10}")
            self.print_time_series_info(data_copy)

        return data_copy['label'].astype(int).to_numpy()

    @staticmethod
    def _compute_returns(data: pd.DataFrame):
        """Simply create a new column and store the return value to know what was the hourly evolution"""
        data['returns'] = data['close'].pct_change().fillna(0)
