"""
Python script used to define functions to labelize time series data.
"""
from abc import abstractmethod, ABC

import numpy as np
import pandas as pd

from helpers.classes import BTC, Tweets

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
        pass

    @abstractmethod
    def apply_granularity(self, data: pd.DataFrame, granularity: str) -> pd.DataFrame:
        """
        Apply the granularity to the data.
        :param data: The data to apply the granularity.
        :param granularity: The granularity to apply.
        :return: The data with the applied granularity.
        """
        pass

    def label(self, rate, threshold) -> int:
        if rate > threshold:
            return 2  # BULLISH
        elif rate < -threshold:
            return 0  # BEARISH
        else:
            return 1  # NEUTRAL

    def print_time_series_info(self, data):
        print(f"First timestamp: {data.index[0]}\n"
              f"Last timestamp: {data.index[-1]}\n"
              f"Number of elements: {len(data)}\n"
              f"Number of NaN: {data.isna().sum().sum()}\n"
              f"Number of each label: {data['label'].value_counts()}")


class TweetsLabelization(Tweets, Labelization):
    def __init__(self, clean_tweets_path: str):
        super().__init__(clean_tweets_path)
        self.data.set_index('timestamp', inplace=True)
        self.data = self.data.loc[START_TIME:END_TIME]

    def apply_granularity(self, data: pd.DataFrame, granularity: str = 'h') -> pd.DataFrame:
        return data.resample(granularity).agg({'weighted_sentiment': 'mean'}).fillna(0)

    def labelize(self, threshold: float = 0.33, granularity: str = 'h') -> np.array:
        """
        Transform the data into a time series of labels.
        :param threshold: The limit around 0 [-threshold,+threshold] to map sentiment scores between NEUTRAL and BEARISH/BULLISH.
        :param granularity: The time granularity to apply. Must be more than hour.
        :return: The labelized time series data.
        """
        data_copy = self.data.copy()
        mapped_sentiment = {'BEARISH': -1, 'NEUTRAL': 0, 'BULLISH': 1}
        data_copy['mapped_sentiment'] = data_copy['sentiment'].map(mapped_sentiment).copy()
        data_copy['weighted_sentiment'] = (data_copy['mapped_sentiment'] * data_copy['sentiment_score']).copy()
        data_copy = self.apply_granularity(data_copy, granularity)
        data_copy['label'] = data_copy['weighted_sentiment'].apply(lambda x: self.label(x, threshold))
        print(f"{'#' * 10} Tweets Labelization {'#' * 10}")
        self.print_time_series_info(data_copy)
        return data_copy['label'].astype(int).to_numpy()

class BtcLabelization(BTC, Labelization):
    def __init__(self, clean_btc_path: str):
        super().__init__(clean_btc_path)
        self.data = self.data.loc[START_TIME:END_TIME]

    def apply_granularity(self, data: pd.DataFrame, granularity: str = 'h') -> pd.DataFrame:
        return data.resample(granularity).agg({'close': 'last'}).dropna()

    def labelize(self, threshold: float = 0.01, granularity: str = 'h') -> np.array:
        """
        Transform the data into a time series of labels.
        :param threshold: The limit around 0 [-threshold,+threshold] to map returns between NEUTRAL and BEARISH/BULLISH.
        :param granularity: The time granularity to apply. Must be more than hour.
        :return: The labelized time series data.
        """
        data_copy = self.data.copy()
        data_copy = self.apply_granularity(data_copy, granularity)
        self._compute_returns(data_copy)
        data_copy['label'] = data_copy['returns'].apply(lambda x: self.label(x, threshold))
        print(f"{'#' * 10} BTC Labelization {'#' * 10}")
        self.print_time_series_info(data_copy)
        return data_copy['label'].astype(int).to_numpy()

    def _compute_returns(self, data: pd.DataFrame):
        """Simply create a new column and store the return value to know what was the hourly evolution"""
        data['returns'] = data['close'].pct_change().fillna(0)

def main() -> None:
    tweets_labelization = TweetsLabelization('../clean_data/twitter.parquet')
    btc_labelization = BtcLabelization('../clean_data/btc.parquet')

    tweets_labels = tweets_labelization.labelize(granularity='h')
    btc_labels = btc_labelization.labelize(granularity='h')

if __name__ == '__main__':
    main()