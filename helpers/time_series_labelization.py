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
    def apply_granularity(self, granularity: str = 'h') -> None:
        """
        Apply the granularity to the data.
        :param granularity: The granularity to apply.
        """
        pass

    def label(self, rate, threshold) -> int:
        if rate > threshold:
            return 2  # BULLISH
        elif rate < -threshold:
            return 0  # BEARISH
        else:
            return 1  # NEUTRAL


class TweetsLabelization(Tweets, Labelization):
    def __init__(self, clean_tweets_path: str):
        super().__init__(clean_tweets_path)
        self.data.set_index('timestamp', inplace=True)
        self.data = self.data.loc[START_TIME:END_TIME]

    def apply_granularity(self, granularity: str = 'h') -> None:
        self.data = self.data.resample(granularity).agg({'weighted_sentiment': 'mean'}).fillna(0)

    def labelize(self, threshold: float = 0.33, granularity: str = 'h') -> np.array:
        """
        Transform the data into a time series of labels.
        :param threshold: The limit around 0 [-threshold,+threshold] to map sentiment scores between NEUTRAL and BEARISH/BULLISH.
        :param granularity: The time granularity to apply. Must be more than hour.
        :return: The labelized time series data.
        """
        mapped_sentiment = {'BEARISH': -1, 'NEUTRAL': 0, 'BULLISH': 1}
        self.data['mapped_sentiment'] = self.data['sentiment'].map(mapped_sentiment).copy()
        self.data['weighted_sentiment'] = (self.data['mapped_sentiment'] * self.data['sentiment_score']).copy()
        self.apply_granularity(granularity)
        self.data['final_sentiment'] = self.data['weighted_sentiment'].apply(lambda x: self.label(x, threshold))
        print(self.data.head())
        return self.data['final_sentiment'].astype(int).to_numpy()


class BtcLabelization(BTC, Labelization):
    def __init__(self, clean_btc_path: str):
        super().__init__(clean_btc_path)
        self.data = self.data.loc[START_TIME:END_TIME]

    def apply_granularity(self, granularity: str = 'h'):
        self.data = self.data.resample(granularity).agg({'close': 'last'}).dropna()

    def labelize(self, threshold: float = 0.001, granularity: str = 'h') -> np.array:
        """
        Transform the data into a time series of labels.
        :param threshold: The limit around 0 [-threshold,+threshold] to map returns between NEUTRAL and BEARISH/BULLISH.
        :param granularity: The time granularity to apply. Must be more than hour.
        :return: The labelized time series data.
        """
        self.apply_granularity(granularity)
        self._compute_returns()
        self.data['label'] = self.data['returns'].apply(lambda x: self.label(x, threshold))
        print(self.data.head())
        return self.data['label'].astype(int).to_numpy()

    def _compute_returns(self):
        """Simply create a new column and store the return value to know what was the hourly evolution"""
        self.data['returns'] = self.data['close'].pct_change().fillna(0)


def main() -> None:
    tweets_labelization = TweetsLabelization('../clean_data/twitter.parquet')
    btc_labelization = BtcLabelization('../clean_data/btc.parquet')

    tweets_labels = tweets_labelization.labelize(granularity='h')
    btc_labels = btc_labelization.labelize(granularity='h')

    print(f"Shape of the tweets labels: {tweets_labels.shape}")
    print(f"Shape of the btc labels: {btc_labels.shape}")

if __name__ == '__main__':
    main()