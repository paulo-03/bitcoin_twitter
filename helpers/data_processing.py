"""
This python script is used to define both BTCProcessing and TweetProcessing class to perform the pre-processing
step to construct the clean data which will populate the created clean_data folder.
"""

import re
import pandas as pd
from .classes import BTC, Tweets

class BTCProcessing(BTC):
    def __init__(self, file_path):
        super().__init__(file_path)

    def hourly_granularity(self):
        """Transform our minute granularity Bitcoin price evolution to an hourly basis"""
        self.data = self.data.resample('H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'quote_asset_volume': 'sum',
            'number_of_trades': 'sum'
        }).dropna()

    def compute_returns(self):
        """Simply create a new column and store the return value to know what was the hourly evolution"""
        self.data['returns'] = self.data['close'].pct_change()

    def save_clean_data(self):
        ...


class TweetsProcessing(Tweets):
    def __init__(self, file_path: str):
        super().__init__(file_path)

    def remove_non_english_tweets(self):
        ...

    def clean_tweets(self):
        self.data['cleaned_text'] = self.data[r'text\r'].apply(self._clean_text)

    @staticmethod
    def _clean_text(text):
        """Remove URLs, mentions, hashtags symbol and special character of all tweets to allow the sentiment classifier
        to be more accurate"""
        text = re.sub(r"http\S+", "", text)  # Remove URLs
        text = re.sub(r"@\w+", "", text)  # Remove mentions
        text = re.sub(r"#", "", text)  # Remove hashtags symbol
        text = re.sub(r"[^\w\s]", "", text)  # Remove special characters
        return text.lower()
