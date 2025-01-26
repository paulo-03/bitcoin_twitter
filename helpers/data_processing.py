"""
This python script is used to define both BTCProcessing and TweetProcessing class to perform the pre-processing
step to construct the clean data which will populate the created clean_data folder.
"""

import os
import re
import torch
from tqdm import tqdm
from transformers import pipeline
from .classes import BTC, Tweets


class BTCProcessing(BTC):
    def __init__(self, data_path: str, clean_folder: str):
        super().__init__(data_path)
        self.clean_folder = clean_folder
        # Create folder to store the cleaned pre-process data
        os.makedirs(name=self.clean_folder, exist_ok=True)

    def hourly_granularity(self):
        """Transform our minute granularity Bitcoin price evolution to an hourly basis"""
        self.data = self.data.resample('h').agg({'close': 'last'})
        # Interpolate missing values
        self.data['close'] = self.data['close'].interpolate(method='linear')

    def compute_returns(self):
        """Simply create a new column and store the return value to know what was the hourly evolution"""
        self.data['returns'] = self.data['close'].pct_change()

    def save_clean_data(self):
        self.data.to_parquet(
            os.path.join(self.clean_folder, 'btc.parquet'), index=True
        )


class TweetsProcessing(Tweets):
    def __init__(self, data_path: str, clean_folder: str):
        super().__init__(data_path)

        self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'  # Check if GPU is available
        self.clean_folder = clean_folder

        # Define the regex pattern for emojis, found on stack overflows, used for tweets cleaning
        self.emoji_pattern = re.compile(
            pattern="[\U0001F600-\U0001F64F]"  # Emoticons
                    "|[\U0001F300-\U0001F5FF]"  # Symbols & Pictographs
                    "|[\U0001F680-\U0001F6FF]"  # Transport & Map Symbols
                    "|[\U0001F700-\U0001F77F]"  # Alchemical Symbols
                    "|[\U0001F780-\U0001F7FF]"  # Geometric Shapes Extended
                    "|[\U0001F800-\U0001F8FF]"  # Supplemental Arrows-C
                    "|[\U0001F900-\U0001F9FF]"  # Supplemental Symbols and Pictographs
                    "|[\U0001FA00-\U0001FA6F]"  # Chess Symbols
                    "|[\U0001FA70-\U0001FAFF]"  # Symbols and Pictographs Extended-A
                    "|[\U00002702-\U000027B0]"  # Dingbats
                    "|[\U000024C2-\U0001F251]"  # Enclosed Characters
                    "|[\U0001F1E0-\U0001F1FF]"  # Flags
                    "]+", flags=re.UNICODE
        )

        # Create folder to store the cleaned pre-process data
        os.makedirs(name=self.clean_folder, exist_ok=True)

    def select_pertinent_tweets(self, threshold: int = 100):
        """Only keep the tweets consider as pertinent in function of likes, replies or retweets performances"""
        # Select rows where likes, replies, or retweets are at least x
        total_tweets = self.data.shape[0]  # Retrieve the total number of tweets
        self.data = self.data[(self.data['likes'] >= threshold) | (self.data['replies'] >= threshold) | (
                self.data['retweets'] >= threshold)]
        total_pertinent_tweets = self.data.shape[0]  # Retrieve the total number of pertinent tweets

        # Give feed-back to user
        print(f"xPertinence Selection Report:\n"
              f"\t-Threshold: {threshold}\n"
              f"\t-Total number of original tweets: {total_tweets}\n"
              f"\t-Number of pertinent tweets, wrt threshold: {total_pertinent_tweets} "
              f"({total_pertinent_tweets / total_tweets * 100:.2f}%)")

    def clean_tweets(self):
        """Remove URLs, mentions, hashtags symbol and special character of all tweets to allow the sentiment classifier
        to be more accurate"""
        tqdm.pandas(desc=f"xCleaning tweets", unit='tweets')
        self.data['cleaned_text'] = self.data['text'].progress_apply(self._clean_text)

    @staticmethod
    def _clean_text(text) -> str:
        """Private function called by clean_tweets"""
        text = re.sub(r"http\S+", "", text)  # Remove URLs
        text = re.sub(r"@\w+", "", text)  # Remove mentions
        text = re.sub(r"#", "", text)  # Remove hashtags symbol
        text = re.sub(r'[\n\r\t]+', " ", text)  # Regex pattern to match \n, \r, \t and their duplicates
        text = re.sub(r"\s+", " ", text)  # Normalize multiple spaces into a single space
        # TODO: Sentiment Classifier model seams to retrieve information from emojis, then might be good to keep them
        # text = self.emoji_pattern.sub(r'', text)

        return text.strip().lower()  # Trim any leading/trailing spaces and lower all cases

    def remove_non_english_tweets(self, batch_size=16384):
        """Detect tweets languages in batches and remove non-English ones."""
        # Initialize Hugging Face pipeline
        lang_classifier = pipeline(
            task="text-classification",
            model="papluca/xlm-roberta-base-language-detection",
            device=self.device
        )

        # Process tweets in batches and store the results
        texts = self.data['text'].tolist()
        languages = []
        for i in tqdm(range(0, len(texts), batch_size), desc="xDetecting languages"):
            batch = texts[i:i + batch_size]
            batch_results = lang_classifier(batch)
            languages.extend([result['label'] for result in batch_results])

        # Add language predictions to the DataFrame
        self.data['lang'] = languages

        # Analyze detected languages
        tweets_number = self.data.shape[0]
        unique_lang = self.data['lang'].unique()

        # Store the tweets in foreign languages for qualitative assessment
        self.data[self.data['lang'] != 'en'].to_parquet(
            os.path.join(self.clean_folder, 'foreign_lang_tweets.parquet'), index=False
        )

        # Filter only 'en' tweets
        self.data = self.data[self.data['lang'] == 'en']
        num_en_tweets = self.data.shape[0]
        num_other_tweets = tweets_number - num_en_tweets

        # Feedback to the user
        print(f"Language Detection Report:\n"
              f"\t-Languages detected: {len(unique_lang)}\n\t{unique_lang}\n"
              f"\t-Number of English tweets kept: {num_en_tweets} ({num_en_tweets / tweets_number * 100:.2f}%)\n"
              f"\tNote: Tweets from other languages have been stored into {self.clean_folder}/foreign_lang_tweets.csv "
              f"({num_other_tweets} tweets)")

    def tweets_sentiment_analysis(self, batch_size=128):
        """Classify each tweet into one of the three following categories: 'NEUTRAL', 'BULLISH', 'BEARISH'"""
        # Create a sentiment analysis pipeline
        sentiment_classifier = pipeline(
            "sentiment-analysis",
            model="StephanAkkerman/FinTwitBERT-sentiment",
            device=self.device
        )

        # Process tweets in batches and store the results
        texts = self.data['cleaned_text'].tolist()
        sentiments = []
        sentiments_score = []

        for i in tqdm(range(0, len(texts), batch_size), desc="xDetecting sentiments"):
            batch = texts[i:i + batch_size]  # Get the current batch
            batch_results = sentiment_classifier(batch, batch_size=batch_size)

            # Extract sentiments and scores
            sentiments.extend([result['label'] for result in batch_results])
            sentiments_score.extend([result['score'] for result in batch_results])

        # Add results back to the DataFrame
        self.data['sentiment'] = sentiments
        self.data['sentiment_score'] = sentiments_score

        # Add sentiment predictions to the DataFrame
        self.data['sentiment'] = sentiments
        self.data['sentiment_score'] = sentiments_score
        infos = self.data[['sentiment', 'sentiment_score', 'text']].groupby(by='sentiment').agg({
            'text': 'count',  # Count tweets per sentiment
            'sentiment_score': 'mean'  # Average sentiment score per sentiment
        }).rename(columns={'text': 'tweet_count', 'sentiment_score': 'avg_sentiment_score'})

        # Feedback to the user
        print(f"Sentiment Analysis Report:\n"
              f"\t-Number of tweets considered as 'BULLISH': {infos.loc['BULLISH', 'tweet_count']} "
              f"(Avg. score: {infos.loc['BULLISH', 'avg_sentiment_score']:.2f})\n"
              f"\t-Number of tweets considered as 'NEUTRAL': {infos.loc['NEUTRAL', 'tweet_count']} "
              f"(Avg. score: {infos.loc['NEUTRAL', 'avg_sentiment_score']:.2f})\n"
              f"\t-Number of tweets considered as 'BEARISH': {infos.loc['BEARISH', 'tweet_count']} "
              f"(Avg. score: {infos.loc['BEARISH', 'avg_sentiment_score']:.2f})\n")

    def save_clean_data(self):
        self.data.to_parquet(
            os.path.join(self.clean_folder, 'twitter.parquet'), index=False
        )
