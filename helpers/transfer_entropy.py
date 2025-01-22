"""
Class for computing the transfer entropy between the time series of bitcoin returns classified [Bearish, Neutral, Bullish]
and the tweets sentiment classified [Bearish, Neutral, Bullish].
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyinform.transferentropy import transfer_entropy
from scipy.stats import chi2


class TweetsToBtcTransferEntropy:
    def __init__(self, tweets_sentiment: np.ndarray, btc_returns: np.ndarray):
        self.tweets_sentiment = tweets_sentiment
        self.btc_returns = btc_returns
        self.length = len(btc_returns)
        self.df = 12  # degrees of freedom = ν = nA * (nA+ - 1) * (nB - 1) = 3 * (3-1) * (3-1) = 12

    def compute_transfer_entropy(self, delay: int = 0, k: int = 1) -> float:
        """
        Compute the transfer entropy between the two time series with a given delay.
        :param delay: The lag between the two time series.
        :param k: The history length.
        :return: The transfer entropy value.
        """
        if delay == 0:
            source = self.tweets_sentiment
            target = self.btc_returns
        else:
            source = self.tweets_sentiment[:-delay]
            target = self.btc_returns[delay:]
        return transfer_entropy(source=source, target=target, k=k)

    def compute_p_value(self, TE: float) -> float:
        """
        Compute the p-value of the transfer entropy value being significant (i.e. not due to randomness).
        :param TE: A transfer entropy value between the two time series.
        :return: The p-value.
        """
        res = 2 * self.length * TE
        return 1 - chi2.cdf(res, self.df)

    def transfer_entropy_significance_threshold(self, alpha: float = 0.01) -> float:
        """
        Compute the transfer entropy significance threshold for a given alpha.
        :param alpha: The significance level.
        :return: The transfer entropy significance threshold.
        """
        return chi2.ppf(1 - alpha, self.df) / (2 * self.length)

    def plot_transfer_entropy_on_lags(self, delays: list = range(0, 200), k: int = 1,
                                      moving_average_window: int = 0) -> None:
        """
        Plot the transfer entropy values for different delays.
        :param delays: The list of delays to compute the transfer entropy.
        :param k: The history length.
        :param moving_average_window: The window size for the moving average.
        """

        transfer_entropy_values = [self.compute_transfer_entropy(delay, k) for delay in delays]

        plt.figure(figsize=(12, 6))
        sns.lineplot(x=delays, y=transfer_entropy_values, drawstyle='steps-post')
        # Regression line
        # sns.regplot(x=delays, y=transfer_entropy_values, scatter=False, color='red', label='Regression Line')
        # Moving average
        if moving_average_window > 0:
            sns.lineplot(x=delays,
                         y=pd.Series(transfer_entropy_values).rolling(window=moving_average_window, center=True,
                                                                      min_periods=1).mean(), color='red',
                         label='Moving Average')

        plt.axhline(y=self.transfer_entropy_significance_threshold(), color='pink', linestyle='--')

        plt.fill_between(delays, 0, self.transfer_entropy_significance_threshold(),
                         color='pink', alpha=0.3, label='not significant')

        plt.xlabel('Time shift [hour]')
        plt.ylabel('Transfer Entropy [bit]')
        plt.title('Transfer Entropy from Tweets Sentiment to BTC Returns')
        plt.legend()
        plt.show()

    def plot_transfer_entropy_on_history_lengths(self, ks: list = range(1, 17)) -> None:
        """
        Plot the mean transfer entropy values for different history lengths.
        :param max_delay: The maximum delay to compute the transfer entropy.
        :param ks: The list of history lengths to compute the transfer entropy.
        """

        transfer_entropy_values = [self.compute_transfer_entropy(0, k) for k in ks]

        plt.figure(figsize=(12, 6))
        sns.lineplot(x=ks, y=transfer_entropy_values)
        plt.xlabel('History length (k)')
        plt.ylabel('Transfer Entropy [bit]')
        plt.title('Transfer Entropy for Different History Lengths')
        plt.show()


def main() -> None:
    # Example
    tweets_sentiment = np.load('../data/tweets_time_series.npy')
    btc_returns = np.load('../data/btc_time_series.npy')
    te = TweetsToBtcTransferEntropy(tweets_sentiment, btc_returns)
    te.plot_transfer_entropy_on_lags(delays=list(range(0, 2000)), k=1)

    # Random test
    # random1 = np.random.randint(0, 3, size=len(btc_returns))
    # random2 = np.random.randint(0, 3, size=len(btc_returns))
    # te_random = TweetsToBtcTransferEntropy(random1, random2)
    # te_random.plot_transfer_entropy_on_lags(delays=list(range(1, 200)))


if __name__ == '__main__':
    main()
