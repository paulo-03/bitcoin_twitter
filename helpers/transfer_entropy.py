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

    def compute_p_value(self, TE: float, length: int) -> float:
        """
        Compute the p-value of the transfer entropy value being significant (i.e. not due to randomness).

        :param TE: A transfer entropy value between the two time series.
        :param length: The length of the time series.
        :return: The p-value.
        """
        res = 2 * length * TE
        return 1 - chi2.cdf(res, self.df)

    def transfer_entropy_significance_threshold(self, length: int, alpha: float = 0.01) -> float:
        """
        Compute the transfer entropy significance threshold for a given alpha.

        :param length: The length of the time series.
        :param alpha: The significance level.
        :return: The transfer entropy significance threshold.
        """
        return chi2.ppf(1 - alpha, self.df) / (2 * length)

    def compute_mean_transfer_entropy(self, delays: list = range(0, 200), k: int = 1) -> float:
        """
        Compute the mean transfer entropy value for different delays.

        :param delays: The list of delays to compute the transfer entropy.
        :param k: The history length.
        :return: The mean transfer entropy value.
        """
        return float(np.mean([self.compute_transfer_entropy(delay, k) for delay in delays]))

    def plot_transfer_entropy_on_lags(self, delays: list = range(0, 200), k: int = 1,
                                      moving_average_window: int = 0, case: str = "tweet_to_btc_hours") -> None:
        """
        Plot the transfer entropy values for different delays.

        :param case: The case to plot the transfer entropy for.
        :param delays: The list of delays to compute the transfer entropy.
        :param k: The history length.
        :param moving_average_window: The window size for the moving average.
        """
        # Compute the Transfer Entropy values for a fix history window, but for few different delays
        transfer_entropy_values = [self.compute_transfer_entropy(delay, k) for delay in delays]

        #Delays will induce length reduction since we truncate time series values that are not in the shifted time series intersection
        time_serie_length = [self.length - delay for delay in delays]

        # Compute the different significance thresholds since time series length evolve, significance threshold too
        significance_thresholds = [self.transfer_entropy_significance_threshold(length) for length in time_serie_length]

        # Plot the results
        plt.figure(figsize=(8, 4))
        sns.lineplot(x=delays, y=transfer_entropy_values, drawstyle='steps-post', label='Transfer Entropy')
        sns.lineplot(x=delays, y=significance_thresholds, color='pink', linestyle='--', label='Significance Threshold')
        plt.fill_between(delays, 0, significance_thresholds, color='pink', alpha=0.3, label='Not Significant')

        # Moving average to smooth the TE evolution in function of delays
        if moving_average_window > 0:
            sns.lineplot(x=delays,
                         y=pd.Series(transfer_entropy_values).rolling(window=moving_average_window, center=True,
                                                                      min_periods=1).mean(), color='red',
                         label='Moving Average')

        if case == "tweet_to_btc_hours":
            plt.xlabel('Time shift [hour]')
            plt.ylabel(r"$TE^{T \rightarrow B} [bit]$")
            plt.title('Transfer Entropy from Tweets Sentiment to BTC Returns')

        elif case == "btc_to_tweet_hours":
            plt.xlabel('Time shift [hour]')
            plt.ylabel(r"$TE^{B \rightarrow T} [bit]$")
            plt.title('Transfer Entropy from BTC Returns to Tweets Sentiment')

        elif case == "tweet_to_btc_days":
            plt.xlabel('Time shift [day]')
            plt.ylabel(r"$TE^{T \rightarrow B} [bit]$")
            plt.title('Transfer Entropy from Tweets Sentiment to BTC Returns')

        elif case == "btc_to_tweet_days":
            plt.xlabel('Time shift [day]')
            plt.ylabel(r"$TE^{B \rightarrow T} [bit]$")
            plt.title('Transfer Entropy from BTC Returns to Tweets Sentiment')

        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{case}.pdf")
        plt.show()

    def plot_transfer_entropy_on_history_lengths(self, ks: list = range(1, 17)) -> None:
        """
        Plot the mean transfer entropy values for different history lengths.
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
    # Load the data
    tweets_sentiment = np.load('../data/tweets_time_series.npy')
    btc_returns = np.load('../data/btc_time_series.npy')

    # Run our code for a specific case, i.e. history window k=1
    te = TweetsToBtcTransferEntropy(tweets_sentiment, btc_returns)
    te.plot_transfer_entropy_on_lags(delays=list(range(0, 2000)), k=1)


if __name__ == '__main__':
    # Run the script to easily test our implementation by choosing your testing configuration
    time_serie_size = 18000
    random_x = np.random.randint(0, 3, size=time_serie_size)
    random_y = np.random.randint(0, 3, size=time_serie_size)
    k = 1

    TE = transfer_entropy(source=random_x, target=random_y, k=k)
    res = 2 * time_serie_size * TE
    pvalue = 1 - chi2.cdf(res, 12)  # freedom degree is computed following the Professor Challet paper.

    print("Transfer Entropy results:\n"
          f"\t- TE score: {TE}\n"
          f"\t- P-Value: {pvalue}")
