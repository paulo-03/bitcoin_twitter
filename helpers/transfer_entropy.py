'''
Class for computing the transfer entropy between the time series of bitcoin returns classified [Bearish, Neutral, Bullish]
and the tweets sentiment classified [Bearish, Neutral, Bullish].
'''

from pyinform.transferentropy import transfer_entropy
import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
import seaborn as sns

class TweetsToBtcTransferEntropy:
    def __init__(self, btc_returns: np.ndarray, tweets_sentiment: np.ndarray):
        self.btc_returns = btc_returns
        self.tweets_sentiment = tweets_sentiment
        self.length = len(btc_returns)
        self.df = 12  # degrees of freedom = ν = nA * (nA+ - 1) * (nB - 1) = 3 * (3-1) * (3-1) = 12

    def compute_transfer_entropy(self, delay: int = 1) -> float:
        """
        Compute the transfer entropy between the two time series with a given delay.
        :param delay: The lag between the two time series.
        :return: The transfer entropy value.
        """
        return transfer_entropy(self.btc_returns, self.tweets_sentiment, k=delay) #source and target are reversed in the function

    def compute_p_value(self, TE: float) -> float:
        """
        Compute the p-value of the transfer entropy value being significant (i.e. not due to randomness).
        :param TE: A transfer entropy value between the two time series.
        :return: The p-value.
        """
        res = 2 * self.length * TE
        return 1 - chi2.cdf(res, self.df)

    def plot_transfer_entropy_on_lags(self, delays: list = range(1, 15)) -> None:
        """
        Plot the transfer entropy values for different delays.
        :param delays: The list of delays to compute the transfer entropy.
        """
        transfer_entropy_values = []
        p_value_significant = []
        for delay in delays:

            TE = self.compute_transfer_entropy(delay=delay)
            transfer_entropy_values.append(TE)
            p_value_significant.append(self.compute_p_value(TE) < 0.01)

        plt.figure(figsize=(12, 6))
        for i in range(len(delays) - 1):
            plt.plot(delays[i:i + 2], transfer_entropy_values[i:i + 2],
                     color='#1f77b4' if p_value_significant[i] else '#ff7f0e',
                     label='Significant (p < 0.01)' if p_value_significant[i] else 'Not Significant (p > 0.01)')

        # Remove duplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.xlabel('Time shift [hour]')
        plt.ylabel('Transfer Entropy [bit]')
        plt.title('Transfer Entropy between Tweets Sentiment and BTC Returns')
        plt.show()


def main() -> None:
    # Example
    seed = 42
    np.random.seed(seed)
    tweets_sentiment = np.random.randint(0, 3, 18000)  # Source series
    btc_returns = np.roll(tweets_sentiment, 8)  # Target series influenced by source
    te = TweetsToBtcTransferEntropy(btc_returns, tweets_sentiment)
    te.plot_transfer_entropy_on_lags()

if __name__ == '__main__':
    main()
