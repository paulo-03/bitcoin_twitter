'''
Class for computing the transfer entropy between the time series of bitcoin returns classified [Bearish, Neutral, Bullish]
and the tweets sentiment classified [Bearish, Neutral, Bullish].
'''

from pyinform.transferentropy import transfer_entropy
import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
import seaborn as sns

class TransferEntropy:
    def __init__(self, btc_returns: np.ndarray, tweets_sentiment: np.ndarray):
        self.btc_returns = btc_returns
        self.tweets_sentiment = tweets_sentiment
        self.length = len(btc_returns)
        self.df = 12  # degrees of freedom = 3*(3-1)*(3-1) = 12

    def compute_transfer_entropy(self, delay: int = 1) -> float:
        """
        Compute the transfer entropy between the two time series.
        :param delay: The lag between the two time series.
        :return: The transfer entropy value.
        """
        return transfer_entropy(self.btc_returns, self.tweets_sentiment, k=delay) #source and target are reversed in the function

    def compute_p_value(self, delay: int = 1) -> float:
        """
        Compute the p-value of the transfer entropy value.
        :param delay: The lag between the two time series.
        :return: The p-value.
        """
        res = 2 * self.length * self.compute_transfer_entropy(delay)
        return 1 - chi2.cdf(res, self.df)

    def plot_transfer_entropy(self, delays: list) -> None:
        """
        Plot the transfer entropy values for different delays.
        :param delays: The list of delays to compute the transfer entropy.
        """
        pass

def main() -> None:
    # Example
    seed = 42
    np.random.seed(seed)
    tweets_sentiment = np.random.randint(0, 3, 18000)  # Source series
    btc_returns = np.roll(tweets_sentiment, 20)  # Target series influenced by source
    te = TransferEntropy(btc_returns, tweets_sentiment)
    print("Transfer Entropy (k=1):", te.compute_transfer_entropy(delay=1))
    print("Transfer Entropy (k=2):", te.compute_transfer_entropy(delay=2))
    print("Transfer Entropy (k=3):", te.compute_transfer_entropy(delay=3))

    print("p-value (k=1):", te.compute_p_value(delay=1), "=>", te.compute_p_value(delay=1) < 0.05)
    print("p-value (k=2):", te.compute_p_value(delay=2), "=>", te.compute_p_value(delay=2) < 0.05)
    print("p-value (k=3):", te.compute_p_value(delay=3), "=>", te.compute_p_value(delay=3) < 0.05)

if __name__ == '__main__':
    main()
