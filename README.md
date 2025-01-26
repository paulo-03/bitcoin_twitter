# Tweeting for Bitcoin: Analyzing Price Dynamics through Transfer Entropy

This project explores the dynamic relationship between Twitter sentiment and Bitcoin price movements using **Transfer Entropy (TE)** — a non-parametric and model-free measure capable of capturing non-linear and time-delayed dependencies in financial systems. By analyzing tick-level price data from Binance and sentiment-labeled tweets, this study quantifies the directional flow of information between social media activity and cryptocurrency returns.

***Authors:*** [Rami Atassi](mailto:rami.atassi@epfl.ch), [Mahmoud Dokmak](mahmoud.dokmak@epfl.ch) & [Paulo Ribeiro](paulo.ribeirodecarvalho@epfl.ch).

`Bitcoin` · `Twitter` · `Time Series Analysis` · `Symbolic Transfer Entropy`

## Key Features
- **Data Sources**:
  - **Cryptocurrency Prices**: Minute-level Bitcoin price data from Binance.
  - **Twitter Sentiment**: A dataset of Bitcoin-related tweets analyzed using a specialized financial sentiment classifier.
- **Preprocessing**:
  - Cleaning and aligning price and sentiment data at hourly intervals.
  - Sentiment analysis categorizing tweets as *Bullish*, *Bearish*, or *Neutral* using a fine-tuned NLP model.
- **Analysis**:
  - Computation of Transfer Entropy to detect information flow and lead-lag relationships.
  - Benchmarking the significance of TE through bootstrap statistical tests approximation.
  - Exploration of optimal time delays for maximal information transfer.

## Results
The project provides insights into the influence of Twitter sentiment on Bitcoin's price dynamics, highlighting key moments where public sentiment correlates with price fluctuations. The findings are presented through visualizations of lead-lag relationships.

## Future Work
- Scaling the analysis to include additional cryptocurrencies and social platforms and/or media specialized in cryptocurrencies.
- Enhancing the robustness of the sentiment classification pipeline.
- Applying findings to inform algorithmic trading strategies.

---
Discover the full implementation and results in the repository. A detailed paper about the project is also available for download. Contributions are welcome — feel free to contact us by clicking on our names!
