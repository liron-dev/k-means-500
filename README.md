# 🧬 AI Portfolio Optimizer
An end-to-end financial engineering tool that leverages unsupervised machine learning (K-Means) and genetic optimization (Optuna) to build high-alpha portfolios. The system clusters the S&P 500 by price behavior, prunes outliers, and synthesizes a simplified 4-ETF proxy portfolio.
## 🛠️ Installation & Setup
### 1. Prerequisites
Ensure you have **Python 3.10+** installed.
### 2. Install Dependencies
Run the following command to install the required libraries:

```bash 
pip install flask pandas numpy scikit-learn yfinance optuna requests plotly scipy lxml
```

### 3. Run the Application

Execute the main script (app.py).

Note: The first run will download 5 years of historical data for over 700 tickers (S&P 500 + ETFs). This may take 2–5 minutes depending on your connection.

### 4. Access the Dashboard

Open your browser and navigate to:

[**http://127.0.0.1:5000**](http://127.0.0.1:5000/)

⚙️ How It Works (The Pipeline)
------------------------------

### 1. Data Ingestion & Cleaning

Downloads S&P 500 constituents and a curated universe of 200 liquid ETFs. It automatically handles missing values, ticker symbol mapping (e.g., BRK.B to BRK-B), and log return calculations.

### 2. Clustering & Pruning

*   **Behavioral Grouping:** Groups stocks based on return patterns rather than traditional GICS sectors.
    
*   **Dynamic K:** Tests multiple cluster counts ($k=5$ to $12$) to find the highest Silhouette Score.
    
*   **High-Quality Pruning:** The algorithm automatically prints how many stocks were pruned. It removes stocks with negative silhouette scores or extreme outlier volatility while protecting core Blue Chips.
    

### 3. Genetic Optimization (Optuna)

*   **Deep Research:** Runs **Thousands of trials** for both the Stock Cluster weighting and the ETF selection.
    
*   **Stock Optimization:** Maximizes the Sharpe Ratio of the cluster-based portfolio.
    
*   **ETF Synthesis:** Scans the ETF universe for the best "Alpha & Fit" combination.
    

### 4. The "Max-4" ETF Constraint

The optimizer is hard-coded to produce a portfolio of **maximum 4 ETFs**. This ensures the final strategy is easy to execute while maintaining a high correlation to the AI-optimized stock model.

📊 Visualizations
-----------------

The dashboard provides four primary views:
*   **Ultimate ETF Strategy:**  Which will recommand you the best and most optimized etfs to buy.

*   **Performance Comparison:** A Plotly chart comparing the AI Stock model, the 4-ETF Proxy, and the S&P 500.
    
*   **PCA Market Map:** A 2D visualization of how the AI "sees" the market.
    
*   **Deep Dive Clusters:** Interactive tables with 5-year trend sparklines for every retained stock.
    

⚠️ Disclaimer
-------------

This software is for educational and research purposes only. It is not financial advice. All investment strategies involve risk of loss.

**Author:** Liron

**License:** MIT
