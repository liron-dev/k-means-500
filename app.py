from flask import Flask, render_template
import pandas as pd
import yfinance as yf
import requests
import io
import numpy as np

app = Flask(__name__)
cache = {"data": None}


# Wikipedia is the most frequently updated community-driven source for the S&P 500:
def get_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    df = pd.read_html(io.StringIO(response.text))[0]
    return [str(t).replace('.', '-') for t in df.iloc[:, 0].tolist()]


def make_sparkline(values):
    if len(values) < 2: return ""
    width, height = 500, 120

    # This captures the REAL min and max of the stock's 5-year history
    min_val, max_val = np.min(values), np.max(values)
    if max_val == min_val: return ""

    points = []
    for i, val in enumerate(values):
        x = (i / (len(values) - 1)) * width
        # This maps the real growth (e.g. 100 to 300) to the SVG height
        y = height - ((val - min_val) / (max_val - min_val) * height)
        points.append(f"{x},{y}")

    polyline = f'<polyline points="{" ".join(points)}" fill="none" stroke="#0d6efd" stroke-width="1.5"/>'
    return f'<svg width="{width}" height="{height}" style="background:#f8f9fa; border-radius:4px;">{polyline}</svg>'


def process_market_data():
    print("🚀 Step 1: Getting Universe...")
    tickers = get_tickers()
    print("📥 Step 2: Downloading Data...")
    raw_data = yf.download(tickers, period="5y", interval="1wk", auto_adjust=True)
    prices = raw_data['Close'] if isinstance(raw_data.columns, pd.MultiIndex) else raw_data

    # 1. Calculate Returns
    rets = np.log(prices / prices.shift(1)).dropna(how="all")
    max_nans = int(0.10 * len(rets))
    rets = rets.loc[:, rets.isna().sum() <= max_nans]
    rets = rets.ffill().bfill()

    # 2. DATA FOR THE EYES (Visual Graph)
    visual_paths = np.exp(rets.cumsum()) * 100
    visual_vals = visual_paths.T.values

    # 3. DATA FOR THE COMPUTER (K-Means DNA)
    growth_paths = rets.cumsum()
    X = growth_paths.T.values
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd

    # 4. Volatility (Last Std Value)
    rets_vals = rets.T.values
    rets_mu = rets_vals.mean(axis=1, keepdims=True)
    rets_sd = rets_vals.std(axis=1, keepdims=True)
    rets_z = (rets_vals - rets_mu) / rets_sd

    final_list = []
    valid_tickers = rets.columns.tolist()
    for i, ticker in enumerate(valid_tickers):
        # PASS THE VISUAL VALUE to the sparkline
        chart = make_sparkline(visual_vals[i])

        final_list.append({
            "ticker": ticker,
            "chart": chart,
            "last_val": round(float(rets_z[i][-1]), 2),
            "dna": Xz[i].tolist()  # Store the DNA for the next KM step
        })

    cache['data'] = final_list
    print(f"✅ Ready! Processed {len(final_list)} stocks.")


@app.route('/')
def index():
    if not cache['data']:
        return "<h1>Processing...</h1><p>Check terminal. Refresh in 1 minute.</p>"
    return render_template('index.html', stocks=cache['data'], count=len(cache['data']))


if __name__ == '__main__':
    process_market_data()
    app.run(debug=False, port=5000)