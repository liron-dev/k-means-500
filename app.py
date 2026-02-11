from flask import Flask, render_template
import requests, io, os, time, numpy as np, yfinance as yf, pandas as pd

app = Flask(__name__)
cache = {"data": []}

# Scrape S&P 500 tickers from Wikipedia
def get_tickers():
    r = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', headers={"User-Agent": "Mozilla/5.0"})
    df = (pd.read_html(io.StringIO(r.text)))[0]
    return [t.replace('.', '-') for t in df.iloc[:, 0].tolist()]

# Visuals - Generate SVG sparklines representing real 5-year growth
def make_sparkline(v):
    w, h = 500, 120
    mn, mx = np.min(v), np.max(v)
    if mn == mx: return ""
    pts = [f"{(i / (len(v) - 1)) * w},{h - ((n - mn) / (mx - mn) * h)}" for i, n in enumerate(v)]
    return f'<svg width="{w}" height="{h}" style="background:#f8f9fa; border-radius:4px"><polyline points="{" ".join(pts)}" fill="none" stroke="#0d6efd" stroke-width="1.5"/></svg>'

# Download & Process - Handle caching, cleaning, and standardization (Market DNA)
def process_data():
    print("🚀 Step 1: Getting Data...")
    tickers, f = get_tickers(), 'cache_data.csv'
    # Use local cache if file is less than 24 hours old to avoid rate limits
    if os.path.exists(f) and (time.time() - os.path.getmtime(f) < 86400):
        print("💾 Step 2: Using cached data!")
        raw = pd.read_csv(f, index_col=0, header=[0, 1], parse_dates=True)
    else:
        print("⬇️ Step 2: Downloading Data...")
        raw = yf.download(tickers, period="5y", interval="1wk", progress=False)
        raw.to_csv(f)

    # Data Cleaning: Log returns and removal of stocks with >10% missing data
    prices = raw['Close'] if 'Close' in raw.columns else raw
    rets = np.log(prices / prices.shift(1)).dropna(how="all")
    rets = rets.loc[:, rets.isna().mean() < 0.1].ffill().bfill()

    # Logic: X for K-Means (Standardized DNA), V for Visuals (Real Percentage)
    cum_rets = rets.cumsum()
    X, V = cum_rets.T.values, (np.exp(cum_rets) * 100).T.values

    # Standardization: Force Mean=0 and Std=1 for pattern recognition
    Xz = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
    last_z = ((rets.iloc[-1] - rets.mean()) / rets.std()).fillna(0)

    # Build final payload for UI and future clustering
    cache['data'] = [{"ticker": t, "chart": make_sparkline(V[i]), "last_val": round(last_z.loc[t], 2), "dna": Xz[i].tolist()} for i, t in enumerate(rets.columns)]
    print(f"✅ Ready! Processed {len(cache['data'])} stocks.")

@app.route('/')
def index():
    return render_template('index.html', stocks=cache['data'])

if __name__ == '__main__':
    process_data()
    app.run(port=5000)