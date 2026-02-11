from flask import Flask, render_template
import requests, io, os, time
import numpy as np, pandas as pd, yfinance as yf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from scipy.stats import zscore

app = Flask(__name__)
cache = {"clusters": [], "pca_data": []}


# --- Data Fetching ---
def get_tickers():
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', headers=headers)
    return [t.replace('.', '-') for t in (pd.read_html(io.StringIO(r.text)))[0].iloc[:, 0].tolist()]


# --- Visualization ---
def make_sparkline(v):
    w, h = 150, 40
    mn, mx = np.min(v), np.max(v)
    pts = [f"{(i / (len(v) - 1)) * w},{h - ((n - mn) / (mx - mn) * h)}" for i, n in enumerate(v)]
    return f'<svg width="{w}" height="{h}" style="fill:none; stroke:#0d6efd; stroke-width:1.5"><polyline points="{" ".join(pts)}"/></svg>'


# --- Standardization ---
def row_standardize(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True, ddof=0)
    sd[sd == 0.0] = 1.0
    return (X - mu) / sd


# --- Iterative K Selection ---
def try_ks(X: np.ndarray, ks=range(5, 12), random_state=42):
    rows = []
    for k in ks:
        # High n_init helps separate clusters better:
        km = KMeans(n_clusters=k, n_init=40, max_iter=1000, random_state=random_state)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels, metric="euclidean")
        rows.append({"k": k, "silhouette": sil, "model": km, "labels": labels})

    # Return sorted by best silhouette
    return pd.DataFrame(rows)


# --- Main Processing Pipeline ---
def process_data():
    print("🚀 Step 1: Loading Data...")
    f_path = 'cache_data.csv'
    if os.path.exists(f_path) and (time.time() - os.path.getmtime(f_path) < 86400):
        raw = pd.read_csv(f_path, index_col=0, header=[0, 1], parse_dates=True)
    else:
        tickers = get_tickers()
        raw = yf.download(tickers, period="5y", interval="1wk", progress=False)
        raw.to_csv(f_path)

    prices = raw['Close'] if 'Close' in raw.columns else raw
    rets = np.log(prices / prices.shift(1)).dropna(how="all")
    rets = rets.loc[:, rets.isna().mean() < 0.1].ffill().bfill()

    # --- Standardization ---
    X = rets.T.values
    Xz = row_standardize(X)
    tickers = rets.columns.to_numpy()

    # --- Round 1 Clustering ---
    print("🧠 Searching for best K (Round 1)...")
    grid1 = try_ks(Xz)
    best1 = grid1.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]
    print(f"   -> Chosen K={best1['k']} (Sil: {round(best1['silhouette'], 3)})")

    # --- Pruning ---
    print("✂️ Pruning outliers...")
    sil_values = silhouette_samples(Xz, best1['labels'], metric="euclidean")
    centroids = best1['model'].cluster_centers_
    dists = np.linalg.norm(Xz - centroids[best1['labels']], axis=1)

    diag = pd.DataFrame({
        "ticker": tickers,
        "cluster": best1['labels'],
        "silhouette": sil_values,
        "dist": dists
    })
    diag["dist_z"] = diag.groupby("cluster")["dist"].transform(zscore)

    # Prune problematic stocks
    drop_flag = (
            (diag["silhouette"] < 0.0) |
            (diag.groupby("cluster")["silhouette"].transform(lambda s: s <= s.quantile(0.1))) |
            (diag["dist_z"].abs() > 3.0)
    )

    clean_mask = ~drop_flag
    Xz_pruned = Xz[clean_mask]
    tickers_pruned = tickers[clean_mask]
    print(f"   -> Dropped {len(tickers) - len(tickers_pruned)} stocks.")

    # --- Round 2 Clustering ---
    print("🧠 Refitting clusters (Round 2)...")
    grid2 = try_ks(Xz_pruned)
    best2 = grid2.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]
    print(f"   -> Final K={best2['k']} (Sil: {round(best2['silhouette'], 3)})")

    # --- Visualizing with PCA ---
    print("🎨 Generating PCA coordinates...")
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Xz_pruned)

    # --- UI Formatting ---
    cluster_map = {}
    cum_rets = rets[tickers_pruned].cumsum()
    trends = (np.exp(cum_rets) * 100).T.values

    for i, ticker in enumerate(tickers_pruned):
        c_id = int(best2['labels'][i])
        if c_id not in cluster_map: cluster_map[c_id] = []
        cluster_map[c_id].append({
            "ticker": ticker,
            "x": float(coords[i, 0]), "y": float(coords[i, 1]),
            "sparkline": make_sparkline(trends[i]),
            "last_vol": round(float(Xz_pruned[i, -1]), 2)
        })

    cache['clusters'] = [{"id": cid, "members": cluster_map[cid]} for cid in sorted(cluster_map.keys())]
    cache['pca_data'] = [{"ticker": m['ticker'], "cluster": c['id'], "x": m['x'], "y": m['y']} for c in cache['clusters'] for m in c['members']]
    print(f"✅ Done! Final Universe: {len(tickers_pruned)} stocks.")


@app.route('/')
def index():
    return render_template('index.html', clusters=cache['clusters'], pca_data=cache['pca_data'])


if __name__ == '__main__':
    process_data()
    app.run(port=5000, debug=True, use_reloader=False)