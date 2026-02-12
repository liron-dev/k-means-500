from flask import Flask, render_template
import requests, io, os, time
import numpy as np, pandas as pd, yfinance as yf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from scipy.stats import zscore
import optuna
from google import genai

# --- Configuration ---
app = Flask(__name__)
cache = {"clusters": [], "pca_data": [], "performance_data": {}, "weights": {}}
os.environ["GEMINI_API_KEY"] = "KEY"


# --- Data Fetching ---
def get_tickers():
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', headers=headers)
        return [t.replace('.', '-') for t in (pd.read_html(io.StringIO(r.text)))[0].iloc[:, 0].tolist()]
    except Exception as e:
        print(f"❌ Error fetching tickers: {e}")
        return []


def fetch_prices(tickers):
    f_path = 'cache_data.csv'
    if os.path.exists(f_path) and (time.time() - os.path.getmtime(f_path) < 86400):
        raw = pd.read_csv(f_path, index_col=0, header=[0, 1], parse_dates=True)
        if isinstance(raw.columns, pd.MultiIndex):
            return raw['Close'] if 'Close' in raw.columns else raw
        return raw

    print("📥 Downloading 5Y data...")
    raw = yf.download(tickers, period="5y", interval="1wk", progress=False)
    raw.to_csv(f_path)
    return raw['Close'] if 'Close' in raw.columns else raw


# --- AI Naming ---
def get_cluster_name(cluster_id, tickers):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return f"Cluster {cluster_id}"

    try:
        client = genai.Client(api_key=api_key)
        stocks = ", ".join(tickers)
        prompt = f"Given these stock tickers from a cluster: {stocks}. Provide a short, 1-3 word descriptive name for this sector/theme. Do not use quotes."

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"⚠️ AI Naming Failed: {e}")
        return f"Cluster {cluster_id}"


# --- Helper Math Functions ---
def row_standardize(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True, ddof=0)
    sd[sd == 0.0] = 1.0
    return (X - mu) / sd


def try_ks(X: np.ndarray, ks=range(5, 12), random_state=42):
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=40, max_iter=1000, random_state=random_state)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        rows.append({"k": k, "silhouette": sil, "model": km, "labels": labels})
    return pd.DataFrame(rows)


def make_sparkline(v):
    w, h = 450, 60
    mn, mx = np.min(v), np.max(v)
    if mn == mx: return ""
    pts = [f"{(i / (len(v) - 1)) * w},{h - ((n - mn) / (mx - mn) * h)}" for i, n in enumerate(v)]
    return f'<svg width="{w}" height="{h}" style="fill:none; stroke:#0d6efd; stroke-width:2"><polyline points="{" ".join(pts)}"/></svg>'


# --- Portfolio Optimization Logic ---
def build_sector_index(prices_df, min_participation=0.30):
    rets = prices_df.pct_change()
    counts = rets.notna().sum(axis=1)
    min_part = max(1, int(np.ceil(min_participation * len(prices_df.columns))))

    ew_ret = rets.mean(axis=1, skipna=True)
    ew_ret[counts < min_part] = np.nan

    first_ok = ew_ret.first_valid_index()
    if first_ok is None: return pd.Series(dtype=float)

    ew_ret = ew_ret.loc[first_ok:]
    idx = (1 + ew_ret.fillna(0.0)).cumprod() * 100.0
    idx.iloc[0] = 100.0
    return idx


def sharpe_ratio(returns, rf=0.0):
    excess = returns - rf / 252
    if excess.std() == 0: return -1e9
    return (excess.mean() / excess.std()) * (252 ** 0.5)


def optimize_portfolio(sector_returns):
    sectors = sector_returns.columns.tolist()

    def objective(trial):
        w = np.array([trial.suggest_float(s, 0.0, 1.0) for s in sectors])
        if w.sum() == 0: return -1e9
        w = w / w.sum()

        port_ret = (sector_returns * w).sum(axis=1)
        return sharpe_ratio(port_ret)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=500)

    best_w = study.best_params
    total_w = sum(best_w.values())
    final_weights = {k: v / total_w for k, v in best_w.items()}

    return final_weights


# --- Main Pipeline ---
def process_data():
    print("🚀 Step 1: Loading Data...")
    tickers = get_tickers()
    if not tickers: return
    prices = fetch_prices(tickers)

    rets = np.log(prices / prices.shift(1)).dropna(how="all")
    rets = rets.loc[:, rets.isna().mean() < 0.1].ffill().bfill()

    # --- Clustering ---
    print("🧠 Clustering stocks...")
    Xz = row_standardize(rets.T.values)
    initial_count = Xz.shape[0]

    # Round 1
    grid1 = try_ks(Xz)
    best1 = grid1.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]

    # Pruning
    sil_values = silhouette_samples(Xz, best1['labels'])
    centroids = best1['model'].cluster_centers_
    dists = np.linalg.norm(Xz - centroids[best1['labels']], axis=1)

    diag = pd.DataFrame({"t": rets.columns, "c": best1['labels'], "sil": sil_values, "dist": dists})
    diag["z"] = diag.groupby("c")["dist"].transform(zscore)
    drop_mask = (diag["sil"] < -0.07) | (diag.groupby("c")["sil"].transform(lambda s: s <= s.quantile(0.03))) | (
            diag["z"].abs() > 3.0)

    clean_tickers = diag.loc[~drop_mask, "t"].tolist()
    Xz_pruned = Xz[~drop_mask]

    # PRINT PRUNED STATS
    pruned_count = initial_count - Xz_pruned.shape[0]
    print(f"✂️ Pruned {pruned_count} stocks (low correlation/outliers)...")

    # Round 2
    grid2 = try_ks(Xz_pruned)
    best2 = grid2.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]
    labels = best2['labels']

    # ---  Naming & Index Building ---
    print("🏗️ Building Sector Indexes...")
    cluster_map = {}
    sector_indexes = {}

    groups = pd.DataFrame({"ticker": clean_tickers, "cluster": labels})

    for c_id in sorted(groups["cluster"].unique()):
        members = groups[groups["cluster"] == c_id]["ticker"].tolist()

        c_name = get_cluster_name(c_id, members)
        print(f"   -> Named Cluster {c_id}: {c_name}")

        idx_series = build_sector_index(prices[members])
        sector_indexes[c_name] = idx_series
        cluster_map[c_id] = {"name": c_name, "members": members}

    # --- Stacking & Returns ---
    sector_df = pd.DataFrame(sector_indexes).dropna()
    sector_rets = sector_df.pct_change().fillna(0.0)

    # --- Optuna Optimization ---
    print("🧪 Optimizing Portfolio with Optuna...")
    weights = optimize_portfolio(sector_rets)

    w_series = pd.Series(weights)
    opt_ret = (sector_rets * w_series).sum(axis=1)
    opt_idx = (1 + opt_ret).cumprod() * 100
    opt_idx.iloc[0] = 100

    # --- Prep Visuals ---
    print("🎨 Finalizing Visualization Data...")

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Xz_pruned)

    ui_clusters = []
    pca_points = []

    for i, ticker in enumerate(clean_tickers):
        c_id = labels[i]
        c_name = cluster_map[c_id]["name"]
        pca_points.append({
            "ticker": ticker,
            "cluster": c_name,
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1])
        })

    cum_rets = rets[clean_tickers].cumsum()
    trends = (np.exp(cum_rets) * 100).T.values

    for c_id, info in cluster_map.items():
        m_tickers = info["members"]
        m_data = []
        for t in m_tickers:
            idx = clean_tickers.index(t)
            m_data.append({
                "ticker": t,
                "last_vol": round(float(Xz_pruned[idx, -1]), 2),
                "sparkline": make_sparkline(trends[idx])
            })

        ui_clusters.append({
            "id": c_id,
            "name": info["name"],
            "weight": f"{weights.get(info['name'], 0) * 100:.1f}%",
            "count": len(info["members"]),  # Added Count here
            "members": m_data
        })

    perf_data = {
        "dates": sector_df.resample('W').last().index.strftime('%Y-%m-%d').tolist(),
        "sectors": {col: sector_df[col].resample('W').last().values.tolist() for col in sector_df.columns},
        "optimized": opt_idx.resample('W').last().values.tolist()
    }

    cache['clusters'] = ui_clusters
    cache['pca_data'] = pca_points
    cache['performance_data'] = perf_data
    cache['weights'] = weights

    print(f"✅ Done! Sharpe: {sharpe_ratio(opt_ret):.3f}")


@app.route('/')
def index():
    return render_template('index.html', clusters=cache['clusters'], pca_data=cache['pca_data'],
                           perf_data=cache['performance_data'])


if __name__ == '__main__':
    process_data()
    app.run(port=5000, debug=True, use_reloader=False)