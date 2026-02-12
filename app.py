from flask import Flask, render_template
import requests, io, os, time, optuna, numpy as np, pandas as pd, yfinance as yf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from scipy.stats import zscore

app = Flask(__name__)
cache = {}

# Protection list
BLUE_CHIPS = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'BRK-B', 'LLY', 'AVGO', 'JPM', 'V']

def get_data():
    fm, fp = 'meta.csv', 'prices.csv'
    if os.path.exists(fm) and os.path.exists(fp) and (time.time() - os.path.getmtime(fp) < 86400):
        print("⚡ Loading from Cache...")
        return pd.read_csv(fm, index_col=0), pd.read_csv(fp, index_col=0, parse_dates=True)

    print("🚀 Step 1: Downloading New Data...")
    headers = {"User-Agent": "Mozilla/5.0"}
    df = pd.read_html(io.StringIO(requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', headers=headers).text))[0]
    df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
    meta = df.set_index('Symbol')[['GICS Sector', 'GICS Sub-Industry']]
    meta.to_csv(fm)
    raw = yf.download(meta.index.tolist() + ['^GSPC'], period="5y", interval="1wk", progress=False)['Close']
    raw.to_csv(fp)
    return meta, raw

def run_pipeline():
    meta, prices = get_data()
    bench = prices['^GSPC'].pct_change().fillna(0)
    prices_stocks = prices.drop(columns=['^GSPC'])
    rets = np.log(prices_stocks / prices_stocks.shift(1)).iloc[1:].ffill().bfill()

    print("📊 Step 2: Standardizing Data...")
    X = rets.T.values
    mu, sd = X.mean(axis=1, keepdims=True), X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd

    def best_k_loop(data):
        best = None
        for k in range(5, 12):
            km = KMeans(n_clusters=k, n_init=40, max_iter=1000, random_state=42).fit(data)
            sil = silhouette_score(data, km.labels_)
            if best is None or sil > best['sil']: best = {'k': k, 'sil': sil, 'model': km}
        return best['model']

    print("🧠 Round 1: Initial Clustering...")
    km1 = best_k_loop(Xz)

    print("✂️ Step 3: Pruning Outliers...")
    sil = silhouette_samples(Xz, km1.labels_)
    dist = np.linalg.norm(Xz - km1.cluster_centers_[km1.labels_], axis=1)
    df_s = pd.DataFrame({'t': rets.columns, 'c': km1.labels_, 'sil': sil, 'dist': dist})
    df_s['z'] = df_s.groupby('c')['dist'].transform(zscore)

    mask = ((df_s['sil'] < -0.07) |
            (df_s.groupby('c')['sil'].transform(lambda x: x <= x.quantile(0.03))) |
            (df_s['z'].abs() > 3.0)) & (~df_s['t'].isin(BLUE_CHIPS))

    keepers = df_s[~mask].copy()
    print(f"✅ Kept {len(keepers)} stocks, pruned {len(df_s) - len(keepers)} outliers.")

    print("🧠 Round 2: Re-Clustering & Numbering...")
    Xz_clean = Xz[keepers.index]
    km2 = best_k_loop(Xz_clean)
    keepers.loc[:, 'c'] = km2.labels_

    sec_idxs, ui_clusters = {}, []
    pca_pts = PCA(2).fit_transform(Xz_clean)

    def make_spark(t):
        v = (prices_stocks[t] / prices_stocks[t].iloc[0]).values
        pts = [f"{(i / (len(v) - 1)) * 450},{60 - ((n - min(v)) / (max(v) - min(v)) * 60)}" for i, n in enumerate(v)]
        return f'<svg width="450" height="60" style="fill:none;stroke:#0d6efd;stroke-width:2"><polyline points="{" ".join(pts)}"/></svg>'

    for cid in sorted(keepers['c'].unique()):
        ticks = keepers[keepers['c'] == cid]['t'].tolist()
        # CLUSTER NAMING REMOVED -> REPLACED WITH NUMBERING
        display_name = f"Cluster {int(cid) + 1}"

        sec_idxs[display_name] = prices_stocks[ticks].pct_change().mean(axis=1).fillna(0)
        ui_clusters.append({
            'id': int(cid),
            'name': display_name,
            'count': len(ticks),
            'members': [{'ticker': t, 'last_vol': round(float(keepers.loc[keepers['t'] == t, 'z'].iloc[0]), 2),
                         'sparkline': make_spark(t)} for t in ticks]
        })

    print("⚖️ Step 4: Optuna Optimization...")
    def obj(t):
        w = np.array([t.suggest_float(c, 0, 1) for c in sec_idxs])
        w[w < 0.10] = 0
        if w.sum() == 0: return -1e9
        w /= w.sum()
        active = np.count_nonzero(w)
        penalty = 0
        if active > 4: penalty = (active - 4) * 0.1
        if active < 3: penalty = (3 - active) * 0.2
        r = (pd.DataFrame(sec_idxs) * w).sum(axis=1)
        return ((r.mean() / (r.std() + 1e-6)) * np.sqrt(52)) - penalty

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.CmaEsSampler(seed=42))
    optuna.logging.set_verbosity(0)
    study.optimize(obj, n_trials=1000)

    best_p = {k: v for k, v in study.best_params.items() if v >= 0.10}
    total_w = sum(best_p.values())
    final_weights = {k: v / total_w for k, v in best_p.items()} if total_w > 0 else {}
    opt_ret = (pd.DataFrame(sec_idxs) * pd.Series(final_weights)).sum(axis=1)

    for c in ui_clusters:
        c['weight'] = f"{final_weights.get(c['name'], 0) * 100:.1f}%"

    cache.update({
        'clusters': sorted(ui_clusters, key=lambda x: float(x['weight'].strip('%')), reverse=True),
        'pca_data': [{'x': float(pca_pts[i, 0]), 'y': float(pca_pts[i, 1]), 'ticker': t,
                      'cluster': f"Cluster {km2.labels_[i] + 1}"} for i, t in enumerate(keepers['t'])],
        'perf_data': {
            'dates': opt_ret.index.strftime('%Y-%m-%d').tolist(),
            'sectors': {k: ((1 + v).cumprod() * 100).tolist() for k, v in sec_idxs.items()},
            'optimized': ((1 + opt_ret).cumprod() * 100).tolist(),
            'benchmark': ((1 + bench).cumprod() * 100).tolist()
        }
    })
    print(f"✅ Done! Sharpe Ratio: {(opt_ret.mean() / opt_ret.std() * np.sqrt(52)):.2f}")

@app.route('/')
def index(): return render_template('index.html', **cache)

if __name__ == '__main__':
    run_pipeline()
    app.run(port=5000, debug=True, use_reloader=False)