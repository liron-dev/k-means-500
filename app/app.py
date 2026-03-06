from flask import Flask, render_template
import requests, io, os, time, optuna, numpy as np, pandas as pd, yfinance as yf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from scipy.stats import zscore
import warnings


session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://finance.yahoo.com/'
})

# 🤫 Silence Warnings & Logs
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

app = Flask(__name__)
cache = {}

# 🛡️ Protection list (Mega Caps)
BLUE_CHIPS = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'BRK-B', 'LLY', 'AVGO', 'JPM', 'V']
# etf correlation vs sharp index focusing:
SHARPE = 0.7
CORR = 0.3


# 🌎 200 Liquid ETF Universe
ETF_UNIVERSE = list(set([
    'SPY', 'IVV', 'VOO', 'QQQ', 'DIA', 'IWM', 'IJR', 'IJH', 'RSP', 'VTV', 'VUG', 'IWV', 'VT',  # Broad
    'XLK', 'XLF', 'XLV', 'XLE', 'XLY', 'XLI', 'XLU', 'XLB', 'XLC', 'XLRE',  # SPDR Sectors
    'SOXX', 'SMH', 'IGV', 'XSW', 'XSD', 'HACK', 'CIBR', 'SKYY', 'CLOU', 'WCLD', 'FINX',  # Tech/Digital
    'IPAY', 'BOTZ', 'ROBO', 'ARKK', 'ARKW', 'ARKG', 'ARKF', 'PRNT', 'IZRL',  # Innovation
    'ITA', 'PPA', 'XME', 'XOP', 'OIH', 'GDX', 'GDXJ', 'SIL', 'COPX', 'URA',  # Industrial/Resources
    'KRE', 'KBE', 'IAT', 'XBI', 'IBB', 'ITB', 'XHB', 'XRT', 'JETS', 'PEJ', 'PBJ',  # Industry Specific
    'MTUM', 'QUAL', 'VLUE', 'USMV', 'SIZE', 'DGRW', 'NOBL', 'SDY', 'SCHD', 'VIG',  # Factors
    'VYM', 'DVY', 'HDV', 'SPHD', 'RDIV', 'FDL', 'DGRO', 'PEY', 'QDIV',  # Dividend Focus
    'TAN', 'ICLN', 'LIT', 'DRIV', 'PHO', 'MOO', 'PBW', 'QCLN', 'MJ', 'YOLO', 'MSOS',  # ESG/Themes
    'EEM', 'VEA', 'VWO', 'EZU', 'EWJ', 'EWG', 'EWC', 'EWA', 'MCHI', 'FXI', 'INDA',  # International
    'VTI', 'VXUS', 'BND', 'AGG', 'LQD', 'HYG', 'SHY', 'IEF', 'TLT', 'TIP',  # Asset Classes
    'MGK', 'MGV', 'VBK', 'VBR', 'SCHA', 'SCHG', 'SCHV', 'IWO', 'IWN', 'IWP',  # Style Box
    'FTEC', 'FHLC', 'VGT', 'VFH', 'VHT', 'VDE', 'VIS', 'VPU', 'VAW',  # Fidelity/Vanguard Sectors
    'PAVE', 'IFRA', 'GRID', 'BLOK', 'SRVR', 'VNQ', 'REM', 'REZ', 'IYR', 'ICF',  # Infrastructure/REITs
    'AMLP', 'MLPA', 'DBC', 'GSG', 'GLD', 'SLV', 'IAU', 'PALL', 'CPER', 'WOOD',  # Commodities
    'KWEB', 'ASHR', 'EEMV', 'EFA', 'EFV', 'SCZ', 'IEFA', 'EFAW', 'HEFA', 'DBEF',  # Global Qual
    'DSI', 'ESGU', 'SUSA', 'ESGD', 'ESGE', 'XVV'  # ESG
]))


def get_data():
    fm, fp = 'meta.csv', 'prices.csv'
    
    # 1. Try to load existing local files first
    meta_local = None
    prices_local = None
    
    if os.path.exists(fm) and os.path.exists(fp):
        print("📁 Found local CSV files. Loading...")
        try:
            meta_local = pd.read_csv(fm, index_col=0)
            prices_local = pd.read_csv(fp, index_col=0, parse_dates=True)
            # If data is fresh enough (less than 24h), just return it
            if (time.time() - os.path.getmtime(fp) < 86400):
                print("⚡ Data is fresh. Skipping download.")
                return meta_local, prices_local
        except Exception as e:
            print(f"⚠️ Error reading local files: {e}")

    # 2. Try to download new data
    print("🚀 Step 1: Attempting to Download New Data...")
    try:
        # Wikipedia Sector Data
        headers = {"User-Agent": "Mozilla/5.0"}
        wiki_resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', headers=headers, timeout=10)
        df = pd.read_html(io.StringIO(wiki_resp.text))[0]
        df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
        meta = df.set_index('Symbol')[['GICS Sector', 'GICS Sub-Industry']]
        
        # Prices Data
        all_ticks = list(set(meta.index.tolist() + ['^GSPC'] + ETF_UNIVERSE))
        raw = yf.download(all_ticks, period="5y", interval="1wk", progress=False, auto_adjust=True, session=session)['Close']
        
        if raw is not None and not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex): 
                raw.columns = raw.columns.get_level_values(0)
            meta.to_csv(fm)
            raw.to_csv(fp)
            return meta, raw
        else:
            raise ValueError("Yahoo Finance returned empty data.")

    except Exception as e:
        print(f"❌ Download failed: {e}")
        if prices_local is not None:
            print("🔄 Reverting to local CSV files...")
            return meta_local, prices_local
        else:
            print("🚨 CRITICAL: No local data and download failed.")
            return None, None


def get_ultimate_etf_portfolio(target_ret, etf_data):
    """
    Finds the 'Ultimate' ETF portfolio (Max 4 ETFs).
    """
    print(f"🔍 Analyzing {etf_data.shape[1]} ETFs for Alpha & Fit...")

    etf_rets = etf_data.pct_change().fillna(0)
    common = target_ret.index.intersection(etf_rets.index)
    target = target_ret.loc[common]
    candidates = etf_rets.loc[common]

    # Score Candidates
    scores = {}
    for tick in candidates.columns:
        r = candidates[tick]
        sharpe = (r.mean() / (r.std() + 1e-9)) * np.sqrt(52)
        corr = r.corr(target)
        scores[tick] = (SHARPE * sharpe) + (CORR * corr)

    # Shortlist Top 15 (Expanded search space)
    top_candidates = sorted(scores, key=scores.get, reverse=True)[:15]
    print(f"    ↳ Shortlist (Top 15): {top_candidates}")

    cand_data = candidates[top_candidates]

    def objective(trial):
        # Weight selection for the shortlist
        w = np.array([trial.suggest_float(t, 0, 1) for t in top_candidates])

        # Thresholding to force sparsity
        w[w < 0.15] = 0

        # Hard constraints
        if w.sum() == 0: return -1e9
        w /= w.sum()

        # STRICT CONSTRAINT: Max 4 ETFs allowed
        if np.count_nonzero(w) > 4:
            return -1e9

        port_ret = (cand_data * w).sum(axis=1)
        sharpe = (port_ret.mean() / (port_ret.std() + 1e-9)) * np.sqrt(52)

        # Diversity Bonus: Prefer 3 or 4 over just 1 or 2
        active = np.count_nonzero(w)
        bonus = 0.1 if active >= 3 else 0

        return sharpe + bonus

    print("    ↳ Running Thousands Of Genetic Optimization Trials (Deep Research)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1000)

    best_w = {k: v for k, v in study.best_params.items() if v >= 0.15}

    # Final cleanup to ensure strictly <= 4 in case floating point weirdness
    if len(best_w) > 4:
        # Keep top 4 only
        best_w = dict(sorted(best_w.items(), key=lambda item: item[1], reverse=True)[:4])

    tot = sum(best_w.values())
    final_w = {k: v / tot for k, v in best_w.items()} if tot > 0 else {}

    return final_w


def run_pipeline():
    meta, prices = get_data()
    
    if prices is None or meta is None:
        print("🛑 Pipeline halted: No data available.")
        return

    # Ensure all column names are strings for filtering
    prices.columns = prices.columns.astype(str)

    valid_etfs = [c for c in ETF_UNIVERSE if c in prices.columns]
    etf_prices = prices[valid_etfs]
    stock_cols = [c for c in prices.columns if c not in valid_etfs and c != '^GSPC']
    stock_prices = prices[stock_cols]

    bench = prices['^GSPC'].pct_change().fillna(0) if '^GSPC' in prices.columns else prices.iloc[
        :, 0].pct_change().fillna(0)

    print("📊 Step 2: Processing Stocks...")
    rets = np.log(stock_prices / stock_prices.shift(1)).iloc[1:].ffill().bfill()
    rets = rets.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='any')

    X = rets.T.values
    mu, sd = np.nanmean(X, axis=1, keepdims=True), np.nanstd(X, axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    Xz = np.nan_to_num((X - mu) / sd)

    def best_k_loop(data):
        best = None
        for k in range(5, 12):
            km = KMeans(n_clusters=k, n_init=20, max_iter=500, random_state=42).fit(data)
            sil = silhouette_score(data, km.labels_)
            if best is None or sil > best['sil']: best = {'k': k, 'sil': sil, 'model': km}
        return best['model']

    print("🧠 Step 3: Clustering & Pruning...")
    km1 = best_k_loop(Xz)

    sil = silhouette_samples(Xz, km1.labels_)
    dist = np.linalg.norm(Xz - km1.cluster_centers_[km1.labels_], axis=1)
    df_s = pd.DataFrame({'t': rets.columns, 'c': km1.labels_, 'sil': sil, 'dist': dist})
    df_s['z'] = df_s.groupby('c')['dist'].transform(zscore)

    mask = ((df_s['sil'] < -0.07) | (df_s.groupby('c')['sil'].transform(lambda x: x <= x.quantile(0.03))) | (
            df_s['z'].abs() > 3.0)) & (~df_s['t'].isin(BLUE_CHIPS))
    keepers = df_s[~mask].copy()

    # --- NEW: Print Pruning Stats ---
    print(f"✂️  Pruned {len(df_s) - len(keepers)} bad stocks. Retaining {len(keepers)} high-quality assets.")
    # --------------------------------

    Xz_clean = Xz[keepers.index]
    km2 = best_k_loop(Xz_clean)
    keepers.loc[:, 'c'] = km2.labels_

    sec_idxs, ui_clusters = {}, []
    pca_pts = PCA(2).fit_transform(Xz_clean)

    for cid in sorted(keepers['c'].unique()):
        ticks = keepers[keepers['c'] == cid]['t'].tolist()
        name = f"Cluster {int(cid) + 1}"
        sec_idxs[name] = stock_prices[ticks].pct_change().mean(axis=1).fillna(0)

        def spark(t):
            try:
                v = (stock_prices[t] / stock_prices[t].iloc[0]).values
                pts = [f"{(i / (len(v) - 1)) * 450},{60 - ((n - min(v)) / (max(v) - min(v) + 1e-6) * 60)}" for i, n in
                       enumerate(v)]
                return f'<svg width="450" height="60" style="fill:none;stroke:#0d6efd;stroke-width:2"><polyline points="{" ".join(pts)}"/></svg>'
            except:
                return ""

        ui_clusters.append({
            'id': int(cid), 'name': name, 'count': len(ticks),
            'members': [{'ticker': t, 'last_vol': round(float(keepers.loc[keepers['t'] == t, 'z'].iloc[0]), 2),
                         'sparkline': spark(t)} for t in ticks]
        })

    print("⚖️ Step 4: Optimizing Stocks...")

    def obj_stock(t):
        w = np.array([t.suggest_float(c, 0, 1) for c in sec_idxs])
        w[w < 0.10] = 0
        if w.sum() == 0: return -1e9
        w /= w.sum()
        r = (pd.DataFrame(sec_idxs) * w).sum(axis=1)
        return (r.mean() / (r.std() + 1e-9)) * np.sqrt(52) - (0.1 if np.count_nonzero(w) > 4 else 0)

    study = optuna.create_study(direction="maximize")
    study.optimize(obj_stock, n_trials=1000)

    best_w = {k: v for k, v in study.best_params.items() if v >= 0.10}
    tot_w = sum(best_w.values())
    final_w = {k: v / tot_w for k, v in best_w.items()} if tot_w > 0 else {}
    opt_ret = (pd.DataFrame(sec_idxs) * pd.Series(final_w)).sum(axis=1)

    print("🚀 Step 5: Generating ULTIMATE ETF Portfolio...")
    best_etf_combo = get_ultimate_etf_portfolio(opt_ret, etf_prices)

    etf_ret = (etf_prices[list(best_etf_combo.keys())].pct_change().fillna(0) * pd.Series(best_etf_combo)).sum(axis=1)

    stock_sharpe = (opt_ret.mean() / opt_ret.std()) * np.sqrt(52)
    etf_sharpe = (etf_ret.mean() / etf_ret.std()) * np.sqrt(52)

    print(f"🎯 Results -> Stock Sharpe: {stock_sharpe:.2f} | ETF Sharpe: {etf_sharpe:.2f}")

    for c in ui_clusters: c['weight'] = f"{final_w.get(c['name'], 0) * 100:.1f}%"

    common = opt_ret.index.intersection(etf_ret.index)

    cache.update({
        'clusters': sorted(ui_clusters, key=lambda x: float(x['weight'].strip('%')), reverse=True),
        'etf_proxy': [{'ticker': k, 'weight': f"{v * 100:.1f}%"} for k, v in best_etf_combo.items()],
        'etf_sharpe': round(etf_sharpe, 2),
        'stock_sharpe': round(stock_sharpe, 2),
        'pca_data': [{'x': float(pca_pts[i, 0]), 'y': float(pca_pts[i, 1]), 'ticker': t,
                      'cluster': f"Cluster {km2.labels_[i] + 1}"} for i, t in enumerate(keepers['t'])],
        'perf_data': {
            'dates': common.strftime('%Y-%m-%d').tolist(),
            'sectors': {k: ((1 + v.loc[common]).cumprod() * 100).tolist() for k, v in sec_idxs.items()},
            'optimized': ((1 + opt_ret.loc[common]).cumprod() * 100).tolist(),
            'etf_proxy': ((1 + etf_ret.loc[common]).cumprod() * 100).tolist(),
            'benchmark': ((1 + bench.loc[common]).cumprod() * 100).tolist()
        }
    })


@app.route('/')
def index(): return render_template('index.html', **cache)


if __name__ == '__main__':
    if os.path.exists('prices.csv'):
        try:
            df = pd.read_csv('prices.csv', index_col=0, nrows=1)
            if 'SPY' not in df.columns: os.remove('prices.csv')
        except:
            pass

    run_pipeline()
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)