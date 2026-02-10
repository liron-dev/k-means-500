from flask import Flask, render_template
import pandas as pd
import io
import requests

app = Flask(__name__)


# Wikipedia is the most frequently updated community-driven source for the S&P 500:
def get_latest_sp500():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    tables = pd.read_html(io.StringIO(response.text))
    df = tables[0]
    tickers = df.iloc[:, 0].tolist()
    clean_tickers = [str(t).replace('.', '-') for t in tickers]
    return sorted(clean_tickers)


@app.route('/')
def index():
    tickers = get_latest_sp500()
    return render_template('index.html', tickers=tickers, count=len(tickers))


if __name__ == '__main__':
    app.run(debug=True, port=5000)