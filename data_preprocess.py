# main_prepare_data.py

import os
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from io import BytesIO
from fredapi import Fred
import config

def download_bitcoin_data(start_date, end_date, metrics, sleep_time=1):
    url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
    all_data = []
    current = start_date

    while current <= end_date:
        batch_end = min(current + timedelta(days=99), end_date)
        params = {
            "assets": "btc",
            "metrics": ",".join(metrics),
            "frequency": "1d",
            "start_time": current.strftime("%Y-%m-%d"),
            "end_time": batch_end.strftime("%Y-%m-%d")
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            df = pd.json_normalize(response.json()["data"])
            all_data.append(df)

        current = batch_end + timedelta(days=1)
        time.sleep(sleep_time)

    bitcoin_df = pd.concat(all_data, ignore_index=True)
    bitcoin_df.rename(columns={'time': 'Date'}, inplace=True)
    bitcoin_df.set_index('Date', inplace=True)
    bitcoin_df.index = pd.to_datetime(bitcoin_df.index).strftime('%Y-%m-%d')
    return bitcoin_df

def download_stock_data(tickers, period='15y'):
    stock = yf.download(tickers, period=period)['Close'].dropna()
    stock.index = pd.to_datetime(stock.index).strftime('%Y-%m-%d')
    stock.index.name = 'Date'
    return stock

def download_macro_data():
    fred = Fred(api_key='dcb6fc9c918dcd83799557ec4da97330')
    data = {
        'Risk_Free': fred.get_series('DTB3'),
        'OAS': fred.get_series('BAMLH0A0HYM2'),
        'Ten_Year': fred.get_series('DGS10'),
        'Two_Year': fred.get_series('DGS2'),
        'Term_Spread': fred.get_series('DGS10') - fred.get_series('DGS2'),
        'VIX': fred.get_series('VIXCLS'),
        'USD_Index': fred.get_series('DTWEXBGS'),
        'Expected_Inflation': fred.get_series('T10YIE')
    }
    macro = pd.concat(data, axis=1)
    macro.index = pd.to_datetime(macro.index).strftime('%Y-%m-%d')
    macro.index.name = 'Date'
    return macro

def download_sentiment_data():
    url = 'https://www.frbsf.org/wp-content/uploads/news_sentiment_data.xlsx?20240826&2025-03-28'
    response = requests.get(url)
    sentiment = pd.read_excel(BytesIO(response.content), sheet_name="Data")
    sentiment.rename(columns={'News Sentiment': 'US_News_Sentiment'}, inplace=True)
    sentiment['Date'] = pd.to_datetime(sentiment['date']).dt.strftime('%Y-%m-%d')
    sentiment.set_index('Date', inplace=True)
    sentiment.drop(columns=['date'], inplace=True)
    return sentiment


def preprocess_series(df, col, method):
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")

    if method == "log_diff":
        df[col] = np.log(df[col]) - np.log(df[col].shift(1))
    elif method == "diff":
        df[col] = df[col].diff()
    elif method == "log":
        df[col] = np.log(df[col])
    elif method == "pct_change":
        df[col] = df[col].pct_change()
    elif method == "zscore":
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df

def create_lag_target(df, target_col, lags):
    df = df.copy()
    df["target_t_plus_1"] = df[target_col].shift(-1)
    for i in range(lags):
        df[f"lag_{i+1}"] = df[target_col].shift(i)
    return df.dropna()


def main():

    btc_metrics = ["PriceUSD", "BlkCnt", "FeeMeanUSD", "DiffMean", "BlkSizeMeanByte", "RevUSD", "TxTfrValAdjUSD", "HashRate"]

    if os.path.exists(config.BITCOIN):
        bitcoin = pd.read_csv(config.BITCOIN, index_col=0, parse_dates=True)
    else:
        bitcoin = download_bitcoin_data(datetime(2010, 7, 1), datetime(2025, 2, 28), btc_metrics)
        bitcoin = bitcoin[btc_metrics]
        bitcoin.to_csv(config.BITCOIN)

    if os.path.exists(config.EXOG_VARS):
        exog = pd.read_csv(config.EXOG_VARS, index_col=0, parse_dates=True)
    else:
        stock = download_stock_data(['^GSPC', 'URTH'])
        macro = download_macro_data()
        sentiment = download_sentiment_data()
        exog = pd.concat([stock, macro, sentiment], axis=1).dropna()
        exog.to_csv(config.EXOG_VARS)
        
    full_data = pd.concat([bitcoin, exog], axis=1).ffill()
    full_data.index.name = 'Date'
    full_data.to_csv(config.RAW_DATA)

    raw_data = pd.read_csv(config.RAW_DATA, index_col="Date", parse_dates=True)
    raw_data = raw_data[(raw_data.index >= "2012-01-01") & (raw_data.index <= "2023-02-28")]

    preprocess_methods = {
        "PriceUSD": "pct_change",
        "BlkCnt": "zscore",
        "BlkSizeMeanByte": "pct_change",
        "DiffMean": "pct_change",
        "FeeMeanUSD": "zscore",
        "HashRate": "pct_change",
        "RevUSD": "pct_change",
        "TxTfrValAdjUSD": "pct_change",
        "URTH": "pct_change",
        "^GSPC": "pct_change",
        "Risk_Free": "diff",
        "OAS": "zscore",
        "Ten_Year": "pct_change",
        "Two_Year": "pct_change",
        "Term_Spread": "diff",
        "VIX": "zscore",
        "USD_Index": "pct_change",
        "Expected_Inflation": "pct_change",
        "US_News_Sentiment": "zscore"
    }

    for col, method in preprocess_methods.items():
        raw_data = preprocess_series(raw_data, col, method)
    raw_data = raw_data[(raw_data.index >= "2017-01-01") & (raw_data.index <= "2023-02-28")]
    raw_data.to_csv(config.PREPROCESS_DATA)


if __name__ == "__main__":
    main()

    # preprocess_methods = {
    #     "PriceUSD": "pct_change",
    #     "BlkCnt": "raw",
    #     "BlkSizeMeanByte": "pct_change",
    #     "DiffMean": "pct_change",
    #     "FeeMeanUSD": "raw",
    #     "HashRate": "pct_change",
    #     "RevUSD": "pct_change",
    #     "TxTfrValAdjUSD": "pct_change",
    #     "URTH": "pct_change",
    #     "^GSPC": "pct_change",
    #     "Risk_Free": "diff",
    #     "OAS": "raw",
    #     "Ten_Year": "pct_change",
    #     "Two_Year": "pct_change",
    #     "Term_Spread": "diff",
    #     "VIX": "raw",
    #     "USD_Index": "pct_change",
    #     "Expected_Inflation": "pct_change",
    #     "US_News_Sentiment": "raw"
    # }