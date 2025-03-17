#!/usr/bin/env python
import requests
import json
import time
from bs4 import BeautifulSoup
import datetime as dt
import os
import pickle
import yaml
import yfinance as yf
import pandas as pd
import dateutil.relativedelta
import numpy as np
import re

from datetime import date
from datetime import datetime

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return int(obj) if isinstance(obj, np.integer) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Use the current working directory
DIR = os.getcwd()

# Create necessary directories
for subdir in ['data', 'tmp']:
    path = os.path.join(DIR, subdir)
    if not os.path.exists(path):
        os.makedirs(path)

# Load configuration files
def load_config(file_path):
    try:
        with open(file_path, 'r') as stream:
            return yaml.safe_load(stream)
    except FileNotFoundError:
        return None
    except yaml.YAMLError as exc:
        print(f"Error loading YAML file {file_path}: {exc}")
        return None

private_config = load_config(os.path.join(DIR, 'config_private.yaml'))
config = load_config(os.path.join(DIR, 'config.yaml'))

def cfg(key):
    return private_config.get(key) if private_config else config.get(key) if config else None

def is_valid_ticker(ticker):
    if not isinstance(ticker, str):
        return False
    return bool(re.match(r'^[A-Z0-9.-]+$', ticker))

def getSecurities(url, tickerPos=2, tablePos=1, sectorPosOffset=1, universe="N/A"):
    print(f"Fetching securities from {url}")
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    tables = soup.find_all('table', {'class': ['wikitable sortable', 'wikitable']})
    print(f"Found {len(tables)} tables on {url}")

    if not tables:
        print(f"No tables found with BeautifulSoup on {url}. Falling back to pandas.read_html.")
        try:
            tables = pd.read_html(url)
            if not tables:
                raise ValueError(f"No tables found on {url}.")
            table = tables[tablePos - 1] if tablePos - 1 < len(tables) else tables[0]
            table_html = str(table.to_html())
            soup = BeautifulSoup(table_html, 'html.parser')
            table = soup.find('table')
        except Exception as e:
            raise ValueError(f"Failed to fetch tables with pandas on {url}: {str(e)}")
    else:
        if tablePos - 1 >= len(tables):
            print(f"tablePos ({tablePos}) out of range for {len(tables)} tables. Using the first table.")
            table = tables[0]
        else:
            table = tables[tablePos - 1]

    secs = {}
    for row in table.find_all('tr')[1:]:
        cells = row.find_all('td')
        if len(cells) < max(tickerPos, tickerPos + sectorPosOffset):
            continue
        ticker = cells[tickerPos-1].text.strip()
        if not is_valid_ticker(ticker):
            print(f"Skipping invalid ticker: {ticker}")
            continue
        sec = {
            "ticker": ticker,
            "sector": cells[tickerPos-1 + sectorPosOffset].text.strip(),
            "universe": universe
        }
        secs[ticker] = sec
        print(f"Extracted: ticker={sec['ticker']}, sector={sec['sector']}, universe={sec['universe']}")

    with open(os.path.join(DIR, "tmp", "tickers.pickle"), "wb") as f:
        pickle.dump(secs, f)
    return secs

def get_resolved_securities():
    tickers = {}
    if cfg("NQ100"):
        tickers.update(getSecurities('https://en.wikipedia.org/wiki/Nasdaq-100', tickerPos=2, tablePos=1, sectorPosOffset=1, universe="Nasdaq 100"))
    if cfg("SP500"):
        tickers.update(getSecurities('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', tickerPos=1, tablePos=1, sectorPosOffset=3, universe="S&P 500"))
    if cfg("SP400"):
        tickers.update(getSecurities('https://en.wikipedia.org/wiki/List_of_S%26P_400_companies', tickerPos=2, tablePos=1, sectorPosOffset=1, universe="S&P 400"))
    if cfg("SP600"):
        tickers.update(getSecurities('https://en.wikipedia.org/wiki/List_of_S%26P_600_companies', tickerPos=2, tablePos=1, sectorPosOffset=1, universe="S&P 600"))
    return tickers

API_KEY = cfg("API_KEY")
TD_API = cfg("TICKERS_API")
PRICE_DATA_OUTPUT = os.path.join(DIR, "data", "price_history.json")
SECURITIES = list(get_resolved_securities().values())
DATA_SOURCE = cfg("DATA_SOURCE")

def create_price_history_file(tickers_dict):
    with open(PRICE_DATA_OUTPUT, "w") as fp:
        json.dump(tickers_dict, fp, cls=NumpyEncoder)

def enrich_ticker_data(ticker_response, security):
    ticker_response["sector"] = security["sector"]
    ticker_response["universe"] = security["universe"]

def tda_params(apikey, period_type="year", period=1, frequency_type="daily", frequency=1):
    return (
        ("apikey", apikey),
        ("periodType", period_type),
        ("period", period),
        ("frequencyType", frequency_type),
        ("frequency", frequency)
    )

def print_data_progress(ticker, universe, idx, securities, error_text, elapsed_s, remaining_s):
    dt_ref = datetime.fromtimestamp(0)
    dt_e = datetime.fromtimestamp(elapsed_s)
    elapsed = dateutil.relativedelta.relativedelta(dt_e, dt_ref)
    remaining_string = (f"{int(remaining_s // 60)}m {int(remaining_s % 60)}s" if remaining_s and not np.isnan(remaining_s) else "?")
    print(f"{ticker} from {universe}{error_text} ({idx+1} / {len(securities)}). Elapsed: {elapsed.minutes}m {elapsed.seconds}s. Remaining: {remaining_string}.")

def get_remaining_seconds(load_times, idx, total_len):
    if not load_times:
        return np.nan
    window = min(idx + 1, 25)
    load_time_ma = pd.Series(load_times).rolling(window).mean().iloc[-1]
    remaining_seconds = (total_len - (idx + 1)) * load_time_ma
    return remaining_seconds

def load_prices_from_tda(securities):
    print("*** Loading Stocks from TD Ameritrade ***")
    headers = {"Cache-Control": "no-cache"}
    params = tda_params(API_KEY)
    tickers_dict = {}
    start = time.time()
    load_times = []

    for idx, sec in enumerate(securities):
        r_start = time.time()
        response = requests.get(TD_API % sec["ticker"], params=params, headers=headers)
        current_load_time = time.time() - r_start
        load_times.append(current_load_time)
        remaining_seconds = get_remaining_seconds(load_times, idx, len(securities))
        ticker_data = response.json()
        enrich_ticker_data(ticker_data, sec)
        tickers_dict[sec["ticker"]] = ticker_data
        error_text = f' Error with code {response.status_code}' if response.status_code != 200 else ''
        print_data_progress(sec["ticker"], sec["universe"], idx, securities, error_text, time.time() - start, remaining_seconds)

    create_price_history_file(tickers_dict)

def get_yf_data(security, start_date, end_date):
    ticker = security["ticker"]
    escaped_ticker = ticker.replace(".", "-")
    print(f"Fetching data for ticker: {escaped_ticker}")
    try:
        df = yf.download(escaped_ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
        if df.empty:
            print(f"No data found for ticker {escaped_ticker}")
            return None

        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df[required_columns]

        timestamps = [int(ts.timestamp()) for ts in df.index]
        ticker_data = {
            "candles": [
                {
                    "open": float(df['Open'].iloc[i].iloc[0]),    # Access scalar value
                    "close": float(df['Close'].iloc[i].iloc[0]),  # Access scalar value
                    "low": float(df['Low'].iloc[i].iloc[0]),      # Access scalar value
                    "high": float(df['High'].iloc[i].iloc[0]),    # Access scalar value
                    "volume": int(df['Volume'].iloc[i].iloc[0]),  # Access scalar value
                    "datetime": timestamps[i]
                }
                for i in range(len(df))
            ]
        }
        enrich_ticker_data(ticker_data, security)
        return ticker_data
    except Exception as e:
        print(f"Error fetching data for {escaped_ticker}: {str(e)}")
        return None

def load_prices_from_yahoo(securities):
    print("*** Loading Stocks from Yahoo Finance ***")
    today = date.today()
    start_date = today - dt.timedelta(days=365)
    tickers_dict = {}
    start = time.time()
    load_times = []

    for idx, security in enumerate(securities):
        r_start = time.time()
        ticker_data = get_yf_data(security, start_date, today)
        if ticker_data is None:
            continue
        current_load_time = time.time() - r_start
        load_times.append(current_load_time)
        remaining_seconds = get_remaining_seconds(load_times, idx, len(securities))
        tickers_dict[security["ticker"]] = ticker_data
        print_data_progress(security["ticker"], security["universe"], idx, securities, "", time.time() - start, remaining_seconds)

    create_price_history_file(tickers_dict)

def save_data(source, securities):
    if source == "YAHOO":
        load_prices_from_yahoo(securities)
    elif source == "TD_AMERITRADE":
        load_prices_from_tda(securities)
    else:
        raise ValueError(f"Unknown data source: {source}")

def main():
    if not SECURITIES:
        print("No securities found to process. Check configuration and URLs.")
        return
    if not DATA_SOURCE:
        print("Data source not specified in configuration.")
        return
    save_data(DATA_SOURCE, SECURITIES)

if __name__ == "__main__":
    main()
