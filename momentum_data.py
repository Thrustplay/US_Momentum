#!/usr/bin/env python
import requests
import json
import time
import bs4 as bs
import datetime as dt
import os
import pandas_datareader.data as web
import pickle
import requests
from bs4 import BeautifulSoup
import yaml
import yfinance as yf
import pandas as pd
import dateutil.relativedelta
import numpy as np

from datetime import date
from datetime import datetime

# Use the current working directory in Colab
DIR = os.getcwd()

if not os.path.exists(os.path.join(DIR, 'data')):
    os.makedirs(os.path.join(DIR, 'data'))
if not os.path.exists(os.path.join(DIR, 'tmp')):
    os.makedirs(os.path.join(DIR, 'tmp'))

try:
    with open(os.path.join(DIR, 'config_private.yaml'), 'r') as stream:
        private_config = yaml.safe_load(stream)
except FileNotFoundError:
    private_config = None
except yaml.YAMLError as exc:
        print(exc)

try:
    with open(os.path.join(DIR, 'config.yaml'), 'r') as stream:
        config = yaml.safe_load(stream)
except FileNotFoundError:
    config = None
except yaml.YAMLError as exc:
        print(exc)

def cfg(key):
    try:
        return private_config[key]
    except:
        try:
            return config[key]
        except:
            return None

def getSecurities(url, tickerPos=2, tablePos=1, sectorPosOffset=1, universe="N/A"):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Try finding tables with different possible class names
    tables = soup.find_all('table', {'class': 'wikitable sortable'}) or soup.find_all('table', {'class': 'wikitable'})
    print(f"Found {len(tables)} tables on {url} with class 'wikitable sortable' or 'wikitable'")  # Debug output

    if not tables:
        print(f"No tables found with BeautifulSoup on {url}. Falling back to pandas.read_html.")
        try:
            tables = pd.read_html(url)
            if not tables:
                raise ValueError(f"No tables found on {url}. Check the URL or structure.")
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
    for row in table.find_all('tr')[1:]:  # Skip header row
        cells = row.find_all('td')
        if len(cells) < max(tickerPos, tickerPos + sectorPosOffset):
            continue
        sec = {}
        sec["ticker"] = cells[tickerPos-1].text.strip()
        sec["sector"] = cells[tickerPos-1 + sectorPosOffset].text.strip()
        sec["universe"] = universe
        secs[sec["ticker"]] = sec

    with open(os.path.join(DIR, "tmp", "tickers.pickle"), "wb") as f:
        pickle.dump(secs, f)
    return secs

def get_resolved_securities():
    tickers = {}
    if cfg("NQ100"):
        tickers.update(getSecurities('https://en.wikipedia.org/wiki/Nasdaq-100', 2, 1, universe="Nasdaq 100"))
    if cfg("SP500"):
        tickers.update(getSecurities('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies', 1, 1, sectorPosOffset=3, universe="S&P 500"))
    if cfg("SP400"):
        tickers.update(getSecurities('https://en.wikipedia.org/wiki/List_of_S%26P_400_companies', 2, 1, universe="S&P 400"))
    if cfg("SP600"):
        tickers.update(getSecurities('https://en.wikipedia.org/wiki/List_of_S%26P_600_companies', 2, 1, universe="S&P 600"))
    return tickers

API_KEY = cfg("API_KEY")
TD_API = cfg("TICKERS_API")
PRICE_DATA_OUTPUT = os.path.join(DIR, "data", "price_history.json")
SECURITIES = get_resolved_securities().values()
DATA_SOURCE = cfg("DATA_SOURCE")

def create_price_history_file(tickers_dict):
    with open(PRICE_DATA_OUTPUT, "w") as fp:
        json.dump(tickers_dict, fp)

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
    if remaining_s and not np.isnan(remaining_s):
        dt_r = datetime.fromtimestamp(remaining_s)
        remaining = dateutil.relativedelta.relativedelta(dt_r, dt_ref)
        remaining_string = f'{remaining.minutes}m {remaining.seconds}s'
    else:
        remaining_string = "?"
    print(f'{ticker} from {universe}{error_text} ({idx+1} / {len(securities)}). Elapsed: {elapsed.minutes}m {elapsed.seconds}s. Remaining: {remaining_string}.')

def get_remaining_seconds(all_load_times, idx, len):
    load_time_ma = pd.Series(all_load_times).rolling(np.minimum(idx+1, 25)).mean().tail(1).item()
    remaining_seconds = (len - idx) * load_time_ma
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
        now = time.time()
        current_load_time = now - r_start
        load_times.append(current_load_time)
        remaining_seconds = get_remaining_seconds(load_times, idx, len(securities))
        ticker_data = response.json()
        enrich_ticker_data(ticker_data, sec)
        tickers_dict[sec["ticker"]] = ticker_data
        error_text = f' Error with code {response.status_code}' if response.status_code != 200 else ''
        print_data_progress(sec["ticker"], sec["universe"], idx, securities, error_text, now - start, remaining_seconds)

    create_price_history_file(tickers_dict)

def get_yf_data(security, start_date, end_date):
    escaped_ticker = security["ticker"].replace(".", "-")
    print(f"Fetching data for ticker: {escaped_ticker}")  # Debug output
    df = yf.download(escaped_ticker, start=start_date, end=end_date)
    if df.empty:
        print(f"No data found for ticker {escaped_ticker}")
        return None  # Return None for invalid tickers
    yahoo_response = df.to_dict()
    timestamps = list(yahoo_response["Open"].keys())
    timestamps = list(map(lambda timestamp: int(timestamp.timestamp()), timestamps))
    opens = list(yahoo_response["Open"].values())
    closes = list(yahoo_response["Close"].values())
    lows = list(yahoo_response["Low"].values())
    highs = list(yahoo_response["High"].values())
    volumes = list(yahoo_response["Volume"].values())
    ticker_data = {}
    candles = []

    for i in range(0, len(opens)):
        candle = {}
        candle["open"] = opens[i]
        candle["close"] = closes[i]
        candle["low"] = lows[i]
        candle["high"] = highs[i]
        candle["volume"] = volumes[i]
        candle["datetime"] = timestamps[i]
        candles.append(candle)

    ticker_data["candles"] = candles
    enrich_ticker_data(ticker_data, security)
    return ticker_data

def load_prices_from_yahoo(securities):
    print("*** Loading Stocks from Yahoo Finance ***")
    today = date.today()
    start = time.time()
    start_date = today - dt.timedelta(days=1*365)
    tickers_dict = {}
    load_times = []
    for idx, security in enumerate(securities):
        r_start = time.time()
        ticker_data = get_yf_data(security, start_date, today)
        if ticker_data is None:  # Skip invalid tickers
            continue
        now = time.time()
        current_load_time = now - r_start
        load_times.append(current_load_time)
        remaining_seconds = get_remaining_seconds(load_times, idx, len(securities))
        print_data_progress(security["ticker"], security["universe"], idx, securities, "", time.time() - start, remaining_seconds)
        tickers_dict[security["ticker"]] = ticker_data
    create_price_history_file(tickers_dict)

def save_data(source, securities):
    if source == "YAHOO":
        load_prices_from_yahoo(securities)
    elif source == "TD_AMERITRADE":
        load_prices_from_tda(securities)

def main():
    save_data(DATA_SOURCE, SECURITIES)

if __name__ == "__main__":
    main()
