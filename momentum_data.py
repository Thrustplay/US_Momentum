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
import re  # Added for ticker validation

from datetime import date
from datetime import datetime

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
    """Retrieve a configuration value, prioritizing private_config over config."""
    return private_config.get(key) if private_config else config.get(key) if config else None

# Function to validate ticker symbols
def is_valid_ticker(ticker):
    """Check if a string is a valid ticker symbol (alphanumeric, may include dots or hyphens)."""
    if not isinstance(ticker, str):
        return False
    # Valid tickers: uppercase letters, numbers, dots, or hyphens (e.g., "AAPL", "BRK.B", "GOOGL")
    return bool(re.match(r'^[A-Z0-9.-]+$', ticker))

def getSecurities(url, tickerPos=2, tablePos=1, sectorPosOffset=1, universe="N/A"):
    """
    Fetch securities (tickers, sectors, universe) from a Wikipedia page.
    Args:
        url (str): URL of the Wikipedia page.
        tickerPos (int): Column position of the ticker (1-based index).
        tablePos (int): Table position on the page (1-based index).
        sectorPosOffset (int): Offset from tickerPos to the sector column.
        universe (str): Universe name (e.g., 'Nasdaq 100').
    Returns:
        dict: Dictionary of securities with ticker as key.
    """
    print(f"Fetching securities from {url}")
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find tables with class 'wikitable sortable' or 'wikitable'
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
    for row in table.find_all('tr')[1:]:  # Skip header row
        cells = row.find_all('td')
        if len(cells) < max(tickerPos, tickerPos + sectorPosOffset):
            continue
        ticker = cells[tickerPos-1].text.strip()
        # Validate ticker before adding
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

    # Save to pickle file
    with open(os.path.join(DIR, "tmp", "tickers.pickle"), "wb") as f:
        pickle.dump(secs, f)
    return secs

def get_resolved_securities():
    """Fetch securities from configured universes."""
    tickers = {}
    if cfg("NQ100"):
        tickers.update(getSecurities('https://en.wikipedia.org/wiki/Nasdaq-100', tickerPos=2, tablePos=1, sectorPosOffset=1, universe="Nasdaq 100"))
    if cfg("SP500"):
        # S&P 500: tickerPos=1 (first column), tablePos=1 (first table), sector is 4th column (offset=3)
        tickers.update(getSecurities('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', tickerPos=1, tablePos=1, sectorPosOffset=3, universe="S&P 500"))
    if cfg("SP400"):
        tickers.update(getSecurities('https://en.wikipedia.org/wiki/List_of_S%26P_400_companies', tickerPos=2, tablePos=1, sectorPosOffset=1, universe="S&P 400"))
    if cfg("SP600"):
        tickers.update(getSecurities('https://en.wikipedia.org/wiki/List_of_S%26P_600_companies', tickerPos=2, tablePos=1, sectorPosOffset=1, universe="S&P 600"))
    return tickers

# Configuration variables
API_KEY = cfg("API_KEY")
TD_API = cfg("TICKERS_API")
PRICE_DATA_OUTPUT = os.path.join(DIR, "data", "price_history.json")
SECURITIES = list(get_resolved_securities().values())  # Convert to list for iteration
DATA_SOURCE = cfg("DATA_SOURCE")

def create_price_history_file(tickers_dict):
    """Save ticker data to a JSON file."""
    with open(PRICE_DATA_OUTPUT, "w") as fp:
        json.dump(tickers_dict, fp)

def enrich_ticker_data(ticker_response, security):
    """Add sector and universe metadata to ticker data."""
    ticker_response["sector"] = security["sector"]
    ticker_response["universe"] = security["universe"]

def tda_params(apikey, period_type="year", period=1, frequency_type="daily", frequency=1):
    """Return TD Ameritrade API parameters."""
    return (
        ("apikey", apikey),
        ("periodType", period_type),
        ("period", period),
        ("frequencyType", frequency_type),
        ("frequency", frequency)
    )

def print_data_progress(ticker, universe, idx, securities, error_text, elapsed_s, remaining_s):
    """Print progress of data fetching with elapsed and estimated remaining time."""
    dt_ref = datetime.fromtimestamp(0)
    dt_e = datetime.fromtimestamp(elapsed_s)
    elapsed = dateutil.relativedelta.relativedelta(dt_e, dt_ref)
    remaining_string = (f"{int(remaining_s // 60)}m {int(remaining_s % 60)}s" if remaining_s and not np.isnan(remaining_s) else "?")
    print(f"{ticker} from {universe}{error_text} ({idx+1} / {len(securities)}). Elapsed: {elapsed.minutes}m {elapsed.seconds}s. Remaining: {remaining_string}.")

def get_remaining_seconds(load_times, idx, total_len):
    """Calculate estimated remaining time using a moving average of load times."""
    if not load_times:
        return np.nan
    window = min(idx + 1, 25)  # Use a rolling window of up to 25
    load_time_ma = pd.Series(load_times).rolling(window).mean().iloc[-1]
    remaining_seconds = (total_len - (idx + 1)) * load_time_ma
    return remaining_seconds

def load_prices_from_tda(securities):
    """Load price data from TD Ameritrade."""
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
    """
    Fetch historical price data for a single ticker using yfinance.
    Args:
        security (dict): Security metadata with 'ticker' key.
        start_date (date): Start date for data.
        end_date (date): End date for data.
    Returns:
        dict: Formatted ticker data or None if data is invalid.
    """
    ticker = security["ticker"]
    escaped_ticker = ticker.replace(".", "-")  # Handle tickers like BRK.B
    print(f"Fetching data for ticker: {escaped_ticker}")
    try:
        # Fetch data for a single ticker to avoid MultiIndex issues
        df = yf.download(escaped_ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
        if df.empty:
            print(f"No data found for ticker {escaped_ticker}")
            return None

        # Ensure required columns are present
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df[required_columns]  # Select only the columns we need

        # Convert timestamps to integers and extract OHLCV data
        timestamps = df.index.map(lambda x: int(x.timestamp()))
        ticker_data = {
            "candles": [
                {
                    "open": float(df['Open'].iloc[i]),
                    "close": float(df['Close'].iloc[i]),
                    "low": float(df['Low'].iloc[i]),
                    "high": float(df['High'].iloc[i]),
                    "volume": int(df['Volume'].iloc[i]),
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
    """Load price data from Yahoo Finance."""
    print("*** Loading Stocks from Yahoo Finance ***")
    today = date.today()
    start_date = today - dt.timedelta(days=365)  # Fetch 1 year of data
    tickers_dict = {}
    start = time.time()
    load_times = []

    for idx, security in enumerate(securities):
        r_start = time.time()
        ticker_data = get_yf_data(security, start_date, today)
        if ticker_data is None:  # Skip invalid or failed tickers
            continue
        current_load_time = time.time() - r_start
        load_times.append(current_load_time)
        remaining_seconds = get_remaining_seconds(load_times, idx, len(securities))
        tickers_dict[security["ticker"]] = ticker_data
        print_data_progress(security["ticker"], security["universe"], idx, securities, "", time.time() - start, remaining_seconds)

    create_price_history_file(tickers_dict)

def save_data(source, securities):
    """Save price data using the specified data source."""
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
