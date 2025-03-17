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
from scipy.stats import linregress
from datetime import date, datetime

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
for subdir in ['data', 'tmp', 'output']:
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

# Configuration parameters
PRICE_DATA_OUTPUT = os.path.join(DIR, "data", "price_history.json")
MOMENTUM_OUTPUT = os.path.join(DIR, "data", "momentum_ranking.json")
ACCOUNT_VALUE = cfg("CASH")
RISK_FACTOR_CFG = cfg("RISK_FACTOR")
RISK_FACTOR = RISK_FACTOR_CFG or 0.002
MAX_STOCKS = cfg("STOCKS_COUNT_OUTPUT")
SLOPE_DAYS = cfg("MOMENTUM_CALCULATION_PAST_DAYS")
POS_COUNT_TARGET = cfg("POSITIONS_COUNT_TARGET")
MAX_GAP = cfg("EXCLUDE_MAX_GAP_PCT")
EXCLUDE_MA_CROSSES = cfg("EXCLUDE_ALL_MA_CROSSES")

TITLE_RANK = "Rank"
TITLE_TICKER = "Ticker"
TITLE_SECTOR = "Sector"
TITLE_UNIVERSE = "Universe"
TITLE_MOMENTUM = "Momentum (%)"
TITLE_RISK = "ATR20d"
TITLE_PRICE = "Price"
TITLE_SHARES = "Shares"
TITLE_POS_SIZE = "Position ($)"
TITLE_SUM = "Sum ($)"

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
                    "open": float(df['Open'].iloc[i].iloc[0]),
                    "close": float(df['Close'].iloc[i].iloc[0]),
                    "low": float(df['Low'].iloc[i].iloc[0]),
                    "high": float(df['High'].iloc[i].iloc[0]),
                    "volume": int(df['Volume'].iloc[i].iloc[0]),
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

def momentum(closes):
    """Calculates slope of exp. regression normalized by rsquared"""
    returns = np.log(closes)
    indices = np.arange(len(returns))
    slope, _, r, _, _ = linregress(indices, returns)
    return (((np.exp(slope) ** 252) - 1) * 100) * (r**2)

def atr_20(candles):
    """Calculates last 20d ATR"""
    daily_atrs = []
    for idx, candle in enumerate(candles):
        high = candle["high"]
        low = candle["low"]
        prev_close = 0
        if idx > 0:
            prev_close = candles[idx - 1]["close"]
        daily_atr = max(high-low, np.abs(high - prev_close), np.abs(low - prev_close))
        daily_atrs.append(daily_atr)
    return pd.Series(daily_atrs).rolling(20).mean().tail(1).item()

def calc_stocks_amount(account_value, risk_factor, risk_input):
    return (np.floor(account_value * risk_factor / risk_input)).astype(int)

def calc_pos_size(amount, price):
    return np.round(amount * price, 2)

def calc_sums(account_value, pos_size):
    sums = []
    sum_val = 0
    stocks_count = 0
    for position in list(pos_size):
        sum_val += position
        sums.append(sum_val)
        if sum_val < account_value:
            stocks_count += 1
    return sums, stocks_count

def calculate_momentum_and_positions():
    """Calculate momentum, filter stocks, and determine position sizes."""
    print("*** Calculating Momentum Scores and Position Sizes ***")
    try:
        with open(PRICE_DATA_OUTPUT, 'r') as fp:
            tickers_dict = json.load(fp)
    except FileNotFoundError:
        print(f"Price history file {PRICE_DATA_OUTPUT} not found. Run data fetching first.")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {PRICE_DATA_OUTPUT}: {str(e)}")
        return

    momentums = {}
    ranks = []
    for slope_days in SLOPE_DAYS:
        if slope_days not in momentums:
            momentums[slope_days] = []

        for ticker, data in tickers_dict.items():
            try:
                candles = data.get("candles", [])
                if not candles or len(candles) < 250:
                    print(f"Skipping {ticker}: Not enough data ({len(candles)} candles).")
                    continue

                # Sort candles by datetime
                candles = sorted(candles, key=lambda x: x["datetime"])
                closes = [candle["close"] for candle in candles]

                # Calculate 100-day MA and check for crosses
                closes_series = pd.Series(closes)
                slope_series = closes_series.tail(slope_days)
                mas = closes_series.rolling(100).mean().tail(slope_days)
                ma_is_crossed = False
                if EXCLUDE_MA_CROSSES:
                    ma_crosses = slope_series < mas
                    ma_crosses = ma_crosses.where(ma_crosses == True).dropna()
                    ma_is_crossed = ma_crosses.size > 0

                # Calculate gaps in the last 90 days
                diffs = np.abs(slope_series.pct_change().diff()).dropna()
                gaps = diffs[diffs > (MAX_GAP / 100.0)]
                ma = mas.tail(1).item()
                latest_price = closes[-1]

                # Apply filters
                if ma > latest_price or ma_is_crossed:
                    print(f"{ticker} was below its 100d moving average.")
                    continue
                if len(gaps):
                    print(f"{ticker} has a gap > {MAX_GAP}%")
                    continue

                # Calculate momentum and ATR
                mmntm = momentum(pd.Series(closes[-slope_days:]))
                atr = atr_20(candles)
                ranks.append(len(ranks) + 1)
                momentums[slope_days].append((0, ticker, data["sector"], data["universe"], mmntm, atr, latest_price))

            except KeyError:
                print(f"Ticker {ticker} has corrupted data.")

    # Process each slope period
    dfs = []
    for slope_days in SLOPE_DAYS:
        slope_suffix = f'_{slope_days}' if slope_days != SLOPE_DAYS[0] else ''
        df = pd.DataFrame(momentums[slope_days], columns=[TITLE_RANK, TITLE_TICKER, TITLE_SECTOR, TITLE_UNIVERSE, TITLE_MOMENTUM, TITLE_RISK, TITLE_PRICE])
        df = df.sort_values([TITLE_MOMENTUM], ascending=False)
        df[TITLE_RANK] = range(1, len(df) + 1)
        df = df.head(MAX_STOCKS)

        # Position sizing
        risk_factor = RISK_FACTOR
        calc_runs = 2
        for run in range(1, calc_runs + 1):
            if run > 1 and not RISK_FACTOR_CFG and POS_COUNT_TARGET:
                if stocks_count < POS_COUNT_TARGET or (stocks_count - POS_COUNT_TARGET) > 1:
                    risk_factor = RISK_FACTOR * (stocks_count / POS_COUNT_TARGET)
            df[TITLE_SHARES] = calc_stocks_amount(ACCOUNT_VALUE, risk_factor, df[TITLE_RISK])
            df[TITLE_POS_SIZE] = calc_pos_size(df[TITLE_SHARES], df[TITLE_PRICE])
            sums, stocks_count = calc_sums(ACCOUNT_VALUE, df[TITLE_POS_SIZE])
            df[TITLE_SUM] = sums

        # Save to CSV
        output_csv = os.path.join(DIR, "output", f'mmtm_posis{slope_suffix}.csv')
        df.to_csv(output_csv, index=False)
        print(f"Saved position sizes to {output_csv}")

        # Generate watchlist for TradingView
        watchlist_file = os.path.join(DIR, "output", f'Momentum{slope_suffix}.txt')
        with open(watchlist_file, "w") as watchlist:
            first_10_pf = ""
            tv_ticker_count = 0
            for index, row in df.iterrows():
                plus_sign = "" if tv_ticker_count == 0 else "+"
                if row[TITLE_POS_SIZE] > 0 and row[TITLE_SUM] <= ACCOUNT_VALUE and tv_ticker_count < 10:
                    tv_ticker_count += 1
                    first_10_pf = f'{first_10_pf}{plus_sign}{int(row[TITLE_SHARES])}*{row[TITLE_TICKER]}'
            watchlist_stocks = ','.join(df.head(MAX_STOCKS)[TITLE_TICKER])
            watchlist.write(f'{first_10_pf},{watchlist_stocks}')
        print(f"Saved watchlist to {watchlist_file}")

        dfs.append(df)

    # Save momentum rankings to JSON
    momentum_scores = []
    for _, row in dfs[0].iterrows():
        momentum_scores.append({
            "ticker": row[TITLE_TICKER],
            "momentum": row[TITLE_MOMENTUM],
            "sector": row[TITLE_SECTOR],
            "universe": row[TITLE_UNIVERSE],
            "atr_20d": row[TITLE_RISK],
            "price": row[TITLE_PRICE],
            "shares": int(row[TITLE_SHARES]),
            "position_size": row[TITLE_POS_SIZE]
        })
    with open(MOMENTUM_OUTPUT, "w") as fp:
        json.dump(momentum_scores, fp, indent=2)
    print(f"Saved momentum rankings to {MOMENTUM_OUTPUT}")

    print("\nTop 5 momentum stocks with position sizes:")
    for i, row in dfs[0].head(5).iterrows():
        print(f"{i+1}. {row[TITLE_TICKER]} ({row[TITLE_SECTOR]}): Momentum {row[TITLE_MOMENTUM]:.2f}%, "
              f"ATR {row[TITLE_RISK]:.2f}, Shares {int(row[TITLE_SHARES])}, Position ${row[TITLE_POS_SIZE]:.2f}")

    return dfs

def main():
    if not SECURITIES:
        print("No securities found to process. Check configuration and URLs.")
        return
    if not DATA_SOURCE:
        print("Data source not specified in configuration.")
        return
    
    # Fetch and save price data
    save_data(DATA_SOURCE, SECURITIES)
    
    # Calculate momentum and position sizes
    dfs = calculate_momentum_and_positions()
    if dfs:
        print("\nMomentum positions table:")
        print(dfs[0])
    
    if cfg("EXIT_WAIT_FOR_ENTER"):
        input("Press Enter key to exit...")

if __name__ == "__main__":
    main()
