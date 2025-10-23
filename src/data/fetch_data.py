"""Data fetching module - reuse from Project 1."""

import yfinance as yf
import ccxt
import pandas as pd
from pathlib import Path

def fetch_stock_data(ticker, start_date, end_date):
    """Fetch OHLCV data for stocks/ETFs."""
    print(f"Fetching {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return data

def calculate_returns(prices):
    """Calculate log returns."""
    returns = np.log(prices / prices.shift(1))
    return returns.dropna()
