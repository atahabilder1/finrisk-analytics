"""Data preprocessing for risk modeling."""

import pandas as pd
import numpy as np

def prepare_returns(price_data):
    """Prepare returns for risk modeling."""
    returns = {}
    for ticker, df in price_data.items():
        # Use close prices
        close_prices = df['Close'] if 'Close' in df.columns else df['close']
        # Calculate log returns
        returns[ticker] = np.log(close_prices / close_prices.shift(1)).dropna()
    return returns

def align_returns(returns_dict):
    """Align returns across assets to common dates."""
    # Convert dict to DataFrame
    returns_df = pd.DataFrame(returns_dict)
    # Drop any rows with NaN
    returns_df = returns_df.dropna()
    return returns_df
