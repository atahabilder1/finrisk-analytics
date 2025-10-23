"""LSTM model for volatility forecasting."""

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

class LSTMVolatility:
    """LSTM model for volatility prediction."""
    
    def __init__(self, sequence_length=20, units=64, dropout=0.2):
        self.sequence_length = sequence_length
        self.units = units
        self.dropout = dropout
        self.model = None
    
    def build_model(self, input_shape):
        """Build LSTM model architecture."""
        model = keras.Sequential([
            layers.LSTM(self.units, return_sequences=True, input_shape=input_shape),
            layers.Dropout(self.dropout),
            layers.LSTM(self.units // 2),
            layers.Dropout(self.dropout),
            layers.Dense(1, activation='relu')
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.model = model
        return model
    
    def prepare_sequences(self, returns):
        """Prepare sequences for LSTM."""
        # Calculate realized volatility (squared returns as proxy)
        volatility = returns.rolling(window=5).std()
        volatility = volatility.dropna()
        
        X, y = [], []
        for i in range(len(volatility) - self.sequence_length):
            X.append(volatility.iloc[i:i+self.sequence_length].values)
            y.append(volatility.iloc[i+self.sequence_length])
        
        return np.array(X).reshape(-1, self.sequence_length, 1), np.array(y)
