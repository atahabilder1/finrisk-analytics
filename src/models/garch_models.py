"""GARCH family models for volatility forecasting."""

from arch import arch_model
import pandas as pd
import numpy as np

class GARCHVolatility:
    """GARCH(1,1) volatility model."""
    
    def __init__(self, p=1, q=1):
        self.p = p
        self.q = q
        self.model = None
        self.results = None
    
    def fit(self, returns):
        """Fit GARCH model to returns."""
        # Scale returns to percentage
        returns_pct = returns * 100
        
        # Fit GARCH(1,1)
        self.model = arch_model(returns_pct, vol='Garch', p=self.p, q=self.q)
        self.results = self.model.fit(disp='off')
        return self.results
    
    def forecast(self, horizon=1):
        """Forecast volatility."""
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        forecast = self.results.forecast(horizon=horizon)
        return forecast.variance.iloc[-1] / 100  # Convert back from percentage

class EGARCHVolatility:
    """EGARCH model for asymmetric volatility."""
    
    def __init__(self, p=1, o=1, q=1):
        self.p = p
        self.o = o
        self.q = q
        self.model = None
        self.results = None
    
    def fit(self, returns):
        """Fit EGARCH model."""
        returns_pct = returns * 100
        self.model = arch_model(returns_pct, vol='EGARCH', p=self.p, o=self.o, q=self.q)
        self.results = self.model.fit(disp='off')
        return self.results
    
    def forecast(self, horizon=1):
        """Forecast volatility."""
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        forecast = self.results.forecast(horizon=horizon)
        return forecast.variance.iloc[-1] / 100
