"""Value-at-Risk calculation methods."""

import numpy as np
import pandas as pd
from scipy import stats

class VaRCalculator:
    """Calculate VaR using multiple methods."""
    
    @staticmethod
    def historical_var(returns, confidence=0.95):
        """Historical simulation VaR."""
        return np.percentile(returns, (1 - confidence) * 100)
    
    @staticmethod
    def parametric_var(returns, confidence=0.95):
        """Parametric VaR (normal distribution)."""
        mu = returns.mean()
        sigma = returns.std()
        z_score = stats.norm.ppf(1 - confidence)
        return mu + z_score * sigma
    
    @staticmethod
    def parametric_var_t(returns, confidence=0.95):
        """Parametric VaR with Student-t distribution."""
        params = stats.t.fit(returns)
        df, loc, scale = params
        t_score = stats.t.ppf(1 - confidence, df)
        return loc + t_score * scale
    
    @staticmethod
    def cornish_fisher_var(returns, confidence=0.95):
        """Cornish-Fisher VaR (adjusted for skewness/kurtosis)."""
        from scipy.stats import skew, kurtosis
        
        mu = returns.mean()
        sigma = returns.std()
        s = skew(returns)
        k = kurtosis(returns)
        
        z = stats.norm.ppf(1 - confidence)
        z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3*z) * k / 24 - (2*z**3 - 5*z) * s**2 / 36
        
        return mu + z_cf * sigma
    
    @staticmethod
    def expected_shortfall(returns, confidence=0.95):
        """Expected Shortfall (CVaR) - average loss beyond VaR."""
        var = VaRCalculator.historical_var(returns, confidence)
        return returns[returns <= var].mean()
