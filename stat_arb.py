import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller


class StatArb():
  """
    Standardises log returns using rolling volatility for PCA decomposition into key factor returns. Using factor returns,
    splits returns into idiosyncratic and market returns, testing for stationarity of idiosyncratic returns. Trades these idiosyncratic returns if they
    are mean reverting, using z-scores and entry and exit logic
  """
  
  def __init__(self, vol_span = 60, k_components = 30, fit_window = 252, mr_window = 20, adf_pvalue_threshold = 0.05, min_half_life = 1, max_half_life = 25, entry_zscore =  1.75, exit_zscore = 0.75, max_abs_z = 3):
    """
      Parameters for PCA-residual statistical arbitrage 
    """
    # EWMA span (days) used to standardise returns before factor work
    self.vol_span = vol_span
    
    # Factors stripped out before looking for reversion
    self.k_components = k_components
    
    # Trailing days used to estimate factor loading
    self.fit_window = fit_window
    
    # Days of cumulative residual fed to OU/AR(1) fit - horizon of reversion being traded
    self.mr_window = mr_window
    
    # Gate trades based on ADF test of residual
    self.adf_pvalue_threshold = adf_pvalue_threshold
    
    # Trade residuals whose fitted half-lie sits in this band
    self.min_half_life = min_half_life
    self.max_half_life = max_half_life
    
    # Scores required to trade
    self.entry_zscore = entry_zscore
    self.exit_zscore = exit_zscore
    self.max_abs_z = max_abs_z
    
    # Scalar object
    self.scalar = StandardScaler()
    
    
  def _standardise_returns(self, price_data):
    # Only use the first 252 days worth of data (253 after calculating percent change)
    price_data_df = price_data.iloc[-253:, :]
    
    # Calculate log returns
    log_returns_df = np.log(price_data_df / price_data_df.shift(1))
    log_returns_df = log_returns_df.dropna() # Drop the first row as will be NaN
    
    # Standardise returns
    standardised_returns = self.scalar.fit(log_returns_df)
    
    return standardised_returns
    
  
  def _extract_residuals(self):
    # Get standardised returns
    returns_df = self.standardise_returns()
     
  def _analyse_ou_process(self):
    None
    
  def _signal(self):
    None
    
    
    