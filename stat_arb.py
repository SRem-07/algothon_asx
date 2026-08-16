import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller


class StatArb():
  """
    Standardises log returns using rolling volatility for PCA decomposition into key factor returns. Using factor returns,
    splits returns into idiosyncratic and market returns, testing for stationarity of idiosyncratic returns. Trades these idiosyncratic returns if they
    are mean reverting, using z-scores and entry and exit logic
  """

  def __init__(self, vol_span = 60, k_components = None, variance_threshold = 0.70, fit_window = 252,
               mr_window = 90, adf_pvalue_threshold = 0.01, min_half_life = 1, max_half_life = 25,
               entry_zscore =  1.75, exit_zscore = 0.75, max_abs_z = 3):
    """
      Parameters for PCA-residual statistical arbitrage
    """
    # EWMA span (days) used to standardise returns before PCA. Weights recent
    # observations more heavily than distant ones so the standardised returns
    # sit at roughly unit variance around the *current* vol regime rather
    # than the average of the whole fit window.
    self.vol_span = vol_span

    # Factors stripped out before looking for reversion. If None, chosen
    # dynamically each fit via cumulative explained variance
    self.k_components = k_components

    # Cumulative explained-variance target used when k_components is None
    self.variance_threshold = variance_threshold

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


  def _standardise_returns(self, price_data):
    """
      Standardise trailing fit_window days of log returns to feed into the
      PCA model.

      Scaling is done via an exponentially-weighted moving average (EWMA)
      mean and volatility with span self.vol_span:

        mu_t     = (1 - alpha) * mu_{t-1}    + alpha * r_t
        var_t    = (1 - alpha) * var_{t-1}   + alpha * (r_t - mu_t)^2
        z_t      = (r_t - mu_t) / sqrt(var_t)

      where alpha = 2 / (vol_span + 1). Recent observations get more
      weight than distant ones, so the standardised series is
      approximately unit-variance around the *current* vol regime rather
      than the average of the whole fit window. This matters most when a
      regime shift sits inside the window - e.g. Feb-Mar 2020, where a
      static full-window scaler would leave post-shock returns
      systematically over-scaled and pre-shock returns under-scaled,
      distorting the covariance matrix that PCA is about to decompose.

      Stocks without full price history over the fit window are dropped
      here rather than propagating NaNs into PCA - the resulting universe
      per fit is dynamic, which is what walk-forward needs when the
      universe includes names that IPO'd mid-sample.
    """
    # Need fit_window + 1 price rows to get fit_window log returns.
    price_data_df = price_data.iloc[-(self.fit_window + 1):, :]

    # Drop names missing any price in the window - PCA can't take NaNs
    price_data_df = price_data_df.dropna(axis=1)

    # Calculate log returns
    log_returns_df = np.log(price_data_df / price_data_df.shift(1)).dropna()

    # EWMA mean and volatility - see docstring for the recursion
    ewma_mean = log_returns_df.ewm(span=self.vol_span, adjust=False).mean()
    ewma_std = log_returns_df.ewm(span=self.vol_span, adjust=False).std()

    # Drop illiquid names whose recursive EWMA std collapses to zero
    # anywhere in the window - a run of forward-filled prices gives many
    # exactly-zero daily returns, and the resulting 0/0 in the division
    # below would inject NaNs into every row (not just that column) 
    active_cols = ewma_std.min() > 1e-8
    log_returns_df = log_returns_df.loc[:, active_cols]
    ewma_mean = ewma_mean.loc[:, active_cols]
    ewma_std = ewma_std.loc[:, active_cols]

    # First row is NaN because a one-sample EWMA std is undefined; drop it
    standardised_returns = ((log_returns_df - ewma_mean) / ewma_std).dropna()

    return standardised_returns


  def _select_k_components(self, std_returns_df):
    """
      Choose how many PCA components to treat as 'systematic' before residual/
      mean-reversion analysis, based on cumulative explained variance, instead
      of a hardcoded count.

      With ~50 stocks and a fit_window of ~252 days, a fixed
      k_components risks PCA fitting sample noise rather than genuine common
      factors. Picking k via a variance-explained target adapts to how much
      real common structure is actually present in the current window.

      If self.k_components was set explicitly in __init__, skip all of this
      and just return it (manual override).
    """
    if self.k_components is not None:
      return self.k_components
    
    # Run a full PCA, n_components = None to keep min of n_samples or n_features
    full_pca = PCA(n_components=None).fit(std_returns_df)
    
    # Calculate the cumulative variance, from the eigenvalues that explain the amount of variance explained
    # by each principal component
    cumulative_variance = np.cumsum(full_pca.explained_variance_ratio_)
    
    # Search for the smallest index/amaount of components that satisfied the variance threshold
    k = np.searchsorted(cumulative_variance, self.variance_threshold, side="left") + 1
    
    # Return this to use for PCA decomposition in extracting residuals
    return k


  def _extract_residuals(self, price_data):
    # Get standardised returns
    std_returns_df = self._standardise_returns(price_data)

    # Choose k (fixed override or cumulative-explained-variance cutoff)
    k = self._select_k_components(std_returns_df)

    # Generate PCA object and train PCA/get factors
    pca = PCA(n_components=k)
    factors = pca.fit_transform(std_returns_df)

    # Construct systematic returns df
    systematic_returns = pd.DataFrame(
      pca.inverse_transform(factors),
      index = std_returns_df.index,
      columns = std_returns_df.columns
    )

    # Calculate residuals
    daily_residuals = std_returns_df - systematic_returns

    return daily_residuals


  @staticmethod
  def _fit_ar1(x):
    """
      Fit X_t = a + b * X_{t-1} + zeta_t by OLS on a single 1-D array.

      This is the core 'special function' behind the OU fit: kappa,
      half-life, equilibrium mean, and equilibrium std are all closed-form
      transforms of (a, b, resid_std) - see _fit_ou_parameters below.

      Returns:
        a, b, resid_std
      where resid_std is the std of the regression residuals zeta_t.
      Use ddof=2 when computing resid_std (two parameters estimated: a, b).
    """
    
    # Lagged and current values to use for model
    x_lag = x[:-1] # All values except last (used as predictors, no look-forward)
    x_now = x[1:] # All values except first (response)
    
    # Fit AR(1) via OLS
    b, a = np.polyfit(x_lag, x_now, 1) # returns (slope, intercept)
    
    # Calculate the residual term (zeta)
    residual = x_now - (a + b * x_lag) # x_now - x_pred
    
    # Compute the residual standard deviation for OU process
    residual_std = residual.std(ddof=2) # ddof = 2 as two parameters estimated in regression
    
    # Return values
    return a, b, residual_std
    


  def _fit_ou_parameters(self, cumulative_residual):
    """
      Turn an AR(1) fit of a cumulative residual series into OU parameters
      and the current s-score.

      cumulative_residual: 1-D array/Series, the cumsum of a stock's daily
      idiosyncratic residual over the trailing mr_window (a synthetic
      'price' for the idiosyncratic component - this is what actually
      mean-reverts, not the daily residual itself).

      Formulas (standard OU <-> AR(1) mapping):
        kappa     = -ln(b)                         daily mean-reversion speed
        half_life = ln(2) / kappa
        m         = a / (1 - b)                    equilibrium level
        sigma_eq  = resid_std / sqrt(1 - b**2)       stationary std of X_t
        s_score   = (X_t_last - m) / sigma_eq       standadising residual by equilibirum variance

      Returns a dict {kappa, half_life, m, sigma_eq, s_score}, or None if
      b is outside (0, 1) - no valid mean-reverting solution, reject the
      stock rather than let kappa/half_life blow up or go negative.
    """
    x = np.asarray(cumulative_residual)
    a, b, residual_std = self._fit_ar1(x)
    
    # Guard against non-sensical slope terms
    if not (0 < b < 1):
      return None
    
    # Calculate values of importance
    kappa = -np.log(b) # daily mean-reversion speed
    half_life = np.log(2) / kappa # half-life of mean reversion
    m = a / (1 - b) # equilibrium level
    sigma_eq = residual_std / np.sqrt(1 - b**2) # stationary std of residual
    s_score = (x[-1] - m) / sigma_eq # Standardising residual
    
    # Create dictionary of key values and return
    values = {
      "kappa": kappa, 
      "half_life": half_life,
      "m": m,
      "sigma_eq": sigma_eq,
      "s_score": s_score
    }
    
    return values
   

  def _analyse_ou_process(self, price_data):
    """
      For each stock: build the cumulative idiosyncratic residual over the
      trailing mr_window, fit its OU process, and test it for stationarity.

      IMPORTANT: ADF and the AR(1)/OU fit run on the *cumulative* residual
      (a synthetic price series), not the raw daily residual. Daily returns
      are stationary by construction, so testing them directly for a unit
      root doesn't tell you anything about mean-reversion.

      Returns a DataFrame indexed by stock ticker with columns:
        ['p_value', 'half_life', 's_score', 'passes_gate']
    """
    # Get the residuals/idiosyncratic return, and only use data over mean reversion window
    daily_residuals = self._extract_residuals(price_data)
    window = daily_residuals.iloc[-self.mr_window:]

    results = {}
    # For each stock
    for stock in window.columns:
      cumulative = window[stock].cumsum()
      
      # Get p_value for from ADF to comfirm stationarity of cumulative residual
      p_value = adfuller(cumulative, regression = "c")[1]
      
      # Get OU parameters in dictionary
      ou_params = self._fit_ou_parameters(cumulative)
      
      # If OU fit fails (e.g. due to b not in required range)
      if ou_params is None:
        results[stock] = {
          "p_value": p_value,
          "half_life": np.nan,
          "s_score": np.nan,
          "passes_gate": False
        }
        continue
      
      # Extract the half-life and s-score
      half_life = ou_params["half_life"]
      s_score = ou_params["s_score"]
        
      # If the stock passes
      passes_gate = (
        p_value <= self.adf_pvalue_threshold and 
        self.min_half_life <= ou_params["half_life"] <= self.max_half_life
      )
      
      results[stock] = {
        "p_value": p_value,
        "half_life": half_life,
        "s_score": s_score,
        "passes_gate": passes_gate
      }
      
    # Return DataFrame
    return pd.DataFrame.from_dict(results, orient="index")
      


  def _signal(self, price_data):
    """
      Convert each stock's OU analysis into a continuous conviction in
      [-1, 1] (negative = short, positive = long) rather than a flat
      threshold trigger.

        conviction = -sign(s_score) * zscore_scale * pvalue_confidence * half_life_fitness

      zscore_scale: 0 at |s_score| == exit_zscore, ramps linearly to 1 at
        |s_score| == entry_zscore. Clip |s_score| to max_abs_z first so one
        outlier doesn't blow the scale up unchecked, then clip the final
        scale to [0, 1].

      pvalue_confidence: 1 - (p_value / adf_pvalue_threshold), clipped to
        [0, 1] - a stock barely under the ADF threshold gets less weight
        than one that's strongly significant.

      half_life_fitness: weight in [0, 1] peaking at the middle of
        [min_half_life, max_half_life] and decaying toward the edges (e.g.
        a triangular or cosine bump) - a half-life near the band edge is a
        weaker signal than one near the middle.

      Stocks with passes_gate == False get conviction = 0.

      Returns a Series of conviction values indexed by stock ticker.
    """
    # Get statistics from OU process
    ou_results = self._analyse_ou_process(price_data)
    
    # Compute three weight terms
    signals = pd.Series(
      0.0, 
      index = ou_results.index,
      dtype = float
    )
    for stock in ou_results.index:
      # Collect statistics for a particular stock
      row = ou_results.loc[stock]
      
      # If stock didn't pass gate, then signal = 0
      if not row ["passes_gate"]:
        signals.loc[stock] = 0.0
        continue
      
      # Extract values from OU process
      s_score = row["s_score"]
      p_value = row["p_value"]
      half_life = row["half_life"]
      
      # Check that s-scores are sensible
      if not np.isfinite(s_score):
        signals.loc[stock] = 0.0
        continue
      
      # 1. Calculate the z-score
      abs_s_score = min(abs(s_score), self.max_abs_z)
      
      if abs_s_score <= self.exit_zscore: # No signal if already below z_score used to exit trades
        z_score_scale = 0.0
      
      elif abs_s_score >= self.entry_zscore: # Full signal if above entry_zscore
        z_score_scale = 1.0
        
      else: # Scale linearly between these values for between 1 and 0
        z_score_scale = (abs_s_score - self.exit_zscore) / (self.entry_zscore - self.exit_zscore) 
        
      # Clip to ensure is between 0 and 1
      z_score_scale = np.clip(z_score_scale, 0.0, 1.0)
      
      # 2. Calculate ADF p-value confidence
      # A stock that is barely under the p-value threshold gets a smaller weight
      p_value_confidence = 1 - (p_value / self.adf_pvalue_threshold)
      
      # Clip to ensure is between 0 and 1
      p_value_confidence = np.clip(p_value_confidence, 0.0, 1.0)
      
      # 4. Calculate half-life fitness
      # Half-life near the edges is a weaker signla than one towards the middle
      if self.max_half_life == self.min_half_life:
        half_life_fitness = 1.0
        
      else:
        midpoint = (self.min_half_life + self.max_half_life) /2
        
        # Calculations to ensure that half-lifes around the midpoint are weighted higher
        if half_life <= midpoint:
          half_life_fitness = (half_life - self.min_half_life) / (midpoint - self.min_half_life)
          
        else:
          half_life_fitness = (self.max_half_life - half_life) / (self.max_half_life - midpoint)
          
        # Clip to ensure is between 0 and 1
        half_life_fitness = np.clip(half_life_fitness, 0.0, 1.0)
        
      # Calculate the final conviction score
      conviction = (-np.sign(s_score) * z_score_scale * p_value_confidence * half_life_fitness)

      signals.loc[stock] = np.clip(conviction, -1.0, 1.0)

    return signals

