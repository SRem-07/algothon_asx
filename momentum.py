import numpy as np
import pandas as pd


class CrossSectionalMomentum():
  """
    Cross-sectional momentum on the ASX50, exposed through the same
    _signal(price_data) -> pd.Series[ticker -> conviction in [-1, 1]]
    interface as StatArb so the shared Backtester drives either strategy
    without modification.

    At each call, ranks the universe by trailing (lookback - skip)-day
    total return, skipping the most recent `skip` days to sidestep the
    well-documented short-term reversal effect (the classic 12-1 formation
    window). Only names in the top `top_pct` (long) and bottom `bottom_pct`
    (short) quantiles receive non-zero conviction; within each leg,
    conviction ramps linearly from 0 at the quantile boundary to +/- 1 at
    the tail. Each name's conviction is then scaled by the R^2 of a linear
    fit of its log price over the formation window - a stock whose price
    trended smoothly gets more weight than one whose formation return was
    driven by a single jump.
  """

  def __init__(self, lookback=252, skip=21, top_pct=0.3, bottom_pct=0.3, long_only=False):
    self.lookback = lookback
    self.skip = skip
    self.top_pct = top_pct
    self.bottom_pct = bottom_pct
    self.long_only = long_only

  def _momentum_score(self, price_data):
    formation_end = price_data.iloc[-1 - self.skip]
    formation_start = price_data.iloc[-self.lookback]
    return formation_end / formation_start - 1

  def _trend_quality(self, price_data):
    window = np.log(price_data.iloc[-self.lookback : len(price_data) - self.skip])
    n = len(window)
    if n < 3:
      return pd.Series(0.0, index=price_data.columns)

    x = np.arange(n, dtype=float)
    x_dev = x - x.mean()
    y = window.values
    y_dev = y - np.nanmean(y, axis=0)

    denom_x = np.nansum(x_dev ** 2)
    slope = np.nansum(x_dev[:, None] * y_dev, axis=0) / denom_x
    pred_dev = np.outer(x_dev, slope)

    ss_res = np.nansum((y_dev - pred_dev) ** 2, axis=0)
    ss_tot = np.nansum(y_dev ** 2, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
      r2 = 1 - ss_res / ss_tot
    r2 = np.where(np.isfinite(r2), r2, 0.0)

    return pd.Series(np.clip(r2, 0.0, 1.0), index=window.columns)

  def _signal(self, price_data):
    if len(price_data) < self.lookback + 1:
      return pd.Series(0.0, index=price_data.columns)

    scores = self._momentum_score(price_data).dropna()
    conviction = pd.Series(0.0, index=price_data.columns)

    n = len(scores)
    if n == 0:
      return conviction

    ranked = scores.sort_values(ascending=False)

    n_top = max(1, int(np.ceil(n * self.top_pct)))
    top_names = ranked.index[:n_top]
    conviction.loc[top_names] = np.linspace(1.0, 1.0 / n_top, n_top)

    if not self.long_only:
      n_bot = max(1, int(np.ceil(n * self.bottom_pct)))
      bot_names = ranked.index[-n_bot:]
      conviction.loc[bot_names] = np.linspace(-1.0 / n_bot, -1.0, n_bot)

    trend_r2 = self._trend_quality(price_data).reindex(conviction.index).fillna(0.0)
    conviction *= trend_r2

    return conviction.clip(-1.0, 1.0)
