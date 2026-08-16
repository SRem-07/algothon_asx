import numpy as np
import pandas as pd


class CrossSectionalMomentum():
  """
    Ranks stocks each rebalance date by trailing total return, skipping the most recent
    month to avoid short-term reversal (the standard 12-1 formation window). Goes long the
    top quantile and, optionally, short the bottom quantile, inverse-vol weighting names
    within each leg. Positions are held constant between rebalances.
  """

  def __init__(self, lookback=252, skip=21, rebalance_freq=21,
               top_pct=0.3, bottom_pct=0.3, long_short=True, vol_span=60):
    # Trailing window (days) used to compute the momentum score
    self.lookback = lookback

    # Most recent days excluded from the score to avoid short-term reversal
    self.skip = skip

    # Days between rebalances
    self.rebalance_freq = rebalance_freq

    # Fraction of the universe held long / sold short each rebalance
    self.top_pct = top_pct
    self.bottom_pct = bottom_pct

    # If False, only the long leg is traded (useful when shorting isn't available)
    self.long_short = long_short

    # Trailing window (days) used for inverse-vol weighting within each leg
    self.vol_span = vol_span

  def _momentum_scores(self, price_data):
    """
      Score(t) = price(t - skip) / price(t - lookback) - 1, i.e. total return over the
      formation window that ends `skip` days before the rebalance date.
    """
    formation_prices = price_data.shift(self.skip)
    scores = formation_prices / formation_prices.shift(self.lookback - self.skip) - 1

    return scores

  def _leg_weights(self, scores_row, returns_for_vol):
    """
      Splits one cross-section of momentum scores into long/short legs and inverse-vol
      weights names within each leg (falls back to the leg's mean inverse-vol for names
      with zero/undefined trailing vol).
    """
    scores_row = scores_row.dropna()
    weights = pd.Series(0.0, index=scores_row.index)
    n = len(scores_row)

    if n == 0:
      return weights

    ranked = scores_row.sort_values(ascending=False)

    n_long = max(1, int(np.ceil(n * self.top_pct)))
    long_names = ranked.index[:n_long]
    weights[long_names] = self._inverse_vol_weights(returns_for_vol[long_names])

    if self.long_short:
      n_short = max(1, int(np.ceil(n * self.bottom_pct)))
      short_names = ranked.index[-n_short:]
      weights[short_names] = -self._inverse_vol_weights(returns_for_vol[short_names])

    return weights

  def _inverse_vol_weights(self, returns_window):
    inv_vol = 1 / returns_window.std()
    inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan)
    inv_vol = inv_vol.fillna(inv_vol.mean())

    return inv_vol / inv_vol.sum()

  def generate_weights(self, price_data):
    """
      Builds a daily weight matrix (index=dates, columns=tickers). Weights are only
      recomputed every `rebalance_freq` days and held constant in between.
    """
    scores = self._momentum_scores(price_data)
    daily_returns = price_data.pct_change()

    rebalance_dates = scores.index[self.lookback::self.rebalance_freq]
    rebalance_weights = pd.DataFrame(index=rebalance_dates, columns=price_data.columns, dtype=float)

    for date in rebalance_dates:
      vol_window = daily_returns.loc[:date].tail(self.vol_span)
      rebalance_weights.loc[date] = self._leg_weights(scores.loc[date], vol_window)

    weights = rebalance_weights.reindex(price_data.index).ffill().fillna(0.0)

    return weights

  def backtest(self, price_data):
    """
      Applies each day's weights to the *next* day's return (avoids lookahead) and
      returns the daily strategy return series alongside the weight matrix used.
    """
    weights = self.generate_weights(price_data)
    daily_returns = price_data.pct_change()

    strategy_returns = (weights.shift(1) * daily_returns).sum(axis=1)

    return strategy_returns, weights


def performance_summary(strategy_returns, periods_per_year=252):
  """
    Basic annualised return / vol / Sharpe and max drawdown for a daily return series.
  """
  strategy_returns = strategy_returns.dropna()
  cumulative = (1 + strategy_returns).cumprod()

  annual_return = cumulative.iloc[-1] ** (periods_per_year / len(strategy_returns)) - 1
  annual_vol = strategy_returns.std() * np.sqrt(periods_per_year)
  sharpe = annual_return / annual_vol if annual_vol != 0 else np.nan

  running_max = cumulative.cummax()
  drawdown = cumulative / running_max - 1

  return {
    "annual_return": annual_return,
    "annual_vol": annual_vol,
    "sharpe": sharpe,
    "max_drawdown": drawdown.min(),
  }


if __name__ == "__main__":
  from data_loader import get_asx50_data

  price_data = get_asx50_data()
  # Drop tickers with too much missing history (e.g. late IPOs) to keep the cross-section clean
  price_data = price_data.dropna(axis=1, thresh=int(len(price_data) * 0.9))

  strategy = CrossSectionalMomentum()
  strategy_returns, weights = strategy.backtest(price_data)

  print(performance_summary(strategy_returns))
