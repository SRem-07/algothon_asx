import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Backtester:
  """
    Walk-forward backtester for any strategy exposing a
    `_signal(price_data: pd.DataFrame) -> pd.Series` method, where the
    returned Series is indexed by ticker with values roughly in [-1, 1]
    (negative = short conviction, positive = long conviction). StatArb
    and the upcoming momentum strategy are expected to share this output
    format, so the same Backtester drives either without modification.

    At each rebalance date, only price history up to and including that
    date is passed to the strategy - no look-ahead. The resulting
    conviction is normalised into dollar-neutral portfolio weights and
    held until the next rebalance.
  """

  def __init__(self, price_data, strategy, market_data = None, rebalance_freq = 5,
               min_history = 270, transaction_cost_bps = 5, max_daily_move = 0.35,
               benchmark_ticker = "^AFLI", max_position_weight = 0.15):
    # Price data must be a wide DataFrame: index = dates, columns = tickers
    self.price_data = price_data.sort_index()

    # Strategy object exposing _signal(price_data) -> pd.Series[ticker -> conviction]
    self.strategy = strategy

    # Trading days between signal refits/rebalances
    self.rebalance_freq = rebalance_freq

    # Minimum days of price history required before the first rebalance
    self.min_history = min_history

    # One-way transaction cost in basis points, charged on turnover at each rebalance
    self.transaction_cost_bps = transaction_cost_bps

    # Absolute cap on a single stock's daily return before it hits the book.
    # Guards against data vendor artifacts (bad ticks, unadjusted corporate
    # actions, stale/thin pricing on illiquid microcaps) rather than genuine
    # index-eligible large/mid-cap moves, which essentially never exceed this.
    self.max_daily_move = max_daily_move

    # Daily simple returns of every stock, used to mark the book to market each day
    self.stock_returns = price_data.pct_change().clip(-max_daily_move, max_daily_move)

    # Cap on any single name's share of the book. Without this, a rebalance
    # where only one or two stocks pass the OU gate gets renormalised up to
    # 100% gross exposure in those names alone - a concentrated single-stock
    # bet, not a market-neutral book. This is a placeholder for the position
    # sizing the HMM stage is meant to own eventually; capped weight keeps
    # the backtest reflecting the OU signal's quality rather than breadth
    # artifacts of the current universe/thresholds. Excess conviction when
    # few names qualify is simply left unallocated (held in cash) rather
    # than forced to full gross exposure.
    self.max_position_weight = max_position_weight

    # Optional benchmark index price series (e.g. ASX50 spot) for the equity
    # curve plot only - has no effect on the backtest mechanics itself.
    if market_data is not None and benchmark_ticker in market_data.columns:
      self.benchmark_prices = market_data[benchmark_ticker].reindex(self.price_data.index).ffill()
    else:
      self.benchmark_prices = None


  def _conviction_to_weights(self, conviction):
    """
      Normalise per-stock conviction into dollar-neutral weights (sum of
      |weights| <= 1), then cap any single name at max_position_weight.
      Stocks with zero conviction (e.g. failed gates) get zero weight. If
      total conviction is zero (nothing traded that period), hold no
      positions. Capped excess is left unallocated rather than
      redistributed - see the max_position_weight comment in __init__.
    """
    gross = conviction.abs().sum()
    if gross == 0 or not np.isfinite(gross):
      return conviction * 0.0
    weights = conviction / gross
    return weights.clip(-self.max_position_weight, self.max_position_weight)


  def run(self):
    """
      Walk forward through price_data, rebalancing every rebalance_freq
      trading days. Returns a DataFrame indexed by date with columns:
        ['return', 'equity_curve']
    """
    dates = self.price_data.index
    n = len(dates)

    daily_returns = pd.Series(0.0, index=dates)
    current_weights = pd.Series(0.0, index=self.price_data.columns)

    for idx in range(self.min_history, n, self.rebalance_freq):
      # Signal is computed using only data up to and including 'idx' - no look-ahead
      history = self.price_data.iloc[:idx + 1]
      conviction = self.strategy._signal(history)
      new_weights = self._conviction_to_weights(conviction).reindex(
        self.price_data.columns, fill_value=0.0
      )

      # Transaction cost on turnover, charged against the first day of the new holding period
      turnover = (new_weights - current_weights).abs().sum()
      cost = turnover * (self.transaction_cost_bps / 10_000)

      current_weights = new_weights

      # Hold current_weights until the next rebalance (or the end of the data)
      hold_end = min(idx + self.rebalance_freq, n - 1)
      for day in range(idx + 1, hold_end + 1):
        day_return = (current_weights * self.stock_returns.iloc[day]).sum()
        if day == idx + 1:
          day_return -= cost
        daily_returns.iloc[day] = day_return

    equity_curve = (1 + daily_returns).cumprod()

    results = pd.DataFrame({
      "return": daily_returns,
      "equity_curve": equity_curve
    })

    # Trim off the warm-up period before the first rebalance
    return results.iloc[self.min_history:]


  @staticmethod
  def performance_summary(results):
    """
      Headline stats from a Backtester.run() result: annualised return,
      annualised vol, Sharpe ratio (rf = 0), and max drawdown.
    """
    daily_returns = results["return"]
    equity_curve = results["equity_curve"]

    annual_return = daily_returns.mean() * 252
    annual_vol = daily_returns.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else np.nan

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    max_drawdown = drawdown.min()

    return {
      "annual_return": annual_return,
      "annual_vol": annual_vol,
      "sharpe": sharpe,
      "max_drawdown": max_drawdown
    }


  def plot_performance(self, results, title, save_path):
    """
      Plot the equity curve and drawdown, annotate headline stats, and
      save to save_path (PNG). Returns the stats dict that was plotted.
    """
    stats = self.performance_summary(results)
    equity_curve = results["equity_curve"]
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1

    fig, (ax_equity, ax_dd) = plt.subplots(
      2, 1, figsize=(10, 6), sharex=True,
      gridspec_kw={"height_ratios": [3, 1]}
    )

    if self.benchmark_prices is not None:
      benchmark_window = self.benchmark_prices.reindex(equity_curve.index).ffill().bfill()
      benchmark_curve = benchmark_window / benchmark_window.iloc[0]
      ax_equity.plot(benchmark_curve.index, benchmark_curve.values,
                      color="#AB2E0F", linestyle="--", label="Benchmark")
      ax_equity.plot(equity_curve.index, equity_curve.values, color="#1f77b4", label="Strategy")
      ax_equity.legend(loc="upper right", fontsize=8)
    else:
      ax_equity.plot(equity_curve.index, equity_curve.values, color="#1f77b4")

    ax_equity.set_title(title)
    ax_equity.set_ylabel("Growth of $1")
    ax_equity.grid(alpha=0.3)

    stats_text = (
      f"Ann. return: {stats['annual_return']:.1%}\n"
      f"Ann. vol: {stats['annual_vol']:.1%}\n"
      f"Sharpe: {stats['sharpe']:.2f}\n"
      f"Max drawdown: {stats['max_drawdown']:.1%}"
    )
    ax_equity.text(
      0.02, 0.02, stats_text, transform=ax_equity.transAxes,
      va="bottom", ha="left", fontsize=9,
      bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    ax_dd.fill_between(drawdown.index, drawdown.values, 0, color="#d62728", alpha=0.4)
    ax_dd.set_ylabel("Drawdown")
    ax_dd.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    return stats
