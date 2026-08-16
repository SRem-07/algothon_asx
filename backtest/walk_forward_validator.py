import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from backtest.backtester import Backtester


class WalkForwardValidator:
  """
    Walk-forward hyperparameter validation for any strategy class matching
    the interface Backtester already expects: constructed as
    strategy_class(**params) and exposing _signal(price_data) -> pd.Series.
    Both StatArb and the upcoming momentum strategy fit this, so the same
    validator drives either.

    Splits the price history into n_folds sequential out-of-sample blocks
    of test_size trading days. For each fold, every candidate parameter
    combination is scored on an expanding in-sample window (everything
    before the fold), the best-scoring combination is selected, and only
    that combination's returns *during* the fold are kept. Because the
    selection for a fold never looks at data from that fold or later, the
    stitched result across all folds is a genuine out-of-sample read on
    "pick the best params so far, then trade the next block" - unlike a
    single in-sample-optimised backtest, which overstates performance by
    fitting hyperparameters to the very period being scored.

    Cost note: each candidate combination is backtested once over the
    full price history (not once per fold) - a fixed parameter combo
    trades identically regardless of how folds are later drawn for
    selection, since every rebalance already only uses data available up
    to that day. Total cost is O(number of combinations), not
    O(combinations x folds).
  """

  def __init__(self, price_data, strategy_class, param_grid, market_data = None,
               n_folds = 4, test_size = 252, backtester_kwargs = None):
    self.price_data = price_data.sort_index()

    # Strategy class (not instance) - a fresh instance is constructed per
    # candidate parameter combination via strategy_class(**combo)
    self.strategy_class = strategy_class

    # dict of {param_name: [candidate values]} - swept via full grid (cartesian product)
    self.param_grid = param_grid

    # Optional benchmark data, passed through to each Backtester
    self.market_data = market_data

    # Number of sequential out-of-sample blocks to validate on
    self.n_folds = n_folds

    # Trading days per out-of-sample block
    self.test_size = test_size

    # Extra kwargs passed through to every Backtester (rebalance_freq, min_history, etc.)
    self.backtester_kwargs = backtester_kwargs or {}


  def _param_combinations(self):
    keys = list(self.param_grid.keys())
    value_lists = [self.param_grid[k] for k in keys]
    for values in itertools.product(*value_lists):
      yield dict(zip(keys, values))


  def _fold_boundaries(self, min_history):
    """
      n_folds consecutive out-of-sample blocks of test_size trading days.
      The first block starts one test_size after min_history rather than
      immediately at it, so every fold has at least one full block of
      prior track record to select parameters from - a fold sitting
      exactly at the min_history boundary would have zero in-sample days
      and always be skipped. Returns a list of (test_start_idx,
      test_end_idx) positional index pairs (end exclusive).
    """
    n = len(self.price_data)
    boundaries = []
    for i in range(self.n_folds):
      test_start_idx = min_history + (i + 1) * self.test_size
      test_end_idx = min(test_start_idx + self.test_size, n)
      if test_start_idx >= n - 1:
        break
      boundaries.append((test_start_idx, test_end_idx))
    return boundaries


  def run(self, selection_metric = "sharpe"):
    """
      Runs the validation. Returns (oos_results, fold_choices):
        oos_results - DataFrame like Backtester.run()'s output, stitched
          from each fold's out-of-sample segment under its selected params
        fold_choices - list of dicts, one per fold, recording the date
          range, the chosen parameter combination, and its in-sample score
    """
    min_history = self.backtester_kwargs.get("min_history", 270)

    # Backtest every candidate combination once over the full history
    combo_results = {}
    for combo in self._param_combinations():
      strategy = self.strategy_class(**combo)
      backtester = Backtester(self.price_data, strategy, self.market_data, **self.backtester_kwargs)
      combo_results[tuple(sorted(combo.items()))] = backtester.run()

    folds = self._fold_boundaries(min_history)

    oos_segments = []
    fold_choices = []

    for test_start_idx, test_end_idx in folds:
      test_start_date = self.price_data.index[test_start_idx]
      test_end_date = self.price_data.index[test_end_idx - 1]

      best_combo = None
      best_score = -np.inf
      for combo_key, results in combo_results.items():
        # Strictly-before-fold in-sample slice - the fold itself must never
        # inform which params get chosen for that same fold
        in_sample = results.loc[:test_start_date].iloc[:-1]
        if len(in_sample) < 20:
          continue

        score = Backtester.performance_summary(in_sample)[selection_metric]
        if np.isnan(score):
          continue

        if score > best_score:
          best_score = score
          best_combo = combo_key

      if best_combo is None:
        continue

      fold_returns = combo_results[best_combo].loc[test_start_date:test_end_date, "return"]
      oos_segments.append(fold_returns)

      fold_choices.append({
        "test_start": test_start_date,
        "test_end": test_end_date,
        "params": dict(best_combo),
        "in_sample_score": best_score
      })

    oos_returns = pd.concat(oos_segments) if oos_segments else pd.Series(dtype=float)
    equity_curve = (1 + oos_returns).cumprod()
    oos_results = pd.DataFrame({
      "return": oos_returns,
      "equity_curve": equity_curve
    })

    return oos_results, fold_choices


  def plot_results(self, oos_results, fold_choices, title, save_path):
    """
      Plot the stitched out-of-sample equity curve and drawdown, shading
      each fold and annotating headline stats. Returns the stats dict.
    """
    stats = Backtester.performance_summary(oos_results)
    equity_curve = oos_results["equity_curve"]
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1

    fig, (ax_equity, ax_dd) = plt.subplots(
      2, 1, figsize=(10, 6), sharex=True,
      gridspec_kw={"height_ratios": [3, 1]}
    )

    fold_colors = plt.cm.tab10.colors
    for i, fold in enumerate(fold_choices):
      ax_equity.axvspan(fold["test_start"], fold["test_end"],
                         color=fold_colors[i % len(fold_colors)], alpha=0.08)

    ax_equity.plot(equity_curve.index, equity_curve.values, color="#1f77b4")
    ax_equity.set_title(title)
    ax_equity.set_ylabel("Growth of $1 (stitched OOS)")
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
