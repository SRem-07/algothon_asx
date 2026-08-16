import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import get_asx50_data, get_asx_market_data
from stat_arb import StatArb
from backtest.walk_forward_validator import WalkForwardValidator


def main():
  price_data = get_asx50_data().dropna(axis=1)
  market_data = get_asx_market_data()

  # Sweep the two thresholds most directly tied to signal sparsity: how far
  # a residual must be from equilibrium before entering, and how strict the
  # ADF significance gate is.
  param_grid = {
    "entry_zscore": [1.25, 1.75, 2.25],
    "adf_pvalue_threshold": [0.05, 0.10, 0.15]
  }

  validator = WalkForwardValidator(
    price_data, StatArb, param_grid, market_data,
    n_folds=13, test_size=252,
    backtester_kwargs={"rebalance_freq": 5, "min_history": 270, "transaction_cost_bps": 5}
  )

  oos_results, fold_choices = validator.run()

  results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
  os.makedirs(results_dir, exist_ok=True)

  oos_results.to_csv(os.path.join(results_dir, "stat_arb_walk_forward_returns.csv"))

  stats = validator.plot_results(
    oos_results, fold_choices,
    title="PCA Statistical Arbitrage - Walk-Forward Out-of-Sample",
    save_path=os.path.join(results_dir, "stat_arb_walk_forward.png")
  )

  print("STATS:", stats)
  print()
  print("FOLD CHOICES:")
  for fold in fold_choices:
    print(fold)


if __name__ == "__main__":
  main()
