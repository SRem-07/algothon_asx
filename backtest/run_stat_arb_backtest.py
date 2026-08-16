import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import get_asx200_data, get_asx_market_data
from stat_arb import StatArb
from backtest.backtester import Backtester


def main():
  price_data = get_asx200_data()
  market_data = get_asx_market_data()

  # Keep any name with at least 60% of the full sample - late listings still
  # get included from the point their history is long enough for the fit
  # window. The stat-arb signal itself drops names without full history in
  # each individual fit window, so partial-history names contribute only
  # from the point they become usable rather than being excluded outright.
  price_data = price_data.dropna(axis=1, thresh=int(len(price_data) * 0.6))

  strategy = StatArb()
  backtester = Backtester(
    price_data, strategy, market_data,
    rebalance_freq=10, min_history=270, transaction_cost_bps=5
  )

  results = backtester.run()

  results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
  os.makedirs(results_dir, exist_ok=True)

  results.to_csv(os.path.join(results_dir, "stat_arb_returns.csv"))

  stats = backtester.plot_performance(
    results,
    title="PCA Statistical Arbitrage - ASX50",
    save_path=os.path.join(results_dir, "stat_arb_backtest.png")
  )

  print(stats)


if __name__ == "__main__":
  main()
