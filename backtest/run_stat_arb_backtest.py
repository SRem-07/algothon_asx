import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import get_asx50_data, get_asx_market_data
from stat_arb import StatArb
from backtest.backtester import Backtester


def main():
  price_data = get_asx50_data()
  market_data = get_asx_market_data()

  # Drop tickers without a full price history over the sample (e.g. names
  # listed after the start date). A more thorough approach would only
  # require full history within each rolling fit window rather than over
  # the whole sample, letting late listings join partway through - worth
  # revisiting once the universe handling matters more.
  price_data = price_data.dropna(axis=1)

  strategy = StatArb()
  backtester = Backtester(
    price_data, strategy, market_data,
    rebalance_freq=5, min_history=270, transaction_cost_bps=5
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
