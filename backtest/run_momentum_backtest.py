import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import get_asx50_data, get_asx_market_data
from momentum import CrossSectionalMomentum
from backtest.backtester import Backtester


def main():
  price_data = get_asx50_data()
  market_data = get_asx_market_data()

  price_data = price_data.dropna(axis=1)

  strategy = CrossSectionalMomentum()
  backtester = Backtester(
    price_data, strategy, market_data,
    rebalance_freq=21, min_history=270, transaction_cost_bps=5
  )

  results = backtester.run()

  results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
  os.makedirs(results_dir, exist_ok=True)

  results.to_csv(os.path.join(results_dir, "momentum_returns.csv"))

  stats = backtester.plot_performance(
    results,
    title="Cross-Sectional Momentum - ASX50",
    save_path=os.path.join(results_dir, "momentum_backtest.png")
  )

  print(stats)


if __name__ == "__main__":
  main()
