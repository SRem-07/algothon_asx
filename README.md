# algothon_asx
Application of 2026 SIG Algothon strategy and code to the ASX.

## Strategy 1: PCA Statistical Arbitrage

A cross-sectional mean-reversion strategy on the ASX50. Common (market/sector) risk is
stripped out of each stock's returns via PCA, and the leftover idiosyncratic residual is
tested for mean-reversion and traded via an Ornstein-Uhlenbeck (OU) framework, in the
spirit of Avellaneda & Lee (2010). Implementation: [stat_arb.py](stat_arb.py).

### 1. Standardising returns

Daily log returns are computed and standardised (zero mean, unit variance) over a trailing
`fit_window` (default 252 days), so PCA isn't dominated by whichever stocks happen to have
the highest raw volatility:

$$r_{i,t} = \ln\left(\frac{P_{i,t}}{P_{i,t-1}}\right), \qquad z_{i,t} = \frac{r_{i,t} - \bar{r}_i}{\sigma_i}$$

### 2. Factor decomposition and residuals

PCA is fit on the standardised return matrix $Z$ (days $\times$ stocks). Rather than fixing
the number of components $k$ up front, $k$ is chosen as the smallest number of components
whose cumulative explained variance clears a target threshold $\tau$ (default 55%):

$$k = \min\left\{ k : \sum_{j=1}^{k} \lambda_j \Big/ \sum_{j=1}^{N} \lambda_j \geq \tau \right\}$$

where $\lambda_j$ are the PCA eigenvalues. A fixed $k$ risks either under-fitting genuine
common structure or over-fitting sample noise as the number of names or the fit window
changes; a variance-explained target adapts to how much real common structure is actually
present in the current window. The top $k$ components are treated as systematic risk;
what's left is the idiosyncratic residual for each stock:

$$\hat{Z} = F W, \qquad \epsilon_{i,t} = Z_{i,t} - \hat{Z}_{i,t}$$

### 3. Fitting the OU process

The daily residual $\epsilon_{i,t}$ is itself stationary by construction (it's already a
return), so testing it directly for mean-reversion is uninformative. Instead, its
cumulative sum over the trailing `mr_window` (default 60 days) is treated as a synthetic
"price" for the idiosyncratic component — this is the object that can actually wander from
equilibrium and revert:

$$X_{i,t} = \sum_{s=1}^{t} \epsilon_{i,s}$$

$X_t$ is tested for a unit root (ADF) and fit with an AR(1) regression, which maps onto
standard OU parameters in closed form:

$$X_t = a + b X_{t-1} + \zeta_t$$

$$\kappa = -\ln(b), \qquad \text{half-life} = \frac{\ln 2}{\kappa}, \qquad m = \frac{a}{1-b}, \qquad \sigma_{eq} = \frac{\sigma_\zeta}{\sqrt{1 - b^2}}$$

$$s = \frac{X_t - m}{\sigma_{eq}}$$

A stock only continues to the signal stage if it passes both gates: the ADF p-value is
below `adf_pvalue_threshold`, and the fitted half-life falls inside
`[min_half_life, max_half_life]` — fast enough to plausibly revert within the trading
horizon, slow enough that it's not just noise.

### 4. From s-score to conviction

The end goal is a continuous per-stock conviction in $[-1, 1]$, not a flat $\pm 1$ trigger —
a binary signal throws away information the OU fit already produced. Two stocks that both
clear the entry threshold aren't equally good trades: one might have $p = 0.001$ and a
half-life sitting dead-center of the tradeable band, the other might just scrape under the
p-value threshold with a half-life on the edge of the band. Conviction is built as a
product of three independent confidence terms, each scaled to $[0, 1]$, so a stock only
gets a strong signal when it is significant, well-behaved, *and* meaningfully displaced
from equilibrium all at once — weak on any one dimension damps the conviction rather than
either fully including or fully excluding the trade:

$$\text{conviction}_i = -\text{sign}(s_i) \cdot \phi(s_i) \cdot \psi(p_i) \cdot \chi(h_i)$$

**Distance from equilibrium**, ramped between the exit and entry z-scores rather than
switched on abruptly at entry:

$$\phi(s) = \text{clip}\left(\frac{\min(|s|,\ z_{max}) - z_{exit}}{z_{entry} - z_{exit}},\ 0,\ 1\right)$$

**Statistical confidence**, so a stock that barely clears the ADF threshold contributes
less than one that's strongly significant:

$$\psi(p) = \text{clip}\left(1 - \frac{p}{p_{thresh}},\ 0,\ 1\right)$$

**Half-life fitness**, a triangular weight peaking at the midpoint of
`[min_half_life, max_half_life]` and decaying toward the edges of the band, since a
half-life just inside the boundary is a weaker signal than one near the middle:

$$\chi(h) = \begin{cases} \dfrac{h - h_{min}}{h_{mid} - h_{min}} & h \leq h_{mid} \\[6pt] \dfrac{h_{max} - h}{h_{max} - h_{mid}} & h > h_{mid} \end{cases}$$

Because the three terms are multiplied rather than summed or averaged, any one of them
collapsing to zero (e.g. failing significance outright) zeroes the whole conviction — the
conjunctive gating of the original pass/fail design is preserved, just made continuous
within the region where a trade is actually justified. The resulting per-stock conviction
is designed to combine with a momentum-based conviction score and feed a regime-detecting
HMM that ultimately sizes positions — see Roadmap below.

### Backtest

Walk-forward backtest on the ASX50, 2013–2026, rebalanced every 5 trading days, 5bps
transaction costs, capped at 15% gross exposure per name (this cap stands in for the
position sizing the HMM stage will eventually own — see Roadmap). This is the **raw OU
signal in isolation**: no momentum blend, no beta hedge, no HMM sizing yet.

![PCA Statistical Arbitrage backtest](backtest/results/stat_arb_backtest.png)

| Metric | Value |
|---|---|
| Annualised return | -1.5% |
| Annualised volatility | 6.0% |
| Sharpe | -0.26 |
| Max drawdown | -30.7% |

Volatility is an order of magnitude below the ASX50 benchmark and the equity curve stays
largely uncorrelated with it, which is the market-neutrality the PCA residual construction
is meant to deliver. The raw signal isn't profitable on its own yet over this sample, which
is a reasonable starting point rather than a final read — the signal is also fairly sparse
(often only a handful of the 43-stock universe pass the ADF/half-life gate at any given
rebalance), which is exactly what the walk-forward hyperparameter validator in
[backtest/walk_forward_validator.py](backtest/walk_forward_validator.py) is for.

### Backtesting tools

- [backtest/backtester.py](backtest/backtester.py) — walk-forward backtester, driven by any
  strategy exposing `_signal(price_data) -> pd.Series` (ticker → conviction), so it works
  unmodified for the momentum strategy once it shares this interface.
- [backtest/walk_forward_validator.py](backtest/walk_forward_validator.py) — out-of-sample
  hyperparameter validation: sweeps a parameter grid, and for each sequential
  out-of-sample block, selects the best-scoring combination using only data available
  before that block.
- [backtest/run_stat_arb_backtest.py](backtest/run_stat_arb_backtest.py) — runnable script
  that produces the plot above.

### Roadmap

- Momentum strategy, producing a conviction score in the same format
- HMM over book-level regime features, used both to size overall exposure and to weight
  the mean-reversion vs. momentum blend
- Beta hedge on the combined book
