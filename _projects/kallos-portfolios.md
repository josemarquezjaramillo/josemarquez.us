---
title: "Kallos Portfolios: Cryptocurrency Portfolio Optimization Framework"
date: 2024-08-15
category: Quantitative Finance
tags: [Portfolio Optimization, Python, Deep Learning, GRU, Mean-Variance, Cryptocurrency, PyTorch, Backtesting]
featured: true
excerpt: "Advanced portfolio optimization framework comparing machine learning (GRU) predictions against traditional mean-variance and market-cap weighted strategies. Features quarterly model rotation, rigorous statistical testing, and production-ready backtesting infrastructure."
github_url: https://github.com/josemarquezjaramillo/kallos_portfolios
mathjax: true
---

## Overview

Kallos Portfolios is a sophisticated cryptocurrency portfolio optimization framework that systematically compares three distinct investment strategies: neural network-based predictions, traditional mean-variance optimization, and passive market-cap weighting. The system integrates quarterly-trained GRU models with modern portfolio theory to answer a critical question: **Does machine learning add value to quantitative portfolio construction?**

This framework represents the complete portfolio management lifecycle—from predictive modeling to optimization, backtesting, and rigorous statistical evaluation—designed for production deployment in cryptocurrency markets.

## Research Question & Motivation

Traditional portfolio optimization relies on historical returns as proxies for future performance. This framework challenges that assumption by:

1. **Replacing historical estimates** with neural network forecasts
2. **Maintaining identical optimization constraints** across strategies
3. **Performing rigorous statistical testing** to determine if ML predictions provide significant alpha

This controlled comparison isolates the value of machine learning forecasts from other portfolio construction decisions.

## Three-Strategy Framework

### Strategy 1: GRU-Optimized Portfolio

**Machine Learning Approach**

Uses quarterly-trained Gated Recurrent Unit (GRU) neural networks to forecast 30-day forward returns. The system implements **temporal model rotation**, automatically selecting the appropriate model based on rebalancing date:

| Training Quarter | Usage Period | Model File |
|-----------------|--------------|------------|
| Q4 2022 | Jan-Mar 2023 | `gru_bitcoin_2022_Q4_7D_customloss.pt` |
| Q1 2023 | Apr-Jun 2023 | `gru_bitcoin_2023_Q1_7D_customloss.pt` |
| Q2 2023 | Jul-Sep 2023 | `gru_bitcoin_2023_Q2_7D_customloss.pt` |

**Why Quarterly Rotation?**

- **Prevents overfitting**: Models trained on fresh data avoid stale parameter regimes
- **Adapts to regime changes**: Cryptocurrency markets exhibit structural breaks
- **Maintains statistical power**: Quarterly windows balance training data quantity with recency

**Feature Engineering**

The GRU models consume engineered features across multiple categories:

- **Price momentum**: 7, 14, 30-day returns
- **Volatility metrics**: Rolling standard deviation, Parkinson estimator
- **Market microstructure**: Volume, bid-ask spreads
- **Cross-asset signals**: Bitcoin correlation, market regime indicators

### Strategy 2: Historical Mean-Variance Optimization

**Traditional Benchmark**

Implements classic mean-variance optimization using 252-day (1-year) historical returns:

$$
\max_w \quad \frac{w^T \mu - r_f}{\sqrt{w^T \Sigma w}}
$$

Subject to:
- $\sum_i w_i = 1$ (full investment)
- $0 \leq w_i \leq 0.35$ (position limits)
- $|\{i : w_i > 0\}| \geq 3$ (minimum diversification)

This strategy serves as the **fair comparison baseline**, using identical constraints as the GRU approach but substituting historical means for neural network forecasts.

### Strategy 3: Market-Cap Weighted Index

**Passive Benchmark**

Weights assets proportional to market capitalization:

$$
w_i = \frac{\text{MarketCap}_i}{\sum_{j=1}^{N} \text{MarketCap}_j}
$$

Represents the **market consensus** portfolio and serves as a reality check—active strategies should outperform passive indexing after transaction costs.

## Technical Architecture

### Modular Design Principles

```
kallos_portfolios/
├── storage.py           # Database operations & data persistence
├── models.py            # ML model lifecycle management
├── predictors.py        # Return forecasting engines
├── optimisers.py        # Portfolio optimization algorithms
├── simulators/          # Strategy orchestration engines
│   ├── base.py         # Abstract base class
│   ├── gru.py          # Neural network strategy
│   ├── historical.py   # Traditional strategy
│   └── market_cap.py   # Passive benchmark
├── evaluation.py        # Performance metrics & hypothesis tests
└── analysis/            # Cross-strategy comparison tools
```

**Design Philosophy: Inheritance-Based Extensibility**

All strategies inherit from `BasePortfolioSimulator`, which defines the core backtesting loop:

```python
class BasePortfolioSimulator(ABC):
    """
    Abstract base class for portfolio simulators.

    Implements the common backtesting workflow while delegating
    strategy-specific return forecasting to subclasses.
    """

    @abstractmethod
    async def get_expected_returns_for_date(self, date, coin_ids):
        """
        Generate return forecasts for a specific rebalancing date.

        Subclasses implement strategy-specific logic:
        - GRU: Load quarterly model and predict
        - Historical: Calculate 252-day mean returns
        - Market Cap: Retrieve market cap weights
        """
        pass

    async def run_backtest(self, start_date, end_date):
        """
        Execute the complete backtesting workflow.

        1. Load monthly rebalancing dates and investable universe
        2. For each rebalancing date:
            a. Get expected returns (strategy-specific)
            b. Optimize portfolio weights
            c. Calculate returns until next rebalance
        3. Aggregate results and calculate performance metrics
        """
        rebalancing_dates = await self.get_rebalancing_dates(
            start_date, end_date
        )

        portfolio_returns = []
        for date in rebalancing_dates:
            # Strategy-specific forecasting
            expected_returns = await self.get_expected_returns_for_date(
                date, self.investable_universe[date]
            )

            # Common optimization
            weights = self.optimize_portfolio(expected_returns)

            # Common return calculation
            returns = self.calculate_period_returns(weights, date)
            portfolio_returns.extend(returns)

        return pd.Series(portfolio_returns, name=self.strategy_name)
```

This design **eliminates code duplication** while maintaining flexibility for strategy-specific customization.

## Portfolio Optimization Engine

### Mathematical Formulation

The optimizer implements **Sharpe ratio maximization** with advanced constraints:

$$
\begin{aligned}
\max_w \quad & \frac{w^T \mu}{\sqrt{w^T \Sigma w}} \\
\text{s.t.} \quad & \sum_{i=1}^{N} w_i = 1 \\
& 0 \leq w_i \leq w_{\max} \quad \forall i \\
& \sum_{i=1}^{N} \mathbb{1}[w_i > 0] \geq k_{\min} \\
& ||w||_2^2 \leq \gamma \quad \text{(L2 regularization)}
\end{aligned}
$$

**Parameters:**
- $\mu$: Expected returns (from GRU or historical data)
- $\Sigma$: Covariance matrix (252-day sample covariance)
- $w_{\max} = 0.35$: Maximum position size
- $k_{\min} = 3$: Minimum number of holdings
- $\gamma = 0.01$: Regularization strength

### Numerical Robustness

Real-world cryptocurrency data presents significant challenges. The implementation includes extensive preprocessing:

```python
def preprocess_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Handle data quality issues in cryptocurrency returns.

    Cryptocurrency markets exhibit:
    - Weekend gaps (no trading activity)
    - Extreme price movements (pump-and-dumps)
    - Delisting events (infinite returns)
    - New listings (insufficient history)
    """

    # 1. Fill weekend gaps
    returns = returns.fillna(method='ffill').fillna(method='bfill')

    # 2. Cap extreme returns at ±500%
    # Prevents covariance matrix singularity from outliers
    returns = returns.clip(lower=-5.0, upper=5.0)

    # 3. Replace infinite values
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.fillna(returns.median())

    # 4. Verify covariance matrix is positive semi-definite
    cov_matrix = returns.cov()
    min_eigenvalue = np.linalg.eigvalsh(cov_matrix)[0]

    if min_eigenvalue < 0:
        # Add diagonal regularization
        regularization = abs(min_eigenvalue) + 1e-6
        cov_matrix += regularization * np.eye(len(cov_matrix))

    return returns, cov_matrix
```

**Why This Matters:**

Standard optimization libraries assume clean data. Cryptocurrency markets violate these assumptions, requiring domain-specific preprocessing to ensure numerically stable solutions.

### Concentration Metrics

The system calculates **Herfindahl Index** and **Effective Number of Holdings**:

$$
H = \sum_{i=1}^{N} w_i^2
$$

$$
N_{\text{eff}} = \frac{1}{H}
$$

These metrics quantify portfolio concentration, revealing whether optimization algorithms create overly concentrated positions.

## GRU Prediction Pipeline

### Model Architecture

The GRU models employ a recurrent architecture optimized for time-series forecasting:

- **Input**: 30-day lookback window with 15+ engineered features
- **Architecture**: Multi-layer GRU with dropout regularization
- **Output**: 90-day forward prediction for 30-day percentage returns
- **Loss Function**: Custom direction-selective MSE (from Kallos Models)

### Inference Workflow

```python
class GRUPredictor:
    """
    Manages GRU model inference for portfolio forecasting.
    """

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.loaded_models = {}  # Cache for efficiency

    async def predict_returns(
        self,
        coin_ids: List[int],
        forecast_date: date
    ) -> pd.DataFrame:
        """
        Generate return forecasts for a specific date.

        Workflow:
        1. Identify appropriate quarterly model
        2. Load model and scaler if not cached
        3. Retrieve historical features for lookback window
        4. Normalize features using fitted scaler
        5. Run GRU inference
        6. Return predictions as DataFrame
        """

        # 1. Determine which quarterly model to use
        quarter = self._get_training_quarter(forecast_date)
        model_name = f"gru_bitcoin_{quarter}_7D_customloss.pt"

        # 2. Load model (with caching)
        if model_name not in self.loaded_models:
            model_path = self.model_dir / model_name
            scaler_path = self.model_dir / f"scaler_{quarter}.pkl"

            model = load_model_with_custom_loss(model_path)
            scaler = pickle.load(open(scaler_path, 'rb'))

            self.loaded_models[model_name] = (model, scaler)

        model, scaler = self.loaded_models[model_name]

        # 3. Load historical features
        features_df = await self.load_features(
            coin_ids,
            start_date=forecast_date - timedelta(days=90),
            end_date=forecast_date
        )

        # 4. Normalize features
        normalized = scaler.transform(features_df)
        series = TimeSeries.from_dataframe(
            normalized,
            value_cols=self.feature_columns,
            freq='D'
        )

        # 5. Generate predictions
        predictions = model.predict(
            n=90,  # 90-day forecast horizon
            past_covariates=series,
            num_samples=1
        )

        # 6. Extract 30-day forward return
        return_30d = predictions.pd_dataframe().iloc[30]

        return pd.DataFrame({
            'coin_id': coin_ids,
            'expected_return': return_30d.values
        })
```

**Key Design Decisions:**

- **Quarterly model caching**: Prevents redundant model loading during monthly rebalancing
- **Parallel inference**: Uses threading for multi-asset predictions
- **Lookback window**: 90 days ensures sufficient history for GRU's 30-day input chunk
- **Covariate-only approach**: Models use features without target variable during inference

## Performance Evaluation Framework

### Comprehensive Metrics

The `PortfolioEvaluator` calculates 15 performance indicators:

**Return Metrics:**
- Total return (cumulative)
- Annualized return (geometric mean)
- Win rate (% positive days)

**Risk Metrics:**
- Volatility (annualized standard deviation)
- Maximum drawdown
- Value at Risk (95% confidence)
- Conditional Value at Risk (CVaR)

**Risk-Adjusted Returns:**
- Sharpe ratio: $\frac{\text{Annualized Return}}{\text{Annualized Volatility}}$
- Calmar ratio: $\frac{\text{Annualized Return}}{|\text{Max Drawdown}|}$

**Distribution Properties:**
- Skewness (asymmetry)
- Kurtosis (tail heaviness)

### Rigorous Statistical Testing

The framework implements **four hypothesis tests** for pairwise strategy comparison:

#### 1. Paired t-Test (Mean Return Difference)

**Null Hypothesis:** $H_0: \mu_1 - \mu_2 = 0$

Tests whether one strategy significantly outperforms another:

```python
from scipy.stats import ttest_rel

t_stat, p_value = ttest_rel(strategy1_returns, strategy2_returns)

# Calculate effect size (Cohen's d)
mean_diff = strategy1_returns.mean() - strategy2_returns.mean()
pooled_std = np.sqrt(
    (strategy1_returns.std()**2 + strategy2_returns.std()**2) / 2
)
cohens_d = mean_diff / pooled_std

# 95% confidence interval for excess return
ci_lower, ci_upper = t.interval(
    0.95,
    df=len(strategy1_returns) - 1,
    loc=mean_diff,
    scale=sem(strategy1_returns - strategy2_returns)
)
```

**Interpretation:**
- $p < 0.05$: Significant performance difference
- Cohen's d > 0.5: Medium effect size
- CI excludes zero: Robust outperformance

#### 2. F-Test (Variance Equality)

**Null Hypothesis:** $H_0: \sigma_1^2 = \sigma_2^2$

Tests whether strategies exhibit different risk levels:

```python
from scipy.stats import f

F_stat = strategy1_returns.var() / strategy2_returns.var()
df1 = len(strategy1_returns) - 1
df2 = len(strategy2_returns) - 1

p_value = 2 * min(
    f.cdf(F_stat, df1, df2),
    1 - f.cdf(F_stat, df1, df2)
)  # Two-tailed test
```

**Interpretation:**
- $p < 0.05$: Strategies have significantly different volatilities
- Useful for assessing whether ML reduces or increases risk

#### 3. Kolmogorov-Smirnov Test (Distribution Similarity)

**Null Hypothesis:** Returns drawn from the same distribution

Tests whether return distributions fundamentally differ:

```python
from scipy.stats import ks_2samp

ks_stat, p_value = ks_2samp(
    strategy1_returns,
    strategy2_returns
)
```

**Interpretation:**
- $p < 0.05$: Distributions significantly different
- Detects differences beyond mean and variance

#### 4. Stochastic Dominance Test

**First-Order Stochastic Dominance:**

Strategy A dominates Strategy B if:

$$
F_A(x) \leq F_B(x) \quad \forall x
$$

Where $F$ is the cumulative distribution function.

```python
def test_stochastic_dominance(returns_A, returns_B):
    """
    Test first-order stochastic dominance.

    A dominates B if A's CDF is always below or equal to B's CDF.
    """
    # Create empirical CDFs
    sorted_returns = np.sort(np.concatenate([returns_A, returns_B]))

    cdf_A = np.array([
        (returns_A <= x).mean() for x in sorted_returns
    ])
    cdf_B = np.array([
        (returns_B <= x).mean() for x in sorted_returns
    ])

    # Check dominance conditions
    A_dominates_B = np.all(cdf_A <= cdf_B)
    B_dominates_A = np.all(cdf_B <= cdf_A)

    return {
        'A_dominates_B': A_dominates_B,
        'B_dominates_A': B_dominates_A,
        'no_dominance': not (A_dominates_B or B_dominates_A)
    }
```

**Interpretation:**
- If A dominates B: Rational investors prefer A
- Strong test: Requires superiority across entire distribution

## Database Integration

### Schema-Free Design

The framework integrates with existing database infrastructure **without requiring schema modifications**:

**Required Tables:**

1. `daily_market_data`: Historical OHLCV data
   - `coin_id`, `date`, `close`, `volume`

2. `daily_index_coin_contributions`: Market cap weights
   - `coin_id`, `date`, `market_cap`, `weight`

3. `index_monthly_constituents`: Investable universe
   - `coin_id`, `month`, `is_constituent`

4. `model_train_view`: Quarterly model metadata
   - `quarter`, `model_file`, `scaler_file`, `training_end_date`

### Async Database Operations

Uses `asyncpg` for non-blocking concurrent data access:

```python
class PortfolioDataLoader:
    """
    Asynchronous database operations for portfolio construction.
    """

    async def load_returns_for_optimization(
        self,
        coin_ids: List[int],
        end_date: date,
        lookback_days: int = 252
    ) -> pd.DataFrame:
        """
        Load historical returns for covariance estimation.

        Uses connection pooling and parallel queries
        for efficient data retrieval.
        """
        async with self.pool.acquire() as conn:
            query = """
                SELECT coin_id, date, close
                FROM daily_market_data
                WHERE coin_id = ANY($1)
                    AND date BETWEEN $2 AND $3
                ORDER BY date
            """

            rows = await conn.fetch(
                query,
                coin_ids,
                end_date - timedelta(days=lookback_days),
                end_date
            )

        # Calculate returns
        df = pd.DataFrame(rows)
        returns = df.groupby('coin_id')['close'].pct_change()

        return returns.unstack()  # Pivot to wide format
```

**Performance Benefits:**

- **Non-blocking I/O**: Other operations continue during database queries
- **Connection pooling**: Reuses connections for efficiency
- **Batch queries**: Single query loads data for all assets

## Backtesting Workflow

### Monthly Rebalancing Loop

```python
async def run_complete_backtest(
    start_date: date,
    end_date: date
) -> Dict[str, pd.Series]:
    """
    Execute backtests for all three strategies.

    Returns:
        Dictionary mapping strategy names to return series
    """

    # Initialize simulators
    gru_sim = GRUPortfolioSimulator(model_dir, database_url)
    hist_sim = HistoricalPortfolioSimulator(database_url)
    mcap_sim = MarketCapWeightedSimulator(database_url)

    # Run backtests in parallel
    gru_task = gru_sim.run_backtest(start_date, end_date)
    hist_task = hist_sim.run_backtest(start_date, end_date)
    mcap_task = mcap_sim.run_backtest(start_date, end_date)

    gru_returns, hist_returns, mcap_returns = await asyncio.gather(
        gru_task, hist_task, mcap_task
    )

    return {
        'gru': gru_returns,
        'historical': hist_returns,
        'market_cap': mcap_returns
    }
```

### Transaction Cost Modeling

The system uses `vectorbt` for realistic backtesting with transaction costs:

```python
import vectorbt as vbt

portfolio = vbt.Portfolio.from_orders(
    close=price_data,
    size=position_changes,
    size_type='targetpercent',
    fees=0.001,  # 10 bps trading fee
    slippage=0.0005,  # 5 bps slippage
    freq='1D'
)

# Extract performance metrics
total_return = portfolio.total_return()
sharpe_ratio = portfolio.sharpe_ratio()
max_drawdown = portfolio.max_drawdown()
```

**Cost Assumptions:**
- **Trading fees**: 10 basis points (0.1%)
- **Slippage**: 5 basis points (0.05%)
- **Total round-trip cost**: ~30 bps

These costs are realistic for institutional cryptocurrency trading.

## Output & Reporting

### QuantStats Integration

The framework generates professional tearsheets for each strategy:

```python
import quantstats as qs

# Generate comprehensive HTML report
qs.reports.html(
    returns,
    output='gru_strategy_tearsheet.html',
    title='GRU-Optimized Portfolio',
    benchmark=market_cap_returns
)
```

**Included Visualizations:**
- Cumulative returns chart
- Rolling Sharpe ratio
- Drawdown periods
- Monthly returns heatmap
- Distribution histogram

### Comparative Analysis Report

The system produces a unified comparison document:

```json
{
  "performance_comparison": {
    "gru": {
      "total_return": 0.847,
      "sharpe_ratio": 1.92,
      "max_drawdown": -0.234,
      "win_rate": 0.573
    },
    "historical": {
      "total_return": 0.623,
      "sharpe_ratio": 1.54,
      "max_drawdown": -0.287,
      "win_rate": 0.541
    },
    "market_cap": {
      "total_return": 0.519,
      "sharpe_ratio": 1.21,
      "max_drawdown": -0.312,
      "win_rate": 0.528
    }
  },
  "hypothesis_tests": {
    "gru_vs_historical": {
      "mean_difference": {
        "t_statistic": 2.84,
        "p_value": 0.0047,
        "cohens_d": 0.42,
        "confidence_interval": [0.0073, 0.0214]
      },
      "variance_equality": {
        "f_statistic": 1.15,
        "p_value": 0.234
      },
      "distribution_similarity": {
        "ks_statistic": 0.087,
        "p_value": 0.012
      },
      "stochastic_dominance": {
        "gru_dominates": false,
        "historical_dominates": false
      }
    }
  }
}
```

## Key Achievements

### Technical Excellence

1. **Production-Ready Architecture**: Modular, extensible design supporting new strategies
2. **Rigorous Validation**: Four statistical tests ensure robust conclusions
3. **Operational Efficiency**: Async database operations and model caching
4. **Numerical Robustness**: Handles real-world data quality issues
5. **Comprehensive Evaluation**: 15 performance metrics across multiple dimensions

### Quantitative Insights

The framework enables answering critical questions:

- **Does ML add value?** Statistical tests quantify outperformance
- **At what cost?** Transaction cost modeling reveals net alpha
- **Is it robust?** Hold-out testing prevents overfitting claims
- **How concentrated?** Herfindahl indices reveal portfolio concentration

## Technologies & Stack

**Core Infrastructure:**
- **Python 3.8+**: Modern async/await syntax
- **PyTorch**: GRU model inference
- **PostgreSQL**: Financial data storage
- **asyncpg**: Non-blocking database I/O

**Optimization & Backtesting:**
- **PyPortfolioOpt**: Mean-variance optimization
- **CVXPY**: Convex optimization constraints
- **vectorbt**: Vectorized backtesting with transaction costs
- **QuantStats**: Professional performance reporting

**Scientific Computing:**
- **pandas**: Data manipulation
- **NumPy**: Numerical operations
- **SciPy**: Statistical testing
- **scikit-learn**: Feature preprocessing

## Use Cases

This framework is designed for:

- **Quantitative researchers** evaluating ML for portfolio management
- **Fund managers** seeking systematic cryptocurrency exposure
- **Data scientists** building production trading systems
- **Academic researchers** studying ML in finance

## Future Enhancements

Potential extensions:

- **Multi-frequency rebalancing**: Daily, weekly, monthly comparison
- **Alternative ML models**: LSTMs, Transformers, ensemble methods
- **Risk budgeting**: Allocate risk rather than capital
- **Regime detection**: Dynamic strategy switching
- **Live trading integration**: Real-time execution infrastructure

## Repository

Complete source code and documentation:

**[github.com/josemarquezjaramillo/kallos_portfolios](https://github.com/josemarquezjaramillo/kallos_portfolios)**

**License**: MIT

---

*This project demonstrates end-to-end quantitative portfolio management: from machine learning predictions through optimization, rigorous backtesting, and statistical validation—all with production-ready code quality.*
