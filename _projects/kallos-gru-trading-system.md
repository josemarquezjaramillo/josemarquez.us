---
title: "Kallos: GRU-Powered Cryptocurrency Trading System"
date: 2024-12-15
category: Machine Learning Research
tags: [Deep Learning, Portfolio Optimization, Cryptocurrency, GRU, Time Series, Backtesting, MLOps]
featured: true
excerpt: "End-to-end deep learning trading system evaluating whether GRU neural networks can generate investable alpha in cryptocurrency markets. Full MLOps pipeline from data ingestion to portfolio construction with rigorous walk-forward validation."
github_url: https://github.com/josemarquezjaramillo
paper_url: /assets/Evaluating GRU powered trading systems - the case of cryptocurrency markets by Jose Marquez Jaramillo.pdf
mathjax: true
---

## Overview

This project implements and evaluates a complete deep learning trading system for cryptocurrency markets, addressing the research question: **"Do GRU neural network forecasts translate into investable portfolio improvements?"**

Developed as the capstone project for EN.649.740 Machine Learning: Deep Learning at Johns Hopkins University, the system integrates three interconnected components into a production-grade MLOps pipeline:

1. **[Kallos Data Pipeline](/projects/kallos-data-pipeline/)**: Automated ETL infrastructure for cryptocurrency market data
2. **[Kallos Models](/projects/kallos-models-mlops/)**: Deep learning forecasting with walk-forward validation
3. **[Kallos Portfolios](/projects/kallos-portfolios/)**: Portfolio optimization and backtesting framework

The research evaluates whether sophisticated GRU-based price predictions can outperform simple benchmark strategies in real-world backtesting with transaction costs and realistic constraints.

## Research Question & Motivation

### The Challenge

While deep learning models have shown promise in time-series forecasting, **translating model predictions into profitable trading strategies remains an open question**. This project bridges the gap between academic forecasting research and practical portfolio management.

**Key Questions:**
- Can GRU networks capture cryptocurrency price dynamics better than naive baselines?
- Do improved forecasts lead to superior risk-adjusted returns?
- How do different portfolio construction objectives (expected return vs. quadratic utility) affect performance?
- Are observed improvements statistically significant or attributable to chance?

### Hypothesis

We hypothesize that GRU neural networks, with their ability to model long-term dependencies and custom loss functions emphasizing directional accuracy, will produce forecasts that translate into portfolios with:
- Higher Sharpe ratios than naive persistence forecasts
- Superior risk-adjusted returns compared to market-cap weighted benchmarks
- Statistical significance in performance differences

## System Architecture

The Kallos system implements a **three-stage pipeline** ensuring proper temporal separation and preventing data leakage:

```
┌─────────────────────────────────────────────────────────────────┐
│                    1. DATA COLLECTION                           │
│                    (kallos-data)                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CoinGecko API → ETL Pipeline → PostgreSQL Database            │
│                                                                 │
│  • OHLCV data (hourly)                                         │
│  • Market fundamentals (volume, market cap, dominance)         │
│  • 30+ technical indicators (RSI, MACD, Bollinger Bands)       │
│  • Async operations with rate limiting                         │
│                                                                 │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    2. FORECASTING                               │
│                    (kallos_models)                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Walk-Forward Validation                                        │
│  ├─ Hyperparameter Optimization (Optuna, 50 trials)           │
│  ├─ GRU Training (Direction-Selective MSE Loss)                │
│  └─ Price Predictions (7-day horizon)                          │
│                                                                 │
│  Custom Loss Function:                                          │
│  L = MSE + λ · DirectionPenalty                                │
│                                                                 │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    3. PORTFOLIO CONSTRUCTION                    │
│                    (kallos_portfolios)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Portfolio Optimization (Quarterly Rebalance)                   │
│  ├─ Strategy 1: GRU Forecasts → Mean-Variance                 │
│  ├─ Strategy 2: Naive Persistence → Mean-Variance             │
│  └─ Benchmark: 35% Cap-Weighted Market Portfolio              │
│                                                                 │
│  Backtesting & Performance Evaluation                           │
│  ├─ Transaction costs (0.2% per trade)                         │
│  ├─ Weekly rebalancing                                         │
│  ├─ QuantStats metrics                                         │
│  └─ Statistical hypothesis testing                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

**Weekly Cycle:**
1. **Saturday 00:00 UTC**: Data pipeline fetches latest market data
2. **Saturday 02:00 UTC**: Feature engineering and preprocessing
3. **Saturday 04:00 UTC**: GRU model generates 7-day price forecasts
4. **Saturday 06:00 UTC**: Portfolio optimizer computes weights
5. **Sunday 00:00 UTC**: Execute rebalance trades (if required)

**Quarterly Model Rotation:**
- Every 13 weeks: Retrain GRU with latest data
- Walk-forward window: 52 weeks training, 13 weeks validation
- Prevents overfitting through temporal separation

## Methodology

### 1. Data Preprocessing

**Universe Construction:**
- Top 5 cryptocurrencies by market capitalization (excluding stablecoins)
- Weekly snapshots: BTC, ETH, BNB, ADA, SOL (representative period)
- Hourly OHLCV data aggregated to daily

**Feature Engineering:**

| Feature Category | Variables | Transformation |
|-----------------|-----------|----------------|
| Price Features | Close, Open, High, Low | Log returns |
| Volume Metrics | Trading volume, Volume MA | Z-score normalization |
| Technical Indicators | RSI, MACD, BB | Min-max scaling [0,1] |
| Market Microstructure | Bid-ask spread proxy, volatility | Rolling z-score |
| Fundamental Metrics | Market dominance, circulating supply | Log transformation |

**Stationarity Enforcement:**
- Augmented Dickey-Fuller test for unit roots
- First-differencing for non-stationary series
- Ensures model learns stable relationships

### 2. Walk-Forward Validation

Traditional k-fold cross-validation **violates temporal causality** in time series. We implement strict walk-forward splits:

```python
# Temporal split structure
Training Window:   [t-52w ──────────── t-13w]
Validation Window:              [t-13w ──── t]
Hold-out Test:                         [t ──── t+13w]

# No future information leakage
for split in range(n_quarters):
    train_end = split * 13_weeks
    val_start = train_end
    val_end = val_start + 13_weeks

    # Train only on past
    model.fit(data[:train_end])

    # Validate on immediate future
    preds = model.predict(data[val_start:val_end])
```

This mimics **real trading conditions** where models predict the immediate future using only historical data.

### 3. GRU Neural Network Architecture

**Model Configuration (Optimized via Optuna):**

| Hyperparameter | Search Space | Optimal Value |
|----------------|--------------|---------------|
| Input Chunk Length | [14, 90] days | 56 days |
| Hidden Dimension | [32, 256] | 128 |
| GRU Layers | [1, 4] | 2 |
| Dropout | [0.0, 0.7] | 0.35 |
| Learning Rate | [1e-5, 1e-2] | 3.2e-4 |
| Batch Size | {16, 32, 64} | 32 |
| Direction Penalty (λ) | [0.0, 2.0] | 0.8 |

**Custom Loss Function:**

The standard MSE loss treats all prediction errors equally. For trading, **directional accuracy matters more than magnitude precision**. We implement:

$$
\mathcal{L}_{\text{custom}} = \text{MSE} + \lambda \cdot \mathcal{L}_{\text{direction}}
$$

Where the direction penalty is:

$$
\mathcal{L}_{\text{direction}} = \frac{1}{n}\sum_{i=1}^{n} \mathbb{1}[\text{sign}(\Delta y_i) \neq \text{sign}(\Delta \hat{y}_i)] \cdot |\Delta y_i|
$$

This penalizes directional mistakes proportionally to the magnitude of the true price move, incentivizing the model to **prioritize getting the direction right**.

**Optimization Process:**
- 50 Optuna trials with multi-objective optimization
- Objectives: Minimize RMSE, Maximize Direction Accuracy
- Pruning: Early stopping for underperforming trials
- Persistent study storage in PostgreSQL

### 4. Portfolio Construction

We evaluate **two portfolio objectives** to test robustness:

#### Objective 1: Maximize Expected Returns (MER)

Classic Markowitz mean-variance optimization:

$$
\begin{aligned}
\max_w \quad & w^T \mu - \frac{\gamma}{2} w^T \Sigma w \\
\text{s.t.} \quad & \sum_{i=1}^{N} w_i = 1 \\
& 0 \leq w_i \leq 0.35 \quad \forall i
\end{aligned}
$$

Where:
- $w$ = portfolio weights
- $\mu$ = expected returns (from forecasts)
- $\Sigma$ = covariance matrix (60-day rolling window)
- $\gamma$ = risk aversion parameter (= 1.0)
- 35% position caps prevent concentration

#### Objective 2: Maximize Quadratic Utility (MQU)

Utility-based optimization incorporating investor risk preferences:

$$
U(w) = w^T \mu - \frac{\alpha}{2} w^T \Sigma w
$$

Where $\alpha$ is the risk-aversion coefficient calibrated to match typical cryptocurrency investor profiles ($\alpha = 2.0$).

**Strategy Comparison:**

| Strategy | Return Forecast | Optimization |
|----------|----------------|--------------|
| **GRU-MER** | GRU predictions | Maximize expected returns |
| **Naive-MER** | Persistence (today = tomorrow) | Maximize expected returns |
| **GRU-MQU** | GRU predictions | Maximize quadratic utility |
| **Naive-MQU** | Persistence | Maximize quadratic utility |
| **Benchmark** | N/A | 35% cap-weighted market portfolio |

### 5. Backtesting Framework

**Implementation Details:**
- Platform: vectorbt with QuantStats integration
- Transaction costs: 0.2% per trade (realistic for exchanges)
- Rebalancing frequency: Weekly (Sunday 00:00 UTC)
- Portfolio constraints: Long-only, 35% position caps
- Initial capital: $100,000

**Performance Metrics:**

| Category | Metrics |
|----------|---------|
| Returns | Total return, CAGR, monthly/annual returns |
| Risk | Volatility, max drawdown, Calmar ratio |
| Risk-Adjusted | Sharpe ratio, Sortino ratio, Omega ratio |
| Downside Protection | VaR (95%), CVaR, worst month |
| Benchmark Comparison | Alpha, beta, tracking error, information ratio |

### 6. Statistical Validation

To ensure results aren't due to chance, we conduct **rigorous hypothesis testing**:

**Sharpe Ratio Comparison (Jobson-Korkie Test):**

$$
t = \frac{S_1 - S_2}{\sqrt{\text{Var}(S_1 - S_2)}}
$$

Where:
- $S_1, S_2$ = Sharpe ratios of two strategies
- Null hypothesis: $S_1 = S_2$ (no difference)
- Alternative: $S_1 > S_2$ (strategy 1 superior)

**Return Distribution Tests:**
- Kolmogorov-Smirnov test for distribution equality
- Mann-Whitney U test for median returns
- Significance level: $\alpha = 0.05$

## Data & Features

### Universe Composition

The **Top-5 cryptocurrency universe** during the study period:

| Rank | Asset | Market Cap | Volatility | Role |
|------|-------|-----------|------------|------|
| 1 | Bitcoin (BTC) | $800B+ | 45% annual | Digital gold |
| 2 | Ethereum (ETH) | $400B+ | 55% annual | Smart contracts |
| 3 | Binance Coin (BNB) | $80B+ | 60% annual | Exchange token |
| 4 | Cardano (ADA) | $40B+ | 70% annual | PoS platform |
| 5 | Solana (SOL) | $35B+ | 80% annual | High-throughput L1 |

**Stablecoins excluded** to focus on volatile, tradable assets with genuine price discovery.

### Technical Indicators

The feature set includes **30+ indicators** spanning multiple categories:

**Momentum Indicators:**
- RSI (Relative Strength Index, 14-day)
- MACD (12/26/9 configuration)
- Stochastic Oscillator
- Rate of Change (ROC)

**Trend Indicators:**
- Moving Averages (SMA 20/50/200, EMA 12/26)
- Bollinger Bands (20-day, 2σ)
- Parabolic SAR
- ADX (Average Directional Index)

**Volume Indicators:**
- On-Balance Volume (OBV)
- Volume-Weighted Average Price (VWAP)
- Accumulation/Distribution Line
- Chaikin Money Flow

**Volatility Indicators:**
- ATR (Average True Range)
- Bollinger Band Width
- Keltner Channels
- Historical Volatility (30-day)

### Data Quality Measures

**Validation Checks:**
- Outlier detection: Winsorization at 1st/99th percentiles
- Missing data: Forward-fill up to 3 hours, drop otherwise
- Stale price detection: Flag assets with < 10% daily volume
- Cross-exchange validation: Verify prices against multiple sources

**Data Coverage:**
- Period: January 2020 – December 2024 (5 years)
- Frequency: Hourly → Aggregated to daily
- Total observations: 1,825 days × 5 assets = 9,125 asset-days

## Results & Analysis

### Overall Performance Summary

The backtesting evaluation spans **13 quarters** with quarterly model retraining. Below are the cumulative results:

#### Maximize Expected Returns (MER) Objective

| Metric | GRU-MER | Naive-MER | Benchmark | Best |
|--------|---------|-----------|-----------|------|
| **Total Return** | 142.3% | 135.7% | 156.2% | **Benchmark** |
| **CAGR** | 28.4% | 27.1% | 31.2% | **Benchmark** |
| **Sharpe Ratio** | 1.12 | 1.08 | 1.21 | **Benchmark** |
| **Sortino Ratio** | 1.68 | 1.62 | 1.79 | **Benchmark** |
| **Max Drawdown** | -32.1% | -34.5% | -38.7% | **GRU-MER** |
| **Calmar Ratio** | 0.88 | 0.79 | 0.81 | **GRU-MER** |
| **Win Rate** | 54.2% | 51.8% | 52.3% | **GRU-MER** |
| **Avg Win/Loss** | 1.31 | 1.24 | 1.28 | **GRU-MER** |

**Key Findings:**
- GRU-MER achieved **4.1% higher Sharpe ratio** vs. Naive-MER (1.12 vs. 1.08)
- However, **benchmark outperformed** both (Sharpe 1.21) due to concentrated exposure during bull runs
- GRU showed **superior downside protection**: 6.6pp lower max drawdown than benchmark
- Difference between GRU and Naive **not statistically significant** (p = 0.18, Jobson-Korkie test)

#### Maximize Quadratic Utility (MQU) Objective

| Metric | GRU-MQU | Naive-MQU | Benchmark | Best |
|--------|---------|-----------|-----------|------|
| **Total Return** | 118.6% | 119.2% | 156.2% | **Benchmark** |
| **CAGR** | 23.7% | 23.8% | 31.2% | **Benchmark** |
| **Sharpe Ratio** | 1.05 | 1.04 | 1.21 | **Benchmark** |
| **Sortino Ratio** | 1.54 | 1.53 | 1.79 | **Benchmark** |
| **Max Drawdown** | -28.3% | -28.9% | -38.7% | **GRU-MQU** |
| **Calmar Ratio** | 0.84 | 0.82 | 0.81 | **GRU-MQU** |
| **Volatility** | 22.6% | 22.9% | 25.8% | **GRU-MQU** |

**Key Findings:**
- Under quadratic utility, **GRU and Naive converged** (virtually identical performance)
- Higher risk aversion ($\alpha = 2.0$) led to **more conservative portfolios** → lower volatility (22.6% vs. 25.8%)
- Both GRU/Naive underperformed benchmark in returns but showed **10pp lower max drawdown**
- Risk-adjusted performance (Calmar ratio) favored **GRU due to drawdown control**

### Statistical Hypothesis Testing

#### Sharpe Ratio Differences (Jobson-Korkie Test)

**MER Objective:**

| Comparison | Δ Sharpe | t-statistic | p-value | Significant? |
|------------|----------|-------------|---------|--------------|
| GRU-MER vs. Naive-MER | +0.04 | 1.35 | 0.18 | No |
| GRU-MER vs. Benchmark | -0.09 | -2.73 | 0.007 | **Yes** (worse) |
| Naive-MER vs. Benchmark | -0.13 | -3.12 | 0.002 | **Yes** (worse) |

**MQU Objective:**

| Comparison | Δ Sharpe | t-statistic | p-value | Significant? |
|------------|----------|-------------|---------|--------------|
| GRU-MQU vs. Naive-MQU | +0.01 | 0.42 | 0.67 | No |
| GRU-MQU vs. Benchmark | -0.16 | -4.21 | <0.001 | **Yes** (worse) |

**Interpretation:**
- GRU forecasts **did not produce statistically significant Sharpe improvements** over naive persistence
- Both strategies significantly underperformed the benchmark
- Higher risk aversion (MQU) eliminated any GRU advantage

#### Return Distribution Analysis

**Kolmogorov-Smirnov Test (Distribution Equality):**

| Comparison | KS Statistic | p-value | Interpretation |
|------------|--------------|---------|----------------|
| GRU-MER vs. Naive-MER | 0.11 | 0.42 | Distributions similar |
| GRU-MQU vs. Naive-MQU | 0.08 | 0.68 | Distributions similar |

**Mann-Whitney U Test (Median Returns):**

| Comparison | U Statistic | p-value | Median Difference |
|------------|-------------|---------|-------------------|
| GRU-MER vs. Naive-MER | 1523 | 0.31 | +0.12% weekly |
| GRU-MQU vs. Naive-MQU | 1489 | 0.58 | +0.03% weekly |

**Conclusion:** No statistically significant differences in return distributions or medians between GRU and Naive strategies.

### Forecast Quality Analysis

While portfolio performance was mixed, we examine the **quality of GRU predictions themselves**:

#### Prediction Metrics (7-Day Horizon)

| Asset | RMSE (%) | MAE (%) | Direction Accuracy | Naive RMSE (%) |
|-------|----------|---------|-------------------|----------------|
| BTC | 4.2 | 3.1 | 56.3% | 5.8 |
| ETH | 5.7 | 4.3 | 54.1% | 7.2 |
| BNB | 6.8 | 5.2 | 52.8% | 8.5 |
| ADA | 8.1 | 6.4 | 51.7% | 9.9 |
| SOL | 9.3 | 7.2 | 50.9% | 11.2 |
| **Average** | **6.8%** | **5.2%** | **53.2%** | **8.5%** |

**Key Insights:**
- GRU achieved **20% lower RMSE** than naive persistence (6.8% vs. 8.5%)
- Direction accuracy of **53.2%** is modest but above random (50%)
- Forecasting difficulty increases with market cap rank (BTC easiest, SOL hardest)
- Custom loss function successfully balanced magnitude and directional errors

#### Forecast Error Distribution

The residual distribution analysis reveals important patterns:

**Bitcoin Forecast Residuals:**
- Mean: -0.08% (slight downward bias)
- Median: -0.02%
- Std Dev: 4.18%
- Skewness: -0.23 (left tail)
- Kurtosis: 3.87 (fat tails)

**Interpretation:**
- Residuals approximately normal but with **fat tails** (kurtosis > 3)
- GRU underestimates extreme moves (both up and down)
- Consistent with cryptocurrency market's jump dynamics

### Quarterly Performance Breakdown

Performance varied significantly across quarters due to changing market regimes:

| Quarter | Market Regime | GRU-MER Return | Naive-MER Return | Benchmark Return |
|---------|---------------|----------------|------------------|------------------|
| Q1 2023 | Bull (BTC +70%) | +18.2% | +16.5% | +22.3% |
| Q2 2023 | Sideways | +3.1% | +2.8% | +1.9% |
| Q3 2023 | Bear (-15%) | -8.4% | -9.7% | -12.1% |
| Q4 2023 | Recovery | +11.3% | +10.8% | +14.2% |
| Q1 2024 | Strong bull | +22.1% | +20.3% | +28.7% |
| Q2 2024 | Volatile | +5.7% | +4.9% | +6.2% |

**Pattern Analysis:**
- **Bull markets**: Benchmark's concentration advantage dominates (higher beta)
- **Sideways markets**: GRU shows advantage through superior asset selection
- **Bear markets**: GRU's risk management (lower drawdowns) provides value
- **Volatile markets**: GRU's direction accuracy yields modest edge

### Why Didn't GRU Outperform?

Despite improved forecasting metrics, the GRU strategy failed to significantly beat the benchmark. Several factors explain this:

**1. Forecast Horizon Mismatch**
- Models optimized for 7-day forecasts
- Portfolio rebalancing: weekly
- Optimal forecast horizon may differ from rebalancing frequency

**2. Transaction Costs**
- 0.2% per trade erodes theoretical alpha
- GRU strategies had 23% higher turnover than benchmark
- Annual cost drag: ~4.6% for GRU vs. 3.2% for benchmark

**3. Optimization Objective Alignment**
- Models trained to minimize RMSE + direction penalty
- Portfolio optimization maximizes Sharpe ratio
- **Misalignment**: Better point forecasts ≠ better portfolio inputs

**4. Cryptocurrency Market Efficiency**
- Top-5 assets are highly liquid and widely analyzed
- Most alpha opportunities already arbitraged away
- Greater potential in mid-cap, less efficient assets

**5. Bull Market Period**
- Study period (2023-2024) featured strong bull market
- Simple market-cap weighting benefits from momentum
- Risk management advantages of GRU obscured by rising tide

## Discussion

### Theoretical Implications

**1. Forecasting ≠ Portfolio Performance**

This research demonstrates the critical **gap between prediction accuracy and investment returns**. The GRU model achieved:
- 20% lower RMSE than naive baseline
- 53.2% directional accuracy
- Statistically significant forecast improvements

Yet these gains **did not translate into superior risk-adjusted returns**. This highlights:
- Portfolio optimization introduces additional complexity
- Transaction costs and constraints matter enormously
- Covariance estimation errors can overwhelm forecast improvements

**2. Walk-Forward Validation is Essential**

Traditional backtesting with fixed train/test splits would have shown **spuriously optimistic results**. Walk-forward validation revealed:
- Quarterly model retraining necessary to maintain performance
- Performance degrades 15-20% within 13 weeks without retraining
- Hyperparameters stable across walk-forward splits (good generalization)

**3. Multi-Objective Optimization Value**

The custom direction-selective loss function proved valuable:
- Standard MSE: 50.1% direction accuracy
- Custom loss (λ=0.8): 53.2% direction accuracy
- Trade-off: Slight increase in RMSE (+0.3pp) acceptable

For trading applications, **correctly predicting direction is more valuable than minimizing magnitude errors**.

### Practical Considerations

**1. When GRU Might Add Value**
- **Bear markets**: Lower drawdowns justify lower returns
- **Sideways markets**: Superior asset selection more visible
- **Mid-cap cryptocurrencies**: Less efficient, more predictable
- **Higher frequency**: Daily rebalancing may better utilize forecasts

**2. Production Deployment Challenges**
- **Model drift**: Quarterly retraining required (computationally expensive)
- **Infrastructure**: GPU training, database management, scheduling
- **Monitoring**: Track forecast quality, detect regime changes
- **Latency**: Inference must complete before trading window

**3. Cost-Benefit Analysis**

**Costs:**
- Development: ~200 hours (data + models + backtesting)
- Infrastructure: $500/month (cloud compute, databases)
- Transaction costs: 4.6% annual drag
- Opportunity cost: Lower returns than buy-and-hold

**Benefits:**
- Risk management: 6.6pp lower max drawdown
- Learning experience: Complete MLOps pipeline
- Extensibility: Framework supports other assets/strategies
- Intellectual property: Proprietary forecasting system

**Verdict:** For a retail investor, **costs likely exceed benefits**. For institutional asset managers with scale, lower transaction costs, and risk mandates, the system could be viable.

### Comparison to Literature

Our results align with academic consensus on financial ML:

| Study | Finding | Alignment |
|-------|---------|-----------|
| Krauss et al. (2017) | Deep learning beats linear models in stocks | ✓ (GRU > Naive) |
| Sezer et al. (2020) | CNNs achieve 55-60% direction accuracy in crypto | ✓ (53.2% for GRU) |
| De Prado (2018) | Transaction costs eliminate most ML alpha | ✓✓ (major factor) |
| Hsu et al. (2021) | Out-of-sample performance degrades quickly | ✓ (quarterly retraining needed) |
| Jansen (2020) | Portfolio construction matters more than forecasts | ✓✓ (key insight) |

**Novel Contribution:** This work provides **end-to-end evaluation** from data ingestion to portfolio performance, unlike most studies focusing on isolated forecasting metrics.

### Limitations & Future Work

**Current Limitations:**

1. **Limited Universe**: Top-5 cryptocurrencies only
   - Missing mid-cap opportunities
   - No cross-asset diversification

2. **Simple Portfolio Optimization**: Mean-variance framework
   - Assumes normally distributed returns (violated in crypto)
   - Sensitive to covariance estimation errors

3. **Fixed Rebalancing**: Weekly schedule
   - Potentially suboptimal frequency
   - No dynamic rebalancing based on signals

4. **No Alternative Architectures**: GRU only
   - Transformers may capture longer dependencies
   - Ensemble methods could improve robustness

5. **Market Regime Ignorance**: No regime detection
   - GRU performs differently in bull/bear markets
   - Adaptive strategies could exploit this

**Future Enhancements:**

**1. Advanced Portfolio Methods**
- Hierarchical Risk Parity (less covariance-sensitive)
- Black-Litterman with GRU views
- Kelly Criterion for position sizing

**2. Multi-Asset Universe**
- Expand to Top-20 cryptocurrencies
- Include stablecoins for cash allocation
- Cross-asset strategies (crypto + equities)

**3. Ensemble Forecasting**
- Combine GRU, LSTM, Transformer
- Weighted averaging based on recent performance
- Bayesian model averaging for uncertainty quantification

**4. Regime-Adaptive Strategies**
- Hidden Markov Models for regime detection
- Bull/bear-specific portfolio strategies
- Dynamic risk budgeting based on volatility regime

**5. Reinforcement Learning**
- Directly optimize trading decisions (not forecasts)
- Deep Q-Learning for portfolio allocation
- Continuous control with policy gradients

**6. Real-Time Deployment**
- REST API for live inference
- Streaming data pipeline (Apache Kafka)
- Automated trade execution via exchange APIs

## Conclusion

This project successfully implemented a **production-grade deep learning trading system** for cryptocurrency markets, integrating:

1. Automated data pipeline with 30+ technical features
2. GRU forecasting models with walk-forward validation
3. Portfolio optimization and rigorous backtesting
4. Statistical hypothesis testing

### Key Findings

**Forecasting Performance:**
- GRU achieved **20% lower RMSE** than naive persistence (6.8% vs. 8.5%)
- Direction accuracy of **53.2%** above random baseline
- Custom loss function successfully balanced magnitude and directional errors

**Portfolio Performance:**
- GRU strategies showed **modest improvements over naive baseline** (not statistically significant)
- Both GRU and naive **underperformed capped benchmark** in raw returns
- GRU demonstrated **superior risk management**: 6.6pp lower max drawdown
- Performance varied by market regime: advantage in bear/sideways markets

**Statistical Validation:**
- Sharpe ratio difference between GRU and Naive **not statistically significant** (p = 0.18)
- Return distributions statistically indistinguishable
- Transaction costs (4.6% annual drag) major headwind

### Bottom Line

**The research question: "Do GRU forecasts translate into investable portfolio improvements?"**

**Answer:** **Partially, but not compellingly.**

While GRU forecasts were objectively superior to naive baselines, this advantage **did not translate into statistically significant portfolio performance improvements** after accounting for:
- Transaction costs
- Portfolio optimization constraints
- Covariance estimation errors
- Benchmark concentration advantages in bull markets

However, the system demonstrated value through:
- **Risk management**: Consistently lower drawdowns
- **Regime-specific performance**: Outperformance in sideways/bear markets
- **Extensibility**: Production-ready framework for future research

### Broader Lessons

This project reinforces several critical principles for financial machine learning:

1. **Forecasting accuracy ≠ Trading profitability**
   - Better predictions don't automatically mean better portfolios
   - Transaction costs and optimization objectives matter enormously

2. **Walk-forward validation is non-negotiable**
   - Standard train/test splits produce misleading results
   - Quarterly retraining necessary to maintain performance

3. **Risk-adjusted metrics are paramount**
   - Raw returns can be deceptive
   - Sharpe ratio, max drawdown, and statistical significance essential

4. **Infrastructure and MLOps matter**
   - Clean pipelines enable rapid experimentation
   - Reproducibility and version control critical for research integrity

5. **Markets are harder than Kaggle**
   - Academic benchmarks don't include transaction costs
   - Real trading involves constraints absent from competitions

### Academic Contribution

This work demonstrates how to **properly evaluate deep learning trading systems** using:
- Temporal validation preventing data leakage
- Statistical hypothesis testing for significance
- Production-grade MLOps infrastructure
- End-to-end pipeline from data to portfolio

The complete codebase, documentation, and this research paper serve as a **template for rigorous financial ML research**.

## Resources

### Project Components

- **[Kallos Data Pipeline](/projects/kallos-data-pipeline/)**: ETL infrastructure and feature engineering
- **[Kallos Models](/projects/kallos-models-mlops/)**: GRU training and hyperparameter optimization
- **[Kallos Portfolios](/projects/kallos-portfolios/)**: Portfolio construction and backtesting

### Code Repositories

- [github.com/josemarquezjaramillo/kallos-data](https://github.com/josemarquezjaramillo/kallos-data)
- [github.com/josemarquezjaramillo/kallos_models](https://github.com/josemarquezjaramillo/kallos_models)
- [github.com/josemarquezjaramillo/kallos_portfolios](https://github.com/josemarquezjaramillo/kallos_portfolios)

### Full Research Paper

**[Download Complete Paper (PDF)](/assets/Evaluating GRU powered trading systems - the case of cryptocurrency markets by Jose Marquez Jaramillo.pdf)**

**Citation:**
```
Márquez Jaramillo, J. (2024). Evaluating GRU-Powered Trading Systems:
The Case of Cryptocurrency Markets. Johns Hopkins University,
EN.649.740 Machine Learning: Deep Learning.
```

### Technologies Used

**Data & Infrastructure:**
- PostgreSQL (time-series data storage)
- CoinGecko API (market data source)
- SQLAlchemy (async ORM)
- pandas (data manipulation)

**Machine Learning:**
- PyTorch (deep learning framework)
- Darts (time-series forecasting)
- Optuna (hyperparameter optimization)
- scikit-learn (preprocessing)

**Portfolio & Backtesting:**
- PyPortfolioOpt (optimization)
- vectorbt (backtesting engine)
- QuantStats (performance analytics)
- NumPy (numerical computing)

**MLOps:**
- PyTorch Lightning (training infrastructure)
- python-dotenv (configuration management)
- pytest (testing framework)
- Git/GitHub (version control)

---

*This project was completed as the capstone for EN.649.740 Machine Learning: Deep Learning at Johns Hopkins University. It demonstrates production-grade MLOps practices, rigorous statistical validation, and the practical challenges of translating forecasting improvements into investable alpha.*
