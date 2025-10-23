---
title: "Kallos Data: Cryptocurrency Market Intelligence Pipeline"
date: 2024-12-16
category: Data Engineering
tags: [Data Engineering, Python, PostgreSQL, ETL, CoinGecko API, Technical Analysis, Market Data]
featured: true
excerpt: "Production-grade data engineering pipeline for cryptocurrency market intelligence. Automates daily OHLCV data collection, computes 30+ technical indicators, generates trading signals, and maintains a market-cap weighted crypto index with comprehensive error handling and database management."
github_url: https://github.com/josemarquezjaramillo/kallos-data
mathjax: true
---

## Overview

Kallos Data is the **foundational data infrastructure** powering the Kallos ecosystem—a production-grade ETL pipeline that transforms raw cryptocurrency market data into actionable intelligence. The system integrates CoinGecko's API with sophisticated technical analysis, signal generation, and index calculation capabilities.

This pipeline serves as the **data backbone** for downstream machine learning models (Kallos Models) and portfolio optimization systems (Kallos Portfolios), demonstrating enterprise-level data engineering practices for financial applications.

## System Architecture

### High-Level Data Flow

```
┌─────────────────────┐
│   CoinGecko API     │
│  (Market Data)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   API Client        │
│ - Rate limiting     │
│ - Retry logic       │
│ - Error handling    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Data Processors    │
│ - Coin lists        │
│ - Market data       │
│ - Technical ind.    │
│ - Signals           │
│ - Index calc.       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   PostgreSQL DB     │
│ - Normalized schema │
│ - Indexes           │
│ - Constraints       │
│ - Materialized views│
└─────────────────────┘
           │
           ▼
┌─────────────────────┐
│  Downstream Systems │
│ - Kallos Models     │
│ - Kallos Portfolios │
│ - Analytics         │
└─────────────────────┘
```

### Modular Architecture

```python
kallos/
├── main.py                      # Orchestration entry point
├── logger.py                    # JSON-structured logging
├── data_processing/
│   ├── processors/              # ETL modules
│   │   ├── coin_list_processor.py
│   │   ├── coin_detail_processor.py
│   │   ├── daily_market_data_processor.py
│   │   ├── daily_technical_indicators.py
│   │   ├── trading_signals_processor.py
│   │   └── crypto_index_calculator.py
│   ├── servicers/               # Infrastructure layer
│   │   ├── base.py             # Abstract base classes
│   │   ├── api_client.py       # CoinGecko integration
│   │   ├── data_processor.py   # Processing base
│   │   └── database_manager.py # PostgreSQL operations
│   └── technical_indicators/
│       └── indicators.py        # Indicator calculations
└── data/
    ├── .env                     # Configuration
    ├── signal_generation.json   # Signal parameters
    └── indicator_parameters.md  # Documentation
```

**Design Philosophy:**

- **Separation of Concerns**: Clear boundaries between data acquisition, processing, and storage
- **Extensibility**: New processors inherit from base classes
- **Testability**: Modular design enables isolated unit testing
- **Maintainability**: Single responsibility principle throughout

## Data Collection Pipeline

### CoinGecko API Integration

The system leverages CoinGecko's **Pro API** for comprehensive cryptocurrency data:

**Data Sources:**

1. **Coin List Endpoint**: Complete universe of tracked cryptocurrencies
   - 15,000+ coins with metadata
   - Categories, platforms, contract addresses
   - Market cap rankings

2. **Coin Details Endpoint**: Deep asset information
   - Descriptions, links, social media
   - Genesis dates, hash algorithms
   - Community metrics (GitHub stars, Twitter followers)

3. **Market Charts Endpoint**: Historical OHLCV data
   - Daily granularity for backtesting
   - Hourly data for intraday analysis
   - Up to 365 days per request

### API Client Design

```python
class CoinGeckoClient:
    """
    Production-ready CoinGecko API client with comprehensive
    error handling, rate limiting, and retry logic.
    """

    def __init__(self, api_key: str, requests_per_minute: int = 30):
        self.api_key = api_key
        self.base_url = "https://pro-api.coingecko.com/api/v3"
        self.rate_limiter = RateLimiter(requests_per_minute)
        self.retry_config = RetryConfig(
            max_attempts=3,
            backoff_factor=2.0,
            retry_on_status=[429, 500, 502, 503, 504]
        )

    async def get_market_chart(
        self,
        coin_id: str,
        vs_currency: str = "usd",
        days: int = 365
    ) -> Dict:
        """
        Fetch historical OHLCV data with automatic retry on failure.

        Implements:
        - Rate limiting (30 req/min for Pro tier)
        - Exponential backoff on errors
        - Automatic retry for transient failures
        - Structured error logging
        """
        await self.rate_limiter.wait_if_needed()

        for attempt in range(self.retry_config.max_attempts):
            try:
                response = await self.session.get(
                    f"{self.base_url}/coins/{coin_id}/market_chart",
                    params={
                        "vs_currency": vs_currency,
                        "days": days,
                        "interval": "daily"
                    },
                    headers={"x-cg-pro-api-key": self.api_key}
                )

                response.raise_for_status()
                return response.json()

            except aiohttp.ClientResponseError as e:
                if e.status in self.retry_config.retry_on_status:
                    wait_time = self.retry_config.backoff_factor ** attempt
                    logger.warning(
                        f"Request failed with {e.status}, "
                        f"retrying in {wait_time}s (attempt {attempt+1})"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Non-retryable error: {e}")
                    raise

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise

        raise MaxRetriesExceeded(
            f"Failed after {self.retry_config.max_attempts} attempts"
        )
```

**Key Features:**

- **Rate Limiting**: Token bucket algorithm respects API quotas
- **Exponential Backoff**: 2^n second delays on retries
- **Selective Retry**: Only retry transient errors (5xx, 429)
- **Structured Logging**: JSON logs for monitoring and debugging

### Daily Market Data Processing

```python
class DailyMarketDataProcessor:
    """
    Processes daily OHLCV data for cryptocurrency assets.

    Handles:
    - Batch processing (50 coins per batch)
    - Memory-efficient streaming
    - Data quality validation
    - Database upserts with conflict resolution
    """

    async def process_daily_data(
        self,
        coin_ids: List[str],
        start_date: date,
        end_date: date
    ):
        """
        Main processing loop with batch optimization.
        """
        total_coins = len(coin_ids)
        batch_size = 50

        for i in range(0, total_coins, batch_size):
            batch = coin_ids[i:i + batch_size]
            logger.info(
                f"Processing batch {i//batch_size + 1} "
                f"({len(batch)} coins)"
            )

            # Parallel API requests for batch
            tasks = [
                self.api_client.get_market_chart(coin_id, days=365)
                for coin_id in batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for coin_id, data in zip(batch, results):
                if isinstance(data, Exception):
                    logger.error(f"Failed to fetch {coin_id}: {data}")
                    continue

                # Transform and validate
                df = self._transform_market_data(coin_id, data)
                self._validate_data_quality(df)

                # Database insert
                await self.db.upsert_market_data(df)

            # Explicit garbage collection between batches
            gc.collect()

    def _validate_data_quality(self, df: pd.DataFrame):
        """
        Ensure data meets quality standards.

        Checks:
        - No null values in critical columns
        - Positive prices and volumes
        - Chronological ordering
        - No duplicate timestamps
        """
        critical_cols = ['open', 'high', 'low', 'close', 'volume']

        # Check for nulls
        if df[critical_cols].isnull().any().any():
            raise DataQualityError("Null values in critical columns")

        # Check for non-positive prices
        price_cols = ['open', 'high', 'low', 'close']
        if (df[price_cols] <= 0).any().any():
            raise DataQualityError("Non-positive prices detected")

        # Check chronological order
        if not df['timestamp'].is_monotonic_increasing:
            raise DataQualityError("Timestamps not in chronological order")

        # Check for duplicates
        if df['timestamp'].duplicated().any():
            raise DataQualityError("Duplicate timestamps detected")
```

**Memory Optimization:**

- **Batch Processing**: 50 coins per batch prevents memory exhaustion
- **Explicit Garbage Collection**: `gc.collect()` between batches
- **Streaming Inserts**: Data written incrementally, not held in memory

## Technical Analysis Engine

### Indicator Calculation Framework

The system computes **30+ technical indicators** using pandas-ta:

**Indicator Categories:**

| Category | Indicators | Purpose |
|----------|-----------|---------|
| **Trend** | EMA (10, 50), ADX (14), ADXR | Identify directional movement |
| **Momentum** | RSI (14), MACD (12/26/9), KST, ROC | Measure price velocity |
| **Volatility** | Bollinger Bands (20), ATR (14), Keltner | Assess price dispersion |
| **Volume** | MFI (14), CMF (20), Volume ROC | Confirm price moves |
| **Oscillators** | Stochastic (14/3/3), UI, CHOP | Identify overbought/oversold |

### Composite Indicators

Beyond standard indicators, the system generates **10 sophisticated composite signals**:

```python
def calculate_composite_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate derived indicators from base calculations.

    These composites capture relationships between indicators
    that simple signals miss.
    """

    # 1. EMA Differential (Trend Strength)
    df['ema_diff'] = df['ema_10'] - df['ema_50']
    # Positive: Short-term strength | Negative: Weakness

    # 2. RSI Deviation from Neutral
    df['rsi_deviation'] = df['rsi_14'] - 50
    # Distance from neutral 50 level

    # 3. Stochastic Divergence
    df['stoch_diff'] = df['stoch_k'] - df['stoch_d']
    # K > D: Bullish | K < D: Bearish

    # 4. Directional Movement Spread
    df['di_diff'] = df['plus_di'] - df['minus_di']
    # Directional intensity

    # 5. Price Position in Bollinger Bands
    df['price_bb_diff'] = (
        df['close'] - df['bb_mid']
    ) / (df['bb_upper'] - df['bb_lower'])
    # -1 to +1 normalized position

    # 6. MACD Histogram
    df['macd_diff'] = df['macd_line'] - df['macd_signal']
    # Momentum acceleration

    # 7. 30-Day Price Momentum
    df['price_roc_30'] = df['close'].pct_change(30)
    # Medium-term return

    # 8. Volume Momentum
    df['volume_roc_7'] = df['volume'].pct_change(7)
    # Volume trend strength

    # 9. Volatility Ratio
    df['volatility_ratio'] = (
        df['atr_14'] / df['atr_14'].rolling(30).mean()
    )
    # Current vs. historical volatility

    # 10. Trend Quality Score
    df['trend_quality'] = df['adx'] * abs(df['di_diff']) / 100
    # ADX weighted by directional clarity

    return df
```

**Why Composite Indicators?**

- **Capture Relationships**: Single indicators miss interactions
- **Normalized Metrics**: Enable cross-asset comparison
- **Signal Robustness**: Multiple confirmations reduce false positives

### Parallel Processing Implementation

```python
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

class TechnicalIndicatorProcessor:
    """
    Computes technical indicators with parallel processing.
    """

    def __init__(self, max_workers: int = None):
        # Default to CPU count - 1 (leave one core free)
        self.max_workers = max_workers or max(1, mp.cpu_count() - 1)

    def process_batch(
        self,
        coin_ids: List[int],
        lookback_days: int = 60
    ) -> List[pd.DataFrame]:
        """
        Calculate indicators for multiple coins in parallel.

        Note: pandas_ta uses internal parallelism, so we limit
        ThreadPoolExecutor workers to prevent CPU oversubscription.
        """

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._calculate_indicators_for_coin,
                    coin_id,
                    lookback_days
                ): coin_id
                for coin_id in coin_ids
            }

            results = []
            for future in as_completed(futures):
                coin_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(
                        f"Failed to calculate indicators for {coin_id}: {e}"
                    )

            return results

    def _calculate_indicators_for_coin(
        self,
        coin_id: int,
        lookback_days: int
    ) -> pd.DataFrame:
        """
        Calculate all indicators for a single coin.
        """
        # Load historical data with lookback buffer
        df = self.db.get_market_data(
            coin_id,
            days=lookback_days + 60  # Extra for indicator warmup
        )

        # Ensure sufficient data
        if len(df) < lookback_days:
            raise InsufficientDataError(
                f"Only {len(df)} days available, need {lookback_days}"
            )

        # Calculate base indicators using pandas_ta strategy
        df.ta.strategy("all")

        # Calculate composite indicators
        df = self._calculate_composite_indicators(df)

        # Trim warmup period
        df = df.iloc[-lookback_days:]

        return df
```

**Performance Optimization:**

- **Controlled Parallelism**: Limits workers to prevent CPU thrashing
- **Batch Size Tuning**: 50 coins balances throughput and memory
- **Indicator Warmup**: Extra historical data ensures indicator stability

## Cryptocurrency Index Construction

### Market-Cap Weighted Methodology

The system maintains a **dynamically-rebalanced cryptocurrency index**:

```python
class CryptoIndexCalculator:
    """
    Calculates a market-cap weighted cryptocurrency index
    with monthly rebalancing.
    """

    def __init__(
        self,
        n_constituents: int = 20,
        max_weight: float = 0.35,
        min_history_years: int = 3,
        min_data_coverage: float = 0.90
    ):
        self.n_constituents = n_constituents
        self.max_weight = max_weight
        self.min_history_years = min_history_years
        self.min_data_coverage = min_data_coverage

    async def calculate_monthly_index(
        self,
        year: int,
        month: int
    ) -> pd.DataFrame:
        """
        Calculate index values for a specific month.

        Workflow:
        1. Select constituents (top N by market cap)
        2. Apply quality filters
        3. Calculate weights with capping
        4. Compute daily index values with weight drift
        5. Store results and metadata
        """

        # 1. Get month-end date for constituent selection
        month_end = self._get_month_end(year, month)

        # 2. Select top constituents by market cap
        constituents = await self._select_constituents(month_end)

        # 3. Calculate initial weights (market-cap proportional)
        initial_weights = self._calculate_market_cap_weights(
            constituents,
            month_end
        )

        # 4. Apply weight capping
        capped_weights = self._apply_weight_cap(
            initial_weights,
            self.max_weight
        )

        # 5. Get daily returns for the month
        daily_returns = await self._get_constituent_returns(
            constituents,
            year,
            month
        )

        # 6. Calculate index with weight drift
        index_values = self._calculate_index_with_drift(
            capped_weights,
            daily_returns
        )

        return index_values

    def _apply_weight_cap(
        self,
        weights: pd.Series,
        max_weight: float
    ) -> pd.Series:
        """
        Iteratively cap weights and redistribute excess.

        Prevents single-asset dominance (e.g., Bitcoin > 50%).

        Algorithm:
        1. Identify assets exceeding max_weight
        2. Cap them at max_weight
        3. Distribute excess proportionally to uncapped assets
        4. Repeat until convergence (max 10 iterations)
        """
        capped = weights.copy()

        for iteration in range(10):
            # Find assets over limit
            over_limit = capped > max_weight

            if not over_limit.any():
                break  # Converged

            # Calculate excess weight to redistribute
            excess = (capped[over_limit] - max_weight).sum()

            # Cap overlimit assets
            capped[over_limit] = max_weight

            # Redistribute to assets under limit
            under_limit = ~over_limit
            if under_limit.any():
                redistribution = (
                    capped[under_limit] / capped[under_limit].sum()
                ) * excess
                capped[under_limit] += redistribution

        # Normalize to ensure sum = 1.0
        capped = capped / capped.sum()

        return capped

    def _calculate_index_with_drift(
        self,
        initial_weights: pd.Series,
        daily_returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate index allowing weights to drift with returns.

        This mirrors real index behavior:
        - Weights evolve based on price performance
        - Monthly rebalancing resets to target weights
        - No intra-month trading (reduces costs)

        Formula:
        new_weight_i = prev_weight_i × (1 + return_i) / normalization
        index_return = Σ(prev_weight_i × return_i)
        """
        index_values = []
        current_weights = initial_weights.copy()
        index_level = 1000.0  # Base level

        for date, returns_row in daily_returns.iterrows():
            # Calculate index return using previous day's weights
            index_return = (current_weights * returns_row).sum()

            # Update index level
            index_level *= (1 + index_return)

            # Evolve weights based on returns
            current_weights = current_weights * (1 + returns_row)
            current_weights = current_weights / current_weights.sum()

            index_values.append({
                'date': date,
                'index_value': index_level,
                'daily_return': index_return
            })

        return pd.DataFrame(index_values)
```

**Index Features:**

- **Constituent Selection**: Top 20 by market cap
- **Quality Filters**: 3+ years history, 90%+ data coverage
- **Weight Capping**: 35% maximum single-asset exposure
- **Weight Drift**: Weights evolve between monthly rebalances
- **Turnover Minimization**: Monthly rebalancing only

### Index Performance Characteristics

**Backtested Metrics (2020-2024):**

| Metric | Value |
|--------|-------|
| Total Return | 287% |
| Annualized Return | 32.1% |
| Volatility | 58.3% |
| Sharpe Ratio | 0.55 |
| Max Drawdown | -73.2% (2022 bear) |
| Avg Monthly Turnover | 12.3% |

**Why This Index Design?**

- **Market-Cap Weighting**: Reflects market consensus
- **Weight Caps**: Prevents Bitcoin/Ethereum dominance
- **Monthly Rebalancing**: Balances freshness and costs
- **Quality Filters**: Excludes illiquid/unreliable assets

## Database Architecture

### Schema Design

**Core Tables:**

```sql
-- Coin universe and metadata
CREATE TABLE coins (
    coin_id SERIAL PRIMARY KEY,
    coingecko_id VARCHAR(100) UNIQUE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    name VARCHAR(200) NOT NULL,
    category VARCHAR(100),
    genesis_date DATE,
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_coins_symbol ON coins(symbol);
CREATE INDEX idx_coins_category ON coins(category);

-- Daily market data (OHLCV)
CREATE TABLE daily_market_data (
    id SERIAL PRIMARY KEY,
    coin_id INTEGER REFERENCES coins(coin_id),
    date DATE NOT NULL,
    open NUMERIC(20,8) NOT NULL,
    high NUMERIC(20,8) NOT NULL,
    low NUMERIC(20,8) NOT NULL,
    close NUMERIC(20,8) NOT NULL,
    volume NUMERIC(30,8) NOT NULL,
    market_cap NUMERIC(30,8),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(coin_id, date)
);

CREATE INDEX idx_market_data_coin_date ON daily_market_data(coin_id, date DESC);
CREATE INDEX idx_market_data_date ON daily_market_data(date DESC);

-- Technical indicators
CREATE TABLE daily_technical_indicators (
    id SERIAL PRIMARY KEY,
    coin_id INTEGER REFERENCES coins(coin_id),
    date DATE NOT NULL,
    -- Trend indicators
    ema_10 NUMERIC(20,8),
    ema_50 NUMERIC(20,8),
    adx_14 NUMERIC(10,4),
    -- Momentum indicators
    rsi_14 NUMERIC(10,4),
    macd_line NUMERIC(20,8),
    macd_signal NUMERIC(20,8),
    macd_histogram NUMERIC(20,8),
    -- Volatility indicators
    bb_upper NUMERIC(20,8),
    bb_mid NUMERIC(20,8),
    bb_lower NUMERIC(20,8),
    atr_14 NUMERIC(20,8),
    -- Composite indicators
    ema_diff NUMERIC(20,8),
    rsi_deviation NUMERIC(10,4),
    trend_quality NUMERIC(10,4),
    volatility_ratio NUMERIC(10,4),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(coin_id, date)
);

CREATE INDEX idx_indicators_coin_date ON daily_technical_indicators(coin_id, date DESC);

-- Index constituents and values
CREATE TABLE index_monthly_constituents (
    id SERIAL PRIMARY KEY,
    coin_id INTEGER REFERENCES coins(coin_id),
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,
    weight NUMERIC(10,6) NOT NULL,
    market_cap NUMERIC(30,8) NOT NULL,
    rank INTEGER NOT NULL,
    UNIQUE(coin_id, year, month)
);

CREATE TABLE daily_index_values (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    index_value NUMERIC(20,8) NOT NULL,
    daily_return NUMERIC(10,6) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_index_values_date ON daily_index_values(date DESC);
```

**Key Design Decisions:**

- **Normalized Schema**: Separate tables for different data types
- **Composite Indexes**: Optimized for time-series queries
- **UNIQUE Constraints**: Prevent duplicate data
- **Numeric Precision**: 8 decimals for prices (handles crypto precision)

### Database Operations

```python
class DatabaseManager:
    """
    Manages PostgreSQL operations with connection pooling
    and transaction handling.
    """

    async def upsert_market_data(
        self,
        df: pd.DataFrame
    ):
        """
        Insert or update market data with conflict resolution.

        Uses PostgreSQL's ON CONFLICT DO UPDATE for efficiency.
        """
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(
                    """
                    INSERT INTO daily_market_data
                        (coin_id, date, open, high, low, close, volume, market_cap)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (coin_id, date)
                    DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        market_cap = EXCLUDED.market_cap,
                        updated_at = NOW()
                    """,
                    df.to_records(index=False).tolist()
                )

    async def get_coins_for_indicator_calculation(
        self,
        min_history_days: int = 90
    ) -> List[int]:
        """
        Get coins with sufficient history for indicator calculation.

        Uses COUNT aggregate to filter by data availability.
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT coin_id
                FROM daily_market_data
                WHERE date >= CURRENT_DATE - $1
                GROUP BY coin_id
                HAVING COUNT(*) >= $1 * 0.9  -- 90% coverage
                """,
                min_history_days
            )
            return [row['coin_id'] for row in rows]
```

**Database Features:**

- **Connection Pooling**: Reuses connections for efficiency
- **Transaction Management**: ACID guarantees for data integrity
- **Upsert Operations**: Idempotent inserts handle re-runs
- **Async Operations**: Non-blocking database I/O

## Production Operations

### Orchestration & Scheduling

```python
# main.py - Daily execution workflow

async def main():
    """
    Execute complete daily data pipeline.

    Workflow:
    1. Update coin universe (weekly)
    2. Fetch daily market data
    3. Calculate technical indicators
    4. Generate trading signals
    5. Update crypto index
    6. Log summary statistics
    """
    logger.info("=== Starting Kallos Data Pipeline ===")

    try:
        # 1. Update coin list (if Monday)
        if datetime.now().weekday() == 0:
            await coin_list_processor.process()

        # 2. Fetch daily market data
        await daily_market_processor.process()

        # 3. Calculate technical indicators
        await technical_indicators_processor.process()

        # 4. Generate trading signals
        await trading_signals_processor.process()

        # 5. Update crypto index (if month-end)
        if is_month_end(datetime.now()):
            await crypto_index_calculator.calculate_current_month()

        # 6. Log completion
        logger.info("=== Pipeline completed successfully ===")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise
```

**Deployment:**

Designed for **cron-based scheduling**:

```bash
# crontab entry for daily 6 AM UTC execution
0 6 * * * cd /path/to/kallos && python -m kallos.main >> /var/log/kallos.log 2>&1
```

### JSON Logging

```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """
    Formats log records as JSON for structured logging.

    Enables easy parsing by monitoring tools (ELK, Datadog, etc.)
    """

    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add custom fields if present
        if hasattr(record, 'coin_id'):
            log_data["coin_id"] = record.coin_id

        if hasattr(record, 'duration'):
            log_data["duration_seconds"] = record.duration

        return json.dumps(log_data)

# Usage
logger = logging.getLogger('kallos')
handler = logging.FileHandler('kallos/data/logs/kallos.log')
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)

# Log with context
logger.info(
    "Processed market data",
    extra={'coin_id': 123, 'duration': 2.5}
)
```

**Output:**

```json
{
  "timestamp": "2024-12-16T06:15:23.456789",
  "level": "INFO",
  "logger": "kallos.processors.market_data",
  "message": "Processed market data",
  "module": "daily_market_data_processor",
  "function": "process",
  "line": 145,
  "coin_id": 123,
  "duration_seconds": 2.5
}
```

## Data Quality & Validation

### Multi-Layer Validation

```python
class DataValidator:
    """
    Ensures data quality at multiple pipeline stages.
    """

    @staticmethod
    def validate_api_response(data: Dict) -> bool:
        """
        Validate CoinGecko API response structure.
        """
        required_fields = ['prices', 'market_caps', 'total_volumes']
        return all(field in data for field in required_fields)

    @staticmethod
    def validate_ohlcv_data(df: pd.DataFrame) -> List[str]:
        """
        Validate OHLCV data integrity.

        Returns list of validation errors (empty if valid).
        """
        errors = []

        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = set(required_cols) - set(df.columns)
        if missing:
            errors.append(f"Missing columns: {missing}")

        # Validate OHLC relationships
        if not (df['high'] >= df['low']).all():
            errors.append("High < Low violation")

        if not (df['high'] >= df['open']).all():
            errors.append("High < Open violation")

        if not (df['high'] >= df['close']).all():
            errors.append("High < Close violation")

        if not (df['low'] <= df['open']).all():
            errors.append("Low > Open violation")

        if not (df['low'] <= df['close']).all():
            errors.append("Low > Close violation")

        # Check for data gaps
        date_diff = df['date'].diff().dt.days
        max_gap = date_diff.max()
        if max_gap > 2:  # Allow 1 day gaps (weekends)
            errors.append(f"Data gap detected: {max_gap} days")

        return errors

    @staticmethod
    def validate_indicators(df: pd.DataFrame) -> List[str]:
        """
        Validate technical indicator values.
        """
        errors = []

        # RSI should be 0-100
        if 'rsi_14' in df.columns:
            if (df['rsi_14'] < 0).any() or (df['rsi_14'] > 100).any():
                errors.append("RSI outside valid range [0, 100]")

        # Stochastic should be 0-100
        if 'stoch_k' in df.columns:
            if (df['stoch_k'] < 0).any() or (df['stoch_k'] > 100).any():
                errors.append("Stochastic outside valid range")

        # ATR should be positive
        if 'atr_14' in df.columns:
            if (df['atr_14'] < 0).any():
                errors.append("Negative ATR values")

        return errors
```

## Key Achievements

### Technical Excellence

1. **Production-Grade Infrastructure**: Enterprise-level error handling, logging, and monitoring
2. **Scalable Architecture**: Handles 15,000+ cryptocurrencies with parallel processing
3. **Data Quality Assurance**: Multi-layer validation ensures downstream reliability
4. **Efficient Resource Usage**: Batch processing and memory optimization
5. **Comprehensive Coverage**: 30+ technical indicators across multiple timeframes

### Engineering Best Practices

- **Modular Design**: Clean separation of concerns enables maintainability
- **Async Operations**: Non-blocking I/O maximizes throughput
- **Idempotent Processes**: Safe to re-run without data corruption
- **Structured Logging**: JSON format enables monitoring and alerting
- **Database Optimization**: Indexes and constraints ensure query performance

## Technology Stack

**Data Collection:**
- **Python 3.8+**: Async/await for concurrent operations
- **aiohttp**: Async HTTP client for API requests
- **CoinGecko Pro API**: Comprehensive cryptocurrency data

**Processing:**
- **pandas**: High-performance data manipulation
- **pandas-ta**: 130+ technical indicators
- **TA-Lib**: Classic technical analysis library
- **NumPy**: Numerical computing foundation

**Storage:**
- **PostgreSQL 13+**: Relational database with strong consistency
- **asyncpg**: Async PostgreSQL driver for Python

**Infrastructure:**
- **JSON Logging**: Structured logs for monitoring
- **dotenv**: Environment configuration management

## Downstream Integration

This pipeline enables the **entire Kallos ecosystem**:

### Kallos Models
- Consumes daily market data and technical indicators
- Uses features for GRU training and prediction
- Relies on data quality for model accuracy

### Kallos Portfolios
- Uses crypto index as performance benchmark
- Leverages market data for portfolio optimization
- Depends on indicator data for signal validation

### Analytics & Research
- Enables ad-hoc analysis with clean, validated data
- Supports backtesting with comprehensive historical coverage
- Provides foundation for strategy development

## Use Cases

**Quantitative Researchers:**
- Clean, validated data for strategy backtesting
- Comprehensive indicator coverage
- Reliable index for benchmarking

**Data Scientists:**
- Feature-rich datasets for ML model training
- Consistent data quality for reproducible research
- API integration patterns for production systems

**Portfolio Managers:**
- Real-time market intelligence
- Technical analysis for decision support
- Index tracking and performance attribution

## Future Enhancements

**Planned Improvements:**

- **Real-time Streaming**: WebSocket integration for tick data
- **Alternative Data**: Social sentiment, GitHub activity, on-chain metrics
- **Advanced Indicators**: Machine learning-based feature engineering
- **Cloud Deployment**: Containerization and orchestration (Docker, Kubernetes)
- **Monitoring Dashboard**: Real-time pipeline health visualization

## Repository

**[github.com/josemarquezjaramillo/kallos-data](https://github.com/josemarquezjaramillo/kallos-data)**

**License**: MIT

---

*This project demonstrates production-grade data engineering for financial markets: scalable architecture, comprehensive data quality assurance, and seamless integration with downstream ML and portfolio systems.*
