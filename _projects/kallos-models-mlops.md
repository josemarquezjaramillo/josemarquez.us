---
title: "Kallos Models: MLOps Framework for Cryptocurrency Forecasting"
date: 2024-12-15
category: Machine Learning
tags: [MLOps, Python, Deep Learning, Time Series, Cryptocurrency, PyTorch, Optuna, Darts]
featured: true
excerpt: "Production-grade MLOps framework for training and deploying deep learning time-series models for cryptocurrency price prediction. Features walk-forward validation, multi-objective optimization, and end-to-end CLI workflow."
github_url: https://github.com/josemarquezjaramillo/kallos_models
mathjax: true
---

## Overview

Kallos Models is a sophisticated Python package implementing a complete MLOps workflow for cryptocurrency price forecasting using deep learning. The framework emphasizes production-ready practices including proper time-series validation, hyperparameter optimization, and modular architecture designed for real-world deployment.

Unlike typical academic implementations, this project addresses the full machine learning lifecycle: data loading, preprocessing, model training, hyperparameter tuning, and rigorous evaluation—all with proper handling of temporal dependencies critical to financial forecasting.

## Key Features

### Production-Grade MLOps Workflow

The system implements a **three-step pipeline** designed to prevent common pitfalls in time-series modeling:

1. **Hyperparameter Tuning**: Walk-forward cross-validation with Optuna
2. **Model Training**: Production model building with optimal parameters
3. **Hold-Out Evaluation**: Assessment on completely unseen data

This separation ensures **no data leakage** between tuning, training, and testing phases—a critical requirement often overlooked in financial ML projects.

### Walk-Forward Validation

Traditional k-fold cross-validation violates temporal ordering in time series data. Kallos Models implements **walk-forward splits** that respect causality:

```python
# Pseudocode representation of the approach
for split in walk_forward_splits(data, n_splits):
    train_data = data[:split.train_end]
    val_data = data[split.val_start:split.val_end]

    # Train only on past data
    model.fit(train_data)

    # Validate on immediate future
    predictions = model.predict(val_data)
    metrics.append(evaluate(predictions, val_data))
```

This methodology mirrors real-world trading scenarios where models must predict the immediate future using only historical information.

### Multi-Objective Optimization

The framework uses Optuna's multi-objective optimization to balance competing goals:

**Objective 1: Prediction Accuracy**
$$
\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
$$

**Objective 2: Directional Accuracy**
$$
\text{Direction Accuracy} = \frac{1}{n}\sum_{i=1}^{n} \mathbb{1}[\text{sign}(y_i - y_{i-1}) = \text{sign}(\hat{y}_i - y_{i-1})]
$$

This dual optimization recognizes that in trading applications, **correctly predicting price direction** is often more valuable than minimizing absolute error.

## Technical Architecture

### Supported Models

The framework provides a factory pattern supporting three state-of-the-art architectures via the Darts library:

**1. GRU (Gated Recurrent Unit)**
- Efficient RNN variant with gating mechanisms
- Lower computational cost than LSTM
- Effective for medium-length sequences

**2. LSTM (Long Short-Term Memory)**
- Classic architecture for sequential data
- Captures long-term dependencies
- Proven track record in financial forecasting

**3. Transformer**
- Attention-based architecture
- Parallelizable training
- Superior performance on complex patterns

### Custom Loss Function

The framework implements a **Direction-Selective MSE Loss** that penalizes directional errors more heavily than magnitude errors:

$$
\mathcal{L} = \text{MSE} + \lambda \cdot \text{DirectionPenalty}
$$

Where $\lambda$ is a tunable hyperparameter controlling the trade-off between accuracy and directional correctness.

### Model Architecture Factory

```python
def create_model(architecture, params, pl_trainer_kwargs):
    """
    Factory function for instantiating forecasting models.

    Abstracts differences between RNN and Transformer architectures,
    providing a unified interface for model creation.
    """
    if architecture in ['gru', 'lstm']:
        # RNN-based models
        model = BlockRNNModel(
            model=architecture.upper(),
            input_chunk_length=params['input_chunk_length'],
            output_chunk_length=1,  # Single-step forecasting
            hidden_dim=params['hidden_dim'],
            n_rnn_layers=params['n_rnn_layers'],
            dropout=params['dropout'],
            optimizer_kwargs={'lr': params['learning_rate']},
            loss_fn=DirectionSelectiveMSELoss(
                lambda_param=params['lambda']
            ),
            pl_trainer_kwargs=pl_trainer_kwargs,
            random_state=42
        )
    elif architecture == 'transformer':
        # Attention-based model
        model = TransformerModel(
            input_chunk_length=params['input_chunk_length'],
            output_chunk_length=1,
            d_model=params['hidden_dim'],  # Parameter normalization
            nhead=params['n_heads'],
            num_encoder_layers=params['n_encoder_layers'],
            dropout=params['dropout'],
            # ... additional configuration
        )

    return model
```

**Key Design Decisions:**

- **Fixed random seed (42)**: Ensures reproducibility across runs
- **Single-step forecasting**: `output_chunk_length=1` optimized for next-value prediction
- **Parameter normalization**: Maps RNN `hidden_dim` to Transformer `d_model` for consistent interface
- **PyTorch Lightning integration**: Leverages enterprise-grade training infrastructure

## Hyperparameter Optimization

### Comprehensive Search Space

The Optuna tuner explores a carefully designed hyperparameter space:

| Parameter | Range | Distribution | Notes |
|-----------|-------|--------------|-------|
| `input_chunk_length` | 14-90 | Linear | Lookback window size |
| `hidden_dim` | 32-256 | Linear | Model capacity |
| `n_rnn_layers` | 1-4 | Integer | Network depth |
| `dropout` | 0.0-0.7 | Uniform | Regularization |
| `learning_rate` | 1e-5 to 1e-2 | Log scale | Optimization rate |
| `lambda` | 0.0-2.0 | Uniform | Direction penalty weight |
| `batch_size` | 16, 32, 64 | Categorical | Training batch size |
| `n_epochs` | 10, 25, 50 | Categorical | Training duration |

**Logarithmic scaling** for learning rate and lambda reflects their exponential impact on training dynamics.

### Persistent Study Management

```python
# Study persistence enables resumable optimization
study = optuna.create_study(
    study_name=f"{architecture}_optimization",
    storage=f"postgresql://user:pass@host/db",
    directions=['minimize', 'maximize'],  # Multi-objective
    load_if_exists=True  # Resume existing studies
)

# Check if already completed
if current_study_trials >= n_trials:
    print(f"Study already complete with {current_study_trials} trials")
    return study.best_params
```

This approach supports:
- **Resumable optimization**: Interrupted runs continue from last trial
- **Cross-session persistence**: Results survive system restarts
- **Collaborative tuning**: Multiple processes can contribute trials

## Data Pipeline

### Database Integration

The framework loads financial data directly from PostgreSQL, enabling:

- **Real-time data access**: Query latest market data
- **Efficient storage**: Normalized database schemas
- **Version control**: Track data lineage and updates

### Feature Engineering Pipeline

```python
class FeaturePreprocessor:
    """
    Modular preprocessing for different feature categories.

    Supports distinct transformations for:
    - Price features (log returns, normalization)
    - Technical indicators (scaling, clipping)
    - Market microstructure (differencing, standardization)
    """

    def __init__(self, feature_groups):
        self.pipelines = {}
        for group, features in feature_groups.items():
            self.pipelines[group] = self._build_pipeline(group)

    def fit_transform(self, data):
        # Apply category-specific transformations
        transformed = {}
        for group, pipeline in self.pipelines.items():
            transformed[group] = pipeline.fit_transform(
                data[self.feature_groups[group]]
            )
        return pd.concat(transformed.values(), axis=1)
```

**Sophisticated handling** of different feature types ensures appropriate scaling and normalization for heterogeneous financial data.

## Command-Line Interface

The framework provides an intuitive CLI for the complete workflow:

### Step 1: Hyperparameter Tuning

```bash
python main.py tune \
    --architecture lstm \
    --target btc_close \
    --covariates volume,sentiment,volatility \
    --n-trials 100 \
    --train-start 2020-01-01 \
    --train-end 2023-12-31 \
    --val-start 2024-01-01 \
    --val-end 2024-06-30
```

**Output**: `best_params.json` with optimal hyperparameters

### Step 2: Final Model Training

```bash
python main.py train \
    --architecture lstm \
    --params best_params.json \
    --output-dir models/lstm_btc_v1/
```

**Output**:
- `model.pt` - Trained PyTorch Lightning model
- `scaler.pkl` - Fitted preprocessing pipeline

### Step 3: Hold-Out Evaluation

```bash
python main.py evaluate \
    --model models/lstm_btc_v1/model.pt \
    --scaler models/lstm_btc_v1/scaler.pkl \
    --test-start 2024-07-01 \
    --test-end 2024-12-31 \
    --output-dir results/
```

**Output**:
- `metrics.json` - RMSE, MAE, MAPE, direction accuracy
- `forecast_vs_actual.png` - Visualization of predictions

## Configuration Management

### Flexible Configuration System

```json
{
  "database": {
    "host": "${DB_HOST}",
    "port": 5432,
    "database": "crypto_data",
    "credentials_env": true
  },
  "training": {
    "accelerator": "gpu",
    "devices": 1,
    "precision": "16-mixed"
  },
  "tuning": {
    "n_trials": 50,
    "timeout": 3600,
    "n_jobs": 4
  },
  "evaluation": {
    "metrics": ["rmse", "mae", "mape", "direction_accuracy"],
    "plot_format": "png"
  }
}
```

**Environment variable support** keeps sensitive credentials out of version control.

## Development Practices

### Code Quality Standards

- **PEP 8 Compliance**: Enforced via flake8 or black
- **Type Hints**: Comprehensive type annotations for maintainability
- **Docstrings**: NumPy-style documentation for all public APIs
- **Testing**: pytest framework for unit and integration tests

### GitFlow Branching Model

```
main        ← Production-ready code
  ↑
develop     ← Integration branch
  ↑
feature/*   ← New capabilities
hotfix/*    ← Emergency fixes
release/*   ← Version preparation
```

This structure supports **multiple concurrent developments** while maintaining stability.

## Technologies & Dependencies

### Core Stack

- **Python 3.8+**: Modern language features and type hints
- **Darts 0.21.0+**: Professional time-series forecasting library
- **PyTorch**: Deep learning framework with GPU acceleration
- **Optuna 3.0+**: State-of-the-art hyperparameter optimization
- **PostgreSQL**: Robust database for financial data storage

### Supporting Libraries

- **pandas**: High-performance data manipulation
- **scikit-learn**: Feature preprocessing pipelines
- **PyTorch Lightning**: Training infrastructure and callbacks
- **python-dotenv**: Environment variable management

## Key Achievements

### Engineering Excellence

1. **Production-Ready Architecture**: Modular design supporting extension and maintenance
2. **Temporal Integrity**: Proper handling of time-series dependencies throughout
3. **Multi-Objective Optimization**: Balances prediction accuracy with trading signal quality
4. **Persistent Infrastructure**: Database-backed studies and data management
5. **Clean Abstractions**: Factory patterns and unified interfaces across architectures

### MLOps Best Practices

- **Reproducibility**: Fixed random seeds and versioned dependencies
- **Experiment Tracking**: Optuna studies persist all trials and parameters
- **Model Versioning**: Explicit model and scaler checkpointing
- **Configuration Management**: Separation of code and configuration
- **Data Validation**: Hold-out evaluation prevents overfitting

## Use Cases

This framework is designed for:

- **Quantitative researchers** building cryptocurrency trading models
- **Data scientists** requiring rigorous time-series validation
- **ML engineers** deploying production forecasting systems
- **Academic researchers** studying financial time-series prediction

## Future Enhancements

Potential extensions include:

- **Ensemble methods**: Combining multiple architectures
- **Online learning**: Continuous model updates with new data
- **Multi-asset forecasting**: Cross-cryptocurrency predictions
- **Risk metrics**: VaR and CVaR alongside point forecasts
- **Production deployment**: REST API and real-time inference

## Lessons Learned

Building this framework reinforced several critical principles:

1. **Time-series validation is non-negotiable**: Standard ML validation creates data leakage
2. **Multi-objective optimization matters**: Single metrics miss important trade-offs
3. **Persistence enables experimentation**: Database-backed studies support iterative improvement
4. **Clean architecture scales**: Modular design simplifies debugging and extension
5. **Configuration management is critical**: Separating code from settings enables flexibility

## Repository

The complete source code, documentation, and setup instructions are available on GitHub:

**[github.com/josemarquezjaramillo/kallos_models](https://github.com/josemarquezjaramillo/kallos_models)**

**License**: MIT - Free for commercial and academic use

---

*This project demonstrates production-grade MLOps practices for financial machine learning, emphasizing rigorous validation, clean architecture, and real-world deployability.*
