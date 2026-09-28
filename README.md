# Multi-Modal Market Forecasting from Limit Order Book Data

This project studies short-horizon price predictability using high-frequency limit order book (LOB) states and order-flow data for **TSLA, AMZN, and BRK.B**.

The goal is to investigate whether information from order-book structure and market activity can be used to predict future price movements, and whether any statistical predictability remains useful after realistic trading frictions are considered.

The project covers the full research pipeline:

- data inspection and preprocessing
- calendar-time resampling
- feature engineering
- regression and classification
- classical machine learning models
- Transformer-based sequence modeling
- model interpretation
- trading strategy evaluation
- transaction-cost and latency stress testing

## Project Structure

```text
market-forecasting/
├── notebooks/
│   ├── 01_data_inspection.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_classification.ipynb
│   ├── 05_interpretation.ipynb
│   ├── 06_sequence_model.ipynb
│   └── 07_trading_strategy.ipynb
│
├── outputs/
│   ├── all_regression_test_results.csv
│   ├── classification_validation_results.csv
│   ├── regression_test_results.csv
│   ├── regression_validation_results.csv
│   ├── transformer_classification_test_results.csv
│   └── transformer_regression_test_results.csv
│
├── requirements.txt
└── README.md
```

## Data Representation

The raw data consists of high-frequency order-book states and message events.

Because the original observations arrive in irregular event time, the data is converted to a **500 ms calendar-time grid** during the regular trading session.

For each interval:

- the latest observable order-book state is retained
- message events are aggregated within the same 500 ms window
- only past information is used when constructing temporal features

The selected forecasting horizons are:

| Asset | Forecasting Horizons |
|---|---|
| TSLA | 500 ms, 2 s, 10 s |
| AMZN | 500 ms, 2 s, 10 s |
| BRK.B | 5 s, 10 s, 30 s |

BRK.B uses longer minimum horizons because its very short-horizon returns contain a much larger proportion of zero price movements.

## Feature Engineering

The final modeling representation contains **19 interpretable features** describing different aspects of market microstructure.

The main feature groups are:

### Order-book state

- bid-ask spread
- level-1 order-book imbalance
- level-5 order-book imbalance
- bid depth
- ask depth
- microprice deviation

### Order flow and market activity

- event count
- execution count
- normalized book flow
- execution imbalance

### Recent price behavior

- 500 ms return
- 1 s return
- 2 s return
- 5 s return
- 10 s return

### Temporal context

- 5 s volatility
- 30 s volatility
- recent event intensity
- recent execution intensity

All temporal predictors are backward-looking.

## Evaluation Protocol

A strict chronological train/validation/test split is used:

| Period | Role |
|---|---|
| March 16–18, 2026 | Training |
| March 19, 2026 | Validation |
| March 20, 2026 | Final held-out test |

No random splitting is used.

To reduce temporal leakage:

- future-return targets are created independently within each trading day
- sequences never cross trading-day boundaries
- feature normalization is fitted only on the training set
- validation data is used for model and threshold selection
- the final test day is kept separate until the final evaluation

## Regression

Regression targets are future log mid-price returns.

The following approaches are compared:

- Persistence baseline
- Training-mean baseline
- Linear Regression
- Ridge Regression
- Histogram Gradient Boosting
- Transformer sequence regression

Regression performance is evaluated using:

- MAE
- RMSE
- R²
- directional accuracy

The results show that predictive gains are generally small and depend on the asset and forecasting horizon.

Simple linear models capture much of the available predictive structure, while the Transformer does not consistently provide better performance.

## Classification

Price movement is also modeled as a three-class classification problem:

- **DOWN**
- **STATIONARY**
- **UP**

Asset-specific thresholds are estimated using training data only.

The following models are compared:

- ZeroR / majority-class baseline
- Logistic Regression
- class-balanced Logistic Regression
- Histogram Gradient Boosting
- Transformer sequence classification

Performance is evaluated using:

- accuracy
- balanced accuracy
- macro F1
- confusion matrices

Class-balanced Logistic Regression provides a strong simple benchmark and performs consistently better than the majority-class baseline.

## Transformer Sequence Model

The Transformer uses the same 19 engineered features as the classical models.

Each input sequence contains **20 consecutive observations at 500 ms intervals**, corresponding to **10 seconds of recent market history**.

The architecture uses:

- 19-dimensional input features
- 64-dimensional feature projection
- learned positional embeddings
- 2 Transformer encoder layers
- 4 attention heads
- task-specific regression or classification output heads

The Transformer captures temporal information comparable to the classical models, but its additional complexity does not consistently improve performance over simpler approaches.

## Model Interpretation

The selected classification models are analyzed using standardized Logistic Regression coefficients.

Across the three assets, **level-1 order-book imbalance** is one of the clearest and most consistent features for distinguishing upward and downward price movements.

Recent short-horizon returns also contain useful information and often show an opposing relationship consistent with short-term reversal behavior.

Market activity and volatility features are more useful for distinguishing directional price movement from stationary observations.

These relationships are interpreted as model associations rather than causal effects.

## Trading Strategy

A simple trading strategy is evaluated using the **AMZN 500 ms class-balanced Logistic Regression model**.

A position is opened only when the predicted UP or DOWN probability exceeds a confidence threshold selected using the validation set.

Possible positions are:

- `+1`: long
- `-1`: short
- `0`: no trade

Each trade is held for one 500 ms forecasting interval.

The economic evaluation uses executable prices:

- long positions enter at the ask and exit at the future bid
- short positions enter at the bid and exit at the future ask

The evaluation also includes:

- bid-ask spread
- transaction costs
- execution latency

The model contains a small directional signal, but the signal is substantially smaller than the effective bid-ask spread.

As a result, the statistical predictability does **not** translate into a profitable strategy under the tested execution assumptions.

This highlights an important distinction between predictive model performance and economic usefulness.

## Latency Stress Test

The trading strategy is also evaluated under delayed execution.

The model, confidence threshold, and holding period are kept fixed while execution is delayed by:

- 0 ms
- 500 ms
- 1 second

Performance deteriorates when execution is delayed, showing that the weak short-horizon signal is also sensitive to latency.

## AI-Assisted Development

AI-assisted tools were used during development as a supporting tool for tasks such as debugging, reviewing implementation choices, and improving code and documentation.

AI-generated suggestions were not used blindly. The final implementation, modeling decisions, evaluation methodology, and experimental results were reviewed and validated throughout the project.

## Data Availability

The original high-frequency limit order book dataset is **not included** in this repository.

The notebooks expect the raw source data to be available locally under:

```text
data/task_data/
```

Processed datasets generated during feature engineering are stored locally under:

```text
data/processed/
```

These directories are excluded from the public repository.

The repository contains the analysis notebooks, modeling code, evaluation outputs, and documentation needed to understand the methodology and reproduce the pipeline with compatible data.

## Environment

The project was developed using **Python 3.11**.

Install the required packages with:

```bash
pip install -r requirements.txt
```

Main dependencies:

- NumPy
- pandas
- PyArrow
- scikit-learn
- matplotlib
- PyTorch
- Jupyter

## Notebook Execution Order

The notebooks are designed to be followed in numerical order:

1. `01_data_inspection.ipynb`
2. `02_feature_engineering.ipynb`
3. `03_modeling.ipynb`
4. `04_classification.ipynb`
5. `05_interpretation.ipynb`
6. `06_sequence_model.ipynb`
7. `07_trading_strategy.ipynb`

`02_feature_engineering.ipynb` creates the processed datasets used by the later notebooks.

## Key Takeaways

The experiments suggest that high-frequency limit order book data contain **weak but detectable short-horizon predictive structure**.

Simple classical models capture a substantial part of this information, while the additional complexity of the Transformer does not provide a consistent improvement.

Most importantly, the trading experiment demonstrates that statistical predictability alone is not sufficient. After incorporating realistic bid-ask execution, transaction costs, and latency, the predictive signal is not strong enough to produce a profitable strategy under the tested assumptions.
