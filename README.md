# Data Pipeline Tool

A comprehensive, modular Python tool for automated **data cleaning**, **exploratory data analysis (EDA)**, and **feature engineering**.

## 🚀 Quick Start

```python
from data_pipeline import DataPipeline

# Initialize and load data
pipeline = DataPipeline()
pipeline.load("your_data.csv")

# Run full pipeline
cleaned_df, final_df = pipeline.run_full_pipeline(
    target_col="target_column",
    problem_type="classification"  # or "regression"
)

# Save results
pipeline.save_data("./output")
```

## 📦 Installation

Ensure you have the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
# Optional for SMOTE:
pip install imbalanced-learn
```

## 🏗️ Project Structure

```
data_pipeline/
├── __init__.py           # Package initialization
├── data_loader.py        # Data loading and validation
├── data_cleaner.py       # Data cleaning operations
├── eda.py                # Exploratory data analysis
├── feature_engineer.py   # Feature engineering
└── pipeline.py           # Main orchestrator
```

## 📋 Features

### 1. Data Loading & Validation
- Load CSV, Excel, JSON, Parquet files
- Accept DataFrame directly
- Detect duplicates, missing values, data types
- Identify constant/near-constant columns
- Detect potential ID columns

### 2. Data Cleaning
- Remove duplicate rows
- Handle missing values (mean/median/mode/drop)
- Fix incorrect data types
- Handle outliers (IQR/Z-score)
- Clean categorical values (whitespace, case)
- Drop low-value columns

### 3. Exploratory Data Analysis
- Summary statistics
- Correlation analysis
- Distribution analysis
- Visualizations:
  - Histograms
  - Boxplots
  - Correlation heatmap
  - Categorical count plots
  - Feature-target relationships

### 4. Feature Engineering
- Categorical encoding (One-Hot, Label)
- Feature scaling (Standard, MinMax)
- Datetime feature extraction
- Polynomial/interaction features
- Feature importance computation
- Drop low-variance features
- Drop highly correlated features
- Handle class imbalance (SMOTE)

## 🔧 Usage Examples

### Full Pipeline (Recommended)

```python
from data_pipeline import DataPipeline

pipeline = DataPipeline()
pipeline.load("data.csv")

cleaned_df, final_df = pipeline.run_full_pipeline(
    target_col="price",
    problem_type="regression",
    show_eda_plots=True
)
```

### Step-by-Step Control

```python
pipeline = DataPipeline()
pipeline.load("data.csv")

# Validate
pipeline.validate()

# Clean with custom settings
pipeline.clean(
    missing_numeric_strategy='median',
    outlier_method='iqr',
    outlier_action='clip'
)

# Analyze
pipeline.analyze(target_col="price", show_plots=True)

# Engineer features
pipeline.engineer_features(
    target_col="price",
    problem_type="regression",
    scale_method='standard'
)

# Get data
cleaned_df = pipeline.get_cleaned_data()
final_df = pipeline.get_final_data()
```

### Using Individual Modules

```python
from data_pipeline import DataLoader, DataCleaner, EDAAnalyzer, FeatureEngineer

# Load
loader = DataLoader()
df = loader.load("data.csv")
loader.validate()

# Clean
cleaner = DataCleaner(df)
cleaner.remove_duplicates()
cleaner.handle_missing_values()
cleaner.handle_outliers()
cleaned_df = cleaner.get_cleaned_data()

# Analyze
eda = EDAAnalyzer(cleaned_df, target_col="price")
eda.run_full_analysis()

# Engineer
engineer = FeatureEngineer(cleaned_df, target_col="price")
engineer.encode_categorical()
engineer.scale_features()
final_df = engineer.get_transformed_data()
```

## ⚙️ Configuration Options

### Cleaning Options
| Parameter | Default | Options |
|-----------|---------|---------|
| `missing_numeric_strategy` | `'median'` | `'mean'`, `'median'`, `'zero'` |
| `missing_categorical_strategy` | `'mode'` | `'mode'`, `'unknown'` |
| `missing_drop_threshold` | `0.4` | 0.0 - 1.0 |
| `outlier_method` | `'iqr'` | `'iqr'`, `'zscore'` |
| `outlier_action` | `'clip'` | `'clip'`, `'remove'`, `'nan'` |

### Feature Engineering Options
| Parameter | Default | Options |
|-----------|---------|---------|
| `scale_method` | `'standard'` | `'standard'`, `'minmax'` |
| `encode_categorical` | `True` | Auto selects one-hot or label |
| `create_polynomial_features` | `False` | Creates interaction terms |
| `correlation_threshold` | `0.95` | Drops one of correlated pairs |
| `handle_imbalance` | `False` | SMOTE for classification |

## 📊 Output

The pipeline produces:
- **Cleaned DataFrame**: Missing values handled, outliers fixed, duplicates removed
- **Final DataFrame**: Model-ready with encoding, scaling, and feature engineering
- **EDA Report**: Statistics, insights, and visualizations
- **Pipeline Report**: Complete log of all transformations

## 🎯 Supported Problem Types

- **Classification**: Includes class imbalance handling
- **Regression**: Continuous target analysis
- **Clustering**: Unsupervised data preparation

## 📝 License

MIT License - Feel free to use and modify.
