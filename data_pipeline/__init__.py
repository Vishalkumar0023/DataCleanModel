"""
Data Pipeline Tool
==================
A comprehensive, modular Python tool for automated data cleaning, 
exploratory data analysis (EDA), and feature engineering.

Modules:
- data_loader: Load and validate datasets
- data_cleaner: Clean and preprocess data
- eda: Exploratory data analysis and visualizations
- feature_engineer: Feature engineering and transformation
- pipeline: Main orchestrator combining all modules
"""

from .data_loader import DataLoader
from .data_cleaner import DataCleaner
from .model_trainer import ModelTrainer

# Lazy imports to avoid loading matplotlib/seaborn/scipy at import time
# Use: from data_pipeline.eda import EDAAnalyzer
# Use: from data_pipeline.feature_engineer import FeatureEngineer
# Use: from data_pipeline.pipeline import DataPipeline

def _get_pipeline():
    from .pipeline import DataPipeline as _DP
    return _DP

# Make DataPipeline available but lazy
import importlib
def __getattr__(name):
    if name == "DataPipeline":
        return _get_pipeline()
    if name == "EDAAnalyzer":
        from .eda import EDAAnalyzer
        return EDAAnalyzer
    if name == "FeatureEngineer":
        from .feature_engineer import FeatureEngineer
        return FeatureEngineer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__version__ = "1.1.0"
__all__ = ["DataPipeline", "DataLoader", "DataCleaner", "EDAAnalyzer", "FeatureEngineer", "ModelTrainer"]
