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

from .pipeline import DataPipeline
from .data_loader import DataLoader
from .data_cleaner import DataCleaner
from .eda import EDAAnalyzer
from .feature_engineer import FeatureEngineer
from .model_trainer import ModelTrainer

__version__ = "1.1.0"
__all__ = ["DataPipeline", "DataLoader", "DataCleaner", "EDAAnalyzer", "FeatureEngineer", "ModelTrainer"]
