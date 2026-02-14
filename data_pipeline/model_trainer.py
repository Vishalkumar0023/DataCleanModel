"""
Model Trainer Module
====================
Advanced ML training with auto-detection, multi-model comparison,
cross-validation, hyperparameter tuning, explainability, and model export.
"""

import pandas as pd
import numpy as np
import joblib
import re
import warnings
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

from sklearn.model_selection import (
    cross_validate, StratifiedKFold, KFold, RandomizedSearchCV
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    r2_score, mean_absolute_error, mean_squared_error,
    classification_report, make_scorer
)

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

warnings.filterwarnings('ignore')


# ─── Column pattern filters ─────────────────────────────────────────────────
ID_PATTERNS = re.compile(
    r'^(id|_id|index|row_?num|serial|sr_?no|unnamed)', re.IGNORECASE
)
URL_PATTERNS = re.compile(
    r'(url|link|href|src|image|img|photo|avatar|thumbnail|path|file)', re.IGNORECASE
)
TARGET_HINTS = [
    'target', 'label', 'class', 'outcome', 'result', 'y',
    'is_', 'has_', 'survived', 'churn', 'fraud', 'default',
    'price', 'salary', 'revenue', 'cost', 'amount', 'score'
]


class ModelTrainer:
    """
    Advanced ML model trainer with auto-detection, multi-model comparison,
    cross-validation, hyperparameter tuning, and explainability.
    
    Example
    -------
    >>> trainer = ModelTrainer(df, target_col='price', problem_type='regression')
    >>> results = trainer.run()
    >>> print(results['best_model'])
    >>> print(results['metrics_dashboard'])
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        problem_type: Optional[str] = None
    ):
        self.original_df = df.copy()
        self.df = df.copy()
        self.target_col = target_col
        self.problem_type = problem_type  # 'classification' or 'regression'
        self.warnings: List[str] = []
        self.log: List[str] = []

        # Will be set during prepare
        self.X = None
        self.y = None
        self.feature_names: List[str] = []
        self.scaler = None
        self.label_encoder = None

        # Results
        self.models: Dict[str, Any] = {}
        self.cv_results: Dict[str, Dict] = {}
        self.best_model_name: str = ''
        self.best_model = None
        self.best_metrics: Dict[str, float] = {}
        self.importances: Dict[str, float] = {}

    # ─── 1. Target Detection ────────────────────────────────────────────
    def detect_target(self) -> str:
        """Auto-detect or validate the target column."""
        cols = self.df.columns.tolist()

        # If already provided, validate it
        if self.target_col:
            if self.target_col in cols:
                self.log.append(f"Target column confirmed: '{self.target_col}'")
                return self.target_col
            else:
                self.warnings.append(
                    f"Specified target '{self.target_col}' not found in data."
                )
                self.target_col = None

        # Try heuristic search
        for hint in TARGET_HINTS:
            for col in cols:
                if hint.lower() == col.lower() or col.lower().startswith(hint.lower()):
                    self.target_col = col
                    self.log.append(f"Auto-detected target column: '{col}' (matched hint '{hint}')")
                    return col

        # Fallback: last column
        self.target_col = cols[-1]
        self.warnings.append(
            f"No obvious target found — using last column '{self.target_col}'. "
            "You can specify a target column explicitly."
        )
        self.log.append(f"Fallback target column: '{self.target_col}'")
        return self.target_col

    # ─── 2. Problem Type Detection ──────────────────────────────────────
    def detect_problem_type(self) -> str:
        """Detect whether this is a classification or regression problem."""
        if self.target_col is None:
            self.detect_target()
            
        y = self.df[self.target_col]
        
        # User override with validation
        if self.problem_type:
            if self.problem_type == 'regression':
                # Validate that target is numeric
                if y.dtype == 'object' or y.dtype.name == 'category':
                    try:
                        pd.to_numeric(y, errors='raise')
                    except Exception:
                        self.warnings.append(
                            f"Regression requested but target '{self.target_col}' is non-numeric. "
                            "Switching to classification."
                        )
                        self.log.append(f"Auto-corrected problem type to 'classification' (target is string)")
                        self.problem_type = 'classification'
                        return 'classification'
            
            self.log.append(f"Problem type specified: {self.problem_type}")
            return self.problem_type

        # Auto-detection
        # Object/category → classification
        if y.dtype == 'object' or y.dtype.name == 'category':
            self.problem_type = 'classification'
        # Few unique values → classification
        elif y.nunique() <= 15 and y.nunique() / len(y) < 0.05:
            self.problem_type = 'classification'
        else:
            self.problem_type = 'regression'

        self.log.append(f"Auto-detected problem type: {self.problem_type}")
        return self.problem_type

    # ─── 3. Feature Preparation ─────────────────────────────────────────
    def _drop_junk_columns(self):
        """Remove ID, URL, image, and metadata columns."""
        cols_to_drop = []
        for col in self.df.columns:
            if col == self.target_col:
                continue
            # ID-like
            if ID_PATTERNS.match(col):
                cols_to_drop.append(col)
                continue
            # URL/image-like
            if URL_PATTERNS.search(col):
                cols_to_drop.append(col)
                continue
            # Columns that are all unique strings (likely IDs)
            if self.df[col].dtype == 'object':
                if self.df[col].nunique() > 0.9 * len(self.df) and len(self.df) > 20:
                    cols_to_drop.append(col)
                    continue

        if cols_to_drop:
            self.df.drop(columns=cols_to_drop, inplace=True)
            self.log.append(f"Dropped {len(cols_to_drop)} junk columns: {cols_to_drop[:5]}{'...' if len(cols_to_drop) > 5 else ''}")

    def _drop_high_correlation(self, threshold: float = 0.9):
        """Remove one of each pair of features correlated above threshold."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col in numeric_cols:
            numeric_cols.remove(self.target_col)
        if len(numeric_cols) < 2:
            return

        corr = self.df[numeric_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > threshold)]

        if to_drop:
            self.df.drop(columns=to_drop, inplace=True)
            self.log.append(f"Dropped {len(to_drop)} highly correlated features (r > {threshold})")

    def _encode_categoricals(self):
        """Label-encode remaining categorical columns."""
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        if self.target_col in cat_cols:
            cat_cols.remove(self.target_col)

        for col in cat_cols:
            le = LabelEncoder()
            mask = self.df[col].notna()
            self.df.loc[mask, col] = le.fit_transform(self.df.loc[mask, col].astype(str))
            self.df[col] = self.df[col].astype(float)

        if cat_cols:
            self.log.append(f"Label-encoded {len(cat_cols)} categorical features")

    def _extract_datetime_parts(self):
        """Extract date parts from datetime columns."""
        dt_cols = self.df.select_dtypes(include=['datetime64']).columns.tolist()
        for col in dt_cols:
            if col == self.target_col:
                continue
            self.df[f'{col}_year'] = self.df[col].dt.year
            self.df[f'{col}_month'] = self.df[col].dt.month
            self.df[f'{col}_day'] = self.df[col].dt.day
            self.df[f'{col}_dayofweek'] = self.df[col].dt.dayofweek
            self.df.drop(columns=[col], inplace=True)
        if dt_cols:
            self.log.append(f"Extracted date parts from {len(dt_cols)} datetime columns")

    def _create_ratio_features(self, max_ratios: int = 5):
        """Generate ratio features from top numeric pairs."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col in numeric_cols:
            numeric_cols.remove(self.target_col)
        if len(numeric_cols) < 2:
            return

        # Use top columns by variance
        variances = self.df[numeric_cols].var().sort_values(ascending=False)
        top_cols = variances.head(min(4, len(numeric_cols))).index.tolist()

        created = 0
        for i in range(len(top_cols)):
            for j in range(i + 1, len(top_cols)):
                if created >= max_ratios:
                    break
                a, b = top_cols[i], top_cols[j]
                denom = self.df[b].replace(0, np.nan)
                ratio = self.df[a] / denom
                if ratio.notna().sum() > 0.5 * len(self.df):
                    name = f'{a}_div_{b}'
                    self.df[name] = ratio.fillna(0)
                    created += 1
            if created >= max_ratios:
                break

        if created:
            self.log.append(f"Created {created} ratio features")

    def _create_bins(self, n_bins: int = 5):
        """Bin the top numeric features."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col in numeric_cols:
            numeric_cols.remove(self.target_col)
        cols = numeric_cols[:5]
        created = 0
        for col in cols:
            try:
                self.df[f'{col}_bin'] = pd.qcut(
                    self.df[col], q=n_bins, labels=False, duplicates='drop'
                )
                created += 1
            except Exception:
                pass
        if created:
            self.log.append(f"Created binned features for {created} columns")

    def prepare_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Full feature preparation pipeline. Returns (X, y)."""
        self.detect_target()
        self.detect_problem_type()

        # Dataset size warning
        if len(self.df) < 100:
            self.warnings.append(
                f"⚠ Small dataset ({len(self.df)} rows). "
                "Results may be unreliable. Consider collecting more data."
            )

        # Feature engineering steps
        self._drop_junk_columns()
        self._extract_datetime_parts()
        self._encode_categoricals()
        self._create_ratio_features()
        self._create_bins()
        self._drop_high_correlation(threshold=0.9)

        # Separate target
        y = self.df[self.target_col].copy()
        X = self.df.drop(columns=[self.target_col]).copy()

        # Drop any remaining non-numeric
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            X.drop(columns=non_numeric, inplace=True)
            self.log.append(f"Dropped {len(non_numeric)} non-numeric columns before training")

        # Fill remaining NaN
        X = X.fillna(X.median())

        # Encode target for classification if needed
        if self.problem_type == 'classification' and y.dtype == 'object':
            self.label_encoder = LabelEncoder()
            y = pd.Series(self.label_encoder.fit_transform(y), name=self.target_col)
            self.log.append("Label-encoded target column")

        y = y.fillna(y.mode()[0] if self.problem_type == 'classification' else y.median())

        # Scale features
        self.feature_names = X.columns.tolist()
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.log.append(f"Scaled {len(self.feature_names)} features with StandardScaler")

        self.X = X_scaled
        self.y = y.values if hasattr(y, 'values') else np.array(y)

        self.log.append(f"Final training shape: X={self.X.shape}, y={self.y.shape}")
        return self.X, self.y

    # ─── 4. Imbalance Detection ─────────────────────────────────────────
    def detect_imbalance(self) -> Dict[str, Any]:
        """Detect class imbalance for classification problems."""
        if self.problem_type != 'classification':
            return {'imbalanced': False, 'reason': 'Regression problem'}

        counts = pd.Series(self.y).value_counts(normalize=True)
        majority_pct = counts.iloc[0]
        minority_pct = counts.iloc[-1]

        imbalanced = majority_pct > 0.70
        info = {
            'imbalanced': imbalanced,
            'majority_class': str(counts.index[0]),
            'majority_pct': round(majority_pct * 100, 1),
            'minority_class': str(counts.index[-1]),
            'minority_pct': round(minority_pct * 100, 1),
            'class_distribution': {str(k): round(v * 100, 1) for k, v in counts.items()}
        }

        if imbalanced:
            self.warnings.append(
                f"Class imbalance detected: majority {info['majority_pct']}% / "
                f"minority {info['minority_pct']}%. Using class weights or SMOTE."
            )
            self.log.append(f"Imbalance detected: {info['class_distribution']}")

        return info

    # ─── 5. Model Definitions ──────────────────────────────────────────
    def _get_models_and_params(self) -> Dict[str, Dict]:
        """Return model instances and hyperparameter grids."""
        imbalance = self.detect_imbalance()
        use_balanced = imbalance.get('imbalanced', False)
        n_classes = len(np.unique(self.y)) if self.problem_type == 'classification' else 0

        if self.problem_type == 'classification':
            models = {
                'Logistic Regression': {
                    'model': LogisticRegression(
                        max_iter=1000, random_state=42,
                        class_weight='balanced' if use_balanced else None
                    ),
                    'params': {
                        'C': [0.01, 0.1, 1, 10],
                        'solver': ['lbfgs', 'liblinear']
                    }
                },
                'Random Forest': {
                    'model': RandomForestClassifier(
                        random_state=42,
                        class_weight='balanced' if use_balanced else None
                    ),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [5, 10, 20, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                },
                'Gradient Boosting': {
                    'model': GradientBoostingClassifier(random_state=42),
                    'params': {
                        'n_estimators': [100, 200],
                        'max_depth': [3, 5, 7],
                        'learning_rate': [0.01, 0.05, 0.1, 0.2],
                        'subsample': [0.8, 1.0]
                    }
                }
            }
            if HAS_XGBOOST:
                models['XGBoost'] = {
                    'model': XGBClassifier(
                        random_state=42, eval_metric='logloss',
                        use_label_encoder=False
                    ),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [3, 5, 7],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'subsample': [0.8, 1.0],
                        'colsample_bytree': [0.8, 1.0]
                    }
                }
        else:  # regression
            models = {
                'Ridge Regression': {
                    'model': Ridge(random_state=42),
                    'params': {
                        'alpha': [0.01, 0.1, 1, 10, 100]
                    }
                },
                'Random Forest': {
                    'model': RandomForestRegressor(random_state=42),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [5, 10, 20, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                },
                'Gradient Boosting': {
                    'model': GradientBoostingRegressor(random_state=42),
                    'params': {
                        'n_estimators': [100, 200],
                        'max_depth': [3, 5, 7],
                        'learning_rate': [0.01, 0.05, 0.1, 0.2],
                        'subsample': [0.8, 1.0]
                    }
                }
            }
            if HAS_XGBOOST:
                models['XGBoost'] = {
                    'model': XGBRegressor(random_state=42),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [3, 5, 7],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'subsample': [0.8, 1.0],
                        'colsample_bytree': [0.8, 1.0]
                    }
                }

        return models

    # ─── 6. Training ───────────────────────────────────────────────────
    def train_all_models(self) -> Dict[str, Dict]:
        """Train all models with cross-validation and hyperparameter tuning."""
        if self.X is None:
            self.prepare_features()

        model_defs = self._get_models_and_params()

        # Handle SMOTE for classification imbalance
        X_train, y_train = self.X, self.y
        imbalance = self.detect_imbalance()
        if imbalance.get('imbalanced') and HAS_SMOTE and self.problem_type == 'classification':
            try:
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                self.log.append(f"Applied SMOTE: {len(self.y)} → {len(y_train)} samples")
            except Exception as e:
                self.log.append(f"SMOTE failed, using original data: {e}")

        # CV strategy
        n_folds = 3
        if self.problem_type == 'classification':
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            scoring_primary = 'f1_weighted'
            scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
            try:
                if len(np.unique(y_train)) == 2:
                    scoring.append('roc_auc')
            except Exception:
                pass
        else:
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            scoring_primary = 'r2'
            scoring = ['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error']

        # Limit search iterations for speed
        n_iter = min(10, max(3, 100 // len(model_defs)))

        for name, config in model_defs.items():
            self.log.append(f"Training {name}...")
            try:
                # Hyperparameter tuning
                search = RandomizedSearchCV(
                    config['model'],
                    config['params'],
                    n_iter=min(n_iter, self._param_combinations(config['params'])),
                    cv=cv,
                    scoring=scoring_primary,
                    random_state=42,
                    n_jobs=1,
                    error_score='raise'
                )
                search.fit(X_train, y_train)
                best_estimator = search.best_estimator_

                # Cross-validate with all metrics
                cv_scores = cross_validate(
                    best_estimator, X_train, y_train,
                    cv=cv, scoring=scoring, n_jobs=1
                )

                # Store results
                metrics = {}
                for metric_name in scoring:
                    key = f'test_{metric_name}'
                    if key in cv_scores:
                        scores = cv_scores[key]
                        # Convert negative metrics
                        if metric_name.startswith('neg_'):
                            scores = -scores
                            clean_name = metric_name.replace('neg_', '')
                        else:
                            clean_name = metric_name
                        metrics[clean_name] = {
                            'mean': round(float(np.mean(scores)), 4),
                            'std': round(float(np.std(scores)), 4)
                        }

                self.models[name] = best_estimator
                self.cv_results[name] = {
                    'metrics': metrics,
                    'best_params': search.best_params_,
                    'best_cv_score': round(float(search.best_score_), 4)
                }
                self.log.append(f"  {name} → best CV score: {search.best_score_:.4f}")

            except Exception as e:
                self.log.append(f"  {name} failed: {str(e)[:100]}")
                self.cv_results[name] = {'error': str(e)}

        return self.cv_results

    def _param_combinations(self, params: Dict) -> int:
        """Calculate total parameter combinations."""
        total = 1
        for v in params.values():
            total *= len(v)
        return total

    # ─── 7. Best Model Selection ───────────────────────────────────────
    def get_best_model(self) -> Dict[str, Any]:
        """Select the best model based on primary metric."""
        if not self.cv_results:
            self.train_all_models()

        primary_metric = 'f1_weighted' if self.problem_type == 'classification' else 'r2'

        best_score = -np.inf
        for name, result in self.cv_results.items():
            if 'error' in result:
                continue
            score = result['metrics'].get(primary_metric, {}).get('mean', -np.inf)
            if score > best_score:
                best_score = score
                self.best_model_name = name

        if self.best_model_name:
            self.best_model = self.models[self.best_model_name]
            self.best_metrics = self.cv_results[self.best_model_name]['metrics']
            self.log.append(f"Best model: {self.best_model_name} ({primary_metric}={best_score:.4f})")

        return {
            'name': self.best_model_name,
            'metrics': self.best_metrics,
            'params': self.cv_results.get(self.best_model_name, {}).get('best_params', {})
        }

    # ─── 8. Feature Importance ─────────────────────────────────────────
    def get_feature_importance(self, top_n: int = 10) -> Dict[str, float]:
        """Extract top feature importances from the best model."""
        if self.best_model is None:
            self.get_best_model()
        if self.best_model is None:
            return {}

        importances = None
        try:
            if hasattr(self.best_model, 'feature_importances_'):
                importances = self.best_model.feature_importances_
            elif hasattr(self.best_model, 'coef_'):
                coef = self.best_model.coef_
                if coef.ndim > 1:
                    importances = np.mean(np.abs(coef), axis=0)
                else:
                    importances = np.abs(coef)
        except Exception:
            pass

        if importances is None or len(importances) != len(self.feature_names):
            return {}

        imp_dict = dict(zip(self.feature_names, importances.tolist()))
        imp_dict = dict(sorted(imp_dict.items(), key=lambda x: x[1], reverse=True))

        self.importances = {k: round(v, 6) for k, v in list(imp_dict.items())[:top_n]}
        return self.importances

    # ─── 9. Metrics Dashboard ──────────────────────────────────────────
    def get_metrics_dashboard(self) -> Dict[str, Any]:
        """Build a complete metrics dashboard."""
        if not self.cv_results:
            self.train_all_models()

        dashboard = {
            'problem_type': self.problem_type,
            'target_column': self.target_col,
            'dataset_shape': {
                'rows': int(self.X.shape[0]),
                'features': int(self.X.shape[1])
            },
            'models_compared': {},
            'best_model': self.get_best_model(),
            'feature_importance': self.get_feature_importance(),
            'reliability': self.get_reliability_score(),
            'warnings': self.warnings,
            'log': self.log,
            'explanation': self.explain_model()  # Add explanation
        }

        # Per-model summary
        for name, result in self.cv_results.items():
            if 'error' in result:
                dashboard['models_compared'][name] = {'error': result['error']}
            else:
                dashboard['models_compared'][name] = {
                    'metrics': result['metrics'],
                    'best_params': result['best_params']
                }

        return dashboard

    # ─── 15. Explainability (SHAP) ─────────────────────────────────────
    def explain_model(self) -> Dict[str, Any]:
        """Generate SHAP explanations for the best model."""
        if self.best_model is None or self.X is None:
            return {}

        try:
            import shap
            import matplotlib.pyplot as plt
            import io
            import base64
            
            # Use a small sample for speed
            sample_size = min(100, len(self.X))
            X_sample = self.X[:sample_size]
            
            # Choose explainer
            # Tree models vs Linear vs Generic
            model_type = type(self.best_model).__name__
            if 'Forest' in model_type or 'Boost' in model_type or 'Tree' in model_type or 'XGB' in model_type:
                explainer = shap.TreeExplainer(self.best_model)
            elif 'Linear' in model_type or 'Ridge' in model_type or 'Logistic' in model_type:
                explainer = shap.LinearExplainer(self.best_model, X_sample)
            else:
                explainer = shap.KernelExplainer(self.best_model.predict, X_sample)

            shap_values = explainer.shap_values(X_sample)
            
            # Handling classification (list of arrays) vs regression (array)
            if isinstance(shap_values, list):
                # binary classification -> take index 1 (positive class)
                # multi-class -> take index 1 just for demo
                shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                shap_vals = shap_values

            # PLOT 1: Summary Plot (Bar)
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_vals, X_sample, feature_names=self.feature_names, plot_type="bar", show=False)
            summary_bar = self._fig_to_base64(fig)
            plt.close(fig)

            # PLOT 2: Summary Plot (Dot/Beeswarm)
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_vals, X_sample, feature_names=self.feature_names, show=False)
            summary_dot = self._fig_to_base64(fig)
            plt.close(fig)
            
            return {
                'summary_bar': summary_bar,
                'summary_dot': summary_dot,
                'available': True
            }

        except Exception as e:
            self.log.append(f"SHAP explanation failed: {e}")
            return {'error': str(e), 'available': False}

    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        import io
        import base64
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"

    # ─── 14. Full Pipeline Run ─────────────────────────────────────────
    def run(self) -> Dict[str, Any]:
        """
        Run the full ML training pipeline:
        1. Detect target & problem type
        2. Prepare features
        3. Train all models with CV + tuning
        4. Select best model
        5. Get feature importance
        6. Build metrics dashboard
        """
        self.prepare_features()
        self.train_all_models()
        self.get_best_model() # Ensure best model is selected
        dashboard = self.get_metrics_dashboard()
        return dashboard

    # ─── 10. Reliability Score ─────────────────────────────────────────
    def get_reliability_score(self) -> Dict[str, Any]:
        """
        Compute a 0–100 reliability score based on:
        - Dataset size (bigger = better)
        - CV score variance (lower = better)
        - Primary metric quality
        """
        score = 50  # baseline
        reasons = []

        # Dataset size factor (0–25 pts)
        n = self.X.shape[0] if self.X is not None else len(self.df)
        if n >= 10000:
            score += 25
            reasons.append("Large dataset (+25)")
        elif n >= 1000:
            pts = int(15 + (n - 1000) / 900 * 10)
            score += pts
            reasons.append(f"Medium dataset (+{pts})")
        elif n >= 100:
            pts = int(5 + (n - 100) / 900 * 10)
            score += pts
            reasons.append(f"Small dataset (+{pts})")
        else:
            score -= 15
            reasons.append("Very small dataset (-15)")

        # CV variance factor (0–15 pts)
        if self.best_metrics:
            primary = 'f1_weighted' if self.problem_type == 'classification' else 'r2'
            std = self.best_metrics.get(primary, {}).get('std', 0.1)
            if std < 0.02:
                score += 15
                reasons.append("Very stable CV scores (+15)")
            elif std < 0.05:
                score += 10
                reasons.append("Stable CV scores (+10)")
            elif std < 0.1:
                score += 5
                reasons.append("Moderate CV variance (+5)")
            else:
                score -= 5
                reasons.append("High CV variance (-5)")

        # Metric quality factor (0–10 pts)
        if self.best_metrics:
            primary = 'f1_weighted' if self.problem_type == 'classification' else 'r2'
            val = self.best_metrics.get(primary, {}).get('mean', 0)
            if val >= 0.9:
                score += 10
                reasons.append("Excellent primary metric (+10)")
            elif val >= 0.75:
                score += 5
                reasons.append("Good primary metric (+5)")
            elif val < 0.5:
                score -= 10
                reasons.append("Poor primary metric (-10)")

        score = max(0, min(100, score))
        return {
            'score': score,
            'grade': 'A' if score >= 85 else 'B' if score >= 70 else 'C' if score >= 50 else 'D',
            'reasons': reasons
        }

    # ─── 11. Export ────────────────────────────────────────────────────
    def export_model(self, path: str) -> str:
        """Export the best model and metadata to a .pkl file."""
        if self.best_model is None:
            self.get_best_model()

        export_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'target_col': self.target_col,
            'problem_type': self.problem_type,
            'metrics': self.best_metrics,
            'feature_importance': self.importances,
            'exported_at': datetime.now().isoformat(),
            'reliability': self.get_reliability_score()
        }

        joblib.dump(export_data, path)
        self.log.append(f"Model exported to '{path}'")
        return path

    # ─── 13. Comparison Logic ──────────────────────────────────────────
    def train_naive_baseline(self) -> Dict[str, Any]:
        """
        Train a 'Naive' model on raw data for comparison.
        Strategy: 'Just Encoding' + Minimal Imputation (to prevent crash).
        """
        try:
            # work on a copy of original_df
            df = self.original_df.copy()
            
            # 1. Drop IDs (essential) but keep almost everything else
            for col in df.columns:
                if col == self.target_col: continue
                if ID_PATTERNS.match(col) or URL_PATTERNS.search(col):
                    df.drop(columns=[col], inplace=True)
            
            # 2. Handle Target
            if self.target_col not in df.columns:
                return {} 
            
            y = df[self.target_col]
            X = df.drop(columns=[self.target_col])
            
            # 3. Naive Preprocessing (The "Lazy" Way - using Pipeline to avoid leakage)
            
            # Identify columns
            num_cols = X.select_dtypes(include=[np.number]).columns
            cat_cols = X.select_dtypes(exclude=[np.number]).columns
            
            # Preprocessing for numeric data: simple mean imputation
            num_transformer = SimpleImputer(strategy='mean')
            
            # Preprocessing for categorical data: constant fill + ordinal encoding
            # OrdinalEncoder is used because it handles 2D arrays (unlike LabelEncoder)
            # and we can handle unknown values by encoding them as -1
            cat_transformer = make_pipeline(
                SimpleImputer(strategy='constant', fill_value='missing'),
                OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            )
            
            preprocessor = make_column_transformer(
                (num_transformer, num_cols),
                (cat_transformer, cat_cols),
                remainder='passthrough'
            )
            
            # Encode target if necessary
            if self.problem_type == 'classification' and y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y.astype(str))
            
            # 4. Train Naive Model (Simple Decision Tree)
            if self.problem_type == 'classification':
                from sklearn.tree import DecisionTreeClassifier
                model = DecisionTreeClassifier(random_state=42)
                scoring = 'accuracy'
            else:
                from sklearn.tree import DecisionTreeRegressor
                model = DecisionTreeRegressor(random_state=42)
                scoring = 'r2'
                
            # Create full pipeline
            # Note: We don't scale or do anything fancy. Just impute -> encode -> tree.
            pipeline = make_pipeline(preprocessor, model)
            
            # Quick CV
            n_folds = 3
            if self.problem_type == 'classification':
                cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            else:
                cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
                
            try:
                cv_scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)
                mean_score = np.mean(cv_scores['test_score'])
            except ValueError as ve:
                # Fallback for very small datasets where cv=3 might fail
                if "n_splits" in str(ve):
                    model.fit(preprocessor.fit_transform(X, y), y)
                    mean_score = model.score(preprocessor.transform(X), y)
                else:
                    raise ve
            
            return {
                "score": round(mean_score, 4),
                "metric": scoring,
                "model_type": "Decision Tree (Baseline)"
            }
            
        except Exception as e:
            self.log.append(f"Naive baseline failed: {e}")
            return {"score": 0.0, "error": str(e)}

    def run_full_comparison(self) -> Dict[str, Any]:
        """
        Run both Naive (Raw) and Advanced (Cleaned) pipelines and compare.
        """
        # 1. Train Naive
        naive_results = self.train_naive_baseline()
        
        # 2. Run Advanced Pipeline
        advanced_dashboard = self.run()
        
        # 3. Compare
        advanced_score = 0
        if 'best_model' in advanced_dashboard and 'metrics' in advanced_dashboard['best_model']:
            metrics = advanced_dashboard['best_model']['metrics']
            # classification -> Accuracy or F1
            # regression -> R2
            if self.problem_type == 'classification':
                # Try accuracy first for direct comparison
                if 'accuracy' in metrics:
                    advanced_score = metrics['accuracy']['mean']
                elif 'f1_weighted' in metrics:
                    advanced_score = metrics['f1_weighted']['mean']
            else:
                if 'r2' in metrics:
                    advanced_score = metrics['r2']['mean']
        
        naive_score = naive_results.get('score', 0)
        
        # Calculate Improvement
        improvement = 0
        if naive_score != 0:
            improvement = ((advanced_score - naive_score) / abs(naive_score)) * 100
        elif advanced_score > 0:
            improvement = 100 # Infinite improvement
            
        # Add comparison to dashboard
        advanced_dashboard['comparison'] = {
            'raw_score': naive_score,
            'cleaned_score': advanced_score,
            'improvement_pct': round(improvement, 1),
            'metric': naive_results.get('metric', 'score')
        }
        
        return advanced_dashboard

    # ─── 14. Full Pipeline Run ─────────────────────────────────────────
    def run(self) -> Dict[str, Any]:
        """
        Run the full ML training pipeline:
        1. Detect target & problem type
        2. Prepare features
        3. Train all models with CV + tuning
        4. Select best model
        5. Get feature importance
        6. Build metrics dashboard
        """
        self.prepare_features()
        self.train_all_models()
        dashboard = self.get_metrics_dashboard()
        return dashboard
