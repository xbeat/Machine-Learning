## Mastering ML Pipelines Common Mistakes to Avoid
Slide 1: Data Validation: The Foundation of ML Pipelines

Data validation serves as the critical first line of defense in machine learning pipelines, ensuring data quality, consistency, and reliability. Implementing comprehensive validation checks prevents downstream issues and maintains the integrity of your machine learning models throughout their lifecycle.

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class DataValidator:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.validation_results = {}
    
    def check_missing_values(self, threshold: float = 0.1) -> Dict:
        missing_pct = self.df.isnull().mean()
        cols_above_threshold = missing_pct[missing_pct > threshold]
        self.validation_results['missing_values'] = cols_above_threshold
        return {'columns_above_threshold': cols_above_threshold.to_dict()}
    
    def check_data_types(self) -> Dict:
        return {'dtypes': self.df.dtypes.to_dict()}
    
    def validate_numerical_range(self, 
                               columns: List[str], 
                               min_values: Dict[str, float],
                               max_values: Dict[str, float]) -> Dict:
        out_of_range = {}
        for col in columns:
            mask = ~self.df[col].between(min_values[col], max_values[col])
            if mask.any():
                out_of_range[col] = self.df.loc[mask, col].tolist()
        return {'out_of_range_values': out_of_range}

# Example usage
data = pd.DataFrame({
    'age': [25, 30, -5, 150, 45],
    'salary': [50000, 60000, 75000, np.nan, 55000]
})

validator = DataValidator(data)
missing_check = validator.check_missing_values(threshold=0.1)
range_check = validator.validate_numerical_range(
    columns=['age'], 
    min_values={'age': 0}, 
    max_values={'age': 120}
)

print("Missing values check:", missing_check)
print("Range validation:", range_check)
```

Slide 2: Pipeline Modularity and Configuration Management

Pipeline modularity ensures maintainability, reusability, and easier debugging by breaking down complex workflows into discrete, testable components. Using configuration files instead of hardcoded values increases flexibility and adaptability to changing requirements.

```python
import yaml
from dataclasses import dataclass
from typing import Optional, List, Dict
import joblib

@dataclass
class ModelConfig:
    model_name: str
    hyperparameters: Dict
    features: List[str]
    target: str
    test_size: float
    random_state: int

class Pipeline:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.model = None
        
    def _load_config(self, config_path: str) -> ModelConfig:
        with open(config_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        return ModelConfig(**config_dict)
    
    def preprocess(self, data: pd.DataFrame) -> tuple:
        X = data[self.config.features]
        y = data[self.config.target]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        model_class = getattr(sklearn.ensemble, self.config.model_name)
        self.model = model_class(**self.config.hyperparameters)
        self.model.fit(X_train, y_train)
    
    def save_model(self, path: str) -> None:
        joblib.dump(self.model, path)

# Example config.yaml:
"""
model_name: RandomForestClassifier
hyperparameters:
  n_estimators: 100
  max_depth: 10
features:
  - feature1
  - feature2
target: target_column
test_size: 0.2
random_state: 42
"""

# Usage
pipeline = Pipeline('config.yaml')
X_train, X_test, y_train, y_test = pipeline.preprocess(data)
pipeline.train(X_train, y_train)
pipeline.save_model('model.joblib')
```

Slide 3: Automated Data Schema Validation

Schema validation ensures data consistency by enforcing strict type checking and structural validation. This prevents silent failures in production pipelines by catching data inconsistencies early in the process before they propagate through the system.

```python
import pandera as pa
from pandera.typing import DataFrame, Series
from typing import Optional

class DataSchema(pa.SchemaModel):
    age: Series[int] = pa.Field(ge=0, le=120)
    income: Series[float] = pa.Field(ge=0)
    education_years: Series[int] = pa.Field(ge=0, le=30)
    employment_status: Series[str] = pa.Field(isin=['employed', 'unemployed', 'student'])
    
    @pa.check('income')
    def validate_income_distribution(cls, series: Series[float]) -> bool:
        # Check for extreme outliers using IQR
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        return series.between(lower_bound, upper_bound).all()

def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    try:
        validated_df = DataSchema.validate(df)
        print("Data validation successful!")
        return validated_df
    except pa.errors.SchemaError as e:
        print(f"Validation failed: {str(e)}")
        raise

# Example usage
data = pd.DataFrame({
    'age': [25, 35, 45],
    'income': [50000.0, 75000.0, 90000.0],
    'education_years': [16, 18, 14],
    'employment_status': ['employed', 'employed', 'student']
})

validated_data = validate_dataset(data)
```

Slide 4: Feature Engineering Pipeline

A robust feature engineering pipeline ensures reproducible data transformations and maintains consistency between training and inference. This implementation demonstrates a modular approach to feature creation with proper validation and error handling.

```python
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Dict, Optional
import numpy as np

class FeatureEngineeringPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 numerical_features: List[str],
                 categorical_features: List[str],
                 derived_features_config: Dict[str, dict]):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.derived_features_config = derived_features_config
        self.feature_statistics = {}
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        # Calculate and store statistics for numerical features
        self.feature_statistics['numerical'] = {
            col: {
                'mean': X[col].mean(),
                'std': X[col].std(),
                'median': X[col].median()
            } for col in self.numerical_features
        }
        
        # Calculate and store statistics for categorical features
        self.feature_statistics['categorical'] = {
            col: X[col].value_counts().to_dict()
            for col in self.categorical_features
        }
        
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = X.copy()
        
        # Transform numerical features
        for col in self.numerical_features:
            X_transformed[f'{col}_scaled'] = (
                X_transformed[col] - self.feature_statistics['numerical'][col]['mean']
            ) / self.feature_statistics['numerical'][col]['std']
            
        # Transform categorical features
        for col in self.categorical_features:
            X_transformed[f'{col}_encoded'] = X_transformed[col].map(
                self.feature_statistics['categorical'][col]
            )
            
        # Create derived features
        for feature_name, config in self.derived_features_config.items():
            if config['type'] == 'interaction':
                X_transformed[feature_name] = X_transformed[config['col1']] * X_transformed[config['col2']]
            elif config['type'] == 'polynomial':
                X_transformed[feature_name] = X_transformed[config['col']] ** config['degree']
                
        return X_transformed

# Example usage
feature_config = {
    'income_education_interaction': {
        'type': 'interaction',
        'col1': 'income',
        'col2': 'education_years'
    },
    'age_squared': {
        'type': 'polynomial',
        'col': 'age',
        'degree': 2
    }
}

pipeline = FeatureEngineeringPipeline(
    numerical_features=['age', 'income', 'education_years'],
    categorical_features=['employment_status'],
    derived_features_config=feature_config
)

X_transformed = pipeline.fit_transform(data)
print("Transformed features shape:", X_transformed.shape)
```

Slide 5: Data Pipeline Logging and Monitoring

Implementing comprehensive logging and monitoring in ML pipelines is crucial for tracking data flow, catching anomalies, and maintaining model performance. This system captures key metrics and validation results throughout the pipeline execution.

```python
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional
import mlflow

class PipelineMonitor:
    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.start_time = None
        self.metrics = {}
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        self.logger = logging.getLogger(pipeline_name)
        
    def start_monitoring(self):
        self.start_time = time.time()
        self.logger.info(f"Started pipeline: {self.pipeline_name}")
        mlflow.start_run(run_name=self.pipeline_name)
        
    def log_step(self, step_name: str, metrics: Dict[str, Any]):
        duration = time.time() - self.start_time
        self.metrics[step_name] = metrics
        
        # Log metrics to MLflow
        for metric_name, value in metrics.items():
            mlflow.log_metric(f"{step_name}_{metric_name}", value)
        
        self.logger.info(
            f"Step: {step_name} - Duration: {duration:.2f}s - Metrics: {metrics}"
        )
        
    def log_validation_results(self, validation_results: Dict[str, Any]):
        self.logger.info(f"Validation Results: {validation_results}")
        mlflow.log_params({"validation": validation_results})
        
    def end_monitoring(self):
        total_duration = time.time() - self.start_time
        self.logger.info(
            f"Pipeline completed - Total duration: {total_duration:.2f}s"
        )
        mlflow.end_run()

# Example usage
monitor = PipelineMonitor("feature_engineering_pipeline")
monitor.start_monitoring()

# Log data validation step
monitor.log_step("data_validation", {
    "missing_values": 0.02,
    "outliers_detected": 15,
    "schema_validation": "passed"
})

# Log feature engineering metrics
monitor.log_step("feature_engineering", {
    "features_created": 10,
    "memory_usage_mb": 150,
    "null_values_after": 0
})

monitor.end_monitoring()
```

Slide 6: Pipeline Error Handling and Recovery

Robust error handling mechanisms ensure pipeline resilience and provide clear debugging information. This implementation demonstrates comprehensive exception handling and automatic recovery strategies for common pipeline failures.

```python
from typing import Optional, Callable
import traceback
from functools import wraps
import pickle

class PipelineError(Exception):
    """Base class for pipeline-specific exceptions"""
    pass

class DataValidationError(PipelineError):
    """Raised when data validation fails"""
    pass

class ProcessingError(PipelineError):
    """Raised when data processing fails"""
    pass

def retry_operation(max_attempts: int = 3, 
                   recovery_func: Optional[Callable] = None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        raise PipelineError(
                            f"Operation failed after {max_attempts} attempts: {str(e)}"
                        )
                    if recovery_func:
                        recovery_func(*args, **kwargs)
            return None
        return wrapper
    return decorator

class ResilientPipeline:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        self.error_handler = self._setup_error_handler()
        
    def _setup_error_handler(self):
        handler = logging.FileHandler('pipeline_errors.log')
        handler.setLevel(logging.ERROR)
        return handler
        
    def save_checkpoint(self, data: Any, step_name: str):
        checkpoint_path = f"{self.checkpoint_dir}/checkpoint_{step_name}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(data, f)
            
    def load_checkpoint(self, step_name: str) -> Any:
        checkpoint_path = f"{self.checkpoint_dir}/checkpoint_{step_name}.pkl"
        try:
            with open(checkpoint_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
            
    @retry_operation(max_attempts=3)
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            # Validate data
            if data.isnull().any().any():
                raise DataValidationError("Found null values in data")
                
            # Process data
            processed_data = self._process_step(data)
            self.save_checkpoint(processed_data, 'processing')
            
            return processed_data
            
        except Exception as e:
            self.error_handler.handle(logging.LogRecord(
                'pipeline', logging.ERROR, '', 0, str(e), (), None
            ))
            raise ProcessingError(f"Processing failed: {str(e)}")
            
    def _process_step(self, data: pd.DataFrame) -> pd.DataFrame:
        # Implementation of actual processing logic
        return data

# Example usage
pipeline = ResilientPipeline(checkpoint_dir="./checkpoints")

try:
    processed_data = pipeline.process_data(data)
except PipelineError as e:
    print(f"Pipeline failed: {str(e)}")
    # Load last checkpoint
    processed_data = pipeline.load_checkpoint('processing')
```

Slide 7: Parameterized Feature Selection

Feature selection is crucial for model performance and efficiency. This implementation provides a modular approach to feature selection with configurable parameters and statistical validation of selected features.

```python
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from typing import List, Tuple, Optional

class FeatureSelector:
    def __init__(self, 
                 selection_method: str = 'mutual_info',
                 n_features: Optional[int] = None,
                 threshold: Optional[float] = None):
        self.selection_method = selection_method
        self.n_features = n_features
        self.threshold = threshold
        self.selected_features = None
        self.feature_scores = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureSelector':
        if self.selection_method == 'mutual_info':
            selector = SelectKBest(
                score_func=mutual_info_regression,
                k=self.n_features or 'all'
            )
            selector.fit(X, y)
            self.feature_scores = dict(zip(X.columns, selector.scores_))
            
        elif self.selection_method == 'random_forest':
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            self.feature_scores = dict(zip(X.columns, rf.feature_importances_))
        
        # Select features based on threshold or n_features
        if self.threshold:
            self.selected_features = [
                feature for feature, score in self.feature_scores.items()
                if score > self.threshold
            ]
        else:
            sorted_features = sorted(
                self.feature_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            self.selected_features = [
                f[0] for f in sorted_features[:self.n_features]
            ]
            
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_features]
    
    def get_feature_importance_report(self) -> pd.DataFrame:
        report = pd.DataFrame({
            'Feature': self.feature_scores.keys(),
            'Score': self.feature_scores.values()
        })
        return report.sort_values('Score', ascending=False)

# Example usage
X = pd.DataFrame(np.random.randn(1000, 5), 
                columns=['f1', 'f2', 'f3', 'f4', 'f5'])
y = 2*X['f1'] + 3*X['f2'] + np.random.randn(1000)

selector = FeatureSelector(
    selection_method='random_forest',
    n_features=3
)

selector.fit(X, y)
X_selected = selector.transform(X)
importance_report = selector.get_feature_importance_report()
print("Feature Importance Report:\n", importance_report)
```

Slide 8: Model Versioning and Reproducibility

Ensuring reproducibility in machine learning pipelines requires careful tracking of model versions, dependencies, and experimental configurations. This implementation provides a comprehensive versioning system for ML models and their artifacts.

```python
from dataclasses import dataclass
from datetime import datetime
import json
import hashlib
from typing import Dict, Any, Optional
import os

@dataclass
class ModelVersion:
    model_id: str
    version: str
    timestamp: datetime
    parameters: Dict[str, Any]
    dependencies: Dict[str, str]
    metrics: Dict[str, float]
    
class ModelVersionControl:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.versions_file = os.path.join(base_dir, 'versions.json')
        self.versions = self._load_versions()
        
    def _load_versions(self) -> Dict:
        if os.path.exists(self.versions_file):
            with open(self.versions_file, 'r') as f:
                return json.load(f)
        return {}
        
    def _save_versions(self):
        with open(self.versions_file, 'w') as f:
            json.dump(self.versions, f, default=str)
            
    def _generate_version_hash(self, params: Dict) -> str:
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()[:8]
        
    def create_version(self,
                      model_id: str,
                      parameters: Dict[str, Any],
                      dependencies: Dict[str, str],
                      metrics: Dict[str, float]) -> ModelVersion:
        version = self._generate_version_hash(parameters)
        timestamp = datetime.now()
        
        model_version = ModelVersion(
            model_id=model_id,
            version=version,
            timestamp=timestamp,
            parameters=parameters,
            dependencies=dependencies,
            metrics=metrics
        )
        
        if model_id not in self.versions:
            self.versions[model_id] = {}
            
        self.versions[model_id][version] = {
            'timestamp': str(timestamp),
            'parameters': parameters,
            'dependencies': dependencies,
            'metrics': metrics
        }
        
        self._save_versions()
        return model_version
    
    def get_version(self,
                   model_id: str,
                   version: str) -> Optional[ModelVersion]:
        if model_id in self.versions and version in self.versions[model_id]:
            v = self.versions[model_id][version]
            return ModelVersion(
                model_id=model_id,
                version=version,
                timestamp=datetime.fromisoformat(v['timestamp']),
                parameters=v['parameters'],
                dependencies=v['dependencies'],
                metrics=v['metrics']
            )
        return None

# Example usage
mvc = ModelVersionControl("./model_versions")

model_version = mvc.create_version(
    model_id="random_forest_classifier",
    parameters={
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2
    },
    dependencies={
        "scikit-learn": "1.0.2",
        "pandas": "1.4.0"
    },
    metrics={
        "accuracy": 0.85,
        "f1_score": 0.83
    }
)

print(f"Created version: {model_version.version}")
retrieved_version = mvc.get_version(
    "random_forest_classifier",
    model_version.version
)
```

Slide 9: Pipeline Performance Monitoring

Continuous monitoring of pipeline performance is essential for maintaining model quality in production. This implementation provides real-time monitoring of key performance indicators and data drift detection.

```python
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class DriftMetrics:
    ks_statistic: float
    p_value: float
    is_drift: bool

class PipelineMonitor:
    def __init__(self, 
                 reference_data: pd.DataFrame,
                 drift_threshold: float = 0.05):
        self.reference_data = reference_data
        self.drift_threshold = drift_threshold
        self.reference_statistics = self._compute_statistics(reference_data)
        
    def _compute_statistics(self, data: pd.DataFrame) -> Dict:
        stats_dict = {}
        for column in data.select_dtypes(include=[np.number]).columns:
            stats_dict[column] = {
                'mean': data[column].mean(),
                'std': data[column].std(),
                'median': data[column].median(),
                'distribution': data[column].values
            }
        return stats_dict
        
    def detect_drift(self, 
                    current_data: pd.DataFrame) -> Dict[str, DriftMetrics]:
        drift_results = {}
        
        for column in self.reference_statistics.keys():
            if column in current_data.columns:
                ks_stat, p_value = stats.ks_2samp(
                    self.reference_statistics[column]['distribution'],
                    current_data[column].values
                )
                
                is_drift = p_value < self.drift_threshold
                
                drift_results[column] = DriftMetrics(
                    ks_statistic=ks_stat,
                    p_value=p_value,
                    is_drift=is_drift
                )
                
        return drift_results
    
    def compute_performance_metrics(self,
                                  predictions: np.ndarray,
                                  actual: np.ndarray) -> Dict:
        return {
            'mse': np.mean((predictions - actual) ** 2),
            'mae': np.mean(np.abs(predictions - actual)),
            'r2': 1 - (np.sum((actual - predictions) ** 2) / 
                      np.sum((actual - actual.mean()) ** 2))
        }
        
    def generate_monitoring_report(self,
                                 current_data: pd.DataFrame,
                                 predictions: Optional[np.ndarray] = None,
                                 actual: Optional[np.ndarray] = None) -> Dict:
        report = {
            'drift_analysis': self.detect_drift(current_data),
            'data_statistics': self._compute_statistics(current_data)
        }
        
        if predictions is not None and actual is not None:
            report['performance_metrics'] = self.compute_performance_metrics(
                predictions, actual
            )
            
        return report

# Example usage
reference_data = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 1000),
    'feature2': np.random.normal(5, 2, 1000)
})

monitor = PipelineMonitor(reference_data)

# Simulate drift in current data
current_data = pd.DataFrame({
    'feature1': np.random.normal(0.5, 1, 1000),  # Shifted mean
    'feature2': np.random.normal(5, 2, 1000)
})

# Generate predictions and actual values for performance monitoring
predictions = np.random.normal(0, 1, 1000)
actual = np.random.normal(0, 1, 1000)

report = monitor.generate_monitoring_report(
    current_data,
    predictions,
    actual
)

print("Monitoring Report:")
for feature, metrics in report['drift_analysis'].items():
    print(f"\n{feature} drift analysis:")
    print(f"KS statistic: {metrics.ks_statistic:.4f}")
    print(f"p-value: {metrics.p_value:.4f}")
    print(f"Drift detected: {metrics.is_drift}")
```

Slide 10: Automated Model Retraining Pipeline

An automated model retraining pipeline ensures model performance remains optimal as data distributions change. This implementation provides a comprehensive framework for monitoring, triggering, and executing model retraining.

```python
from sklearn.base import BaseEstimator
from typing import Optional, Callable
import joblib
from datetime import datetime, timedelta

class ModelRetrainingPipeline:
    def __init__(self,
                 model: BaseEstimator,
                 retrain_trigger: Callable[[Dict], bool],
                 performance_threshold: float,
                 model_path: str):
        self.model = model
        self.retrain_trigger = retrain_trigger
        self.performance_threshold = performance_threshold
        self.model_path = model_path
        self.last_training_date = None
        self.training_history = []
        
    def evaluate_model(self,
                      X: pd.DataFrame,
                      y: pd.Series) -> Dict[str, float]:
        predictions = self.model.predict(X)
        metrics = {
            'mse': mean_squared_error(y, predictions),
            'mae': mean_absolute_error(y, predictions),
            'r2': r2_score(y, predictions)
        }
        return metrics
        
    def should_retrain(self,
                      current_metrics: Dict[str, float],
                      new_data: pd.DataFrame) -> bool:
        # Check if model performance has degraded
        if current_metrics['r2'] < self.performance_threshold:
            return True
            
        # Check custom retrain trigger
        if self.retrain_trigger(current_metrics):
            return True
            
        return False
        
    def retrain_model(self,
                     X: pd.DataFrame,
                     y: pd.Series,
                     validation_data: Optional[Tuple] = None):
        try:
            # Fit model on new data
            self.model.fit(X, y)
            
            # Evaluate on validation data if provided
            metrics = {}
            if validation_data:
                X_val, y_val = validation_data
                metrics = self.evaluate_model(X_val, y_val)
                
            # Save training record
            training_record = {
                'timestamp': datetime.now(),
                'metrics': metrics,
                'data_shape': X.shape
            }
            self.training_history.append(training_record)
            
            # Save model
            self.save_model()
            self.last_training_date = datetime.now()
            
            return metrics
            
        except Exception as e:
            logging.error(f"Model retraining failed: {str(e)}")
            raise
            
    def save_model(self):
        joblib.dump(self.model, self.model_path)
        
    def load_model(self):
        self.model = joblib.load(self.model_path)

# Example usage
def custom_retrain_trigger(metrics: Dict[str, float]) -> bool:
    return metrics['r2'] < 0.8 or metrics['mse'] > 0.1

pipeline = ModelRetrainingPipeline(
    model=RandomForestRegressor(n_estimators=100),
    retrain_trigger=custom_retrain_trigger,
    performance_threshold=0.75,
    model_path='model.joblib'
)

# Simulate new data and evaluate model
X_new = pd.DataFrame(np.random.randn(1000, 5))
y_new = np.random.randn(1000)

current_metrics = pipeline.evaluate_model(X_new, y_new)

if pipeline.should_retrain(current_metrics, X_new):
    print("Initiating model retraining...")
    new_metrics = pipeline.retrain_model(X_new, y_new)
    print("Retraining completed. New metrics:", new_metrics)
```

Slide 11: Pipeline Output Validation

Output validation ensures that model predictions meet quality standards before deployment. This implementation provides comprehensive validation checks for model outputs and handles edge cases appropriately.

```python
from typing import Any, Dict, List, Optional, Union
import numpy as np
from dataclasses import dataclass

@dataclass
class ValidationResult:
    passed: bool
    errors: List[str]
    warnings: List[str]

class OutputValidator:
    def __init__(self,
                 expected_range: Optional[Tuple[float, float]] = None,
                 allowed_classes: Optional[List[Any]] = None,
                 custom_checks: Optional[List[Callable]] = None):
        self.expected_range = expected_range
        self.allowed_classes = allowed_classes
        self.custom_checks = custom_checks or []
        
    def validate_predictions(self,
                           predictions: Union[np.ndarray, pd.Series],
                           confidence_scores: Optional[np.ndarray] = None) -> ValidationResult:
        errors = []
        warnings = []
        
        # Check for NaN values
        if np.any(np.isnan(predictions)):
            errors.append("Predictions contain NaN values")
            
        # Validate numerical range
        if self.expected_range:
            min_val, max_val = self.expected_range
            if np.any(predictions < min_val) or np.any(predictions > max_val):
                errors.append(
                    f"Predictions outside expected range [{min_val}, {max_val}]"
                )
                
        # Validate categorical predictions
        if self.allowed_classes:
            invalid_classes = set(predictions) - set(self.allowed_classes)
            if invalid_classes:
                errors.append(
                    f"Invalid prediction classes found: {invalid_classes}"
                )
                
        # Check confidence scores
        if confidence_scores is not None:
            if np.any(confidence_scores < 0) or np.any(confidence_scores > 1):
                errors.append(
                    "Confidence scores must be between 0 and 1"
                )
            if np.mean(confidence_scores) < 0.5:
                warnings.append(
                    "Low average confidence score detected"
                )
                
        # Run custom validation checks
        for check in self.custom_checks:
            try:
                result = check(predictions)
                if not result:
                    errors.append(f"Custom check failed: {check.__name__}")
            except Exception as e:
                errors.append(f"Error in custom check {check.__name__}: {str(e)}")
                
        return ValidationResult(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def validate_prediction_distribution(self,
                                      predictions: np.ndarray,
                                      reference_distribution: np.ndarray,
                                      threshold: float = 0.05) -> ValidationResult:
        errors = []
        warnings = []
        
        # Perform Kolmogorov-Smirnov test
        ks_stat, p_value = stats.ks_2samp(predictions, reference_distribution)
        
        if p_value < threshold:
            warnings.append(
                f"Prediction distribution significantly different from reference "
                f"(KS test p-value: {p_value:.4f})"
            )
            
        # Check for distribution moments
        pred_mean = np.mean(predictions)
        ref_mean = np.mean(reference_distribution)
        
        if abs(pred_mean - ref_mean) > np.std(reference_distribution):
            warnings.append(
                f"Mean of predictions ({pred_mean:.2f}) significantly different "
                f"from reference ({ref_mean:.2f})"
            )
            
        return ValidationResult(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

# Example usage
def check_outliers(predictions: np.ndarray) -> bool:
    z_scores = np.abs(stats.zscore(predictions))
    return np.sum(z_scores > 3) / len(predictions) < 0.01

validator = OutputValidator(
    expected_range=(-10, 10),
    allowed_classes=None,
    custom_checks=[check_outliers]
)

# Generate sample predictions
predictions = np.random.normal(0, 2, 1000)
confidence_scores = np.random.uniform(0.6, 1.0, 1000)

# Validate predictions
result = validator.validate_predictions(predictions, confidence_scores)

print("Validation Results:")
print(f"Passed: {result.passed}")
if result.errors:
    print("Errors:", *result.errors, sep="\n- ")
if result.warnings:
    print("Warnings:", *result.warnings, sep="\n- ")
```

Slide 12: Automated Documentation Generator

Maintaining comprehensive documentation for ML pipelines is crucial for reproducibility and collaboration. This implementation automatically generates documentation for pipeline components and their configurations.

```python
from typing import Dict, Any, Optional
import inspect
import yaml
import json
from datetime import datetime

class PipelineDocumentGenerator:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.documentation = {
            'generated_at': datetime.now().isoformat(),
            'pipeline_components': {},
            'configurations': {},
            'dependencies': {}
        }
        
    def document_component(self,
                         component: Any,
                         component_name: str,
                         description: Optional[str] = None):
        # Extract component information
        doc = inspect.getdoc(component) or "No documentation available"
        signature = str(inspect.signature(component.__init__))
        source = inspect.getsource(component)
        
        self.documentation['pipeline_components'][component_name] = {
            'description': description or doc,
            'signature': signature,
            'source': source,
            'methods': self._document_methods(component)
        }
        
    def _document_methods(self, component: Any) -> Dict:
        methods = {}
        for name, method in inspect.getmembers(component, inspect.isfunction):
            if not name.startswith('_'):  # Skip private methods
                doc = inspect.getdoc(method) or "No documentation available"
                signature = str(inspect.signature(method))
                methods[name] = {
                    'description': doc,
                    'signature': signature
                }
        return methods
        
    def document_configuration(self,
                             config: Dict[str, Any],
                             config_name: str):
        self.documentation['configurations'][config_name] = config
        
    def document_dependencies(self,
                            dependencies: Dict[str, str]):
        self.documentation['dependencies'] = dependencies
        
    def generate_markdown(self) -> str:
        markdown = f"# Pipeline Documentation\n\n"
        markdown += f"Generated at: {self.documentation['generated_at']}\n\n"
        
        # Components
        markdown += "## Pipeline Components\n\n"
        for name, component in self.documentation['pipeline_components'].items():
            markdown += f"### {name}\n\n"
            markdown += f"{component['description']}\n\n"
            markdown += f"**Signature:**\n```python\n{component['signature']}\n```\n\n"
            
            markdown += "**Methods:**\n\n"
            for method_name, method in component['methods'].items():
                markdown += f"#### {method_name}\n"
                markdown += f"{method['description']}\n"
                markdown += f"```python\n{method['signature']}\n```\n\n"
                
        # Configurations
        markdown += "## Configurations\n\n"
        for name, config in self.documentation['configurations'].items():
            markdown += f"### {name}\n\n"
            markdown += f"```yaml\n{yaml.dump(config)}\n```\n\n"
            
        # Dependencies
        markdown += "## Dependencies\n\n"
        for dep, version in self.documentation['dependencies'].items():
            markdown += f"- {dep}: {version}\n"
            
        return markdown
        
    def save_documentation(self):
        # Save markdown
        with open(f"{self.output_dir}/pipeline_docs.md", 'w') as f:
            f.write(self.generate_markdown())
            
        # Save JSON
        with open(f"{self.output_dir}/pipeline_docs.json", 'w') as f:
            json.dump(self.documentation, f, indent=2)

# Example usage
doc_gen = PipelineDocumentGenerator("./docs")

# Document pipeline component
doc_gen.document_component(
    OutputValidator,
    "OutputValidator",
    "Validates model predictions and handles edge cases"
)

# Document configuration
doc_gen.document_configuration({
    'model_params': {
        'n_estimators': 100,
        'max_depth': 10
    },
    'validation_params': {
        'expected_range': [-10, 10],
        'confidence_threshold': 0.5
    }
}, "model_config")

# Document dependencies
doc_gen.document_dependencies({
    'scikit-learn': '1.0.2',
    'pandas': '1.4.0',
    'numpy': '1.21.0'
})

# Generate and save documentation
doc_gen.save_documentation()
```

Slide 13: Pipeline Testing Framework

A comprehensive testing framework ensures pipeline reliability and robustness. This implementation provides unit tests, integration tests, and end-to-end testing capabilities for ML pipelines.

```python
import unittest
from typing import Any, Callable, Dict, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class TestCase:
    input_data: Any
    expected_output: Any
    description: str

class PipelineTestFramework:
    def __init__(self):
        self.test_cases = {}
        self.pipeline_components = {}
        
    def register_component(self,
                         name: str,
                         component: Any,
                         test_cases: List[TestCase]):
        self.pipeline_components[name] = component
        self.test_cases[name] = test_cases
        
    def run_unit_tests(self) -> Dict[str, Dict[str, bool]]:
        results = {}
        for component_name, component in self.pipeline_components.items():
            results[component_name] = {}
            for idx, test_case in enumerate(self.test_cases[component_name]):
                try:
                    output = component(test_case.input_data)
                    passed = np.allclose(output, test_case.expected_output)
                    results[component_name][f"test_{idx}"] = passed
                except Exception as e:
                    results[component_name][f"test_{idx}"] = False
        return results
        
    def run_integration_test(self,
                           component_sequence: List[str],
                           test_data: Any,
                           expected_output: Any) -> bool:
        try:
            current_output = test_data
            for component_name in component_sequence:
                component = self.pipeline_components[component_name]
                current_output = component(current_output)
            return np.allclose(current_output, expected_output)
        except Exception as e:
            print(f"Integration test failed: {str(e)}")
            return False

class MLPipelineTests(unittest.TestCase):
    def setUp(self):
        self.test_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        self.test_labels = np.random.randint(0, 2, 100)
        
    def test_data_validation(self):
        validator = DataValidator()
        validation_result = validator.validate(self.test_data)
        self.assertTrue(validation_result.passed)
        
    def test_feature_engineering(self):
        engineer = FeatureEngineer()
        transformed_data = engineer.transform(self.test_data)
        self.assertEqual(
            transformed_data.shape[1],
            self.test_data.shape[1] * 2  # Assuming feature engineering doubles features
        )
        
    def test_model_training(self):
        model = RandomForestClassifier()
        model.fit(self.test_data, self.test_labels)
        predictions = model.predict(self.test_data)
        self.assertEqual(len(predictions), len(self.test_labels))
        
    def test_end_to_end_pipeline(self):
        pipeline = MLPipeline(
            validator=DataValidator(),
            engineer=FeatureEngineer(),
            model=RandomForestClassifier()
        )
        
        # Train pipeline
        pipeline.train(self.test_data, self.test_labels)
        
        # Make predictions
        predictions = pipeline.predict(self.test_data)
        
        # Assertions
        self.assertEqual(len(predictions), len(self.test_labels))
        self.assertTrue(all(isinstance(p, (int, float)) for p in predictions))
        
    def test_pipeline_persistence(self):
        pipeline = MLPipeline()
        pipeline.train(self.test_data, self.test_labels)
        
        # Save pipeline
        pipeline.save('test_pipeline.pkl')
        
        # Load pipeline
        loaded_pipeline = MLPipeline.load('test_pipeline.pkl')
        
        # Compare predictions
        original_predictions = pipeline.predict(self.test_data)
        loaded_predictions = loaded_pipeline.predict(self.test_data)
        
        np.testing.assert_array_almost_equal(
            original_predictions,
            loaded_predictions
        )

# Example usage
if __name__ == '__main__':
    # Create test framework
    test_framework = PipelineTestFramework()
    
    # Register components with test cases
    test_framework.register_component(
        "data_validator",
        DataValidator(),
        [
            TestCase(
                input_data=pd.DataFrame({'a': [1, 2, 3]}),
                expected_output=True,
                description="Valid data test"
            ),
            TestCase(
                input_data=pd.DataFrame({'a': [1, np.nan, 3]}),
                expected_output=False,
                description="Invalid data test"
            )
        ]
    )
    
    # Run tests
    unit_test_results = test_framework.run_unit_tests()
    print("Unit Test Results:", unit_test_results)
    
    # Run integration test
    integration_result = test_framework.run_integration_test(
        component_sequence=["data_validator", "feature_engineer", "model"],
        test_data=pd.DataFrame({'feature1': [1, 2, 3]}),
        expected_output=np.array([0, 1, 0])
    )
    print("Integration Test Result:", integration_result)
    
    # Run unittest suite
    unittest.main()
```

Slide 14: Additional Resources

*   ArXiv Paper: "A Survey of Machine Learning Pipeline Implementation Tools" [https://arxiv.org/abs/2106.03089](https://arxiv.org/abs/2106.03089)
*   ArXiv Paper: "Best Practices for Machine Learning Engineering" [https://arxiv.org/abs/2109.08256](https://arxiv.org/abs/2109.08256)
*   ArXiv Paper: "Automated Machine Learning: State of the Art and Open Challenges" [https://arxiv.org/abs/2012.05826](https://arxiv.org/abs/2012.05826)
*   ML Pipeline Development Guide: [https://developers.google.com/machine-learning/guides/rules-of-ml](https://developers.google.com/machine-learning/guides/rules-of-ml)
*   Scikit-learn Pipeline Documentation: [https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
*   MLflow Documentation for Pipeline Tracking: [https://www.mlflow.org/docs/latest/index.html](https://www.mlflow.org/docs/latest/index.html)

