## Avoiding Bad Coding Practices in Data Science

Slide 1: Configuration Management Using YAML

Configuration management is crucial for maintaining clean, scalable code in data science projects. Using YAML files to store parameters, model configurations, and data paths enables easier maintenance and reproducibility while eliminating hardcoded values throughout the codebase.

```python
import yaml
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelConfig:
    learning_rate: float
    batch_size: int
    epochs: int
    layers: List[int]
    dropout: Optional[float] = 0.2

def load_config(path: str = "config.yml") -> ModelConfig:
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)
    return ModelConfig(**config_dict)

# Example config.yml content:
"""
learning_rate: 0.001
batch_size: 32
epochs: 100
layers: [64, 32, 16]
dropout: 0.2
"""

config = load_config()
print(f"Model configuration: {config}")
```

Slide 2: Type Annotations and Documentation

Type hints and comprehensive documentation are essential for code maintenance and collaboration. They provide clear interfaces, catch type-related bugs early, and make code self-documenting. Using tools like mypy for static type checking enhances code quality.

```python
from typing import Dict, List, Union
import numpy as np
import pandas as pd

def preprocess_features(
    data: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str]
) -> Dict[str, Union[np.ndarray, sparse.csr_matrix]]:
    """
    Preprocess numeric and categorical features for model training.
    
    Args:
        data: Input DataFrame containing raw features
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names
        
    Returns:
        Dictionary containing processed numeric and categorical features
    """
    numeric_features = data[numeric_cols].to_numpy()
    categorical_features = pd.get_dummies(data[categorical_cols], sparse=True)
    
    return {
        "numeric": numeric_features,
        "categorical": categorical_features
    }
```

Slide 3: Function Modularization for Data Processing

Effective modularization involves breaking down complex data processing tasks into discrete, focused functions. Each function should handle a single responsibility, making the code more maintainable and easier to test while promoting code reuse across projects.

```python
def load_data(filepath: str) -> pd.DataFrame:
    """Load and validate raw data"""
    data = pd.read_csv(filepath)
    validate_schema(data)
    return data

def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in dataset"""
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    df[categorical_cols] = df[categorical_cols].fillna('MISSING')
    return df

def validate_schema(df: pd.DataFrame) -> None:
    """Validate dataframe schema and types"""
    required_cols = ['feature1', 'feature2', 'target']
    if not all(col in df.columns for col in required_cols):
        raise ValueError("Missing required columns")
```

Slide 4: Error Handling and Logging

Robust error handling and logging are crucial for maintaining production-ready data science code. Implementing proper exception handling and detailed logging helps track issues, debug problems, and monitor model performance in production.

```python
import logging
from functools import wraps
from typing import Callable

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def log_exceptions(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            logger.info(f"Starting {func.__name__}")
            result = func(*args, **kwargs)
            logger.info(f"Completed {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper

@log_exceptions
def process_features(data: pd.DataFrame) -> np.ndarray:
    if data.empty:
        raise ValueError("Empty DataFrame provided")
    return data.values
```

Slide 5: Configuration Through Environment Variables

Using environment variables for configuration management enhances security and deployment flexibility. This approach prevents sensitive information from being hardcoded and allows for easy configuration changes across different environments.

```python
import os
from dotenv import load_dotenv
from dataclasses import dataclass

@dataclass
class DatabaseConfig:
    host: str
    port: int
    username: str
    password: str
    database: str

def load_database_config() -> DatabaseConfig:
    load_dotenv()
    return DatabaseConfig(
        host=os.getenv('DB_HOST', 'localhost'),
        port=int(os.getenv('DB_PORT', '5432')),
        username=os.getenv('DB_USER', 'default'),
        password=os.getenv('DB_PASSWORD', ''),
        database=os.getenv('DB_NAME', 'mydb')
    )

# Usage with .env file:
"""
DB_HOST=production.db.server
DB_PORT=5432
DB_USER=admin
DB_PASSWORD=secure_password
DB_NAME=production_db
"""
```

Slide 6: Data Validation Pipeline

Data validation ensures data quality and consistency throughout the machine learning pipeline. Implementing robust validation checks helps catch data issues early and prevents model training on corrupted or invalid data while maintaining code readability and maintainability.

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd

@dataclass
class ValidationReport:
    is_valid: bool
    errors: Dict[str, List[str]]

class DataValidator:
    def __init__(self):
        self.numeric_columns = ['age', 'salary']
        self.categorical_columns = ['department', 'position']
        
    def validate_dataset(self, df: pd.DataFrame) -> ValidationReport:
        errors = {}
        
        # Check missing values
        missing = df.isnull().sum()
        if missing.any():
            errors['missing_values'] = missing[missing > 0].to_dict()
            
        # Validate numeric ranges
        for col in self.numeric_columns:
            if (df[col] < 0).any():
                errors.setdefault('range_errors', []).append(f"{col} contains negative values")
                
        # Validate categorical values
        valid_departments = {'IT', 'HR', 'Sales'}
        invalid_depts = set(df['department']) - valid_departments
        if invalid_depts:
            errors.setdefault('category_errors', []).append(f"Invalid departments: {invalid_depts}")
            
        return ValidationReport(
            is_valid=len(errors) == 0,
            errors=errors
        )

# Example usage
validator = DataValidator()
validation_result = validator.validate_dataset(df)
print(f"Data validation passed: {validation_result.is_valid}")
```

Slide 7: Feature Engineering Pipeline

Feature engineering is crucial for model performance. This implementation demonstrates a clean, modular approach to creating and transforming features while maintaining code clarity and type safety through proper annotations.

```python
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, date_columns: List[str]):
        self.date_columns = date_columns
        
    def fit(self, X: pd.DataFrame, y=None):
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        # Extract date features
        for col in self.date_columns:
            X[f'{col}_year'] = pd.to_datetime(X[col]).dt.year
            X[f'{col}_month'] = pd.to_datetime(X[col]).dt.month
            X[f'{col}_day'] = pd.to_datetime(X[col]).dt.day
            
        # Create interaction features
        numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                X[f'{col1}_{col2}_interaction'] = X[col1] * X[col2]
                
        return X

# Example usage
engineer = FeatureEngineer(date_columns=['transaction_date'])
transformed_data = engineer.fit_transform(raw_data)
```

Slide 8: Model Training Pipeline

A well-structured model training pipeline ensures reproducibility and maintainability. This implementation shows how to create a clean training workflow with proper logging and parameter management.

```python
from dataclasses import dataclass
from typing import Optional, Dict, Any
from sklearn.model_selection import cross_val_score

@dataclass
class TrainingConfig:
    model_params: Dict[str, Any]
    cv_folds: int = 5
    random_seed: int = 42

class ModelTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.cv_scores = None
        np.random.seed(config.random_seed)
        
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        self.model = RandomForestClassifier(**self.config.model_params)
        
        # Perform cross-validation
        self.cv_scores = cross_val_score(
            self.model, X, y,
            cv=self.config.cv_folds,
            scoring='accuracy'
        )
        
        # Train final model
        self.model.fit(X, y)
        
        return {
            'cv_mean': self.cv_scores.mean(),
            'cv_std': self.cv_scores.std(),
            'train_score': self.model.score(X, y)
        }

# Example usage
config = TrainingConfig(
    model_params={'n_estimators': 100, 'max_depth': 10},
    cv_folds=5
)
trainer = ModelTrainer(config)
metrics = trainer.train(X_train, y_train)
```

Slide 9: Model Evaluation Framework

Creating a comprehensive evaluation framework helps assess model performance across multiple metrics. This implementation provides a clean interface for model evaluation while maintaining code modularity.

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

class ModelEvaluator:
    def __init__(self, model, X_test: np.ndarray, y_test: np.ndarray):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = None
        
    def evaluate(self) -> Dict[str, float]:
        self.y_pred = self.model.predict(self.X_test)
        
        metrics = {
            'accuracy': accuracy_score(self.y_test, self.y_pred),
            'precision': precision_score(self.y_test, self.y_pred, average='weighted'),
            'recall': recall_score(self.y_test, self.y_pred, average='weighted'),
            'f1': f1_score(self.y_test, self.y_pred, average='weighted')
        }
        
        return metrics
    
    def save_results(self, filepath: str):
        metrics = self.evaluate()
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=4)
        
# Example usage
evaluator = ModelEvaluator(trained_model, X_test, y_test)
results = evaluator.evaluate()
evaluator.save_results('model_metrics.json')
```

Slide 10: Unit Testing Framework for Data Science Code

A robust testing framework for data science code needs to cover data preprocessing, feature engineering, model training, and evaluation components. Proper unit tests ensure code reliability and make maintenance easier while catching potential issues early.

```python
import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

class TestDataPreprocessing(unittest.TestCase):
    def setUp(self):
        self.raw_data = pd.DataFrame({
            'numeric': [1, 2, None, 4, 5],
            'categorical': ['A', 'B', None, 'A', 'C']
        })
        
    def test_missing_value_handling(self):
        preprocessed = clean_missing_values(self.raw_data)
        self.assertEqual(preprocessed['numeric'].isna().sum(), 0)
        self.assertEqual(preprocessed['categorical'].isna().sum(), 0)
        
    def test_data_validation(self):
        validation_result = validate_data_types(self.raw_data)
        self.assertTrue(validation_result.is_valid)
        self.assertEqual(len(validation_result.errors), 0)

class TestModelTraining(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_classification(
            n_samples=100, 
            n_features=5,
            random_state=42
        )
        
    def test_model_fitting(self):
        model = train_model(self.X, self.y)
        predictions = model.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))
        
    def test_model_performance(self):
        accuracy = evaluate_model(self.X, self.y)
        self.assertGreater(accuracy, 0.5)

if __name__ == '__main__':
    unittest.main()
```

Slide 11: Production Model Deployment

Production deployment requires careful handling of model versioning, input validation, and performance monitoring. This implementation shows a clean approach to deploying machine learning models while maintaining code quality.

```python
from typing import Dict, Any
import joblib
import time

class ProductionModel:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
        self.inference_times = []
        self.prediction_counts = 0
        
    def predict(self, features: Dict[str, Any]) -> float:
        start_time = time.time()
        
        # Validate input
        self._validate_input(features)
        
        # Transform features
        X = self._preprocess_features(features)
        
        # Make prediction
        prediction = float(self.model.predict([X])[0])
        
        # Log metrics
        self._log_inference(time.time() - start_time)
        
        return prediction
    
    def _validate_input(self, features: Dict[str, Any]) -> None:
        required_features = {'feature1', 'feature2', 'feature3'}
        if not required_features.issubset(features.keys()):
            raise ValueError(f"Missing required features: {required_features - features.keys()}")
    
    def _preprocess_features(self, features: Dict[str, Any]) -> np.ndarray:
        return np.array([features[f] for f in sorted(features.keys())])
    
    def _log_inference(self, duration: float) -> None:
        self.inference_times.append(duration)
        self.prediction_counts += 1
```

Slide 12: Real-world Implementation: Customer Churn Prediction

In this example, we implement a complete customer churn prediction pipeline using the best practices discussed. This shows how to combine modular code, proper validation, and clean implementation in a real scenario.

```python
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def process_customer_data(
    data: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    
    # Handle missing values
    data[numeric_features] = data[numeric_features].fillna(data[numeric_features].mean())
    data[categorical_features] = data[categorical_features].fillna('Unknown')
    
    # Create features
    X_numeric = data[numeric_features].values
    X_categorical = pd.get_dummies(data[categorical_features]).values
    
    # Combine features
    X = np.hstack([X_numeric, X_categorical])
    y = data['churn'].values
    
    return X, y

def train_churn_model(X: np.ndarray, y: np.ndarray) -> Dict:
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    
    return {
        'model': model,
        'train_score': train_score,
        'val_score': val_score
    }
```

Slide 13: Additional Resources

arxiv.org/abs/1810.03993 - "Hidden Technical Debt in Machine Learning Systems" arxiv.org/abs/2209.07611 - "Software Engineering for AI/ML: An Annotated Bibliography" arxiv.org/abs/2103.07164 - "Machine Learning Operations (MLOps): Overview, Definition, and Architecture"

