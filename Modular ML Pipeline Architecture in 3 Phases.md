## Modular ML Pipeline Architecture in 3 Phases
Slide 1: Feature Engineering Pipeline Architecture

Feature engineering serves as the foundation of any robust machine learning system. This phase handles data preparation, cleaning, and transformation while maintaining consistency across training and inference. A well-structured feature pipeline ensures reproducibility and efficient feature reuse.

```python
from typing import Dict, List
import pandas as pd
import numpy as np

class FeaturePipeline:
    def __init__(self):
        self.feature_store = {}
        self.transformers = {}
    
    def add_transformer(self, name: str, transform_fn):
        """Register a new feature transformer"""
        self.transformers[name] = transform_fn
        
    def process_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Process raw data through transformation pipeline"""
        features = {}
        for name, transformer in self.transformers.items():
            features[name] = transformer(data)
            self.feature_store[name] = features[name]
        return features

# Example Usage
pipeline = FeaturePipeline()

# Add custom transformers
pipeline.add_transformer(
    "numeric_scaler", 
    lambda df: (df.select_dtypes(include=[np.number]) - df.mean()) / df.std()
)

# Process features
raw_data = pd.read_csv("data.csv")
features = pipeline.process_features(raw_data)
```

Slide 2: Feature Store Implementation

The feature store acts as a centralized repository for computed features, ensuring consistency between training and inference. This implementation provides versioning, caching, and retrieval mechanisms for efficient feature management.

```python
import joblib
from datetime import datetime
from pathlib import Path

class FeatureStore:
    def __init__(self, storage_path: str = "feature_store"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.feature_metadata = {}
        
    def save_feature(self, name: str, feature_data: np.ndarray):
        """Save feature with versioning"""
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        feature_path = self.storage_path / f"{name}_v{version}.joblib"
        
        joblib.dump(feature_data, feature_path)
        self.feature_metadata[name] = {
            'version': version,
            'path': str(feature_path),
            'shape': feature_data.shape
        }
        
    def load_feature(self, name: str, version: str = 'latest'):
        """Load feature from store"""
        if version == 'latest':
            version = self.feature_metadata[name]['version']
        
        feature_path = self.storage_path / f"{name}_v{version}.joblib"
        return joblib.load(feature_path)

# Example Usage
store = FeatureStore()
features = np.random.randn(1000, 10)
store.save_feature("customer_embeddings", features)
loaded_features = store.load_feature("customer_embeddings")
```

Slide 3: Training Pipeline Architecture

The training pipeline orchestrates model development through automated stages including data splitting, model initialization, training loops, and validation. This implementation emphasizes reproducibility and proper experiment tracking.

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow

class TrainingPipeline:
    def __init__(self, model, feature_store):
        self.model = model
        self.feature_store = feature_store
        mlflow.set_experiment("ml_training")
        
    def train(self, features: Dict[str, np.ndarray], labels: np.ndarray,
             test_size: float = 0.2):
        """Execute training pipeline with experiment tracking"""
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                "model_type": type(self.model).__name__,
                "test_size": test_size
            })
            
            # Split data
            X = np.hstack(list(features.values()))
            X_train, X_test, y_train, y_test = train_test_split(
                X, labels, test_size=test_size
            )
            
            # Train and evaluate
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(self.model, "model")
            
            return accuracy

# Example Usage
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
pipeline = TrainingPipeline(model, store)
accuracy = pipeline.train(features, labels)
```

Slide 4: Model Registry Implementation

The model registry maintains versions of trained models, facilitating deployment and rollback capabilities. This implementation provides model metadata tracking and standardized serialization formats.

```python
from typing import Optional
import json
from datetime import datetime

class ModelRegistry:
    def __init__(self, registry_path: str = "model_registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True)
        self.metadata_file = self.registry_path / "metadata.json"
        self.load_metadata()
        
    def load_metadata(self):
        """Load existing model metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
            
    def register_model(self, model, name: str, 
                      metrics: Dict[str, float],
                      production: bool = False):
        """Register new model version"""
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.registry_path / f"{name}_v{version}.joblib"
        
        # Save model and metadata
        joblib.dump(model, model_path)
        self.metadata[name] = {
            'version': version,
            'path': str(model_path),
            'metrics': metrics,
            'production': production
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
            
    def load_model(self, name: str, version: Optional[str] = None):
        """Load model from registry"""
        if version is None:
            version = self.metadata[name]['version']
        
        model_path = self.registry_path / f"{name}_v{version}.joblib"
        return joblib.load(model_path)

# Example Usage
registry = ModelRegistry()
registry.register_model(
    model,
    "customer_classifier",
    metrics={"accuracy": accuracy},
    production=True
)
```

Slide 5: Inference Pipeline Architecture

The inference pipeline handles real-time and batch predictions while ensuring consistent feature processing. This implementation includes prediction caching, monitoring, and automated performance tracking to maintain reliability at scale.

```python
import redis
from typing import Union, List
import numpy as np

class InferencePipeline:
    def __init__(self, model_registry, feature_store):
        self.model_registry = model_registry
        self.feature_store = feature_store
        self.cache = redis.Redis(host='localhost', port=6379, db=0)
        self.prediction_metrics = []
        
    def predict(self, input_data: Union[pd.DataFrame, List[Dict]], 
                model_name: str) -> np.ndarray:
        """Execute inference pipeline with caching and monitoring"""
        # Generate cache key
        cache_key = f"{model_name}_{hash(str(input_data))}"
        
        # Check cache
        if self.cache.exists(cache_key):
            return np.frombuffer(self.cache.get(cache_key))
        
        # Load model and process features
        model = self.model_registry.load_model(model_name)
        
        # Transform input data to DataFrame if needed
        if isinstance(input_data, list):
            input_data = pd.DataFrame(input_data)
            
        # Process features consistently
        features = {}
        for feature_name in self.feature_store.feature_metadata.keys():
            transformer = self.feature_store.load_transformer(feature_name)
            features[feature_name] = transformer(input_data)
            
        # Make prediction
        X = np.hstack(list(features.values()))
        prediction = model.predict(X)
        
        # Cache result
        self.cache.set(cache_key, prediction.tobytes())
        
        # Log metrics
        self.prediction_metrics.append({
            'timestamp': datetime.now(),
            'model': model_name,
            'input_shape': X.shape,
            'latency': time.time() - start_time
        })
        
        return prediction

# Example Usage
inference = InferencePipeline(registry, store)
new_data = pd.read_csv("new_customers.csv")
predictions = inference.predict(new_data, "customer_classifier")
```

Slide 6: Automated Model Retraining System

This implementation creates an automated system for detecting model drift and triggering retraining pipelines when performance degrades. It integrates monitoring metrics with automated decision making for maintenance.

```python
from sklearn.metrics import mean_squared_error
import schedule
import time

class ModelMonitor:
    def __init__(self, inference_pipeline, training_pipeline,
                 drift_threshold: float = 0.1):
        self.inference = inference_pipeline
        self.training = training_pipeline
        self.drift_threshold = drift_threshold
        self.performance_history = []
        
    def calculate_drift(self, ground_truth: np.ndarray, 
                       predictions: np.ndarray) -> float:
        """Calculate model drift based on recent performance"""
        current_error = mean_squared_error(ground_truth, predictions)
        if self.performance_history:
            baseline_error = np.mean(self.performance_history[-10:])
            drift = abs(current_error - baseline_error) / baseline_error
            return drift
        return 0.0
        
    def check_and_retrain(self, new_data: pd.DataFrame, 
                         ground_truth: np.ndarray):
        """Monitor performance and trigger retraining if needed"""
        predictions = self.inference.predict(new_data, "production_model")
        drift = self.calculate_drift(ground_truth, predictions)
        
        if drift > self.drift_threshold:
            print(f"Drift detected: {drift:.2f}. Initiating retraining...")
            
            # Retrain model with updated data
            features = self.feature_store.process_features(new_data)
            accuracy = self.training.train(features, ground_truth)
            
            # Update production model if improved
            if accuracy > self.current_best_accuracy:
                self.model_registry.register_model(
                    self.training.model,
                    "production_model",
                    metrics={"accuracy": accuracy},
                    production=True
                )
                print(f"New model deployed with accuracy: {accuracy:.2f}")

# Example Usage
monitor = ModelMonitor(inference, training_pipeline)
schedule.every().hour.do(
    monitor.check_and_retrain, 
    new_data=get_latest_data(), 
    ground_truth=get_latest_labels()
)

while True:
    schedule.run_pending()
    time.sleep(60)
```

Slide 7: Feature Cross-Validation System

A robust system for validating feature quality and stability across different data distributions. This implementation helps identify unreliable features and ensures consistency between training and inference.

```python
from sklearn.model_selection import KFold
from scipy import stats

class FeatureValidator:
    def __init__(self, n_splits: int = 5, stability_threshold: float = 0.05):
        self.n_splits = n_splits
        self.stability_threshold = stability_threshold
        self.validation_results = {}
        
    def validate_feature_stability(self, 
                                 feature_data: np.ndarray, 
                                 feature_name: str):
        """Validate feature stability across different data splits"""
        kf = KFold(n_splits=self.n_splits, shuffle=True)
        distributions = []
        
        for train_idx, val_idx in kf.split(feature_data):
            train_dist = feature_data[train_idx]
            val_dist = feature_data[val_idx]
            
            # Perform Kolmogorov-Smirnov test
            ks_statistic, p_value = stats.ks_2samp(train_dist, val_dist)
            distributions.append({
                'ks_statistic': ks_statistic,
                'p_value': p_value
            })
        
        # Analyze stability
        avg_p_value = np.mean([d['p_value'] for d in distributions])
        is_stable = avg_p_value > self.stability_threshold
        
        self.validation_results[feature_name] = {
            'is_stable': is_stable,
            'avg_p_value': avg_p_value,
            'distributions': distributions
        }
        
        return is_stable

# Example Usage
validator = FeatureValidator()
features = feature_store.load_feature("customer_embeddings")
is_stable = validator.validate_feature_stability(
    features, 
    "customer_embeddings"
)

if not is_stable:
    print(f"Warning: Feature 'customer_embeddings' shows instability")
    print(f"Average p-value: {validator.validation_results['customer_embeddings']['avg_p_value']:.4f}")
```

Slide 8: Distributed Feature Processing

Implementing distributed feature processing using PySpark for handling large-scale datasets efficiently. This system enables parallel feature computation while maintaining data consistency across multiple nodes.

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import *
import numpy as np

class DistributedFeatureProcessor:
    def __init__(self, app_name: str = "distributed_features"):
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.executor.memory", "4g") \
            .getOrCreate()
        
    def register_feature_udf(self, feature_name: str, feature_fn):
        """Register feature computation as Spark UDF"""
        return_type = ArrayType(DoubleType())
        
        @udf(returnType=return_type)
        def feature_udf(*cols):
            return feature_fn(*cols).tolist()
            
        self.spark.udf.register(f"feature_{feature_name}", feature_udf)
        return feature_udf
        
    def compute_features(self, data_path: str, feature_configs: dict):
        """Distributed feature computation"""
        df = self.spark.read.parquet(data_path)
        
        for feature_name, config in feature_configs.items():
            input_cols = config['input_columns']
            feature_fn = config['function']
            
            # Register and apply transformation
            feature_udf = self.register_feature_udf(
                feature_name, 
                feature_fn
            )
            df = df.withColumn(
                feature_name,
                feature_udf(*input_cols)
            )
        
        return df

# Example Usage
def compute_embeddings(text_col):
    # Simplified embedding computation
    return np.random.randn(100)

processor = DistributedFeatureProcessor()
feature_configs = {
    'text_embeddings': {
        'input_columns': ['text_column'],
        'function': compute_embeddings
    }
}

features_df = processor.compute_features(
    "s3://data/customers.parquet",
    feature_configs
)
```

Slide 9: Real-time Feature Service

Implementation of a high-performance feature serving system for real-time applications. This service provides fast feature retrieval and computation with automatic caching and load balancing.

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import aioredis
import uvicorn

class FeatureService:
    def __init__(self):
        self.app = FastAPI()
        self.redis = None
        self.feature_processors = {}
        self.setup_routes()
        
    async def initialize(self):
        """Initialize Redis connection pool"""
        self.redis = await aioredis.create_redis_pool(
            'redis://localhost'
        )
        
    async def get_cached_feature(self, feature_key: str):
        """Retrieve cached feature vector"""
        cached = await self.redis.get(feature_key)
        if cached:
            return np.frombuffer(cached, dtype=np.float32)
        return None
        
    def setup_routes(self):
        @self.app.post("/compute_features")
        async def compute_features(request: dict):
            feature_key = f"feature_{hash(str(request))}"
            
            # Check cache
            cached = await self.get_cached_feature(feature_key)
            if cached is not None:
                return {"features": cached.tolist()}
            
            # Compute features
            features = {}
            for name, processor in self.feature_processors.items():
                features[name] = await processor(request)
                
                # Cache new features
                await self.redis.set(
                    f"{feature_key}_{name}",
                    np.array(features[name], dtype=np.float32).tobytes(),
                    expire=3600  # 1 hour cache
                )
            
            return {"features": features}
        
    def run(self):
        """Start feature service"""
        uvicorn.run(self.app, host="0.0.0.0", port=8000)

# Example Usage
service = FeatureService()

async def text_embedding_processor(request):
    text = request.get('text', '')
    # Simplified embedding computation
    return np.random.randn(100).tolist()

service.feature_processors['text_embedding'] = text_embedding_processor

if __name__ == "__main__":
    service.run()
```

Slide 10: Automated Model Performance Analysis

A comprehensive system for analyzing model performance across different data segments and time periods. This implementation helps identify performance degradation patterns and potential biases.

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class ModelAnalyzer:
    def __init__(self):
        self.performance_history = []
        self.segment_performance = {}
        
    def analyze_performance(self, y_true: np.ndarray, 
                          y_pred: np.ndarray,
                          segments: dict = None):
        """Analyze model performance with detailed metrics"""
        # Overall performance
        report = classification_report(y_true, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Segment-wise analysis
        if segments:
            for segment_name, segment_mask in segments.items():
                segment_true = y_true[segment_mask]
                segment_pred = y_pred[segment_mask]
                
                self.segment_performance[segment_name] = {
                    'report': classification_report(
                        segment_true, 
                        segment_pred, 
                        output_dict=True
                    ),
                    'confusion_matrix': confusion_matrix(
                        segment_true, 
                        segment_pred
                    )
                }
        
        # Store historical performance
        self.performance_history.append({
            'timestamp': datetime.now(),
            'overall_report': report,
            'confusion_matrix': conf_matrix,
            'segment_performance': self.segment_performance.copy()
        })
        
        return {
            'overall_report': report,
            'segment_performance': self.segment_performance
        }
    
    def plot_confusion_matrix(self, segment: str = None):
        """Plot confusion matrix for specified segment"""
        if segment and segment in self.segment_performance:
            matrix = self.segment_performance[segment]['confusion_matrix']
            title = f"Confusion Matrix - {segment}"
        else:
            matrix = self.performance_history[-1]['confusion_matrix']
            title = "Overall Confusion Matrix"
            
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues')
        plt.title(title)
        plt.show()

# Example Usage
analyzer = ModelAnalyzer()

# Define segments
segments = {
    'high_value_customers': customer_value > 1000,
    'new_customers': account_age < 30
}

# Analyze performance
results = analyzer.analyze_performance(y_true, y_pred, segments)
analyzer.plot_confusion_matrix('high_value_customers')
```

Slide 11: Feature Selection and Importance Analysis

This implementation provides a systematic approach to evaluate feature importance and select optimal feature sets using multiple statistical methods. The system helps identify the most influential features for model performance.

```python
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import shap
import numpy as np

class FeatureImportanceAnalyzer:
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.importance_scores = {}
        self.selected_features = set()
        
    def analyze_importance(self, X: np.ndarray, y: np.ndarray):
        """Analyze feature importance using multiple methods"""
        # Mutual Information
        mi_scores = mutual_info_classif(X, y)
        self.importance_scores['mutual_info'] = dict(
            zip(self.feature_names, mi_scores)
        )
        
        # Random Forest importance
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(X, y)
        rf_importance = rf.feature_importances_
        self.importance_scores['random_forest'] = dict(
            zip(self.feature_names, rf_importance)
        )
        
        # SHAP values
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X)
        shap_importance = np.abs(shap_values).mean(axis=0)
        self.importance_scores['shap'] = dict(
            zip(self.feature_names, shap_importance)
        )
        
    def select_features(self, threshold: float = 0.05):
        """Select features based on importance scores"""
        selected = set()
        
        for method, scores in self.importance_scores.items():
            # Normalize scores
            max_score = max(scores.values())
            normalized_scores = {
                k: v/max_score for k, v in scores.items()
            }
            
            # Select features above threshold
            method_selected = {
                k for k, v in normalized_scores.items() 
                if v > threshold
            }
            selected.update(method_selected)
            
        self.selected_features = selected
        return list(selected)
    
    def plot_importance(self, method: str = 'random_forest'):
        """Plot feature importance scores"""
        scores = self.importance_scores[method]
        sorted_features = sorted(
            scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        plt.figure(figsize=(12, 6))
        features, values = zip(*sorted_features)
        plt.barh(features, values)
        plt.title(f'Feature Importance ({method})')
        plt.xlabel('Importance Score')
        plt.show()

# Example Usage
analyzer = FeatureImportanceAnalyzer(feature_names)
analyzer.analyze_importance(X_train, y_train)
selected_features = analyzer.select_features(threshold=0.05)
analyzer.plot_importance(method='shap')
```

Slide 12: Model Version Control System

A comprehensive version control system for ML models that tracks model artifacts, hyperparameters, and performance metrics. This implementation enables reproducible experiments and easy model rollbacks.

```python
import git
from dataclasses import dataclass
from typing import Dict, Any
import yaml
import hashlib

@dataclass
class ModelVersion:
    version_id: str
    params: Dict[str, Any]
    metrics: Dict[str, float]
    feature_set: List[str]
    timestamp: datetime
    
class ModelVersionControl:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.repo = git.Repo.init(repo_path)
        self.versions_file = self.repo_path / "versions.yaml"
        self.load_versions()
        
    def load_versions(self):
        """Load existing version history"""
        if self.versions_file.exists():
            with open(self.versions_file, 'r') as f:
                self.versions = yaml.safe_load(f)
        else:
            self.versions = {}
            
    def save_model_version(self, 
                          model,
                          params: Dict[str, Any],
                          metrics: Dict[str, float],
                          feature_set: List[str]):
        """Save new model version with metadata"""
        # Generate version ID
        version_id = hashlib.sha256(
            str(params).encode()
        ).hexdigest()[:8]
        
        # Save model artifacts
        model_path = self.repo_path / f"models/model_{version_id}.joblib"
        model_path.parent.mkdir(exist_ok=True)
        joblib.dump(model, model_path)
        
        # Create version entry
        version = ModelVersion(
            version_id=version_id,
            params=params,
            metrics=metrics,
            feature_set=feature_set,
            timestamp=datetime.now()
        )
        
        # Update versions file
        self.versions[version_id] = version.__dict__
        with open(self.versions_file, 'w') as f:
            yaml.dump(self.versions, f)
            
        # Commit changes
        self.repo.index.add([
            str(model_path.relative_to(self.repo_path)),
            str(self.versions_file.relative_to(self.repo_path))
        ])
        self.repo.index.commit(
            f"Model version {version_id} - "
            f"accuracy: {metrics.get('accuracy', 0):.3f}"
        )
        
        return version_id
        
    def load_model_version(self, version_id: str):
        """Load specific model version"""
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
            
        model_path = self.repo_path / f"models/model_{version_id}.joblib"
        model = joblib.load(model_path)
        
        return model, self.versions[version_id]
    
    def compare_versions(self, version_id1: str, version_id2: str):
        """Compare two model versions"""
        v1 = self.versions[version_id1]
        v2 = self.versions[version_id2]
        
        # Compare metrics
        metric_diff = {
            k: v2['metrics'][k] - v1['metrics'][k]
            for k in v1['metrics']
        }
        
        # Compare feature sets
        feature_changes = {
            'added': set(v2['feature_set']) - set(v1['feature_set']),
            'removed': set(v1['feature_set']) - set(v2['feature_set'])
        }
        
        return {
            'metric_differences': metric_diff,
            'feature_changes': feature_changes,
            'param_changes': {
                k: (v1['params'].get(k), v2['params'].get(k))
                for k in set(v1['params']) | set(v2['params'])
                if v1['params'].get(k) != v2['params'].get(k)
            }
        }

# Example Usage
vc = ModelVersionControl("ml_models")
version_id = vc.save_model_version(
    model,
    params=model.get_params(),
    metrics={'accuracy': 0.95},
    feature_set=['f1', 'f2', 'f3']
)

comparison = vc.compare_versions('abc123', version_id)
```

Slide 13: Batch Inference Pipeline

This implementation handles large-scale batch predictions efficiently with automatic data partitioning and parallel processing. The system includes progress tracking and error handling for robust production deployments.

```python
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Iterator, List
import pandas as pd

@dataclass
class BatchPredictionConfig:
    batch_size: int = 1000
    max_workers: int = 4
    timeout: int = 3600
    
class BatchInferencePipeline:
    def __init__(self, model_registry, feature_store, 
                 config: BatchPredictionConfig):
        self.model_registry = model_registry
        self.feature_store = feature_store
        self.config = config
        self.prediction_logs = []
        
    def partition_data(self, data: pd.DataFrame) -> Iterator[pd.DataFrame]:
        """Split data into manageable batches"""
        for i in range(0, len(data), self.config.batch_size):
            yield data.iloc[i:i + self.config.batch_size]
            
    def process_batch(self, batch: pd.DataFrame, model_name: str):
        """Process single batch of data"""
        try:
            # Load model
            model = self.model_registry.load_model(model_name)
            
            # Compute features
            features = {}
            for feature_name in self.feature_store.get_feature_names():
                transformer = self.feature_store.load_transformer(feature_name)
                features[feature_name] = transformer(batch)
                
            # Make predictions
            X = np.hstack([
                features[name] for name in sorted(features.keys())
            ])
            predictions = model.predict(X)
            
            return {
                'success': True,
                'predictions': predictions,
                'batch_size': len(batch)
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'batch_size': len(batch)
            }
            
    async def run_batch_prediction(self, 
                                 data: pd.DataFrame,
                                 model_name: str):
        """Run batch prediction pipeline"""
        results = []
        failed_batches = []
        
        with ProcessPoolExecutor(
            max_workers=self.config.max_workers
        ) as executor:
            # Submit batch processing jobs
            future_to_batch = {
                executor.submit(
                    self.process_batch, batch, model_name
                ): i 
                for i, batch in enumerate(self.partition_data(data))
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(
                future_to_batch, 
                timeout=self.config.timeout
            ):
                batch_idx = future_to_batch[future]
                result = future.result()
                
                if result['success']:
                    results.append(result['predictions'])
                else:
                    failed_batches.append({
                        'batch_idx': batch_idx,
                        'error': result['error']
                    })
                    
        # Combine predictions
        if results:
            final_predictions = np.concatenate(results)
        else:
            final_predictions = np.array([])
            
        # Log prediction run
        self.prediction_logs.append({
            'timestamp': datetime.now(),
            'model_name': model_name,
            'total_records': len(data),
            'successful_predictions': len(final_predictions),
            'failed_batches': failed_batches
        })
        
        return final_predictions, failed_batches

# Example Usage
config = BatchPredictionConfig(batch_size=1000, max_workers=4)
pipeline = BatchInferencePipeline(registry, feature_store, config)

predictions, failures = await pipeline.run_batch_prediction(
    large_dataset,
    "production_model_v1"
)

if failures:
    print(f"Failed batches: {len(failures)}")
    for failure in failures:
        print(f"Batch {failure['batch_idx']}: {failure['error']}")
```

Slide 14: Additional Resources

*   Reliable Machine Learning Systems: [https://arxiv.org/abs/2212.08697](https://arxiv.org/abs/2212.08697)
*   Feature Store Architectures: [https://arxiv.org/abs/2210.15742](https://arxiv.org/abs/2210.15742)
*   ML Pipeline Design Patterns: [https://arxiv.org/abs/2205.09330](https://arxiv.org/abs/2205.09330)
*   Best practices for searching ML-related papers:
    *   Google Scholar with keywords: "ML pipeline architecture", "feature store design", "ML system design"
    *   Papers With Code: [https://paperswithcode.com/task/mlops](https://paperswithcode.com/task/mlops)
    *   ArXiv CS.LG category: [https://arxiv.org/list/cs.LG/recent](https://arxiv.org/list/cs.LG/recent)

