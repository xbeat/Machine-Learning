## Custom Exception Handling for ML Pipelines
Slide 1: Custom Exception Hierarchy for ML Pipelines

Exception handling in machine learning requires specialized error types to handle data processing, model training, and inference failures. Creating a custom exception hierarchy allows precise error identification and appropriate handling strategies.

```python
class MLException(Exception):
    """Base exception class for ML pipeline errors"""
    def __init__(self, message, error_code=None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class DataValidationError(MLException):
    """Raised when data validation fails"""
    pass

class ModelTrainingError(MLException):
    """Raised during model training failures"""
    pass

class InferenceError(MLException):
    """Raised during model inference issues"""
    pass

# Example usage
try:
    if data_quality_score < threshold:
        raise DataValidationError("Data quality below threshold", "ERR_001")
except DataValidationError as e:
    print(f"Error {e.error_code}: {e.message}")
```

Slide 2: Centralized Error Management System

A centralized error handling system provides consistent error management across different components of ML pipelines. This implementation includes error logging, notification, and recovery strategies.

```python
import logging
from functools import wraps
import time

class MLErrorManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.retry_attempts = 3
        self.backoff_factor = 2
    
    def handle_error(self, error, context=None):
        error_id = str(int(time.time()))
        self.logger.error(f"Error ID: {error_id} - {str(error)}")
        
        if context:
            self.logger.error(f"Context: {context}")
        
        return error_id
    
    def retry_operation(self, operation):
        @wraps(operation)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(self.retry_attempts):
                try:
                    return operation(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    wait_time = self.backoff_factor ** attempt
                    time.sleep(wait_time)
            
            self.handle_error(last_exception)
            raise last_exception
        return wrapper

# Usage example
error_manager = MLErrorManager()

@error_manager.retry_operation
def train_model(data):
    # Simulated training
    if not data.is_valid():
        raise DataValidationError("Invalid training data")
    return "Model trained successfully"
```

Slide 3: Input Validation with Type Contracts

Type contracts ensure data consistency and prevent runtime errors by validating inputs before processing. This implementation uses Python's type hints and runtime checking.

```python
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class ModelInputValidator:
    required_features: List[str]
    numeric_ranges: Dict[str, tuple]
    
    def validate_features(self, data: np.ndarray, 
                         feature_names: List[str]) -> bool:
        """Validates input features against requirements"""
        if not all(feat in feature_names for feat in self.required_features):
            raise ValueError(f"Missing required features: {self.required_features}")
        
        for feature, (min_val, max_val) in self.numeric_ranges.items():
            idx = feature_names.index(feature)
            values = data[:, idx]
            if np.any((values < min_val) | (values > max_val)):
                raise ValueError(f"Feature {feature} outside range [{min_val}, {max_val}]")
        
        return True

# Example usage
validator = ModelInputValidator(
    required_features=['age', 'income'],
    numeric_ranges={'age': (0, 120), 'income': (0, 1e6)}
)

def preprocess_data(data: np.ndarray, features: List[str]) -> np.ndarray:
    validator.validate_features(data, features)
    return data  # Add actual preprocessing steps
```

Slide 4: Real-time Error Monitoring System

A comprehensive monitoring system that tracks errors across different stages of ML pipeline execution, collecting metrics and generating alerts when error rates exceed thresholds.

```python
import time
from collections import defaultdict
from threading import Lock
import numpy as np

class MLMonitor:
    def __init__(self, error_threshold=0.1, window_size=3600):
        self.errors = defaultdict(list)
        self.timestamps = defaultdict(list)
        self.lock = Lock()
        self.threshold = error_threshold
        self.window_size = window_size
    
    def record_error(self, component: str, error_type: str):
        with self.lock:
            current_time = time.time()
            self.errors[component].append(error_type)
            self.timestamps[component].append(current_time)
            self._clean_old_records(component)
            
            if self._calculate_error_rate(component) > self.threshold:
                self._trigger_alert(component)
    
    def _clean_old_records(self, component):
        current_time = time.time()
        cutoff_time = current_time - self.window_size
        
        valid_indices = [i for i, ts in enumerate(self.timestamps[component])
                        if ts > cutoff_time]
        
        self.timestamps[component] = [self.timestamps[component][i] 
                                    for i in valid_indices]
        self.errors[component] = [self.errors[component][i] 
                                for i in valid_indices]
    
    def _calculate_error_rate(self, component):
        return len(self.errors[component]) / self.window_size
    
    def _trigger_alert(self, component):
        print(f"ALERT: High error rate detected in {component}")
        print(f"Current rate: {self._calculate_error_rate(component):.2%}")

# Usage example
monitor = MLMonitor(error_threshold=0.05)
monitor.record_error("model_training", "convergence_error")
```

Slide 5: Circuit Breaker Pattern Implementation

The circuit breaker pattern prevents cascading failures in distributed ML systems by automatically stopping operations when error rates exceed acceptable thresholds.

```python
from enum import Enum
import time
from threading import Lock

class CircuitState(Enum):
    CLOSED = "CLOSED"  # Normal operation
    OPEN = "OPEN"      # Stopping operation
    HALF_OPEN = "HALF_OPEN"  # Testing if system recovered

class MLCircuitBreaker:
    def __init__(self, failure_threshold=5, reset_timeout=60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.lock = Lock()
    
    def execute(self, func, *args, **kwargs):
        with self.lock:
            if self._can_execute():
                try:
                    result = func(*args, **kwargs)
                    self._handle_success()
                    return result
                except Exception as e:
                    self._handle_failure()
                    raise e
            else:
                raise Exception("Circuit breaker is OPEN")
    
    def _can_execute(self):
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = CircuitState.HALF_OPEN
                return True
        return False
    
    def _handle_success(self):
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
    
    def _handle_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

# Example usage
def train_model_batch(data):
    # Simulated model training
    if np.random.random() < 0.3:  # 30% chance of failure
        raise Exception("Training failed")
    return "Model trained successfully"

circuit_breaker = MLCircuitBreaker(failure_threshold=3)
try:
    result = circuit_breaker.execute(train_model_batch, data=train_data)
except Exception as e:
    print(f"Operation failed: {str(e)}")
```

Slide 6: Comprehensive Error Logging System

A sophisticated logging system designed specifically for ML pipelines that captures detailed information about errors, model states, and system conditions when failures occur.

```python
import logging
import traceback
import json
from datetime import datetime
import numpy as np

class MLLogger:
    def __init__(self, log_file="ml_pipeline.log"):
        self.logger = logging.getLogger("MLPipeline")
        self.logger.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
    def log_error(self, error, context=None, model_state=None):
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error.__class__.__name__,
            'error_message': str(error),
            'stacktrace': traceback.format_exc(),
            'context': context or {},
            'model_state': self._sanitize_model_state(model_state)
        }
        
        self.logger.error(json.dumps(error_info, indent=2))
        
    def _sanitize_model_state(self, state):
        if state is None:
            return None
            
        sanitized = {}
        for key, value in state.items():
            if isinstance(value, np.ndarray):
                sanitized[key] = {
                    'shape': value.shape,
                    'dtype': str(value.dtype),
                    'stats': {
                        'mean': float(np.mean(value)),
                        'std': float(np.std(value)),
                        'min': float(np.min(value)),
                        'max': float(np.max(value))
                    }
                }
            else:
                sanitized[key] = str(value)
        return sanitized

# Usage example
ml_logger = MLLogger()

try:
    # Simulated model training
    model_state = {
        'weights': np.random.randn(100, 100),
        'learning_rate': 0.001,
        'epoch': 10
    }
    raise ValueError("Gradient explosion detected")
except Exception as e:
    ml_logger.log_error(
        error=e,
        context={'phase': 'training', 'batch_id': 123},
        model_state=model_state
    )
```

Slide 7: Graceful Degradation Implementation

A system that maintains critical functionality when parts of the ML pipeline fail by implementing fallback mechanisms and feature toggles.

```python
from enum import Enum
from typing import Dict, Any, Optional
import json

class FeatureStatus(Enum):
    ACTIVE = "active"
    DEGRADED = "degraded"
    DISABLED = "disabled"

class GracefulDegradation:
    def __init__(self, config_file: str):
        self.features = self._load_config(config_file)
        self.fallbacks = {}
        self.status = {feature: FeatureStatus.ACTIVE 
                      for feature in self.features}
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        with open(config_file, 'r') as f:
            return json.load(f)
    
    def register_fallback(self, feature: str, fallback_func):
        """Register a fallback function for a feature"""
        self.fallbacks[feature] = fallback_func
    
    def execute_feature(self, feature: str, 
                       func, *args, **kwargs) -> Optional[Any]:
        """Execute a feature with graceful degradation"""
        if self.status[feature] == FeatureStatus.DISABLED:
            return None
            
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            self._handle_feature_failure(feature, e)
            return self._execute_fallback(feature, *args, **kwargs)
    
    def _handle_feature_failure(self, feature: str, error: Exception):
        if self.status[feature] == FeatureStatus.ACTIVE:
            self.status[feature] = FeatureStatus.DEGRADED
            print(f"Feature {feature} degraded: {str(error)}")
    
    def _execute_fallback(self, feature: str, *args, **kwargs):
        if feature in self.fallbacks:
            try:
                return self.fallbacks[feature](*args, **kwargs)
            except Exception as e:
                self.status[feature] = FeatureStatus.DISABLED
                print(f"Feature {feature} disabled: {str(e)}")
                return None
        return None

# Example usage
def complex_prediction(data):
    # Simulated complex model prediction
    raise Exception("GPU memory error")

def simple_prediction(data):
    # Fallback to simple model
    return np.mean(data, axis=0)

degradation_handler = GracefulDegradation("config.json")
degradation_handler.register_fallback("prediction", simple_prediction)

result = degradation_handler.execute_feature(
    "prediction", 
    complex_prediction, 
    data=np.random.randn(100, 10)
)
```

Slide 8: Retry Mechanism with Exponential Backoff

A sophisticated retry mechanism that implements exponential backoff and jitter for handling transient failures in distributed ML systems, particularly useful for network-related operations.

```python
import random
import time
from functools import wraps
from typing import Callable, Optional, Any

class RetryHandler:
    def __init__(self, max_attempts: int = 3, 
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 jitter: bool = True):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
    
    def retry_with_backoff(self, retryable_exceptions: tuple = (Exception,)):
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                attempt = 0
                while attempt < self.max_attempts:
                    try:
                        return func(*args, **kwargs)
                    except retryable_exceptions as e:
                        attempt += 1
                        if attempt == self.max_attempts:
                            raise e
                        
                        delay = min(
                            self.base_delay * (2 ** (attempt - 1)),
                            self.max_delay
                        )
                        
                        if self.jitter:
                            delay = delay * random.uniform(0.5, 1.5)
                        
                        print(f"Attempt {attempt} failed. "
                              f"Retrying in {delay:.2f} seconds...")
                        time.sleep(delay)
                return None
            return wrapper
        return decorator

# Example usage
retry_handler = RetryHandler(max_attempts=3, base_delay=2.0)

@retry_handler.retry_with_backoff(retryable_exceptions=(ConnectionError,))
def fetch_training_data(url: str) -> np.ndarray:
    if random.random() < 0.7:  # Simulate 70% failure rate
        raise ConnectionError("Failed to fetch data")
    return np.random.randn(1000, 10)

# Test the retry mechanism
try:
    data = fetch_training_data("http://example.com/data")
    print("Data fetched successfully")
except ConnectionError as e:
    print(f"All retry attempts failed: {str(e)}")
```

Slide 9: Error Metrics Collection and Analysis

A comprehensive system for collecting, analyzing, and visualizing error metrics across different components of an ML pipeline, enabling early detection of systemic issues.

```python
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from typing import Dict, List, Tuple

@dataclass
class ErrorMetric:
    count: int
    mean: float
    std: float
    timestamps: List[float]
    values: List[float]

class ErrorMetricsAnalyzer:
    def __init__(self, window_size: int = 3600):
        self.window_size = window_size
        self.metrics = defaultdict(lambda: defaultdict(list))
        self.thresholds = {}
    
    def add_metric(self, component: str, metric_type: str, 
                   value: float, timestamp: float):
        self.metrics[component][metric_type].append(
            (timestamp, value)
        )
        self._clean_old_metrics(component, metric_type)
    
    def set_threshold(self, component: str, metric_type: str, 
                     threshold: float):
        self.thresholds[(component, metric_type)] = threshold
    
    def get_metrics(self, component: str, metric_type: str) -> ErrorMetric:
        data = self.metrics[component][metric_type]
        if not data:
            return ErrorMetric(0, 0.0, 0.0, [], [])
        
        timestamps, values = zip(*data)
        return ErrorMetric(
            count=len(values),
            mean=np.mean(values),
            std=np.std(values),
            timestamps=list(timestamps),
            values=list(values)
        )
    
    def check_thresholds(self) -> List[Tuple[str, str, float]]:
        violations = []
        for (component, metric_type), threshold in self.thresholds.items():
            metrics = self.get_metrics(component, metric_type)
            if metrics.mean > threshold:
                violations.append(
                    (component, metric_type, metrics.mean)
                )
        return violations
    
    def _clean_old_metrics(self, component: str, metric_type: str):
        current_time = time.time()
        cutoff_time = current_time - self.window_size
        
        self.metrics[component][metric_type] = [
            (ts, val) for ts, val in self.metrics[component][metric_type]
            if ts > cutoff_time
        ]

# Example usage
analyzer = ErrorMetricsAnalyzer(window_size=3600)

# Simulate error metrics collection
for _ in range(100):
    timestamp = time.time()
    analyzer.add_metric(
        "model_training",
        "loss_variance",
        random.uniform(0, 2),
        timestamp
    )

analyzer.set_threshold("model_training", "loss_variance", 1.5)
violations = analyzer.check_thresholds()
for component, metric_type, value in violations:
    print(f"Threshold violated: {component}/{metric_type} = {value:.2f}")
```

Slide 10: Input Validation Framework for ML Pipelines

A robust framework for validating input data and model parameters, ensuring data quality and preventing training failures before they occur.

```python
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np
from enum import Enum

class DataType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"

@dataclass
class ValidationRule:
    data_type: DataType
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    max_missing_pct: float = 0.1
    
class DataValidator:
    def __init__(self):
        self.rules: Dict[str, ValidationRule] = {}
        self.validation_results: Dict[str, List[str]] = {}
    
    def add_rule(self, feature_name: str, rule: ValidationRule):
        self.rules[feature_name] = rule
    
    def validate(self, data: Dict[str, np.ndarray]) -> bool:
        self.validation_results.clear()
        is_valid = True
        
        for feature_name, feature_data in data.items():
            if feature_name not in self.rules:
                continue
                
            rule = self.rules[feature_name]
            feature_errors = self._validate_feature(
                feature_name, feature_data, rule
            )
            
            if feature_errors:
                is_valid = False
                self.validation_results[feature_name] = feature_errors
        
        return is_valid
    
    def _validate_feature(self, feature_name: str, 
                         feature_data: np.ndarray, 
                         rule: ValidationRule) -> List[str]:
        errors = []
        
        # Check missing values
        missing_pct = np.isnan(feature_data).mean()
        if missing_pct > rule.max_missing_pct:
            errors.append(
                f"Missing values ({missing_pct:.2%}) exceed threshold "
                f"({rule.max_missing_pct:.2%})"
            )
        
        # Numeric validation
        if rule.data_type == DataType.NUMERIC:
            if rule.min_value is not None:
                if np.any(feature_data < rule.min_value):
                    errors.append(
                        f"Values below minimum threshold {rule.min_value}"
                    )
            if rule.max_value is not None:
                if np.any(feature_data > rule.max_value):
                    errors.append(
                        f"Values above maximum threshold {rule.max_value}"
                    )
        
        # Categorical validation
        elif rule.data_type == DataType.CATEGORICAL:
            if rule.allowed_values:
                invalid_values = set(feature_data) - set(rule.allowed_values)
                if invalid_values:
                    errors.append(
                        f"Invalid categories found: {invalid_values}"
                    )
        
        return errors

# Example usage
validator = DataValidator()

# Add validation rules
validator.add_rule(
    "age",
    ValidationRule(
        data_type=DataType.NUMERIC,
        min_value=0,
        max_value=120,
        max_missing_pct=0.05
    )
)

validator.add_rule(
    "category",
    ValidationRule(
        data_type=DataType.CATEGORICAL,
        allowed_values=['A', 'B', 'C'],
        max_missing_pct=0.0
    )
)

# Test validation
test_data = {
    'age': np.array([25, 35, np.nan, 150]),
    'category': np.array(['A', 'B', 'D', 'C'])
}

is_valid = validator.validate(test_data)
if not is_valid:
    for feature, errors in validator.validation_results.items():
        print(f"\nValidation errors for {feature}:")
        for error in errors:
            print(f"- {error}")
```

Slide 11: Real-time Error Rate Analysis System

A sophisticated system for analyzing error patterns in real-time, detecting anomalies, and predicting potential system failures before they occur using statistical analysis.

```python
import numpy as np
from scipy import stats
from collections import deque
from datetime import datetime, timedelta

class ErrorRateAnalyzer:
    def __init__(self, window_size: int = 3600, 
                 anomaly_threshold: float = 2.0):
        self.window_size = window_size
        self.anomaly_threshold = anomaly_threshold
        self.error_counts = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        self.baseline_mean = None
        self.baseline_std = None
    
    def update(self, error_count: int, timestamp: datetime = None):
        if timestamp is None:
            timestamp = datetime.now()
            
        self.error_counts.append(error_count)
        self.timestamps.append(timestamp)
        self._update_baseline()
    
    def _update_baseline(self):
        if len(self.error_counts) >= 60:  # Minimum sample size
            self.baseline_mean = np.mean(self.error_counts)
            self.baseline_std = np.std(self.error_counts)
    
    def detect_anomalies(self) -> dict:
        if not self.baseline_mean:
            return {"status": "insufficient_data"}
            
        recent_errors = list(self.error_counts)[-60:]  # Last hour
        z_scores = stats.zscore(recent_errors)
        
        anomalies = []
        for i, z_score in enumerate(z_scores):
            if abs(z_score) > self.anomaly_threshold:
                anomalies.append({
                    "timestamp": self.timestamps[-60 + i],
                    "error_count": recent_errors[i],
                    "z_score": z_score
                })
        
        trend = self._analyze_trend(recent_errors)
        
        return {
            "status": "alert" if anomalies else "normal",
            "anomalies": anomalies,
            "trend": trend,
            "current_rate": recent_errors[-1],
            "baseline_rate": self.baseline_mean,
            "std_deviation": self.baseline_std
        }
    
    def _analyze_trend(self, data: list) -> str:
        if len(data) < 2:
            return "insufficient_data"
            
        slope, _, r_value, p_value, _ = stats.linregress(
            range(len(data)), data
        )
        
        if p_value > 0.05:  # Not statistically significant
            return "stable"
            
        if slope > 0:
            return "increasing"
        return "decreasing"
    
    def predict_next_hour(self) -> dict:
        if len(self.error_counts) < 120:  # Need minimum history
            return {"status": "insufficient_data"}
            
        recent_data = list(self.error_counts)[-120:]
        time_points = np.arange(len(recent_data))
        
        # Fit polynomial regression
        coeffs = np.polyfit(time_points, recent_data, 2)
        poly = np.poly1d(coeffs)
        
        # Predict next hour
        next_hour = poly(len(recent_data) + 60)
        
        return {
            "predicted_rate": max(0, float(next_hour)),
            "confidence": self._calculate_prediction_confidence(
                recent_data, poly(time_points)
            )
        }
    
    def _calculate_prediction_confidence(self, 
                                      actual: list, 
                                      predicted: list) -> float:
        residuals = np.array(actual) - predicted
        rmse = np.sqrt(np.mean(residuals ** 2))
        return 1.0 / (1.0 + rmse)

# Example usage
analyzer = ErrorRateAnalyzer()

# Simulate error rate data
for i in range(200):
    # Generate synthetic error counts with increasing trend
    base_errors = 10
    trend = i / 20
    noise = np.random.normal(0, 2)
    error_count = max(0, int(base_errors + trend + noise))
    
    timestamp = datetime.now() - timedelta(minutes=200-i)
    analyzer.update(error_count, timestamp)

# Analyze current state
analysis = analyzer.detect_anomalies()
prediction = analyzer.predict_next_hour()

print("Current Analysis:")
print(f"Status: {analysis['status']}")
print(f"Trend: {analysis['trend']}")
print(f"Current Rate: {analysis['current_rate']}")
print("\nPrediction:")
print(f"Next Hour Rate: {prediction['predicted_rate']:.2f}")
print(f"Confidence: {prediction['confidence']:.2%}")
```

Slide 12: Fault Isolation in Distributed ML Systems

A robust implementation of fault isolation patterns that prevent cascading failures across distributed ML system components while maintaining partial system functionality.

```python
from enum import Enum
from typing import Dict, List, Callable, Any
import threading
import time

class ComponentStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"

class FaultIsolator:
    def __init__(self):
        self.components = {}
        self.dependencies = {}
        self.health_checks = {}
        self.status_lock = threading.Lock()
        self.component_status = {}
    
    def register_component(self, name: str, 
                          health_check: Callable[[], bool],
                          dependencies: List[str] = None):
        self.components[name] = {
            'health_check': health_check,
            'dependencies': dependencies or []
        }
        self.component_status[name] = ComponentStatus.HEALTHY
        
    def execute_with_isolation(self, component: str, 
                             func: Callable, *args, **kwargs) -> Any:
        if not self._can_execute(component):
            raise RuntimeError(f"Component {component} is not available")
            
        try:
            result = func(*args, **kwargs)
            self._update_status(component, ComponentStatus.HEALTHY)
            return result
        except Exception as e:
            self._handle_component_failure(component)
            raise e
    
    def _can_execute(self, component: str) -> bool:
        if component not in self.components:
            return False
            
        # Check component health
        if self.component_status[component] == ComponentStatus.FAILED:
            return False
            
        # Check dependencies
        for dep in self.components[component]['dependencies']:
            if self.component_status[dep] == ComponentStatus.FAILED:
                return False
                
        return True
    
    def _handle_component_failure(self, component: str):
        with self.status_lock:
            self._update_status(component, ComponentStatus.FAILED)
            self._propagate_failure_impact(component)
    
    def _propagate_failure_impact(self, failed_component: str):
        for component, config in self.components.items():
            if failed_component in config['dependencies']:
                self._update_status(component, ComponentStatus.DEGRADED)
    
    def _update_status(self, component: str, status: ComponentStatus):
        with self.status_lock:
            self.component_status[component] = status
    
    def get_system_health(self) -> Dict[str, ComponentStatus]:
        return {
            component: status 
            for component, status in self.component_status.items()
        }

# Example usage
def health_check_data_pipeline():
    return random.random() > 0.1  # 90% healthy

def health_check_model_training():
    return random.random() > 0.2  # 80% healthy

def health_check_inference():
    return random.random() > 0.05  # 95% healthy

# Create fault isolator
isolator = FaultIsolator()

# Register components with dependencies
isolator.register_component("data_pipeline", health_check_data_pipeline)
isolator.register_component(
    "model_training", 
    health_check_model_training,
    dependencies=["data_pipeline"]
)
isolator.register_component(
    "inference", 
    health_check_inference,
    dependencies=["model_training"]
)

# Example execution with fault isolation
def train_model(data):
    if random.random() < 0.3:  # 30% chance of failure
        raise Exception("Training failed")
    return "Model trained successfully"

try:
    result = isolator.execute_with_isolation(
        "model_training", 
        train_model, 
        data="sample_data"
    )
    print(f"Training result: {result}")
except Exception as e:
    print(f"Training failed: {str(e)}")

# Check system health
system_health = isolator.get_system_health()
for component, status in system_health.items():
    print(f"{component}: {status.value}")
```

Slide 13: Additional Resources

*   "A Survey of System Level Fault Management in Machine Learning Systems" - [https://arxiv.org/abs/2110.03043](https://arxiv.org/abs/2110.03043)
*   "Robust Error Handling Patterns for Distributed ML Systems" - [https://arxiv.org/abs/2103.09877](https://arxiv.org/abs/2103.09877)
*   "Fault Tolerance in Distributed Machine Learning: A Systematic Review" - [https://arxiv.org/abs/2012.15832](https://arxiv.org/abs/2012.15832)
*   "Error Detection and Recovery in Machine Learning Pipeline Systems" - [https://arxiv.org/abs/2106.12789](https://arxiv.org/abs/2106.12789)

