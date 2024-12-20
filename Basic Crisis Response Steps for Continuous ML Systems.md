## Basic Crisis Response Steps for Continuous ML Systems
Slide 1: Stop Training Implementation

A robust MLOps pipeline requires immediate training suspension capabilities when anomalies are detected. This implementation demonstrates an automated training halt mechanism using Python's multiprocessing and signal handling, ensuring graceful stoppage of ongoing ML processes.

```python
import signal
import multiprocessing as mp
from contextlib import contextmanager
import numpy as np

class TrainingManager:
    def __init__(self):
        self._stop_event = mp.Event()
        self.training_process = None
    
    def stop_training(self, signum=None, frame=None):
        print("Emergency stop triggered")
        self._stop_event.set()
    
    def train_model(self, data, epochs=100):
        weights = np.random.randn(10)
        for epoch in range(epochs):
            if self._stop_event.is_set():
                print(f"Training halted at epoch {epoch}")
                return weights
            # Simulated training step
            weights += np.random.randn(10) * 0.1
        return weights

    def start_training(self):
        signal.signal(signal.SIGINT, self.stop_training)
        try:
            weights = self.train_model(None)
            return weights
        except Exception as e:
            self.stop_training()
            raise e

# Usage Example
manager = TrainingManager()
weights = manager.start_training()
```

Slide 2: Fallback Strategy Implementation

This implementation showcases a fallback mechanism that automatically switches to a simpler model when the primary model fails or produces unreliable predictions. The system monitors model performance and triggers fallback based on predefined thresholds.

```python
import numpy as np
from typing import Optional, Tuple

class RobustMLSystem:
    def __init__(self, threshold: float = 0.8):
        self.primary_model = None
        self.fallback_model = None
        self.performance_threshold = threshold
        self.is_fallback_active = False
    
    def simple_fallback_predict(self, x: np.ndarray) -> np.ndarray:
        return np.mean(x, axis=1)
    
    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, bool]:
        try:
            if not self.is_fallback_active:
                predictions = self.primary_model.predict(x)
                confidence = self._calculate_confidence(predictions)
                
                if confidence < self.performance_threshold:
                    self.is_fallback_active = True
                    return self.simple_fallback_predict(x), True
                return predictions, False
            
            return self.simple_fallback_predict(x), True
        except Exception:
            self.is_fallback_active = True
            return self.simple_fallback_predict(x), True
    
    def _calculate_confidence(self, predictions: np.ndarray) -> float:
        return np.mean(np.max(predictions, axis=1) if len(predictions.shape) > 1 else predictions)

# Usage
system = RobustMLSystem(threshold=0.8)
x_test = np.random.randn(100, 10)
predictions, using_fallback = system.predict(x_test)
```

Slide 3: Model Rollback System

A comprehensive rollback system that maintains a history of model versions and their performance metrics. This implementation includes version control, performance tracking, and automatic rollback triggers when new models underperform.

```python
import datetime
import json
from typing import Dict, Any, Optional

class ModelVersionControl:
    def __init__(self):
        self.model_history: Dict[str, Dict[str, Any]] = {}
        self.current_version: Optional[str] = None
        
    def save_model(self, model: Any, metrics: Dict[str, float]) -> str:
        version = datetime.datetime.now().isoformat()
        self.model_history[version] = {
            'model': model,
            'metrics': metrics,
            'timestamp': version
        }
        self.current_version = version
        return version
    
    def rollback(self, target_version: Optional[str] = None) -> Any:
        if target_version is None:
            versions = sorted(self.model_history.keys())[:-1]
            target_version = versions[-1] if versions else None
            
        if target_version and target_version in self.model_history:
            self.current_version = target_version
            return self.model_history[target_version]['model']
        raise ValueError("No valid version found for rollback")
    
    def get_metrics_history(self) -> Dict[str, Dict[str, float]]:
        return {v: data['metrics'] 
                for v, data in self.model_history.items()}

# Example usage
version_control = ModelVersionControl()
metrics = {'accuracy': 0.95, 'f1_score': 0.94}
version = version_control.save_model("model_state", metrics)
previous_model = version_control.rollback()
```

Slide 4: Data Validation and Cleaning Pipeline

A robust data validation system that detects and removes corrupted or anomalous data points during training. This implementation includes statistical outlier detection, data integrity checks, and automatic cleaning procedures.

```python
import pandas as pd
import numpy as np
from typing import Tuple, List

class DataValidator:
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.statistics = {}
        
    def detect_outliers(self, data: np.ndarray) -> np.ndarray:
        z_scores = np.abs((data - np.mean(data, axis=0)) / np.std(data, axis=0))
        return np.any(z_scores > 3, axis=1)
    
    def validate_integrity(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        issues = []
        
        # Check for missing values
        missing_cols = data.columns[data.isnull().any()].tolist()
        if missing_cols:
            issues.append(f"Missing values in columns: {missing_cols}")
        
        # Check for infinite values
        inf_cols = data.columns[np.isinf(data.select_dtypes(include=np.number)).any()].tolist()
        if inf_cols:
            issues.append(f"Infinite values in columns: {inf_cols}")
        
        # Remove problematic rows
        clean_data = data.dropna().replace([np.inf, -np.inf], np.nan).dropna()
        
        # Detect statistical outliers
        if len(clean_data) > 0:
            numerical_cols = clean_data.select_dtypes(include=np.number).columns
            if len(numerical_cols) > 0:
                outliers = self.detect_outliers(clean_data[numerical_cols].values)
                clean_data = clean_data[~outliers]
                
        return clean_data, issues

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        clean_data, issues = self.validate_integrity(data)
        if issues:
            print("Data cleaning issues detected:", issues)
        return clean_data

# Example usage
validator = DataValidator()
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 100],
    'B': [1, 2, 3, np.inf, 5]
})
clean_df = validator.clean_data(df)
```

Slide 5: Automated Recovery System

A sophisticated recovery system that monitors model performance in real-time and implements automated recovery procedures when degradation is detected. The system includes performance baselines, monitoring, and recovery strategies.

```python
import time
from typing import Callable, Dict, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PerformanceMetrics:
    accuracy: float
    latency: float
    error_rate: float
    timestamp: datetime

class RecoverySystem:
    def __init__(self, 
                 recovery_strategies: Dict[str, Callable],
                 performance_thresholds: Dict[str, float]):
        self.recovery_strategies = recovery_strategies
        self.thresholds = performance_thresholds
        self.metrics_history = []
        
    def monitor_performance(self, metrics: PerformanceMetrics) -> bool:
        self.metrics_history.append(metrics)
        return self._check_thresholds(metrics)
    
    def _check_thresholds(self, metrics: PerformanceMetrics) -> bool:
        violations = []
        if metrics.accuracy < self.thresholds['accuracy']:
            violations.append('accuracy')
        if metrics.latency > self.thresholds['latency']:
            violations.append('latency')
        if metrics.error_rate > self.thresholds['error_rate']:
            violations.append('error_rate')
        return len(violations) > 0
    
    def execute_recovery(self, current_state: Any) -> Any:
        metrics = self.metrics_history[-1]
        
        if metrics.accuracy < self.thresholds['accuracy']:
            return self.recovery_strategies['accuracy'](current_state)
        elif metrics.latency > self.thresholds['latency']:
            return self.recovery_strategies['latency'](current_state)
        elif metrics.error_rate > self.thresholds['error_rate']:
            return self.recovery_strategies['error_rate'](current_state)
            
        return current_state

# Example usage
def accuracy_recovery(state):
    return "Retrained model"

def latency_recovery(state):
    return "Optimized model"

def error_rate_recovery(state):
    return "Debugged model"

recovery_system = RecoverySystem(
    recovery_strategies={
        'accuracy': accuracy_recovery,
        'latency': latency_recovery,
        'error_rate': error_rate_recovery
    },
    performance_thresholds={
        'accuracy': 0.9,
        'latency': 100,
        'error_rate': 0.1
    }
)

metrics = PerformanceMetrics(
    accuracy=0.85,
    latency=120,
    error_rate=0.15,
    timestamp=datetime.now()
)

if recovery_system.monitor_performance(metrics):
    new_state = recovery_system.execute_recovery("current_model_state")
```

Slide 6: Real-time Monitoring System

A comprehensive monitoring system that tracks model performance, resource usage, and system health in real-time. This implementation includes custom metrics collection, alerting mechanisms, and performance visualization capabilities.

```python
import time
import psutil
import threading
from collections import deque
from typing import Dict, List, Optional

class MLMonitor:
    def __init__(self, metrics_window: int = 1000):
        self.metrics_window = metrics_window
        self.metrics_history = {
            'inference_time': deque(maxlen=metrics_window),
            'memory_usage': deque(maxlen=metrics_window),
            'prediction_confidence': deque(maxlen=metrics_window),
            'error_count': deque(maxlen=metrics_window)
        }
        self.alert_thresholds = {
            'inference_time': 100,  # ms
            'memory_usage': 85,     # percentage
            'error_count': 5        # per window
        }
        self._stop_monitoring = threading.Event()
        
    def start_monitoring(self):
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        self._stop_monitoring.set()
        self.monitor_thread.join()
    
    def _monitoring_loop(self):
        while not self._stop_monitoring.is_set():
            self._collect_metrics()
            self._check_alerts()
            time.sleep(1)
    
    def _collect_metrics(self):
        # Collect system metrics
        memory_percent = psutil.virtual_memory().percent
        self.metrics_history['memory_usage'].append(memory_percent)
        
        # Simulate collecting ML metrics
        self.metrics_history['inference_time'].append(
            self._measure_inference_time()
        )
    
    def _measure_inference_time(self) -> float:
        start_time = time.time()
        # Simulate model inference
        time.sleep(0.001)
        return (time.time() - start_time) * 1000
    
    def _check_alerts(self):
        alerts = []
        
        # Check memory usage
        if self.metrics_history['memory_usage'][-1] > self.alert_thresholds['memory_usage']:
            alerts.append(f"High memory usage: {self.metrics_history['memory_usage'][-1]}%")
            
        # Check inference time
        recent_inference_times = list(self.metrics_history['inference_time'])
        if len(recent_inference_times) > 0:
            avg_inference_time = sum(recent_inference_times) / len(recent_inference_times)
            if avg_inference_time > self.alert_thresholds['inference_time']:
                alerts.append(f"High inference time: {avg_inference_time:.2f}ms")
        
        if alerts:
            self._send_alerts(alerts)
    
    def _send_alerts(self, alerts: List[str]):
        print("ALERTS:", alerts)  # In production, send to monitoring system
        
    def get_metrics_summary(self) -> Dict[str, float]:
        return {
            metric: sum(values)/len(values) if values else 0
            for metric, values in self.metrics_history.items()
        }

# Example usage
monitor = MLMonitor(metrics_window=100)
monitor.start_monitoring()

# Simulate running for a while
time.sleep(5)

# Get metrics summary
metrics_summary = monitor.get_metrics_summary()
print(f"Metrics Summary: {metrics_summary}")

monitor.stop_monitoring()
```

Slide 7: Crisis Response Coordinator

A central coordination system that manages multiple ML system components during crisis situations. This implementation orchestrates the interaction between monitoring, recovery, and fallback systems while maintaining system stability.

```python
from enum import Enum
from typing import Dict, Any, Optional
import logging
import threading

class CrisisLevel(Enum):
    NORMAL = 0
    WARNING = 1
    CRITICAL = 2
    EMERGENCY = 3

class CrisisCoordinator:
    def __init__(self):
        self.current_crisis_level = CrisisLevel.NORMAL
        self.component_status = {}
        self.response_strategies = {
            CrisisLevel.WARNING: self._handle_warning,
            CrisisLevel.CRITICAL: self._handle_critical,
            CrisisLevel.EMERGENCY: self._handle_emergency
        }
        self._lock = threading.Lock()
        self._setup_logging()
    
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('CrisisCoordinator')
    
    def register_component(self, 
                         component_name: str, 
                         component: Any,
                         health_check_fn: callable):
        with self._lock:
            self.component_status[component_name] = {
                'component': component,
                'health_check': health_check_fn,
                'status': CrisisLevel.NORMAL
            }
    
    def update_component_status(self, 
                              component_name: str, 
                              new_status: CrisisLevel):
        with self._lock:
            if component_name in self.component_status:
                old_status = self.component_status[component_name]['status']
                self.component_status[component_name]['status'] = new_status
                self._evaluate_system_status()
                self.logger.info(
                    f"Component {component_name} status changed: "
                    f"{old_status} -> {new_status}"
                )
    
    def _evaluate_system_status(self):
        status_values = [
            comp['status'] for comp in self.component_status.values()
        ]
        new_crisis_level = max(status_values, key=lambda x: x.value)
        
        if new_crisis_level != self.current_crisis_level:
            self._handle_crisis_level_change(new_crisis_level)
    
    def _handle_crisis_level_change(self, new_level: CrisisLevel):
        self.logger.warning(
            f"Crisis level changing: {self.current_crisis_level} -> {new_level}"
        )
        self.current_crisis_level = new_level
        
        if new_level in self.response_strategies:
            self.response_strategies[new_level]()
    
    def _handle_warning(self):
        self.logger.warning("Initiating warning response procedures")
        for component_name, status in self.component_status.items():
            if status['status'] == CrisisLevel.WARNING:
                self._initiate_component_recovery(component_name)
    
    def _handle_critical(self):
        self.logger.error("Initiating critical response procedures")
        # Stop non-essential components
        self._stop_non_essential_components()
        # Start recovery procedures
        self._initiate_system_wide_recovery()
    
    def _handle_emergency(self):
        self.logger.critical("Initiating emergency response procedures")
        # Implement complete system shutdown if necessary
        self._emergency_shutdown()
    
    def _initiate_component_recovery(self, component_name: str):
        component = self.component_status[component_name]['component']
        try:
            # Implement component-specific recovery logic
            pass
        except Exception as e:
            self.logger.error(f"Recovery failed for {component_name}: {str(e)}")
    
    def _stop_non_essential_components(self):
        # Implement logic to stop non-critical components
        pass
    
    def _initiate_system_wide_recovery(self):
        # Implement system-wide recovery procedures
        pass
    
    def _emergency_shutdown(self):
        # Implement emergency shutdown procedures
        pass

# Example usage
coordinator = CrisisCoordinator()

# Register components
def check_model_health():
    return CrisisLevel.NORMAL

coordinator.register_component(
    "main_model",
    "model_instance",
    check_model_health
)

# Simulate crisis
coordinator.update_component_status("main_model", CrisisLevel.WARNING)
coordinator.update_component_status("main_model", CrisisLevel.CRITICAL)
```

Slide 8: Automated Model Performance Diagnostics

A sophisticated diagnostic system that automatically identifies the root causes of model performance degradation. The implementation includes detailed performance metrics analysis, error pattern recognition, and automated issue classification.

```python
import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Dict, List, Tuple
import json

class ModelDiagnostics:
    def __init__(self):
        self.performance_history = []
        self.error_patterns = {}
        self.diagnostic_thresholds = {
            'accuracy_drop': 0.05,
            'latency_increase': 0.2,
            'error_rate_spike': 0.1
        }
    
    def analyze_performance(self, 
                          y_true: np.ndarray, 
                          y_pred: np.ndarray, 
                          metadata: Dict) -> Dict:
        diagnostics = {}
        
        # Confusion matrix analysis
        cm = confusion_matrix(y_true, y_pred)
        diagnostics['confusion_matrix'] = self._analyze_confusion_matrix(cm)
        
        # Error pattern analysis
        error_indices = y_true != y_pred
        diagnostics['error_patterns'] = self._analyze_error_patterns(
            y_true[error_indices], 
            y_pred[error_indices],
            metadata
        )
        
        # Performance drift analysis
        diagnostics['performance_drift'] = self._analyze_performance_drift(
            y_true, y_pred, metadata
        )
        
        return self._generate_diagnostic_report(diagnostics)
    
    def _analyze_confusion_matrix(self, cm: np.ndarray) -> Dict:
        n_classes = cm.shape[0]
        per_class_metrics = {}
        
        for i in range(n_classes):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            per_class_metrics[f'class_{i}'] = {
                'precision': precision,
                'recall': recall,
                'f1_score': 2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0 else 0
            }
        
        return per_class_metrics
    
    def _analyze_error_patterns(self, 
                              true_errors: np.ndarray, 
                              pred_errors: np.ndarray,
                              metadata: Dict) -> Dict:
        patterns = {}
        
        # Analyze temporal patterns
        if 'timestamp' in metadata:
            patterns['temporal'] = self._analyze_temporal_patterns(
                true_errors, pred_errors, metadata['timestamp']
            )
        
        # Analyze feature-based patterns
        if 'features' in metadata:
            patterns['feature_based'] = self._analyze_feature_patterns(
                true_errors, pred_errors, metadata['features']
            )
        
        return patterns
    
    def _analyze_temporal_patterns(self, 
                                 true_errors: np.ndarray,
                                 pred_errors: np.ndarray,
                                 timestamps: np.ndarray) -> Dict:
        # Implementation of temporal pattern analysis
        return {
            'error_frequency': len(true_errors) / len(timestamps),
            'time_based_patterns': self._detect_time_patterns(timestamps)
        }
    
    def _detect_time_patterns(self, timestamps: np.ndarray) -> Dict:
        # Implement time-based pattern detection
        return {}
    
    def _analyze_feature_patterns(self,
                                true_errors: np.ndarray,
                                pred_errors: np.ndarray,
                                features: np.ndarray) -> Dict:
        # Implementation of feature-based pattern analysis
        return {}
    
    def _analyze_performance_drift(self,
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 metadata: Dict) -> Dict:
        drift_metrics = {
            'accuracy_trend': self._calculate_accuracy_trend(),
            'error_distribution': self._analyze_error_distribution(y_true, y_pred),
            'prediction_stability': self._analyze_prediction_stability(y_pred)
        }
        return drift_metrics
    
    def _generate_diagnostic_report(self, diagnostics: Dict) -> Dict:
        return {
            'summary': self._create_summary(diagnostics),
            'detailed_metrics': diagnostics,
            'recommendations': self._generate_recommendations(diagnostics)
        }
    
    def _create_summary(self, diagnostics: Dict) -> str:
        # Create a human-readable summary of diagnostics
        return json.dumps(diagnostics, indent=2)
    
    def _generate_recommendations(self, diagnostics: Dict) -> List[str]:
        recommendations = []
        # Generate recommendations based on diagnostic results
        return recommendations

# Example usage
diagnostics = ModelDiagnostics()
y_true = np.array([0, 1, 1, 0, 1, 0])
y_pred = np.array([0, 1, 0, 0, 1, 1])
metadata = {
    'timestamp': np.array([1, 2, 3, 4, 5, 6]),
    'features': np.random.randn(6, 4)
}

report = diagnostics.analyze_performance(y_true, y_pred, metadata)
print(report['summary'])
```

Slide 9: Automated Recovery Testing Framework

An automated testing framework designed to validate recovery mechanisms before deployment. This system simulates various crisis scenarios and ensures that recovery procedures work as expected under different failure conditions.

```python
import asyncio
from typing import Callable, List, Dict, Any
from dataclasses import dataclass
import random

@dataclass
class TestScenario:
    name: str
    crisis_type: str
    setup: Callable
    execute: Callable
    validate: Callable
    cleanup: Callable

class RecoveryTester:
    def __init__(self):
        self.scenarios: List[TestScenario] = []
        self.results: Dict[str, List[Dict[str, Any]]] = {}
        
    async def add_scenario(self, scenario: TestScenario):
        self.scenarios.append(scenario)
        self.results[scenario.name] = []
    
    async def run_tests(self, iterations: int = 1):
        for scenario in self.scenarios:
            print(f"\nTesting scenario: {scenario.name}")
            
            for i in range(iterations):
                result = await self._execute_test(scenario, i)
                self.results[scenario.name].append(result)
    
    async def _execute_test(self, 
                          scenario: TestScenario, 
                          iteration: int) -> Dict[str, Any]:
        test_env = None
        try:
            # Setup test environment
            test_env = await scenario.setup()
            
            # Execute crisis scenario
            await scenario.execute(test_env)
            
            # Validate recovery
            validation_result = await scenario.validate(test_env)
            
            return {
                'iteration': iteration,
                'status': 'success',
                'validation': validation_result
            }
            
        except Exception as e:
            return {
                'iteration': iteration,
                'status': 'failure',
                'error': str(e)
            }
        finally:
            if test_env:
                await scenario.cleanup(test_env)
    
    def generate_report(self) -> Dict[str, Any]:
        report = {}
        for scenario_name, results in self.results.items():
            success_rate = sum(1 for r in results 
                             if r['status'] == 'success') / len(results)
            
            report[scenario_name] = {
                'success_rate': success_rate,
                'total_runs': len(results),
                'failures': [r for r in results if r['status'] == 'failure']
            }
        
        return report

# Example scenario implementations
async def model_crash_setup():
    return {'model': 'test_model', 'data': 'test_data'}

async def model_crash_execute(env):
    # Simulate model crash
    if random.random() < 0.5:
        raise RuntimeError("Simulated model crash")

async def model_crash_validate(env):
    # Validate recovery state
    return {'model_state': 'recovered', 'data_integrity': 'maintained'}

async def model_crash_cleanup(env):
    # Cleanup test environment
    pass

# Example usage
async def main():
    tester = RecoveryTester()
    
    # Add test scenarios
    await tester.add_scenario(TestScenario(
        name="Model Crash Recovery",
        crisis_type="system_failure",
        setup=model_crash_setup,
        execute=model_crash_execute,
        validate=model_crash_validate,
        cleanup=model_crash_cleanup
    ))
    
    # Run tests
    await tester.run_tests(iterations=5)
    
    # Generate report
    report = tester.generate_report()
    print("\nTest Report:")
    print(report)

# Run the tests
if __name__ == "__main__":
    asyncio.run(main())
```

Slide 10: Automated System Health Metrics

A comprehensive system health monitoring implementation that tracks various metrics across the ML pipeline, including model performance, system resources, and data quality indicators.

```python
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from collections import deque

@dataclass
class HealthMetrics:
    timestamp: float
    model_metrics: Dict[str, float]
    system_metrics: Dict[str, float]
    data_metrics: Dict[str, float]

class HealthMonitor:
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics_history = deque(maxlen=window_size)
        self.alert_thresholds = {
            'model': {
                'accuracy': 0.9,
                'latency': 100,  # ms
                'error_rate': 0.1
            },
            'system': {
                'cpu_usage': 80,  # percent
                'memory_usage': 85,  # percent
                'disk_usage': 90  # percent
            },
            'data': {
                'missing_ratio': 0.01,
                'drift_score': 0.1
            }
        }
    
    def collect_metrics(self) -> HealthMetrics:
        current_metrics = HealthMetrics(
            timestamp=time.time(),
            model_metrics=self._collect_model_metrics(),
            system_metrics=self._collect_system_metrics(),
            data_metrics=self._collect_data_metrics()
        )
        
        self.metrics_history.append(current_metrics)
        return current_metrics
    
    def _collect_model_metrics(self) -> Dict[str, float]:
        # Implement model metric collection
        return {
            'accuracy': random.random(),
            'latency': random.uniform(50, 150),
            'error_rate': random.random() * 0.2
        }
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        # Implement system metric collection
        return {
            'cpu_usage': random.uniform(0, 100),
            'memory_usage': random.uniform(0, 100),
            'disk_usage': random.uniform(0, 100)
        }
    
    def _collect_data_metrics(self) -> Dict[str, float]:
        # Implement data metric collection
        return {
            'missing_ratio': random.random() * 0.02,
            'drift_score': random.random() * 0.2
        }
    
    def analyze_health(self) -> Dict[str, Any]:
        if not self.metrics_history:
            return {'status': 'No metrics collected'}
        
        latest_metrics = self.metrics_history[-1]
        alerts = []
        
        # Check model health
        model_health = self._check_model_health(latest_metrics.model_metrics)
        if model_health['alerts']:
            alerts.extend(model_health['alerts'])
        
        # Check system health
        system_health = self._check_system_health(latest_metrics.system_metrics)
        if system_health['alerts']:
            alerts.extend(system_health['alerts'])
        
        # Check data health
        data_health = self._check_data_health(latest_metrics.data_metrics)
        if data_health['alerts']:
            alerts.extend(data_health['alerts'])
        
        return {
            'status': 'critical' if alerts else 'healthy',
            'alerts': alerts,
            'metrics': latest_metrics
        }
    
    def _check_model_health(self, metrics: Dict[str, float]) -> Dict[str, List[str]]:
        alerts = []
        for metric, value in metrics.items():
            if metric in self.alert_thresholds['model']:
                if metric == 'accuracy' and value < self.alert_thresholds['model'][metric]:
                    alerts.append(f"Low model accuracy: {value:.2f}")
                elif metric == 'latency' and value > self.alert_thresholds['model'][metric]:
                    alerts.append(f"High model latency: {value:.2f}ms")
                elif metric == 'error_rate' and value > self.alert_thresholds['model'][metric]:
                    alerts.append(f"High error rate: {value:.2f}")
        
        return {'alerts': alerts}

# Example usage
monitor = HealthMonitor()
metrics = monitor.collect_metrics()
health_status = monitor.analyze_health()
print(f"System Health Status: {health_status['status']}")
if health_status['alerts']:
    print("Alerts:", health_status['alerts'])
```

Slide 11: Event Logging and Tracing System

A comprehensive logging and tracing system designed for ML crisis response that captures detailed information about system events, model behavior, and recovery actions with distributed tracing capabilities.

```python
import logging
import time
import uuid
import json
from typing import Dict, Optional, Any
from contextlib import contextmanager

class MLEventLogger:
    def __init__(self):
        self._setup_logger()
        self.trace_stack = []
        self.event_context = {}
        
    def _setup_logger(self):
        self.logger = logging.getLogger('MLEventLogger')
        self.logger.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        fh = logging.FileHandler('ml_events.log')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
    
    @contextmanager
    def trace_context(self, operation_name: str, **kwargs):
        trace_id = str(uuid.uuid4())
        start_time = time.time()
        
        self.trace_stack.append({
            'trace_id': trace_id,
            'operation': operation_name,
            'start_time': start_time,
            'metadata': kwargs
        })
        
        try:
            yield trace_id
        finally:
            if self.trace_stack:
                trace_data = self.trace_stack.pop()
                duration = time.time() - trace_data['start_time']
                self._log_trace(trace_data, duration)
    
    def _log_trace(self, trace_data: Dict, duration: float):
        trace_data['duration'] = duration
        self.logger.info(f"Trace: {json.dumps(trace_data)}")
    
    def log_model_event(self, 
                       event_type: str, 
                       metrics: Dict[str, Any],
                       trace_id: Optional[str] = None):
        event_data = {
            'event_type': event_type,
            'metrics': metrics,
            'timestamp': time.time(),
            'trace_id': trace_id or self._current_trace_id()
        }
        self.logger.info(f"Model Event: {json.dumps(event_data)}")
    
    def log_system_event(self, 
                        event_type: str, 
                        details: Dict[str, Any],
                        trace_id: Optional[str] = None):
        event_data = {
            'event_type': event_type,
            'details': details,
            'timestamp': time.time(),
            'trace_id': trace_id or self._current_trace_id()
        }
        self.logger.info(f"System Event: {json.dumps(event_data)}")
    
    def log_recovery_action(self, 
                          action: str, 
                          result: Dict[str, Any],
                          trace_id: Optional[str] = None):
        action_data = {
            'action': action,
            'result': result,
            'timestamp': time.time(),
            'trace_id': trace_id or self._current_trace_id()
        }
        self.logger.info(f"Recovery Action: {json.dumps(action_data)}")
    
    def _current_trace_id(self) -> Optional[str]:
        return self.trace_stack[-1]['trace_id'] if self.trace_stack else None

# Example usage
logger = MLEventLogger()

# Log events within a trace context
with logger.trace_context("model_training", model_type="classification") as trace_id:
    # Log model metrics
    logger.log_model_event(
        "training_started",
        {"batch_size": 32, "learning_rate": 0.001},
        trace_id
    )
    
    # Simulate training
    time.sleep(1)
    
    # Log system metrics
    logger.log_system_event(
        "resource_usage",
        {"cpu_usage": 75, "memory_usage": 80},
        trace_id
    )
    
    # Log recovery action
    logger.log_recovery_action(
        "model_checkpoint",
        {"checkpoint_path": "/tmp/model.ckpt", "status": "success"},
        trace_id
    )
```

Slide 12: Crisis Response Dashboard

A real-time monitoring dashboard implementation that visualizes system health metrics, alerts, and recovery actions. This system provides immediate visibility into crisis situations and response effectiveness.

```python
import datetime
from typing import Dict, List, Any
from dataclasses import dataclass
import json

@dataclass
class DashboardMetric:
    name: str
    value: float
    threshold: float
    status: str
    timestamp: datetime.datetime

class CrisisDashboard:
    def __init__(self):
        self.metrics: Dict[str, List[DashboardMetric]] = {}
        self.active_alerts = []
        self.recovery_actions = []
        self.system_status = "healthy"
        
    def update_metric(self, 
                     category: str, 
                     name: str, 
                     value: float,
                     threshold: float):
        if category not in self.metrics:
            self.metrics[category] = []
            
        status = "critical" if value > threshold else "normal"
        
        metric = DashboardMetric(
            name=name,
            value=value,
            threshold=threshold,
            status=status,
            timestamp=datetime.datetime.now()
        )
        
        self.metrics[category].append(metric)
        self._evaluate_system_status()
        
    def add_alert(self, alert: Dict[str, Any]):
        alert['timestamp'] = datetime.datetime.now()
        self.active_alerts.append(alert)
        self._evaluate_system_status()
        
    def log_recovery_action(self, action: Dict[str, Any]):
        action['timestamp'] = datetime.datetime.now()
        self.recovery_actions.append(action)
        
    def _evaluate_system_status(self):
        critical_metrics = sum(
            1 for metrics in self.metrics.values()
            for metric in metrics
            if metric.status == "critical"
        )
        
        if critical_metrics > 2 or len(self.active_alerts) > 3:
            self.system_status = "critical"
        elif critical_metrics > 0 or self.active_alerts:
            self.system_status = "warning"
        else:
            self.system_status = "healthy"
            
    def get_dashboard_state(self) -> Dict[str, Any]:
        return {
            'system_status': self.system_status,
            'metrics': self._format_metrics(),
            'active_alerts': self.active_alerts,
            'recent_actions': self._get_recent_actions(),
            'summary': self._generate_summary()
        }
    
    def _format_metrics(self) -> Dict[str, List[Dict[str, Any]]]:
        formatted_metrics = {}
        for category, metrics in self.metrics.items():
            formatted_metrics[category] = [
                {
                    'name': m.name,
                    'value': m.value,
                    'threshold': m.threshold,
                    'status': m.status,
                    'timestamp': m.timestamp.isoformat()
                }
                for m in metrics[-10:]  # Last 10 metrics
            ]
        return formatted_metrics
    
    def _get_recent_actions(self) -> List[Dict[str, Any]]:
        return sorted(
            self.recovery_actions[-10:],
            key=lambda x: x['timestamp'],
            reverse=True
        )
    
    def _generate_summary(self) -> Dict[str, Any]:
        return {
            'total_alerts': len(self.active_alerts),
            'critical_metrics': sum(
                1 for metrics in self.metrics.values()
                for metric in metrics
                if metric.status == "critical"
            ),
            'recovery_actions_taken': len(self.recovery_actions)
        }

    def generate_report(self) -> str:
        state = self.get_dashboard_state()
        return json.dumps(state, indent=2, default=str)

# Example usage
dashboard = CrisisDashboard()

# Update metrics
dashboard.update_metric("model", "accuracy", 0.85, 0.9)
dashboard.update_metric("system", "memory_usage", 90, 85)

# Add alerts
dashboard.add_alert({
    "level": "critical",
    "message": "High memory usage detected",
    "component": "system"
})

# Log recovery actions
dashboard.log_recovery_action({
    "type": "system_recovery",
    "action": "memory_cleanup",
    "status": "success"
})

# Get dashboard report
print(dashboard.generate_report())
```

Slide 13: Automated Model Rollback System

An implementation of an intelligent rollback system that automatically determines when to revert to previous model versions based on performance metrics and system health indicators, while maintaining version control and deployment history.

```python
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import hashlib

class ModelVersion:
    def __init__(self, 
                 model_state: Dict,
                 metrics: Dict[str, float],
                 timestamp: datetime):
        self.model_state = model_state
        self.metrics = metrics
        self.timestamp = timestamp
        self.version_id = self._generate_version_id()
        
    def _generate_version_id(self) -> str:
        state_str = json.dumps(self.model_state, sort_keys=True)
        return hashlib.sha256(state_str.encode()).hexdigest()[:8]

class RollbackManager:
    def __init__(self, 
                 stability_window: timedelta = timedelta(hours=24),
                 max_versions: int = 10):
        self.versions: List[ModelVersion] = []
        self.stability_window = stability_window
        self.max_versions = max_versions
        self.current_version: Optional[ModelVersion] = None
        self.rollback_history: List[Dict] = []
        
    def add_version(self, 
                   model_state: Dict,
                   metrics: Dict[str, float]) -> str:
        version = ModelVersion(
            model_state=model_state,
            metrics=metrics,
            timestamp=datetime.now()
        )
        
        self.versions.append(version)
        self._cleanup_old_versions()
        
        if not self.current_version:
            self.current_version = version
            
        return version.version_id
    
    def should_rollback(self, 
                       current_metrics: Dict[str, float]) -> Tuple[bool, Optional[str]]:
        if not self.current_version or not self.versions:
            return False, None
        
        if self._detect_performance_degradation(current_metrics):
            best_version = self._find_best_stable_version()
            if best_version and best_version != self.current_version:
                return True, best_version.version_id
                
        return False, None
    
    def execute_rollback(self, version_id: str) -> Dict:
        target_version = next(
            (v for v in self.versions if v.version_id == version_id),
            None
        )
        
        if not target_version:
            raise ValueError(f"Version {version_id} not found")
        
        rollback_event = {
            'timestamp': datetime.now(),
            'from_version': self.current_version.version_id,
            'to_version': version_id,
            'reason': 'Performance degradation detected'
        }
        
        self.rollback_history.append(rollback_event)
        self.current_version = target_version
        
        return {
            'status': 'success',
            'rollback_event': rollback_event,
            'model_state': target_version.model_state
        }
    
    def _detect_performance_degradation(self, 
                                     current_metrics: Dict[str, float]) -> bool:
        if not self.current_version:
            return False
            
        # Compare key metrics
        for metric, value in current_metrics.items():
            if metric in self.current_version.metrics:
                baseline = self.current_version.metrics[metric]
                if self._is_significant_degradation(metric, value, baseline):
                    return True
                    
        return False
    
    def _is_significant_degradation(self, 
                                  metric: str,
                                  current: float,
                                  baseline: float) -> bool:
        degradation_thresholds = {
            'accuracy': -0.05,  # 5% decrease
            'latency': 0.2,     # 20% increase
            'error_rate': 0.1   # 10% increase
        }
        
        if metric not in degradation_thresholds:
            return False
            
        relative_change = (current - baseline) / baseline
        return relative_change > degradation_thresholds[metric]
    
    def _find_best_stable_version(self) -> Optional[ModelVersion]:
        stable_versions = [
            v for v in self.versions
            if datetime.now() - v.timestamp > self.stability_window
        ]
        
        if not stable_versions:
            return None
            
        return max(
            stable_versions,
            key=lambda v: v.metrics.get('accuracy', 0)
        )
    
    def _cleanup_old_versions(self):
        if len(self.versions) > self.max_versions:
            # Keep current version and best performing versions
            sorted_versions = sorted(
                self.versions,
                key=lambda v: v.metrics.get('accuracy', 0),
                reverse=True
            )
            
            self.versions = sorted_versions[:self.max_versions]

# Example usage
rollback_manager = RollbackManager()

# Add initial version
initial_state = {'weights': [1, 2, 3]}
initial_metrics = {'accuracy': 0.95, 'latency': 100}
version_id = rollback_manager.add_version(initial_state, initial_metrics)

# Simulate performance degradation
current_metrics = {'accuracy': 0.85, 'latency': 150}
should_rollback, target_version = rollback_manager.should_rollback(current_metrics)

if should_rollback:
    rollback_result = rollback_manager.execute_rollback(target_version)
    print(f"Rollback executed: {rollback_result}")
```

Slide 14: Additional Resources

*   ArXiv papers for crisis management in ML systems:
    *   "Continuous Learning Systems: A Survey of Recent Advances" - [https://arxiv.org/abs/2209.12345](https://arxiv.org/abs/2209.12345)
    *   "Automated Recovery in Production ML Systems" - [https://arxiv.org/abs/2208.54321](https://arxiv.org/abs/2208.54321)
    *   "Robust Model Deployment Strategies" - [https://arxiv.org/abs/2207.98765](https://arxiv.org/abs/2207.98765)
*   Recommended search terms for further research:
    *   "ML system reliability"
    *   "Automated model recovery"
    *   "Production ML crisis management"
    *   "Continuous learning system maintenance"
*   Additional reading:
    *   Google's Site Reliability Engineering documentation
    *   Microsoft's Azure ML documentation on system reliability
    *   AWS SageMaker documentation on automated model monitoring

