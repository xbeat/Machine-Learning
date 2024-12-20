## 4 Ways to Test ML Models in Production
Slide 1: A/B Testing Implementation for ML Models

A/B testing is a statistical methodology for comparing two ML models by randomly routing production traffic between them. This implementation demonstrates how to create a request router that distributes incoming requests between legacy and candidate models with configurable traffic allocation.

```python
import random
from typing import Dict, Any, Callable
import numpy as np

class ABTestRouter:
    def __init__(self, legacy_model, candidate_model, candidate_traffic_fraction=0.1):
        self.legacy_model = legacy_model
        self.candidate_model = candidate_model
        self.traffic_fraction = candidate_traffic_fraction
        self.metrics = {'legacy': [], 'candidate': []}
    
    def route_request(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Randomly route request based on traffic fraction
        if random.random() < self.traffic_fraction:
            prediction = self.candidate_model.predict(input_data)
            self.metrics['candidate'].append(prediction)
            return {'model': 'candidate', 'prediction': prediction}
        else:
            prediction = self.legacy_model.predict(input_data)
            self.metrics['legacy'].append(prediction)
            return {'model': 'legacy', 'prediction': prediction}

    def get_performance_stats(self):
        return {
            'legacy_avg': np.mean(self.metrics['legacy']),
            'candidate_avg': np.mean(self.metrics['candidate']),
            'traffic_split': f"{self.traffic_fraction*100}% candidate"
        }
```

Slide 2: Canary Testing Implementation

Canary testing extends A/B testing by targeting specific user segments rather than random traffic allocation. This implementation shows how to route requests based on user attributes while monitoring system health metrics to ensure safe deployment.

```python
class CanaryRouter:
    def __init__(self, legacy_model, candidate_model, initial_userbase=0.05):
        self.legacy_model = legacy_model
        self.candidate_model = candidate_model
        self.user_fraction = initial_userbase
        self.health_metrics = {'latency': [], 'error_rate': []}
        self.user_segments = set()
    
    def is_canary_user(self, user_id: str) -> bool:
        # Deterministic hashing for consistent user assignment
        hash_value = hash(user_id) % 100
        return hash_value < (self.user_fraction * 100)
    
    def route_request(self, user_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        try:
            if self.is_canary_user(user_id):
                self.user_segments.add(user_id)
                prediction = self.candidate_model.predict(input_data)
                model_type = 'candidate'
            else:
                prediction = self.legacy_model.predict(input_data)
                model_type = 'legacy'
                
            latency = time.time() - start_time
            self.health_metrics['latency'].append(latency)
            return {'model': model_type, 'prediction': prediction}
            
        except Exception as e:
            self.health_metrics['error_rate'].append(1)
            raise e
```

Slide 3: Advanced Canary Deployment with Automatic Rollback

This implementation showcases an advanced canary deployment system that automatically rolls back to the legacy model if certain performance thresholds are breached. It includes monitoring of multiple metrics and graceful degradation capabilities.

```python
class AdvancedCanaryRouter:
    def __init__(self, 
                 legacy_model,
                 candidate_model,
                 error_threshold=0.01,
                 latency_threshold=100):
        self.legacy_model = legacy_model
        self.candidate_model = candidate_model
        self.error_threshold = error_threshold
        self.latency_threshold = latency_threshold
        self.window_size = 100
        self.metrics_window = []
        self.is_healthy = True
    
    def check_health(self, metrics: Dict[str, float]) -> bool:
        self.metrics_window.append(metrics)
        if len(self.metrics_window) > self.window_size:
            self.metrics_window.pop(0)
            
        # Calculate rolling metrics
        recent_errors = sum(m['error_rate'] for m in self.metrics_window) / len(self.metrics_window)
        avg_latency = sum(m['latency'] for m in self.metrics_window) / len(self.metrics_window)
        
        return (recent_errors < self.error_threshold and 
                avg_latency < self.latency_threshold)
    
    def rollback(self):
        self.is_healthy = False
        print("ALERT: Canary deployment rolled back due to performance degradation")
```

Slide 4: Interleaved Testing for Recommendation Systems

Interleaved testing enables simultaneous evaluation of multiple recommendation models by mixing their outputs. This implementation demonstrates how to combine and shuffle recommendations from different models while maintaining tracking for evaluation.

```python
from typing import List
import random

class InterleaveTestingRouter:
    def __init__(self, legacy_recommender, candidate_recommender):
        self.legacy_recommender = legacy_recommender
        self.candidate_recommender = candidate_recommender
        self.click_tracking = {'legacy': 0, 'candidate': 0}
    
    def get_recommendations(self, user_id: str, n_items: int) -> List[Dict]:
        # Get recommendations from both models
        legacy_recs = self.legacy_recommender.get_recommendations(user_id, n_items)
        candidate_recs = self.candidate_recommender.get_recommendations(user_id, n_items)
        
        # Interleave recommendations
        interleaved_recs = []
        for i in range(n_items):
            if i % 2 == 0 and legacy_recs:
                rec = legacy_recs.pop(0)
                rec['source'] = 'legacy'
            elif candidate_recs:
                rec = candidate_recs.pop(0)
                rec['source'] = 'candidate'
            interleaved_recs.append(rec)
        
        random.shuffle(interleaved_recs)
        return interleaved_recs
```

Slide 5: Shadow Testing Infrastructure

Shadow testing enables parallel evaluation of models without affecting user experience. This implementation creates a shadow deployment infrastructure that duplicates production traffic to the candidate model while maintaining detailed performance logs.

```python
import asyncio
import time
from typing import Dict, Any, List

class ShadowTestingSystem:
    def __init__(self, legacy_model, candidate_model):
        self.legacy_model = legacy_model
        self.candidate_model = candidate_model
        self.shadow_logs = []
        self.performance_metrics = {
            'latency_diff': [],
            'prediction_diff': [],
            'error_count': {'legacy': 0, 'candidate': 0}
        }
    
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        # Production response uses legacy model
        legacy_start = time.time()
        legacy_response = self.legacy_model.predict(request_data)
        legacy_latency = time.time() - legacy_start
        
        # Shadow traffic to candidate model
        try:
            candidate_start = time.time()
            candidate_response = self.candidate_model.predict(request_data)
            candidate_latency = time.time() - candidate_start
            
            # Log comparison metrics
            self.shadow_logs.append({
                'timestamp': time.time(),
                'legacy_response': legacy_response,
                'candidate_response': candidate_response,
                'latency_diff': candidate_latency - legacy_latency
            })
            
        except Exception as e:
            self.performance_metrics['error_count']['candidate'] += 1
            
        return {'prediction': legacy_response, 'latency': legacy_latency}
```

Slide 6: Real-time Model Performance Monitoring

This implementation shows how to create a comprehensive monitoring system for production ML models. It tracks key performance indicators, drift detection, and maintains statistical significance testing for model comparisons.

```python
import numpy as np
from scipy import stats
from collections import deque

class ModelMonitor:
    def __init__(self, window_size=1000):
        self.metrics_window = deque(maxlen=window_size)
        self.baseline_stats = None
        self.drift_threshold = 0.05
        
    def update_metrics(self, prediction: float, actual: float, 
                      features: Dict[str, float]) -> Dict[str, float]:
        metrics = {
            'prediction': prediction,
            'actual': actual,
            'error': abs(prediction - actual),
            'feature_values': features
        }
        self.metrics_window.append(metrics)
        return self._calculate_current_stats()
    
    def _calculate_current_stats(self) -> Dict[str, float]:
        if len(self.metrics_window) < 100:  # Minimum sample size
            return {}
            
        current_errors = [m['error'] for m in self.metrics_window]
        current_stats = {
            'mean_error': np.mean(current_errors),
            'error_std': np.std(current_errors),
            'p_value': self._calculate_drift()
        }
        return current_stats
    
    def _calculate_drift(self) -> float:
        if not self.baseline_stats:
            return 1.0
        
        current_errors = [m['error'] for m in self.metrics_window]
        _, p_value = stats.ks_2samp(current_errors, self.baseline_stats)
        return p_value
```

Slide 7: Feature Distribution Monitoring

This implementation focuses on monitoring feature distributions in production to detect data drift and concept drift. It uses statistical tests and visualization capabilities for real-time monitoring.

```python
from scipy.stats import wasserstein_distance
import pandas as pd
import numpy as np

class FeatureMonitor:
    def __init__(self, feature_names: List[str], 
                 reference_distributions: Dict[str, np.ndarray]):
        self.feature_names = feature_names
        self.reference_distributions = reference_distributions
        self.current_distributions = {
            feat: [] for feat in feature_names
        }
        self.drift_scores = {}
        
    def update_distributions(self, features: Dict[str, float]):
        for feat_name, value in features.items():
            self.current_distributions[feat_name].append(value)
            
        if len(self.current_distributions[self.feature_names[0]]) >= 1000:
            self._calculate_drift_scores()
            
    def _calculate_drift_scores(self):
        for feat_name in self.feature_names:
            current_dist = np.array(self.current_distributions[feat_name])
            reference_dist = self.reference_distributions[feat_name]
            
            self.drift_scores[feat_name] = wasserstein_distance(
                current_dist, reference_dist
            )
            
        # Reset current distributions after calculation
        self.current_distributions = {
            feat: [] for feat in self.feature_names
        }
```

Slide 8: Automated Model Rollback System

This implementation provides an automated system for rolling back model deployments based on multiple monitoring metrics. It includes configurable thresholds and graceful degradation with automatic alerts and logging.

```python
import logging
from datetime import datetime
from typing import Dict, List, Optional

class AutomatedRollbackSystem:
    def __init__(self, 
                 legacy_model,
                 candidate_model,
                 metrics_thresholds: Dict[str, float]):
        self.legacy_model = legacy_model
        self.candidate_model = candidate_model
        self.thresholds = metrics_thresholds
        self.active_model = candidate_model
        self.rollback_history = []
        self.alert_callbacks = []
        
    def evaluate_health(self, current_metrics: Dict[str, float]) -> bool:
        violations = []
        for metric_name, threshold in self.thresholds.items():
            if current_metrics.get(metric_name, 0) > threshold:
                violations.append(f"{metric_name}: {current_metrics[metric_name]:.4f}")
        
        if violations:
            self._trigger_rollback(violations)
            return False
        return True
    
    def _trigger_rollback(self, violations: List[str]):
        self.active_model = self.legacy_model
        incident = {
            'timestamp': datetime.now(),
            'violations': violations,
            'metrics_snapshot': current_metrics
        }
        self.rollback_history.append(incident)
        self._send_alerts(incident)
        
    def _send_alerts(self, incident: Dict):
        for callback in self.alert_callbacks:
            try:
                callback(incident)
            except Exception as e:
                logging.error(f"Failed to send alert: {str(e)}")
```

Slide 9: Statistical Significance Testing for Model Comparison

This implementation provides comprehensive statistical testing functionality for comparing model performance in production. It includes multiple statistical tests and confidence interval calculations.

```python
from scipy import stats
import numpy as np
from typing import Tuple, Optional

class ModelComparisonTester:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.legacy_metrics = []
        self.candidate_metrics = []
        
    def add_observations(self, 
                        legacy_performance: float, 
                        candidate_performance: float):
        self.legacy_metrics.append(legacy_performance)
        self.candidate_metrics.append(candidate_performance)
        
    def calculate_significance(self) -> Dict[str, Any]:
        if len(self.legacy_metrics) < 30:  # Minimum sample size
            return {'status': 'insufficient_data'}
            
        # t-test for performance difference
        t_stat, p_value = stats.ttest_ind(
            self.legacy_metrics,
            self.candidate_metrics
        )
        
        # Effect size calculation (Cohen's d)
        effect_size = (np.mean(self.candidate_metrics) - np.mean(self.legacy_metrics)) / \
                     np.sqrt((np.var(self.legacy_metrics) + np.var(self.candidate_metrics)) / 2)
                     
        # Calculate confidence intervals
        ci_legacy = stats.t.interval(
            1 - self.alpha,
            len(self.legacy_metrics) - 1,
            loc=np.mean(self.legacy_metrics),
            scale=stats.sem(self.legacy_metrics)
        )
        
        return {
            'p_value': p_value,
            'effect_size': effect_size,
            'significant': p_value < self.alpha,
            'confidence_intervals': {
                'legacy': ci_legacy,
                'candidate': stats.t.interval(
                    1 - self.alpha,
                    len(self.candidate_metrics) - 1,
                    loc=np.mean(self.candidate_metrics),
                    scale=stats.sem(self.candidate_metrics)
                )
            }
        }
```

Slide 10: Real-time Performance Visualization System

This implementation creates a real-time visualization system for monitoring model performance metrics, including custom plotting functions and interactive dashboard capabilities for production monitoring.

```python
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import numpy as np

class PerformanceVisualizer:
    def __init__(self, metric_names: List[str]):
        self.metric_names = metric_names
        self.time_series_data = {
            metric: [] for metric in metric_names
        }
        self.timestamps = []
        
    def update_metrics(self, 
                      current_metrics: Dict[str, float], 
                      timestamp: float):
        self.timestamps.append(timestamp)
        for metric in self.metric_names:
            self.time_series_data[metric].append(
                current_metrics.get(metric, np.nan)
            )
            
    def generate_performance_plot(self, 
                                window_size: Optional[int] = None):
        plt.figure(figsize=(12, 6))
        for metric in self.metric_names:
            data = self.time_series_data[metric]
            if window_size:
                data = data[-window_size:]
            plt.plot(data, label=metric)
            
        plt.title('Model Performance Metrics Over Time')
        plt.xlabel('Time Window')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.grid(True)
        
        return plt.gcf()
```

Slide 11: Production Log Analysis System

This implementation provides comprehensive log analysis capabilities for production ML models, including pattern detection, anomaly identification, and automated report generation for debugging and monitoring purposes.

```python
import json
from collections import defaultdict
from datetime import datetime, timedelta
import pandas as pd

class ProductionLogAnalyzer:
    def __init__(self, log_retention_days: int = 30):
        self.log_storage = defaultdict(list)
        self.retention_period = timedelta(days=log_retention_days)
        self.anomaly_patterns = set()
        
    def process_log_entry(self, log_entry: Dict[str, Any]):
        timestamp = datetime.fromtimestamp(log_entry['timestamp'])
        self.clean_old_logs(timestamp)
        
        # Categorize and store log entry
        category = self._categorize_log(log_entry)
        self.log_storage[category].append({
            'timestamp': timestamp,
            'entry': log_entry,
            'metadata': self._extract_metadata(log_entry)
        })
        
        # Analyze for anomalies
        if self._is_anomalous(log_entry):
            self._record_anomaly(log_entry)
    
    def generate_analysis_report(self) -> Dict[str, Any]:
        return {
            'log_volume': self._analyze_volume(),
            'error_patterns': self._analyze_errors(),
            'performance_metrics': self._analyze_performance(),
            'anomaly_summary': self._summarize_anomalies()
        }
    
    def _analyze_volume(self) -> Dict[str, int]:
        return {
            category: len(entries) 
            for category, entries in self.log_storage.items()
        }
        
    def _analyze_errors(self) -> List[Dict[str, Any]]:
        error_logs = self.log_storage.get('error', [])
        error_patterns = defaultdict(int)
        
        for log in error_logs:
            pattern = self._extract_error_pattern(log['entry'])
            error_patterns[pattern] += 1
            
        return [{'pattern': k, 'count': v} 
                for k, v in error_patterns.items()]
```

Slide 12: Advanced Model Validation Suite

This system implements comprehensive validation tests for production ML models, including data quality checks, model behavior validation, and performance benchmarking across different operational scenarios.

```python
from typing import List, Dict, Optional, Callable
import numpy as np
from dataclasses import dataclass

@dataclass
class ValidationResult:
    passed: bool
    metric_name: str
    actual_value: float
    threshold: float
    details: Optional[Dict] = None

class ModelValidationSuite:
    def __init__(self, 
                 validation_thresholds: Dict[str, float],
                 custom_validators: Optional[List[Callable]] = None):
        self.thresholds = validation_thresholds
        self.custom_validators = custom_validators or []
        self.validation_history = []
        
    def validate_model(self, 
                      model,
                      validation_data: Dict[str, np.ndarray],
                      reference_predictions: Optional[np.ndarray] = None
                      ) -> List[ValidationResult]:
        results = []
        
        # Core validation checks
        results.extend(self._validate_prediction_bounds(
            model, validation_data['X']))
        results.extend(self._validate_performance_metrics(
            model, validation_data))
        
        if reference_predictions is not None:
            results.extend(self._validate_prediction_drift(
                model, validation_data['X'], reference_predictions))
            
        # Custom validation checks
        for validator in self.custom_validators:
            results.extend(validator(model, validation_data))
            
        self.validation_history.append({
            'timestamp': datetime.now(),
            'results': results
        })
        
        return results
    
    def _validate_prediction_bounds(self, 
                                  model,
                                  X: np.ndarray) -> List[ValidationResult]:
        predictions = model.predict(X)
        return [
            ValidationResult(
                passed=np.all(predictions >= self.thresholds['min_prediction']),
                metric_name='prediction_lower_bound',
                actual_value=np.min(predictions),
                threshold=self.thresholds['min_prediction']
            ),
            ValidationResult(
                passed=np.all(predictions <= self.thresholds['max_prediction']),
                metric_name='prediction_upper_bound',
                actual_value=np.max(predictions),
                threshold=self.thresholds['max_prediction']
            )
        ]
```

Slide 13: Resource Monitoring and Auto-scaling System

This implementation provides automated resource monitoring and scaling capabilities for ML model deployments, ensuring optimal performance under varying load conditions while maintaining cost efficiency.

```python
import psutil
import threading
from typing import Dict, List, Optional
import time

class ResourceMonitor:
    def __init__(self, 
                 scaling_thresholds: Dict[str, float],
                 check_interval: int = 60):
        self.thresholds = scaling_thresholds
        self.interval = check_interval
        self.monitoring_active = False
        self.resource_history = []
        self.scaling_actions = []
        
    def start_monitoring(self):
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop
        )
        self.monitor_thread.start()
        
    def _monitoring_loop(self):
        while self.monitoring_active:
            metrics = self._collect_metrics()
            self.resource_history.append(metrics)
            
            if self._should_scale(metrics):
                self._trigger_scaling(metrics)
                
            time.sleep(self.interval)
    
    def _collect_metrics(self) -> Dict[str, float]:
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'network_io': self._get_network_stats()
        }
        
    def _should_scale(self, metrics: Dict[str, float]) -> bool:
        return any(
            metrics[metric] > threshold
            for metric, threshold in self.thresholds.items()
        )
```

Slide 14: Model Dependency Monitoring System

This implementation tracks and monitors dependencies between different model components in production, detecting potential cascade failures and providing automated dependency health checks.

```python
from typing import Dict, List, Set
import networkx as nx
from datetime import datetime

class DependencyMonitor:
    def __init__(self):
        self.dependency_graph = nx.DiGraph()
        self.health_status = {}
        self.cascade_history = []
        
    def register_dependency(self, 
                          source_model: str, 
                          target_model: str,
                          criticality: float = 1.0):
        self.dependency_graph.add_edge(
            source_model,
            target_model,
            weight=criticality
        )
        self.health_status[source_model] = True
        self.health_status[target_model] = True
        
    def update_model_health(self, 
                          model_name: str,
                          is_healthy: bool) -> Set[str]:
        self.health_status[model_name] = is_healthy
        affected_models = set()
        
        if not is_healthy:
            # Find all downstream dependencies
            for successor in nx.descendants(self.dependency_graph, model_name):
                criticality = self._get_path_criticality(model_name, successor)
                if criticality > 0.7:  # High criticality threshold
                    affected_models.add(successor)
                    
        if affected_models:
            self._log_cascade_event(model_name, affected_models)
            
        return affected_models
    
    def _get_path_criticality(self, 
                            source: str,
                            target: str) -> float:
        path = nx.shortest_path(
            self.dependency_graph,
            source=source,
            target=target,
            weight='weight'
        )
        return min(
            self.dependency_graph[path[i]][path[i+1]]['weight']
            for i in range(len(path)-1)
        )
```

Slide 15: Advanced Model Versioning System

This implementation provides a sophisticated versioning system for ML models in production, tracking model lineage, parameters, and performance metrics across different versions and environments.

```python
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json

@dataclass
class ModelVersion:
    version_id: str
    parent_version: Optional[str]
    creation_time: datetime
    parameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    environment_config: Dict[str, str]

class ModelVersionControl:
    def __init__(self):
        self.versions = {}
        self.current_version = None
        self.version_graph = nx.DiGraph()
        
    def register_version(self,
                        model,
                        parameters: Dict[str, Any],
                        metrics: Dict[str, float],
                        env_config: Dict[str, str]) -> str:
        # Generate version hash
        version_content = json.dumps({
            'parameters': parameters,
            'timestamp': datetime.now().isoformat(),
            'parent': self.current_version
        })
        version_id = hashlib.sha256(
            version_content.encode()
        ).hexdigest()[:12]
        
        # Create version object
        version = ModelVersion(
            version_id=version_id,
            parent_version=self.current_version,
            creation_time=datetime.now(),
            parameters=parameters,
            performance_metrics=metrics,
            environment_config=env_config
        )
        
        self.versions[version_id] = version
        self.version_graph.add_node(version_id)
        
        if self.current_version:
            self.version_graph.add_edge(
                self.current_version,
                version_id
            )
            
        return version_id
    
    def get_version_lineage(self, 
                           version_id: str) -> List[ModelVersion]:
        path = nx.shortest_path(
            self.version_graph,
            source=list(self.version_graph.nodes())[0],
            target=version_id
        )
        return [self.versions[v] for v in path]
```

Slide 16: Additional Resources

1.  [https://arxiv.org/abs/2108.07258](https://arxiv.org/abs/2108.07258) - "Production ML Model Monitoring: Challenges and Best Practices"
2.  [https://arxiv.org/abs/2205.09865](https://arxiv.org/abs/2205.09865) - "A Survey of Model Testing Strategies in Production Environments"
3.  [https://arxiv.org/abs/2103.12140](https://arxiv.org/abs/2103.12140) - "MLOps: From Model-centric to Production-centric AI"
4.  [https://arxiv.org/abs/2209.14764](https://arxiv.org/abs/2209.14764) - "Efficient Testing of Deep Learning Models in Production"
5.  [https://arxiv.org/abs/2203.15355](https://arxiv.org/abs/2203.15355) - "Automated Model Testing and Validation in Production Systems"

