## Advantages of Using Pure Python over Anaconda
Slide 1: Pure Python Project Setup

Understanding proper Python project structure is crucial for maintainable codebases. We'll create a modern Python project with virtual environments, dependencies management, and proper package organization that demonstrates the advantages of Pure Python over distributions.

```python
# Project structure setup
project_root/
├── src/
│   ├── __init__.py
│   └── main.py
├── tests/
│   ├── __init__.py
│   └── test_main.py
├── requirements.txt
├── setup.py
└── venv/

# Terminal commands
python -m venv venv
source venv/bin/activate  # Unix
.\venv\Scripts\activate   # Windows

# requirements.txt
numpy==1.21.0
pandas==1.3.0
scikit-learn==0.24.2
```

Slide 2: Virtual Environment Management

Pure Python's virtual environments provide isolated package management per project, preventing dependency conflicts and ensuring reproducible environments. This approach is more explicit and controllable than Anaconda's global package management.

```python
# Create and manage virtual environment
import subprocess
import sys

def setup_project_env():
    # Create virtual environment
    subprocess.run([sys.executable, "-m", "venv", "venv"])
    
    # Install requirements
    if sys.platform == "win32":
        pip_path = "venv\\Scripts\\pip"
    else:
        pip_path = "venv/bin/pip"
    
    subprocess.run([pip_path, "install", "-r", "requirements.txt"])

if __name__ == "__main__":
    setup_project_env()
```

Slide 3: Dependency Management

Pure Python's pip package manager allows precise control over project dependencies. This example demonstrates how to manage, freeze, and install dependencies while maintaining minimal footprint compared to Anaconda's bulk installation.

```python
import pkg_resources
import subprocess
from pathlib import Path

def manage_dependencies():
    # Get installed packages
    installed = {pkg.key: pkg.version for pkg 
                in pkg_resources.working_set}
    
    # Save current environment
    with open('requirements.txt', 'w') as f:
        for package, version in installed.items():
            f.write(f"{package}=={version}\n")
    
    # Install specific version
    subprocess.run(["pip", "install", 
                   "numpy==1.21.0", "--no-cache-dir"])

    return installed

print(manage_dependencies())
```

Slide 4: Project-Specific Package Installation

In Pure Python, packages are installed within the project's virtual environment, maintaining isolation. This script demonstrates package management and verification within a specific project context.

```python
import site
import os
from pathlib import Path

def verify_package_location():
    # Get virtual environment site-packages
    venv_path = Path(site.getsitepackages()[0])
    
    # List installed packages
    packages = [p for p in venv_path.glob("*-info")]
    
    # Package installation verification
    def is_package_local(package_name):
        return any(p.name.startswith(package_name) 
                  for p in packages)
    
    packages_status = {
        "numpy": is_package_local("numpy"),
        "pandas": is_package_local("pandas"),
        "scikit-learn": is_package_local("scikit_learn")
    }
    
    return packages_status

print(verify_package_location())
```

Slide 5: Minimal Build Process

Pure Python enables creating lightweight, production-ready builds. This implementation shows how to create a minimal package distribution without unnecessary dependencies, reducing deployment costs and complexity.

```python
from setuptools import setup, find_packages
import json

def create_minimal_build():
    # Read project dependencies
    with open('requirements.txt') as f:
        required = f.read().splitlines()
    
    # Define package metadata
    setup(
        name="ml_project",
        version="0.1.0",
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        install_requires=required,
        python_requires=">=3.8",
    )
    
    # Calculate package size
    package_info = {
        "dependencies": len(required),
        "packages": len(find_packages(where="src"))
    }
    
    return package_info

print(create_minimal_build())
```

Slide 6: Data Science Project Structure

Pure Python allows for a cleaner, more organized data science project structure. This implementation demonstrates how to set up a machine learning project with proper separation of concerns and minimal dependencies.

```python
from pathlib import Path
import json

def create_ds_project():
    # Create project structure
    structure = {
        "data": ["raw", "processed", "interim"],
        "models": ["trained", "evaluations"],
        "notebooks": [],
        "src": ["data", "features", "models", "visualization"]
    }
    
    for directory, subdirs in structure.items():
        base_dir = Path(directory)
        base_dir.mkdir(exist_ok=True)
        
        for subdir in subdirs:
            (base_dir / subdir).mkdir(exist_ok=True)
            
        if directory == "src":
            (base_dir / "__init__.py").touch()
            for subdir in subdirs:
                (base_dir / subdir / "__init__.py").touch()
    
    return structure

print(json.dumps(create_ds_project(), indent=2))
```

Slide 7: Custom Environment Configuration

Managing environment configurations in Pure Python provides greater flexibility and control compared to Anaconda's approach. This implementation shows how to handle different environments efficiently.

```python
import yaml
import os
from typing import Dict, Any

class EnvironmentConfig:
    def __init__(self, env_name: str):
        self.env_name = env_name
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        config_path = f"config/{self.env_name}.yaml"
        if not os.path.exists(config_path):
            return self._create_default_config()
            
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_default_config(self) -> Dict[str, Any]:
        config = {
            "data_path": "data/",
            "model_path": "models/",
            "log_level": "INFO",
            "max_workers": 4
        }
        os.makedirs("config", exist_ok=True)
        with open(f"config/{self.env_name}.yaml", 'w') as f:
            yaml.dump(config, f)
        return config

# Usage example
dev_config = EnvironmentConfig("development")
print(dev_config.config)
```

Slide 8: Efficient Package Management

Pure Python's pip allows for precise control over package versions and dependencies. This script demonstrates efficient package management and version control.

```python
import subprocess
import json
from typing import Dict, List

class PackageManager:
    @staticmethod
    def get_installed_packages() -> Dict[str, str]:
        result = subprocess.run(
            ["pip", "list", "--format=json"],
            capture_output=True,
            text=True
        )
        return {
            pkg["name"]: pkg["version"] 
            for pkg in json.loads(result.stdout)
        }
    
    @staticmethod
    def check_dependencies(requirements_file: str) -> List[str]:
        with open(requirements_file, 'r') as f:
            required = f.read().splitlines()
        
        installed = PackageManager.get_installed_packages()
        missing = []
        
        for req in required:
            package = req.split('==')[0]
            if package not in installed:
                missing.append(package)
        
        return missing

# Usage example
pkg_manager = PackageManager()
print("Installed packages:", pkg_manager.get_installed_packages())
print("Missing dependencies:", 
      pkg_manager.check_dependencies("requirements.txt"))
```

Slide 9: Advanced Model Development Setup

Demonstrating how Pure Python enables clean machine learning model development with minimal dependencies while maintaining full control over the development environment.

```python
from pathlib import Path
from typing import Optional, Dict, Any
import pickle
import json
import time

class MLProject:
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.project_path = Path(f"projects/{project_name}")
        self._setup_project()
        
    def _setup_project(self):
        # Create project directories
        dirs = ["models", "data", "logs", "configs"]
        for dir_name in dirs:
            (self.project_path / dir_name).mkdir(parents=True, 
                                               exist_ok=True)
    
    def save_model(self, model: Any, 
                  model_name: str, 
                  metadata: Optional[Dict] = None):
        model_path = self.project_path / "models" / f"{model_name}.pkl"
        meta_path = self.project_path / "models" / f"{model_name}_meta.json"
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        if metadata is None:
            metadata = {}
        metadata.update({
            "saved_at": time.time(),
            "model_name": model_name
        })
        
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return model_path, meta_path

# Usage example
project = MLProject("classification_project")
dummy_model = {"type": "random_forest"}
model_path, meta_path = project.save_model(
    dummy_model, 
    "rf_classifier",
    {"accuracy": 0.95}
)
print(f"Model saved at: {model_path}")
print(f"Metadata saved at: {meta_path}")
```

Slide 10: Production Deployment Setup

Pure Python's lightweight nature makes it ideal for production deployments. This example shows how to prepare a model for production while maintaining minimal dependencies.

```python
from typing import Dict, Any
import json
import hashlib
import datetime

class ProductionDeployment:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.deployment_info = self._init_deployment_info()
    
    def _init_deployment_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "deployment_id": self._generate_deployment_id(),
            "deployment_date": datetime.datetime.now().isoformat(),
            "dependencies": self._get_dependencies(),
            "status": "initialized"
        }
    
    def _generate_deployment_id(self) -> str:
        timestamp = datetime.datetime.now().isoformat()
        return hashlib.md5(
            f"{self.model_name}_{timestamp}".encode()
        ).hexdigest()[:12]
    
    def _get_dependencies(self) -> Dict[str, str]:
        with open("requirements.txt", 'r') as f:
            deps = {}
            for line in f:
                if "==" in line:
                    name, version = line.strip().split("==")
                    deps[name] = version
        return deps
    
    def prepare_deployment(self) -> Dict[str, Any]:
        self.deployment_info["status"] = "ready"
        self._save_deployment_config()
        return self.deployment_info
    
    def _save_deployment_config(self):
        config_path = f"deployments/{self.deployment_info['deployment_id']}.json"
        with open(config_path, 'w') as f:
            json.dump(self.deployment_info, f, indent=2)

# Usage example
deployment = ProductionDeployment("sentiment_analyzer")
deployment_info = deployment.prepare_deployment()
print(json.dumps(deployment_info, indent=2))
```

Slide 11: Performance Monitoring Setup

A crucial advantage of Pure Python is the ability to implement lightweight yet powerful monitoring systems. This implementation shows how to track model performance and resource usage efficiently.

```python
import time
import psutil
import json
from datetime import datetime
from typing import Dict, List

class PerformanceMonitor:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.metrics: List[Dict] = []
        
    def capture_metrics(self, prediction_count: int) -> Dict:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_info = psutil.Process().memory_info()
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "model_name": self.model_name,
            "cpu_percent": cpu_percent,
            "memory_mb": memory_info.rss / (1024 * 1024),
            "prediction_count": prediction_count,
        }
        
        self.metrics.append(metrics)
        return metrics
    
    def save_metrics(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump({
                "model_name": self.model_name,
                "metrics": self.metrics
            }, f, indent=2)

# Usage example
monitor = PerformanceMonitor("text_classifier")
for i in range(3):
    metrics = monitor.capture_metrics(100 * (i + 1))
    print(f"Captured metrics: {metrics}")
    time.sleep(1)

monitor.save_metrics("performance_log.json")
```

Slide 12: Automated Testing Framework

Pure Python enables creation of comprehensive testing frameworks without unnecessary dependencies. This implementation demonstrates how to set up automated testing for machine learning models.

```python
import unittest
from typing import Any, Dict, List
import numpy as np
from pathlib import Path

class MLModelTest(unittest.TestCase):
    def setUp(self):
        self.test_data_path = Path("tests/test_data")
        self.test_data_path.mkdir(parents=True, exist_ok=True)
        
    def generate_test_data(self, 
                          n_samples: int = 1000) -> Dict[str, np.ndarray]:
        np.random.seed(42)
        X = np.random.randn(n_samples, 10)
        y = np.random.randint(0, 2, n_samples)
        return {"X": X, "y": y}
    
    def test_model_predictions(self):
        class DummyModel:
            def predict(self, X):
                return np.ones(len(X))
        
        model = DummyModel()
        test_data = self.generate_test_data()
        
        predictions = model.predict(test_data["X"])
        
        self.assertEqual(len(predictions), len(test_data["X"]))
        self.assertTrue(np.all(predictions >= 0))
        self.assertTrue(np.all(predictions <= 1))
    
    def test_model_performance(self):
        def calculate_metrics(y_true, y_pred) -> Dict[str, float]:
            accuracy = np.mean(y_true == y_pred)
            return {"accuracy": accuracy}
        
        test_data = self.generate_test_data()
        dummy_predictions = np.ones(len(test_data["y"]))
        
        metrics = calculate_metrics(test_data["y"], dummy_predictions)
        
        self.assertGreater(metrics["accuracy"], 0)
        self.assertLess(metrics["accuracy"], 1)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
```

Slide 13: Experimental Results Tracking

Pure Python allows for efficient tracking of machine learning experiments without the overhead of additional frameworks. This implementation provides a clean way to log and compare experimental results.

```python
from datetime import datetime
import json
from typing import Dict, List, Optional
import hashlib

class ExperimentTracker:
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.experiments: List[Dict] = []
        
    def log_experiment(self,
                      model_params: Dict,
                      metrics: Dict,
                      dataset_info: Optional[Dict] = None) -> str:
        experiment_id = self._generate_experiment_id()
        
        experiment = {
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "model_parameters": model_params,
            "metrics": metrics,
            "dataset_info": dataset_info or {},
            "project_name": self.project_name
        }
        
        self.experiments.append(experiment)
        self._save_experiment(experiment)
        
        return experiment_id
    
    def _generate_experiment_id(self) -> str:
        timestamp = datetime.now().isoformat()
        unique_string = f"{self.project_name}_{timestamp}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:8]
    
    def _save_experiment(self, experiment: Dict):
        filename = f"experiments/{experiment['experiment_id']}.json"
        with open(filename, 'w') as f:
            json.dump(experiment, f, indent=2)
    
    def get_best_experiment(self, 
                          metric_name: str,
                          higher_is_better: bool = True) -> Dict:
        sorted_experiments = sorted(
            self.experiments,
            key=lambda x: x["metrics"][metric_name],
            reverse=higher_is_better
        )
        return sorted_experiments[0]

# Usage example
tracker = ExperimentTracker("text_classification")
experiment_id = tracker.log_experiment(
    model_params={"learning_rate": 0.01, "max_depth": 5},
    metrics={"accuracy": 0.92, "f1_score": 0.90},
    dataset_info={"size": 10000, "features": 100}
)
print(f"Logged experiment: {experiment_id}")
best_exp = tracker.get_best_experiment("accuracy")
print(f"Best experiment: {best_exp}")
```

Slide 14: Additional Resources

*   "Reproducible Machine Learning with Pure Python"
    *   Search on Google Scholar for: "Python Environment Management in Production ML Systems"
*   "Efficient Model Deployment Strategies"
    *   [https://arxiv.org/abs/2006.10165](https://arxiv.org/abs/2006.10165)
*   "Best Practices for ML Production Systems"
    *   [https://arxiv.org/abs/1909.00177](https://arxiv.org/abs/1909.00177)
*   "Scalable Machine Learning Pipeline Design"
    *   Search for: "MLOps Best Practices with Python" on Google Scholar
*   "Minimalistic Approaches to Large Scale ML Systems"
    *   Search for: "Lightweight ML Systems Design" on Google Scholar

