## Automating Software Deployment with CICD Pipelines
Slide 1: Setting Up a Basic CI/CD Pipeline with GitHub Actions

GitHub Actions provides a robust framework for automating software workflows. This implementation demonstrates creating a basic CI/CD pipeline that automatically runs tests and deploys a Python application when changes are pushed to the main branch.

```python
# .github/workflows/main.yml
name: Python CI/CD Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run tests
      run: |
        python -m pytest tests/
```

Slide 2: Automated Testing Configuration

Pytest configuration for the CI/CD pipeline requires specific settings to ensure consistent test execution across environments. This implementation shows how to set up pytest with custom markers and test discovery patterns.

```python
# pytest.ini
[pytest]
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Tests that take longer than 1 second
testpaths = tests
addopts = -v --strict-markers
```

Slide 3: Unit Testing in CI/CD

A comprehensive unit testing strategy is crucial for maintaining code quality in CI/CD pipelines. This example demonstrates writing testable code with dependency injection and implementing corresponding unit tests.

```python
# app/calculator.py
class Calculator:
    def add(self, a: float, b: float) -> float:
        return a + b

    def divide(self, a: float, b: float) -> float:
        if b == 0:
            raise ValueError("Division by zero")
        return a / b

# tests/test_calculator.py
import pytest
from app.calculator import Calculator

def test_calculator_add():
    calc = Calculator()
    assert calc.add(2, 3) == 5
    assert calc.add(-1, 1) == 0
    assert calc.add(0.1, 0.2) == pytest.approx(0.3)

def test_calculator_divide_by_zero():
    calc = Calculator()
    with pytest.raises(ValueError, match="Division by zero"):
        calc.divide(1, 0)
```

Slide 4: Integration Testing Setup

Integration testing verifies the interaction between different components of the application. This implementation showcases how to set up integration tests using pytest fixtures and environment variables.

```python
# tests/conftest.py
import pytest
import os
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

@pytest.fixture(scope="session")
def db_engine():
    database_url = os.getenv("TEST_DATABASE_URL")
    engine = create_engine(database_url)
    yield engine
    engine.dispose()

@pytest.fixture
def db_session(db_engine) -> Generator[Session, None, None]:
    connection = db_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()
```

Slide 5: Continuous Deployment Configuration

The deployment phase requires careful configuration of environment variables and secrets. This implementation demonstrates setting up secure deployment configurations using environment variables and configuration files.

```python
# config/deploy.py
import os
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class DeploymentConfig:
    app_name: str
    environment: str
    db_url: str
    secret_key: str
    debug_mode: bool

    @classmethod
    def from_env(cls) -> 'DeploymentConfig':
        return cls(
            app_name=os.getenv('APP_NAME', 'myapp'),
            environment=os.getenv('ENVIRONMENT', 'production'),
            db_url=os.getenv('DATABASE_URL'),
            secret_key=os.getenv('SECRET_KEY'),
            debug_mode=os.getenv('DEBUG', 'False').lower() == 'true'
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'APP_NAME': self.app_name,
            'ENVIRONMENT': self.environment,
            'DATABASE_URL': self.db_url,
            'SECRET_KEY': self.secret_key,
            'DEBUG': str(self.debug_mode)
        }
```

Slide 6: Docker Integration for CI/CD

Docker containerization ensures consistent environments across development, testing, and production stages. This implementation demonstrates creating a Dockerfile and docker-compose configuration for a Python application in a CI/CD context.

```python
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create non-root user
RUN useradd -m appuser
USER appuser

CMD ["python", "main.py"]

# docker-compose.yml
version: '3.8'
services:
  web:
    build: .
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - SECRET_KEY=${SECRET_KEY}
    ports:
      - "8000:8000"
    depends_on:
      - db
  db:
    image: postgres:13
    environment:
      - POSTGRES_PASSWORD=${DB_PASSWORD}
      - POSTGRES_DB=appdb
```

Slide 7: Automated Database Migrations

Database migrations must be automated and version-controlled within the CI/CD pipeline. This implementation shows how to manage database schemas using SQLAlchemy and Alembic with automated migration detection.

```python
# migrations/env.py
from alembic import context
from sqlalchemy import engine_from_config
from logging.config import fileConfig
import os

config = context.config
fileConfig(config.config_file_name)

def run_migrations():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix='sqlalchemy.',
        poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True
        )
        with context.begin_transaction():
            context.run_migrations()

# alembic/versions/001_initial.py
def upgrade():
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('username', sa.String(50), unique=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now())
    )

def downgrade():
    op.drop_table('users')
```

Slide 8: Continuous Monitoring Setup

Implementing continuous monitoring is essential for maintaining system health. This code demonstrates setting up a monitoring system using Python's logging framework and custom metrics collection.

```python
# monitoring/metrics.py
import time
import logging
from functools import wraps
from prometheus_client import Counter, Histogram, start_http_server

# Initialize metrics
REQUEST_COUNT = Counter('app_requests_total', 'Total app requests')
REQUEST_LATENCY = Histogram('app_request_latency_seconds', 'Request latency')

def setup_monitoring(port=8000):
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Start metrics endpoint
    start_http_server(port)

def monitor(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        REQUEST_COUNT.inc()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            REQUEST_LATENCY.observe(time.time() - start_time)
    
    return wrapper

# Usage example
@monitor
def process_request():
    time.sleep(0.1)  # Simulate work
    return "processed"
```

Slide 9: Security Testing Integration

Security testing must be integrated into the CI/CD pipeline to identify vulnerabilities early. This implementation shows how to set up automated security scanning using Python's security testing tools.

```python
# security/scan.py
import subprocess
import json
from typing import List, Dict

class SecurityScanner:
    def __init__(self, project_path: str):
        self.project_path = project_path
        
    def run_bandit_scan(self) -> Dict:
        """Run Bandit security scanner on Python code"""
        cmd = [
            'bandit',
            '-r',
            self.project_path,
            '-f', 'json'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return json.loads(result.stdout)
    
    def check_dependencies(self) -> List[Dict]:
        """Check dependencies for known vulnerabilities"""
        cmd = [
            'safety',
            'check',
            '--json'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return json.loads(result.stdout)
    
    def generate_report(self) -> Dict:
        return {
            'bandit_results': self.run_bandit_scan(),
            'dependency_check': self.check_dependencies()
        }

# Usage in CI pipeline
if __name__ == '__main__':
    scanner = SecurityScanner('./src')
    report = scanner.generate_report()
    
    # Fail if high severity issues found
    if any(issue['severity'] == 'HIGH' for issue in report['bandit_results']):
        exit(1)
```

Slide 10: Load Testing Framework

Load testing ensures application performance under stress. This implementation demonstrates creating a load testing framework using Python's asyncio and aiohttp libraries.

```python
# load_testing/framework.py
import asyncio
import aiohttp
import time
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class LoadTestResult:
    requests_per_second: float
    average_response_time: float
    error_rate: float
    status_codes: dict

class LoadTester:
    def __init__(self, target_url: str, num_requests: int, concurrency: int):
        self.target_url = target_url
        self.num_requests = num_requests
        self.concurrency = concurrency
        self.results: List[float] = []
        self.errors = 0
        self.status_codes = {}

    async def make_request(self, session: aiohttp.ClientSession) -> Optional[float]:
        start_time = time.time()
        try:
            async with session.get(self.target_url) as response:
                self.status_codes[response.status] = \
                    self.status_codes.get(response.status, 0) + 1
                return time.time() - start_time
        except Exception:
            self.errors += 1
            return None

    async def run(self) -> LoadTestResult:
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.make_request(session)
                for _ in range(self.num_requests)
            ]
            
            results = await asyncio.gather(*tasks)
            valid_results = [r for r in results if r is not None]
            
            return LoadTestResult(
                requests_per_second=len(valid_results) / sum(valid_results),
                average_response_time=sum(valid_results) / len(valid_results),
                error_rate=self.errors / self.num_requests,
                status_codes=self.status_codes
            )

# Usage example
async def main():
    tester = LoadTester(
        target_url="http://localhost:8000",
        num_requests=1000,
        concurrency=10
    )
    results = await tester.run()
    print(f"Results: {results}")

if __name__ == "__main__":
    asyncio.run(main())
```

Slide 11: Automated Error Reporting

Error reporting and tracking are crucial components of a robust CI/CD pipeline. This implementation demonstrates setting up automated error tracking with detailed context capture and notification systems.

```python
# error_tracking/tracker.py
import sys
import traceback
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass
import json

@dataclass
class ErrorReport:
    timestamp: str
    error_type: str
    message: str
    stacktrace: str
    context: Dict[str, Any]

class ErrorTracker:
    def __init__(self, app_name: str, environment: str):
        self.app_name = app_name
        self.environment = environment
        self.logger = logging.getLogger(__name__)
        
    def capture_context(self) -> Dict[str, Any]:
        return {
            'app_name': self.app_name,
            'environment': self.environment,
            'python_version': sys.version,
            'timestamp': datetime.utcnow().isoformat()
        }

    def track_error(self, error: Exception, additional_context: Optional[Dict] = None) -> ErrorReport:
        context = self.capture_context()
        if additional_context:
            context.update(additional_context)

        report = ErrorReport(
            timestamp=datetime.utcnow().isoformat(),
            error_type=error.__class__.__name__,
            message=str(error),
            stacktrace=traceback.format_exc(),
            context=context
        )
        
        self._store_error(report)
        self._notify_error(report)
        return report
    
    def _store_error(self, report: ErrorReport) -> None:
        with open('error_logs.jsonl', 'a') as f:
            f.write(json.dumps(vars(report)) + '\n')
    
    def _notify_error(self, report: ErrorReport) -> None:
        # Example notification via logging
        self.logger.error(
            f"Error in {self.app_name}: {report.error_type} - {report.message}"
        )

# Usage example
tracker = ErrorTracker("MyApp", "production")

try:
    raise ValueError("Invalid configuration")
except Exception as e:
    report = tracker.track_error(e, {'user_id': 123})
```

Slide 12: Metrics Collection and Analysis

Collecting and analyzing metrics is essential for monitoring application performance. This implementation shows how to create a metrics collection system with statistical analysis capabilities.

```python
# metrics/collector.py
import time
import statistics
from collections import deque
from typing import Dict, List, Optional
import numpy as np

class MetricsCollector:
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics: Dict[str, deque] = {}
        self.timestamps: Dict[str, deque] = {}
    
    def record(self, metric_name: str, value: float) -> None:
        if metric_name not in self.metrics:
            self.metrics[metric_name] = deque(maxlen=self.window_size)
            self.timestamps[metric_name] = deque(maxlen=self.window_size)
        
        self.metrics[metric_name].append(value)
        self.timestamps[metric_name].append(time.time())
    
    def get_statistics(self, metric_name: str) -> Dict[str, float]:
        if not self.metrics.get(metric_name):
            return {}
        
        values = list(self.metrics[metric_name])
        return {
            'count': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
            'percentile_95': np.percentile(values, 95),
            'percentile_99': np.percentile(values, 99),
            'min': min(values),
            'max': max(values)
        }
    
    def get_rate(self, metric_name: str, window_seconds: Optional[float] = None) -> float:
        if not self.metrics.get(metric_name):
            return 0.0
            
        timestamps = list(self.timestamps[metric_name])
        if not timestamps:
            return 0.0
            
        if window_seconds is None:
            window_seconds = timestamps[-1] - timestamps[0]
            
        recent_count = sum(1 for t in timestamps 
                          if t > timestamps[-1] - window_seconds)
        
        return recent_count / window_seconds

# Usage example
collector = MetricsCollector()

# Simulate metric collection
for _ in range(100):
    collector.record('response_time', np.random.normal(0.1, 0.02))
    collector.record('requests', 1)

print(collector.get_statistics('response_time'))
print(f"Request rate: {collector.get_rate('requests', 60)} requests/second")
```

Slide 13: Automated Documentation Generation

Documentation must stay synchronized with code changes in a CI/CD pipeline. This implementation demonstrates automated documentation generation with code analysis and markup generation.

```python
# docs/generator.py
import inspect
import ast
from typing import Dict, List, Optional
from pathlib import Path
import markdown

class DocGenerator:
    def __init__(self, source_dir: Path, output_dir: Path):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
    def analyze_module(self, module_path: Path) -> Dict:
        with open(module_path) as f:
            node = ast.parse(f.read())
            
        classes = {}
        functions = {}
        
        for item in node.body:
            if isinstance(item, ast.ClassDef):
                classes[item.name] = self._analyze_class(item)
            elif isinstance(item, ast.FunctionDef):
                functions[item.name] = self._analyze_function(item)
                
        return {
            'classes': classes,
            'functions': functions
        }
    
    def _analyze_class(self, node: ast.ClassDef) -> Dict:
        methods = {}
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods[item.name] = self._analyze_function(item)
                
        return {
            'docstring': ast.get_docstring(node),
            'methods': methods
        }
    
    def _analyze_function(self, node: ast.FunctionDef) -> Dict:
        return {
            'docstring': ast.get_docstring(node),
            'arguments': [arg.arg for arg in node.args.args],
            'returns': self._get_return_annotation(node)
        }
    
    def _get_return_annotation(self, node: ast.FunctionDef) -> Optional[str]:
        if node.returns:
            return ast.unparse(node.returns)
        return None
    
    def generate_markdown(self, analysis: Dict) -> str:
        content = ["# API Documentation\n"]
        
        if analysis['classes']:
            content.append("## Classes\n")
            for class_name, class_info in analysis['classes'].items():
                content.append(f"### {class_name}\n")
                if class_info['docstring']:
                    content.append(f"{class_info['docstring']}\n")
                
                for method_name, method_info in class_info['methods'].items():
                    content.append(f"#### {method_name}\n")
                    if method_info['docstring']:
                        content.append(f"{method_info['docstring']}\n")
                        
        if analysis['functions']:
            content.append("## Functions\n")
            for func_name, func_info in analysis['functions'].items():
                content.append(f"### {func_name}\n")
                if func_info['docstring']:
                    content.append(f"{func_info['docstring']}\n")
                
        return '\n'.join(content)
    
    def generate_docs(self) -> None:
        for python_file in self.source_dir.glob('**/*.py'):
            analysis = self.analyze_module(python_file)
            markdown_content = self.generate_markdown(analysis)
            
            output_file = self.output_dir / f"{python_file.stem}.md"
            output_file.write_text(markdown_content)

# Usage example
doc_gen = DocGenerator(
    source_dir=Path('./src'),
    output_dir=Path('./docs')
)
doc_gen.generate_docs()
```

Slide 14: Additional Resources

1.  [https://arxiv.org/abs/2103.00586](https://arxiv.org/abs/2103.00586) - "Continuous Integration and Delivery for Machine Learning Applications"
2.  [https://arxiv.org/abs/2003.05991](https://arxiv.org/abs/2003.05991) - "DevOps for ML: An End-to-End Approach"
3.  [https://arxiv.org/abs/2106.08937](https://arxiv.org/abs/2106.08937) - "MLOps: Opportunities, Challenges, and Future Directions"
4.  [https://arxiv.org/abs/2205.07147](https://arxiv.org/abs/2205.07147) - "Automated Testing in CI/CD Pipelines: A Systematic Review"

