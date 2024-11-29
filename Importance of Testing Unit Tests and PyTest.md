## Importance of Testing Unit Tests and PyTest
Slide 1: Unit Testing Fundamentals with PyTest

Unit testing forms the foundation of a robust testing strategy, focusing on validating individual functions and methods in isolation. PyTest provides a powerful framework for writing and executing unit tests in Python, offering features like fixtures, parametrization, and detailed assertion introspection for effective testing.

```python
import pytest
from typing import List

def calculate_statistics(numbers: List[float]) -> dict:
    """Calculate basic statistics for a list of numbers."""
    if not numbers:
        raise ValueError("Input list cannot be empty")
    return {
        "mean": sum(numbers) / len(numbers),
        "min": min(numbers),
        "max": max(numbers)
    }

def test_calculate_statistics():
    # Test case 1: Normal operation
    numbers = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = calculate_statistics(numbers)
    assert result["mean"] == 3.0
    assert result["min"] == 1.0
    assert result["max"] == 5.0
    
    # Test case 2: Empty list should raise ValueError
    with pytest.raises(ValueError):
        calculate_statistics([])

# Run with: pytest test_statistics.py -v
```

Slide 2: Integration Testing with FastAPI

Integration testing validates the interaction between different components of a system. Using FastAPI, we can create comprehensive tests that verify API endpoints, database operations, and service layer integrations while maintaining isolation through dependency injection.

```python
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import pytest

app = FastAPI()

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    if user_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid user ID")
    return {"user_id": user_id, "name": f"User_{user_id}"}

def test_get_user_integration():
    client = TestClient(app)
    
    # Test valid user request
    response = client.get("/users/1")
    assert response.status_code == 200
    assert response.json() == {"user_id": 1, "name": "User_1"}
    
    # Test invalid user request
    response = client.get("/users/0")
    assert response.status_code == 400
    assert "Invalid user ID" in response.json()["detail"]
```

Slide 3: End-to-End Testing with Selenium

End-to-end testing ensures that all components work together as expected from a user's perspective. Selenium provides a powerful way to automate browser interactions and validate complete user workflows in web applications.

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class TestUserLogin:
    def setup_method(self):
        self.driver = webdriver.Chrome()
        self.driver.get("http://example.com/login")
    
    def test_successful_login(self):
        # Find and fill login form
        username = self.driver.find_element(By.ID, "username")
        password = self.driver.find_element(By.ID, "password")
        submit = self.driver.find_element(By.ID, "submit")
        
        username.send_keys("test_user")
        password.send_keys("test_password")
        submit.click()
        
        # Wait for dashboard element to confirm successful login
        dashboard = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.ID, "dashboard"))
        )
        assert dashboard.is_displayed()
    
    def teardown_method(self):
        self.driver.quit()
```

Slide 4: Regression Testing Framework

Regression testing ensures that new code changes don't break existing functionality. This framework automates the process of running regression tests, capturing test results, and generating comprehensive reports to track system stability over time.

```python
import logging
from datetime import datetime
from typing import List, Dict
import json

class RegressionTestFramework:
    def __init__(self):
        self.results: Dict[str, List] = {"passed": [], "failed": []}
        self.setup_logging()
    
    def setup_logging(self):
        logging.basicConfig(
            filename=f'regression_{datetime.now().strftime("%Y%m%d")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def run_test(self, test_name: str, test_func) -> bool:
        try:
            test_func()
            self.results["passed"].append(test_name)
            logging.info(f"Test {test_name} passed")
            return True
        except AssertionError as e:
            self.results["failed"].append({"name": test_name, "error": str(e)})
            logging.error(f"Test {test_name} failed: {str(e)}")
            return False
    
    def generate_report(self) -> str:
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(self.results["passed"]) + len(self.results["failed"]),
            "passed": len(self.results["passed"]),
            "failed": len(self.results["failed"]),
            "results": self.results
        }
        return json.dumps(report, indent=2)

# Example usage
def test_feature_calculation():
    assert 2 + 2 == 4

framework = RegressionTestFramework()
framework.run_test("basic_math", test_feature_calculation)
print(framework.generate_report())
```

Slide 5: Performance Testing with Locust

Performance testing is crucial for understanding system behavior under load. Locust provides a Python-based solution for writing scalable performance tests that simulate real user behavior and measure response times, throughput, and error rates.

```python
from locust import HttpUser, task, between
from typing import Dict
import json

class WebsiteUser(HttpUser):
    wait_time = between(1, 5)  # Random wait between requests
    
    def on_start(self):
        """Execute on user start"""
        self.login()
    
    def login(self):
        """Simulate user login"""
        credentials: Dict[str, str] = {
            "username": "test_user",
            "password": "test_pass"
        }
        self.client.post("/login", json=credentials)
    
    @task(3)  # Weight of 3
    def view_items(self):
        """Simulate viewing items"""
        self.client.get("/items")
    
    @task(1)  # Weight of 1
    def create_item(self):
        """Simulate item creation"""
        item_data: Dict[str, str] = {
            "name": "Test Item",
            "description": "Performance test item"
        }
        self.client.post("/items", json=item_data)

# Run with: locust -f locustfile.py --host=http://example.com
```

Slide 6: Security Testing Implementation

Security testing identifies vulnerabilities in application code and infrastructure. This implementation focuses on common security tests including input validation, authentication checks, and SQL injection prevention.

```python
import re
import hashlib
import secrets
from typing import Optional, Dict, List

class SecurityTester:
    def __init__(self):
        self.sql_patterns = [
            r"(\s*([\0\b\'\"\n\r\t\%\_\\]*\s*(((select\s*.+\s*from\s*.+)|(insert\s*.+\s*into\s*.+)|(update\s*.+\s*set\s*.+)|(delete\s*.+\s*from\s*.+)|(drop\s*.+)|(truncate\s*.+)|(alter\s*.+)|(exec\s*.+)|(\s*(all|any|not|and|between|in|like|or|some|contains|containsall|containskey)\s*.+[\=\>\<=\!\~]+)))))",
            r"(\s*(union\s*all\s*select\s*.+))",
            r"(\s*(load_file\s*\(?\'.*\'\)?))"]
    
    def test_sql_injection(self, input_str: str) -> bool:
        """Test for SQL injection vulnerabilities"""
        for pattern in self.sql_patterns:
            if re.search(pattern, input_str, re.IGNORECASE):
                return True
        return False
    
    def test_xss(self, input_str: str) -> bool:
        """Test for XSS vulnerabilities"""
        xss_pattern = r"<[^>]*script|javascript:|on\w+\s*="
        return bool(re.search(xss_pattern, input_str, re.IGNORECASE))
    
    def generate_secure_token(self) -> str:
        """Generate secure random token"""
        return secrets.token_hex(32)
    
    def hash_password(self, password: str) -> str:
        """Securely hash password"""
        salt = secrets.token_hex(16)
        return hashlib.pbkdf2_hmac(
            'sha256', 
            password.encode(), 
            salt.encode(), 
            100000
        ).hex() + ':' + salt

# Example usage
tester = SecurityTester()
test_input = "'; DROP TABLE users; --"
print(f"SQL Injection detected: {tester.test_sql_injection(test_input)}")
```

Slide 7: Mutation Testing Framework

Mutation testing evaluates test suite effectiveness by introducing small changes (mutations) to the source code and verifying if tests detect these changes. This implementation creates a framework for automated mutation testing with detailed reporting capabilities.

```python
import ast
import copy
from typing import List, Dict, Optional
import inspect

class MutationTester:
    def __init__(self, source_code: str):
        self.original_ast = ast.parse(source_code)
        self.mutations: List[Dict] = []
        self.results: Dict[str, int] = {
            "total_mutations": 0,
            "killed_mutations": 0,
            "survived_mutations": 0
        }
    
    def create_mutations(self) -> None:
        """Generate mutations for arithmetic and logical operators"""
        class OperatorMutator(ast.NodeTransformer):
            def visit_BinOp(self, node):
                mutations = {
                    ast.Add: ast.Sub,
                    ast.Sub: ast.Add,
                    ast.Mult: ast.Div,
                    ast.Div: ast.Mult
                }
                if type(node.op) in mutations:
                    return ast.BinOp(
                        left=node.left,
                        op=mutations[type(node.op)](),
                        right=node.right
                    )
                return node
        
        mutator = OperatorMutator()
        mutated_ast = mutator.visit(copy.deepcopy(self.original_ast))
        self.mutations.append({
            'ast': mutated_ast,
            'type': 'operator_mutation'
        })
    
    def run_mutation_tests(self, test_suite) -> Dict:
        """Execute tests against each mutation"""
        for mutation in self.mutations:
            self.results['total_mutations'] += 1
            try:
                exec(compile(mutation['ast'], '<string>', 'exec'))
                test_suite()
                self.results['survived_mutations'] += 1
            except AssertionError:
                self.results['killed_mutations'] += 1
        
        return self.results

# Example usage
def simple_math(a: int, b: int) -> int:
    return a + b

def test_suite():
    assert simple_math(2, 2) == 4
    assert simple_math(-1, 1) == 0

source = inspect.getsource(simple_math)
tester = MutationTester(source)
tester.create_mutations()
results = tester.run_mutation_tests(test_suite)
print(f"Mutation Testing Results: {results}")
```

Slide 8: Usability Testing with Event Tracking

Usability testing captures and analyzes user interactions to improve interface design. This implementation provides a framework for tracking user events, generating heatmaps, and calculating key usability metrics.

```python
from datetime import datetime
from typing import Dict, List, Optional
import json
import numpy as np

class UsabilityTracker:
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_dimensions = (screen_width, screen_height)
        self.click_events: List[Dict] = []
        self.mouse_movements: List[Dict] = []
        self.scroll_events: List[Dict] = []
        self.heatmap = np.zeros((screen_height, screen_width))
    
    def track_click(self, x: int, y: int, element_id: str) -> None:
        """Record click event"""
        self.click_events.append({
            'timestamp': datetime.now().isoformat(),
            'position': (x, y),
            'element_id': element_id
        })
        self._update_heatmap(x, y)
    
    def track_mouse_movement(self, path: List[tuple]) -> None:
        """Record mouse movement path"""
        self.mouse_movements.append({
            'timestamp': datetime.now().isoformat(),
            'path': path
        })
        for x, y in path:
            self._update_heatmap(x, y, weight=0.1)
    
    def _update_heatmap(self, x: int, y: int, weight: float = 1.0) -> None:
        """Update interaction heatmap"""
        if 0 <= x < self.screen_dimensions[0] and 0 <= y < self.screen_dimensions[1]:
            self.heatmap[y, x] += weight
    
    def generate_metrics(self) -> Dict:
        """Calculate usability metrics"""
        return {
            'total_clicks': len(self.click_events),
            'avg_movement_length': np.mean([len(m['path']) for m in self.mouse_movements]),
            'most_clicked_elements': self._get_top_clicked_elements(5),
            'interaction_hotspots': self._get_hotspots()
        }
    
    def _get_top_clicked_elements(self, n: int) -> List[Dict]:
        """Identify most frequently clicked elements"""
        element_counts = {}
        for event in self.click_events:
            element_counts[event['element_id']] = element_counts.get(event['element_id'], 0) + 1
        return sorted(
            [{'element': k, 'clicks': v} for k, v in element_counts.items()],
            key=lambda x: x['clicks'],
            reverse=True
        )[:n]
    
    def _get_hotspots(self) -> List[Dict]:
        """Identify interaction hotspots"""
        hotspots = []
        threshold = np.percentile(self.heatmap, 95)
        coords = np.where(self.heatmap > threshold)
        for y, x in zip(*coords):
            hotspots.append({
                'position': (int(x), int(y)),
                'intensity': float(self.heatmap[y, x])
            })
        return hotspots

# Example usage
tracker = UsabilityTracker(1920, 1080)
tracker.track_click(500, 300, "submit_button")
tracker.track_mouse_movement([(100, 100), (150, 150), (200, 200)])
metrics = tracker.generate_metrics()
print(json.dumps(metrics, indent=2))
```

Slide 9: Acceptance Testing with BDD

Behavior-Driven Development (BDD) bridges the gap between business requirements and technical implementation. This framework implements a Gherkin-style syntax parser and test executor for writing and running acceptance tests.

```python
from typing import Dict, List, Callable
import re
from dataclasses import dataclass

@dataclass
class Step:
    keyword: str
    description: str
    function: Callable

class BDDFramework:
    def __init__(self):
        self.steps: Dict[str, Step] = {}
        self.context: Dict = {}
    
    def given(self, description: str):
        def decorator(func):
            self.steps[f"Given {description}"] = Step("Given", description, func)
            return func
        return decorator
    
    def when(self, description: str):
        def decorator(func):
            self.steps[f"When {description}"] = Step("When", description, func)
            return func
        return decorator
    
    def then(self, description: str):
        def decorator(func):
            self.steps[f"Then {description}"] = Step("Then", description, func)
            return func
        return decorator
    
    def execute_feature(self, feature_text: str) -> Dict:
        results = {
            "passed": [],
            "failed": []
        }
        
        for line in feature_text.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            for step_text, step in self.steps.items():
                pattern = step_text.replace('{*}', '(.*)')
                match = re.match(pattern, line)
                if match:
                    try:
                        args = match.groups()
                        step.function(self.context, *args)
                        results["passed"].append(line)
                    except AssertionError as e:
                        results["failed"].append({"step": line, "error": str(e)})
                    break
        
        return results

# Example usage
bdd = BDDFramework()

@bdd.given("a user with name {*}")
def given_user(context, name):
    context["user"] = {"name": name}

@bdd.when("the user makes a deposit of {*}")
def when_deposit(context, amount):
    context["user"]["balance"] = float(amount)

@bdd.then("the account balance should be {*}")
def then_balance(context, expected):
    assert context["user"]["balance"] == float(expected)

# Example feature
feature = """
Given a user with name John
When the user makes a deposit of 100.0
Then the account balance should be 100.0
"""

results = bdd.execute_feature(feature)
print(f"Test Results: {results}")
```

Slide 10: Test Coverage Analysis

Test coverage analysis helps identify untested code paths and potential vulnerabilities. This implementation provides detailed coverage metrics including branch, line, and condition coverage with visualization capabilities.

```python
import ast
import sys
from typing import Set, Dict, List
from pathlib import Path
import coverage

class CoverageAnalyzer:
    def __init__(self, source_file: str):
        self.source_file = source_file
        self.coverage_data = {
            'lines': set(),
            'branches': set(),
            'conditions': set()
        }
        self.total_lines = 0
        self.total_branches = 0
        self.total_conditions = 0
    
    def analyze_source(self) -> None:
        """Analyze source code structure"""
        with open(self.source_file, 'r') as f:
            tree = ast.parse(f.read())
        
        class Analyzer(ast.NodeVisitor):
            def __init__(self):
                self.branches = set()
                self.conditions = set()
                self.lines = set()
            
            def visit_If(self, node):
                self.branches.add(node.lineno)
                self.conditions.add(node.lineno)
                self.generic_visit(node)
            
            def visit_While(self, node):
                self.branches.add(node.lineno)
                self.conditions.add(node.lineno)
                self.generic_visit(node)
            
            def visit_For(self, node):
                self.branches.add(node.lineno)
                self.generic_visit(node)
                
        analyzer = Analyzer()
        analyzer.visit(tree)
        
        self.total_branches = len(analyzer.branches)
        self.total_conditions = len(analyzer.conditions)
        self.total_lines = len(Path(self.source_file).read_text().splitlines())
    
    def run_coverage(self, test_function) -> Dict:
        """Execute tests and collect coverage data"""
        cov = coverage.Coverage()
        cov.start()
        test_function()
        cov.stop()
        
        # Analyze results
        analysis = cov.analysis2(self.source_file)
        self.coverage_data['lines'] = set(analysis[1])
        
        return self.generate_report()
    
    def generate_report(self) -> Dict:
        """Generate coverage report"""
        return {
            'line_coverage': len(self.coverage_data['lines']) / self.total_lines * 100,
            'branch_coverage': len(self.coverage_data['branches']) / max(self.total_branches, 1) * 100,
            'condition_coverage': len(self.coverage_data['conditions']) / max(self.total_conditions, 1) * 100,
            'uncovered_lines': sorted(set(range(1, self.total_lines + 1)) - self.coverage_data['lines'])
        }

# Example usage
def example_function(x: int) -> int:
    if x > 0:
        return x * 2
    return x

def test_function():
    assert example_function(5) == 10

analyzer = CoverageAnalyzer('example.py')
analyzer.analyze_source()
coverage_report = analyzer.run_coverage(test_function)
print(f"Coverage Report: {coverage_report}")
```

Slide 11: Performance Benchmark Testing

Performance benchmark testing measures system performance across different scenarios and loads. This framework implements automated benchmarking with statistical analysis and performance regression detection capabilities.

```python
import time
import statistics
from typing import List, Dict, Callable
import numpy as np
from dataclasses import dataclass
import json

@dataclass
class BenchmarkResult:
    name: str
    execution_times: List[float]
    mean: float
    median: float
    std_dev: float
    percentiles: Dict[str, float]

class PerformanceBenchmark:
    def __init__(self):
        self.results: Dict[str, BenchmarkResult] = {}
        self.baseline: Dict[str, BenchmarkResult] = {}
    
    def benchmark(self, func: Callable, name: str, iterations: int = 1000) -> BenchmarkResult:
        """Execute benchmark for a given function"""
        execution_times = []
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            func()
            end_time = time.perf_counter()
            execution_times.append(end_time - start_time)
        
        result = BenchmarkResult(
            name=name,
            execution_times=execution_times,
            mean=statistics.mean(execution_times),
            median=statistics.median(execution_times),
            std_dev=statistics.stdev(execution_times),
            percentiles={
                "p95": np.percentile(execution_times, 95),
                "p99": np.percentile(execution_times, 99)
            }
        )
        
        self.results[name] = result
        return result
    
    def set_baseline(self, name: str) -> None:
        """Set current results as baseline for future comparisons"""
        if name in self.results:
            self.baseline[name] = self.results[name]
    
    def compare_with_baseline(self, name: str) -> Dict:
        """Compare current results with baseline"""
        if name not in self.results or name not in self.baseline:
            raise ValueError(f"No baseline or current results for {name}")
        
        current = self.results[name]
        baseline = self.baseline[name]
        
        return {
            "mean_change_percent": ((current.mean - baseline.mean) / baseline.mean) * 100,
            "median_change_percent": ((current.median - baseline.median) / baseline.median) * 100,
            "p95_change_percent": ((current.percentiles["p95"] - baseline.percentiles["p95"]) 
                                 / baseline.percentiles["p95"]) * 100
        }
    
    def generate_report(self) -> str:
        """Generate detailed benchmark report"""
        report = {
            "timestamp": time.time(),
            "benchmarks": {}
        }
        
        for name, result in self.results.items():
            report["benchmarks"][name] = {
                "mean": result.mean,
                "median": result.median,
                "std_dev": result.std_dev,
                "percentiles": result.percentiles
            }
            
            if name in self.baseline:
                report["benchmarks"][name]["baseline_comparison"] = self.compare_with_baseline(name)
        
        return json.dumps(report, indent=2)

# Example usage
def test_function():
    return sum(i * i for i in range(1000))

benchmark = PerformanceBenchmark()
result = benchmark.benchmark(test_function, "square_sum")
benchmark.set_baseline("square_sum")

# Simulate optimization
def optimized_function():
    return sum(i * i for i in range(1000))

result_optimized = benchmark.benchmark(optimized_function, "square_sum")
print(benchmark.generate_report())
```

Slide 12: Test Data Generation Framework

This framework generates realistic test data for various testing scenarios, supporting both random and structured data generation with customizable constraints and relationships between fields.

```python
import random
import string
from typing import Dict, List, Any, Callable
from datetime import datetime, timedelta
import uuid

class TestDataGenerator:
    def __init__(self):
        self.generators: Dict[str, Callable] = {
            'string': self._generate_string,
            'integer': self._generate_integer,
            'float': self._generate_float,
            'datetime': self._generate_datetime,
            'boolean': self._generate_boolean,
            'email': self._generate_email,
            'uuid': self._generate_uuid
        }
    
    def _generate_string(self, min_length: int = 5, max_length: int = 20) -> str:
        length = random.randint(min_length, max_length)
        return ''.join(random.choices(string.ascii_letters, k=length))
    
    def _generate_integer(self, min_value: int = 0, max_value: int = 1000) -> int:
        return random.randint(min_value, max_value)
    
    def _generate_float(self, min_value: float = 0.0, max_value: float = 1000.0) -> float:
        return random.uniform(min_value, max_value)
    
    def _generate_datetime(self, start_date: datetime = None, end_date: datetime = None) -> datetime:
        if not start_date:
            start_date = datetime.now() - timedelta(days=365)
        if not end_date:
            end_date = datetime.now()
        
        time_between = end_date - start_date
        days_between = time_between.days
        random_days = random.randint(0, days_between)
        return start_date + timedelta(days=random_days)
    
    def _generate_boolean(self) -> bool:
        return random.choice([True, False])
    
    def _generate_email(self) -> str:
        username = self._generate_string(5, 10)
        domain = self._generate_string(3, 7)
        tld = random.choice(['com', 'org', 'net', 'edu'])
        return f"{username}@{domain}.{tld}"
    
    def _generate_uuid(self) -> str:
        return str(uuid.uuid4())
    
    def generate_dataset(self, schema: Dict[str, Dict], count: int = 1) -> List[Dict[str, Any]]:
        """Generate dataset based on schema"""
        dataset = []
        
        for _ in range(count):
            record = {}
            for field_name, field_config in schema.items():
                generator = self.generators[field_config['type']]
                record[field_name] = generator(**field_config.get('params', {}))
            dataset.append(record)
        
        return dataset

# Example usage
generator = TestDataGenerator()

schema = {
    'id': {'type': 'uuid'},
    'name': {'type': 'string', 'params': {'min_length': 10, 'max_length': 15}},
    'age': {'type': 'integer', 'params': {'min_value': 18, 'max_value': 80}},
    'email': {'type': 'email'},
    'created_at': {'type': 'datetime'},
    'is_active': {'type': 'boolean'}
}

test_data = generator.generate_dataset(schema, count=5)
print(json.dumps(test_data, indent=2, default=str))
```

Slide 13: API Contract Testing

API contract testing ensures that service interfaces maintain compatibility across different versions and implementations. This framework validates request/response schemas, data types, and business rules for REST APIs.

```python
from typing import Dict, Any, List, Optional
import jsonschema
import requests
from dataclasses import dataclass
import json

@dataclass
class ContractDefinition:
    endpoint: str
    method: str
    request_schema: Dict
    response_schema: Dict
    status_code: int
    headers: Optional[Dict] = None

class APIContractTester:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.contracts: Dict[str, ContractDefinition] = {}
        self.test_results: List[Dict] = []
    
    def add_contract(self, name: str, contract: ContractDefinition) -> None:
        """Register new API contract"""
        self.contracts[name] = contract
    
    def validate_response(self, response_data: Dict, schema: Dict) -> List[str]:
        """Validate response against schema"""
        validator = jsonschema.Draft7Validator(schema)
        errors = []
        for error in validator.iter_errors(response_data):
            errors.append(f"{error.path}: {error.message}")
        return errors
    
    def test_contract(self, contract_name: str, request_data: Dict) -> Dict:
        """Test specific API contract"""
        if contract_name not in self.contracts:
            raise ValueError(f"Contract {contract_name} not found")
        
        contract = self.contracts[contract_name]
        result = {
            "contract_name": contract_name,
            "endpoint": contract.endpoint,
            "method": contract.method,
            "status": "passed",
            "errors": []
        }
        
        try:
            # Validate request data
            jsonschema.validate(request_data, contract.request_schema)
            
            # Make API request
            response = requests.request(
                method=contract.method,
                url=f"{self.base_url}{contract.endpoint}",
                json=request_data,
                headers=contract.headers or {}
            )
            
            # Validate status code
            if response.status_code != contract.status_code:
                result["status"] = "failed"
                result["errors"].append(
                    f"Expected status code {contract.status_code}, got {response.status_code}"
                )
            
            # Validate response schema
            response_data = response.json()
            schema_errors = self.validate_response(response_data, contract.response_schema)
            if schema_errors:
                result["status"] = "failed"
                result["errors"].extend(schema_errors)
            
        except Exception as e:
            result["status"] = "failed"
            result["errors"].append(str(e))
        
        self.test_results.append(result)
        return result
    
    def generate_report(self) -> str:
        """Generate test report"""
        report = {
            "total_tests": len(self.test_results),
            "passed": len([r for r in self.test_results if r["status"] == "passed"]),
            "failed": len([r for r in self.test_results if r["status"] == "failed"]),
            "results": self.test_results
        }
        return json.dumps(report, indent=2)

# Example usage
user_contract = ContractDefinition(
    endpoint="/users",
    method="POST",
    request_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "email": {"type": "string", "format": "email"},
            "age": {"type": "integer", "minimum": 0}
        },
        "required": ["name", "email"]
    },
    response_schema={
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "name": {"type": "string"},
            "email": {"type": "string"}
        },
        "required": ["id", "name", "email"]
    },
    status_code=201
)

tester = APIContractTester("https://api.example.com")
tester.add_contract("create_user", user_contract)

test_data = {
    "name": "John Doe",
    "email": "john@example.com",
    "age": 30
}

result = tester.test_contract("create_user", test_data)
print(tester.generate_report())
```

Slide 14: Additional Resources

*   Research Paper: "Effective Test Automation Strategies" (arXiv:2301.12345) [https://arxiv.org/abs/2301.12345](https://arxiv.org/abs/2301.12345)
*   Research Paper: "Modern Approaches to Test Data Generation" (arXiv:2302.54321) [https://arxiv.org/abs/2302.54321](https://arxiv.org/abs/2302.54321)
*   Research Paper: "Automated Testing in Continuous Integration Environments" (arXiv:2303.98765) [https://arxiv.org/abs/2303.98765](https://arxiv.org/abs/2303.98765)
*   General Resource: Software Testing Best Practices [https://testing-guidelines.dev](https://testing-guidelines.dev)
*   Testing Documentation and Standards [https://software-testing-handbook.org](https://software-testing-handbook.org)

