## Test-Driven Development in Python
Slide 1: Test-Driven Development Fundamentals

Test-Driven Development (TDD) is a software development methodology where tests are written before the actual code implementation. This approach ensures that code meets requirements from the start and maintains high test coverage throughout development.

```python
# Example of basic TDD workflow for a simple calculator function
import unittest

class TestCalculator(unittest.TestCase):
    def test_add_numbers(self):
        # Write test first
        result = add_numbers(2, 3)
        self.assertEqual(result, 5)

def add_numbers(a, b):
    # Implement the function to make test pass
    return a + b

if __name__ == '__main__':
    unittest.main()

# Output:
# ..
# ----------------------------------------------------------------------
# Ran 1 test in 0.001s
# OK
```

Slide 2: Red-Green-Refactor Cycle

The Red-Green-Refactor cycle is the core principle of TDD, where developers write a failing test first (Red), implement the minimum code to pass the test (Green), and then optimize the code while maintaining test coverage (Refactor).

```python
import unittest

class TestStringOperations(unittest.TestCase):
    def test_string_reverse(self):
        # Red: Write failing test
        self.assertEqual(reverse_string("hello"), "olleh")

def reverse_string(text):
    # Green: Implement minimum code to pass
    return text[::-1]

# Refactor: Optimize while maintaining test passing
def reverse_string_optimized(text):
    return ''.join(reversed(text))

if __name__ == '__main__':
    unittest.main()
```

Slide 3: Test Fixtures and Setup

Test fixtures provide a consistent test environment by setting up necessary preconditions for tests. This ensures reliable and reproducible test execution across different test cases within a test suite.

```python
import unittest

class TestDatabaseOperations(unittest.TestCase):
    def setUp(self):
        # Setup fixture - runs before each test
        self.test_data = {
            'users': [{'id': 1, 'name': 'Alice'},
                     {'id': 2, 'name': 'Bob'}]
        }
        self.db = MockDatabase(self.test_data)
    
    def tearDown(self):
        # Cleanup after each test
        self.db.close()
    
    def test_user_retrieval(self):
        user = self.db.get_user(1)
        self.assertEqual(user['name'], 'Alice')

class MockDatabase:
    def __init__(self, data):
        self.data = data
    
    def get_user(self, user_id):
        return next(user for user in self.data['users'] 
                   if user['id'] == user_id)
    
    def close(self):
        self.data = None

if __name__ == '__main__':
    unittest.main()
```

Slide 4: Mocking External Dependencies

External dependencies like databases or API calls need to be mocked during testing to ensure consistent behavior and faster test execution. Python's unittest.mock provides powerful tools for creating mock objects.

```python
from unittest.mock import Mock, patch
import unittest
import requests

class TestUserService:
    def get_user_data(self, user_id):
        response = requests.get(f"http://api.example.com/users/{user_id}")
        return response.json()

class TestUserServiceMock(unittest.TestCase):
    @patch('requests.get')
    def test_get_user_data(self, mock_get):
        # Configure mock
        mock_response = Mock()
        mock_response.json.return_value = {"id": 1, "name": "Alice"}
        mock_get.return_value = mock_response

        # Test with mock
        service = TestUserService()
        result = service.get_user_data(1)
        
        self.assertEqual(result["name"], "Alice")
        mock_get.assert_called_with("http://api.example.com/users/1")

if __name__ == '__main__':
    unittest.main()
```

Slide 5: Parameterized Testing

Parameterized testing allows running the same test with different input parameters, reducing code duplication and ensuring comprehensive test coverage across various scenarios and edge cases.

```python
import unittest
from parameterized import parameterized

class TestMathOperations(unittest.TestCase):
    @parameterized.expand([
        ("positive", 4, 2, 2),
        ("zero", 0, 5, 0),
        ("negative", -10, 2, -5),
        ("floating", 5.5, 2, 2.75)
    ])
    def test_division(self, name, input_a, input_b, expected):
        result = divide(input_a, input_b)
        self.assertEqual(result, expected)

def divide(a, b):
    return a / b

if __name__ == '__main__':
    unittest.main()

# Output:
# ....
# ----------------------------------------------------------------------
# Ran 4 tests in 0.002s
# OK
```

Slide 6: Test Coverage Analysis

Test coverage analysis helps identify untested code paths and ensures comprehensive testing. Python's coverage.py tool provides detailed reports about which parts of the code are executed during tests and highlights potential gaps.

```python
# Install: pip install coverage
# Run: coverage run -m unittest test_calculator.py
# Report: coverage report -m

import unittest

class Calculator:
    def add(self, a, b):
        return a + b
    
    def divide(self, a, b):
        if b == 0:
            raise ValueError("Division by zero")
        return a / b

class TestCalculator(unittest.TestCase):
    def setUp(self):
        self.calc = Calculator()
    
    def test_add(self):
        self.assertEqual(self.calc.add(3, 5), 8)
    
    def test_divide(self):
        self.assertEqual(self.calc.divide(10, 2), 5)
        with self.assertRaises(ValueError):
            self.calc.divide(5, 0)

# Output from coverage report:
# Name                 Stmts   Miss  Cover   Missing
# --------------------------------------------------
# calculator.py           10      0   100%
```

Slide 7: Property-Based Testing

Property-based testing generates random test cases based on specified properties that should hold true for any input. This approach can uncover edge cases that might be missed with traditional unit testing.

```python
from hypothesis import given, strategies as st
import unittest

class TestStringOperations(unittest.TestCase):
    @given(st.text())
    def test_reverse_string_property(self, text):
        # Property: reversing a string twice returns original
        self.assertEqual(
            reverse_string(reverse_string(text)),
            text
        )
    
    @given(st.text(), st.text())
    def test_concatenation_property(self, text1, text2):
        # Property: length of concatenation equals sum of lengths
        self.assertEqual(
            len(text1 + text2),
            len(text1) + len(text2)
        )

def reverse_string(text):
    return text[::-1]

if __name__ == '__main__':
    unittest.main()
```

Slide 8: Test-Driven API Development

Test-Driven Development applied to API design ensures robust endpoint implementation and clear documentation through comprehensive test cases that define expected behavior and responses.

```python
import unittest
from fastapi.testclient import TestClient
from fastapi import FastAPI, HTTPException

app = FastAPI()

class UserDatabase:
    def __init__(self):
        self.users = {}

    def add_user(self, user_id: int, name: str):
        self.users[user_id] = {"id": user_id, "name": name}
        
    def get_user(self, user_id: int):
        if user_id not in self.users:
            raise HTTPException(status_code=404, detail="User not found")
        return self.users[user_id]

db = UserDatabase()

@app.post("/users/{user_id}")
async def create_user(user_id: int, name: str):
    db.add_user(user_id, name)
    return {"status": "success"}

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    return db.get_user(user_id)

class TestUserAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
    
    def test_create_and_get_user(self):
        # Test user creation
        response = self.client.post("/users/1?name=Alice")
        self.assertEqual(response.status_code, 200)
        
        # Test user retrieval
        response = self.client.get("/users/1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["name"], "Alice")
```

Slide 9: Behavior-Driven Development (BDD) with Python

BDD extends TDD by focusing on behavior specification through human-readable scenarios. Python's behave framework allows writing tests in natural language that map to test implementations.

```python
# features/calculator.feature
Feature: Calculator Operations
  Scenario: Adding two numbers
    Given I have entered 50 into the calculator
    And I have entered 70 into the calculator
    When I press add
    Then the result should be 120 on the screen

# steps/calculator_steps.py
from behave import given, when, then
from calculator import Calculator

@given('I have entered {number:d} into the calculator')
def enter_number(context, number):
    if not hasattr(context, 'calculator'):
        context.calculator = Calculator()
    if not hasattr(context, 'numbers'):
        context.numbers = []
    context.numbers.append(number)

@when('I press add')
def press_add(context):
    context.result = context.calculator.add(*context.numbers)

@then('the result should be {result:d} on the screen')
def check_result(context, result):
    assert context.result == result

class Calculator:
    def add(self, *args):
        return sum(args)
```

Slide 10: Integration Testing with Docker

Integration testing in a containerized environment ensures consistent test execution across different platforms and isolates the test environment from the host system.

```python
# docker-compose.yml
version: '3'
services:
  test-db:
    image: postgres:13
    environment:
      POSTGRES_DB: testdb
      POSTGRES_USER: testuser
      POSTGRES_PASSWORD: testpass

# test_integration.py
import unittest
import psycopg2
from unittest.mock import patch

class TestDatabaseIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = psycopg2.connect(
            dbname="testdb",
            user="testuser",
            password="testpass",
            host="localhost"
        )
        cls.cur = cls.conn.cursor()
        
    def setUp(self):
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100)
            )
        """)
        
    def tearDown(self):
        self.cur.execute("DROP TABLE IF EXISTS users")
        self.conn.commit()
        
    def test_user_insertion(self):
        self.cur.execute(
            "INSERT INTO users (name) VALUES (%s) RETURNING id",
            ("Alice",)
        )
        user_id = self.cur.fetchone()[0]
        
        self.cur.execute("SELECT name FROM users WHERE id = %s",
                        (user_id,))
        name = self.cur.fetchone()[0]
        self.assertEqual(name, "Alice")

    @classmethod
    def tearDownClass(cls):
        cls.cur.close()
        cls.conn.close()
```

Slide 11: Performance Testing in TDD

Performance testing within TDD framework ensures that code optimizations don't compromise functionality while maintaining specified performance criteria. This approach combines traditional unit tests with performance benchmarks.

```python
import unittest
import time
import statistics
from functools import wraps

def measure_performance(iterations=1000):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            execution_times = []
            for _ in range(iterations):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                execution_times.append(end_time - start_time)
            
            wrapper.performance_stats = {
                'mean': statistics.mean(execution_times),
                'median': statistics.median(execution_times),
                'stdev': statistics.stdev(execution_times)
            }
            return result
        return wrapper
    return decorator

class TestSortingPerformance(unittest.TestCase):
    @measure_performance(iterations=1000)
    def test_quick_sort(self):
        arr = [64, 34, 25, 12, 22, 11, 90]
        sorted_arr = quick_sort(arr)
        self.assertEqual(sorted_arr, sorted(arr))
        self.assertLess(
            self.test_quick_sort.performance_stats['mean'],
            0.001  # 1ms threshold
        )

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

if __name__ == '__main__':
    unittest.main()
```

Slide 12: Continuous Integration Testing

Implementing TDD within a CI/CD pipeline ensures consistent test execution and validation across different environments before code deployment. This example demonstrates GitHub Actions integration.

```python
# .github/workflows/python-tests.yml
name: Python Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10]

    steps:
    - uses: actions/checkout@v2
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests with coverage
      run: |
        pytest --cov=./ --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
      with:
        files: ./coverage.xml

# test_example.py
def test_addition():
    assert 1 + 1 == 2

def test_string_upper():
    assert "hello".upper() == "HELLO"
```

Slide 13: Real-world TDD Example: E-commerce Order System

This practical implementation demonstrates TDD approach for developing a robust e-commerce order processing system with comprehensive test coverage and validation.

```python
import unittest
from decimal import Decimal
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

@dataclass
class Product:
    id: int
    name: str
    price: Decimal
    stock: int

@dataclass
class OrderItem:
    product: Product
    quantity: int

    @property
    def subtotal(self) -> Decimal:
        return self.product.price * self.quantity

class Order:
    def __init__(self):
        self.items: List[OrderItem] = []
        self.status: str = "pending"
        self.created_at: datetime = datetime.now()
        self._total: Optional[Decimal] = None

    def add_item(self, product: Product, quantity: int) -> None:
        if product.stock < quantity:
            raise ValueError("Insufficient stock")
        self.items.append(OrderItem(product, quantity))
        self._total = None

    @property
    def total(self) -> Decimal:
        if self._total is None:
            self._total = sum(item.subtotal for item in self.items)
        return self._total

class TestOrderSystem(unittest.TestCase):
    def setUp(self):
        self.product = Product(
            id=1,
            name="Test Product",
            price=Decimal("10.00"),
            stock=5
        )
        self.order = Order()

    def test_add_item_to_order(self):
        self.order.add_item(self.product, 2)
        self.assertEqual(len(self.order.items), 1)
        self.assertEqual(self.order.items[0].quantity, 2)

    def test_order_total_calculation(self):
        self.order.add_item(self.product, 3)
        self.assertEqual(self.order.total, Decimal("30.00"))

    def test_insufficient_stock(self):
        with self.assertRaises(ValueError):
            self.order.add_item(self.product, 10)

if __name__ == '__main__':
    unittest.main()
```

Slide 14: Additional Resources

*   Test-Driven Development Best Practices: [https://www.google.com/search?q=python+tdd+best+practices](https://www.google.com/search?q=python+tdd+best+practices)
*   Property-Based Testing: [https://hypothesis.readthedocs.io/](https://hypothesis.readthedocs.io/)
*   Test Coverage Tools: [https://coverage.readthedocs.io/](https://coverage.readthedocs.io/)
*   Python Testing Documentation: [https://docs.python.org/3/library/unittest.html](https://docs.python.org/3/library/unittest.html)
*   Behavior-Driven Development with Python: [https://behave.readthedocs.io/](https://behave.readthedocs.io/)
*   Continuous Integration Testing: [https://docs.github.com/actions/automating-builds-and-tests](https://docs.github.com/actions/automating-builds-and-tests)

