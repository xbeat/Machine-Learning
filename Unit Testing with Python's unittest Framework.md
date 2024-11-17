## Unit Testing with Python's unittest Framework
Slide 1: Setting Up Basic Unit Tests

Unit testing in Python leverages the unittest module, which provides a rich set of tools for constructing and running tests. The TestCase class serves as the foundation for creating test cases, offering various assertion methods to validate expected outcomes against actual results.

```python
import unittest

class SimpleTest(unittest.TestCase):
    def setUp(self):
        # This method runs before each test
        self.value_a = 10
        self.value_b = 20
    
    def test_addition(self):
        # Test basic addition
        result = self.value_a + self.value_b
        self.assertEqual(result, 30, "Addition test failed")
    
    def test_subtraction(self):
        # Test basic subtraction
        result = self.value_b - self.value_a
        self.assertEqual(result, 10, "Subtraction test failed")

if __name__ == '__main__':
    unittest.main()

# Output:
# ..
# ----------------------------------------------------------------------
# Ran 2 tests in 0.001s
# OK
```

Slide 2: Test Fixtures and Setup Methods

Test fixtures establish a consistent testing environment by preparing necessary resources before tests and cleaning up afterward. The setUp and tearDown methods are crucial fixture methods that ensure each test starts with a clean slate.

```python
import unittest
import tempfile
import os

class TestWithFixtures(unittest.TestCase):
    def setUp(self):
        # Create temporary test file
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, 'test.txt')
        with open(self.test_file, 'w') as f:
            f.write('Test data')
    
    def tearDown(self):
        # Clean up temporary files
        os.remove(self.test_file)
        os.rmdir(self.test_dir)
    
    def test_file_content(self):
        with open(self.test_file, 'r') as f:
            content = f.read()
        self.assertEqual(content, 'Test data')

if __name__ == '__main__':
    unittest.main()
```

Slide 3: Advanced Assertions

The unittest framework provides a comprehensive set of assertion methods beyond simple equality checks, enabling precise validation of various conditions, types, and expected behaviors in test cases.

```python
import unittest

class AdvancedAssertionsTest(unittest.TestCase):
    def test_assertions(self):
        # Test for equality
        self.assertEqual(2 + 2, 4)
        
        # Test for inequality
        self.assertNotEqual(2 + 2, 5)
        
        # Test for approximate equality
        self.assertAlmostEqual(3.14159, 3.14160, places=4)
        
        # Test for truthiness
        self.assertTrue(bool([1, 2, 3]))
        
        # Test for falsiness
        self.assertFalse(bool([]))
        
        # Test for presence in collection
        self.assertIn(3, [1, 2, 3, 4])
        
        # Test for type
        self.assertIsInstance("test", str)
        
        # Test for exceptions
        with self.assertRaises(ZeroDivisionError):
            1 / 0

if __name__ == '__main__':
    unittest.main()
```

Slide 4: Testing Exceptions

Exception handling verification is crucial in unit testing, ensuring that code properly raises and handles expected exceptions under specific conditions.

```python
import unittest

class ExceptionTest(unittest.TestCase):
    def test_exception_context(self):
        # Test exception with context manager
        with self.assertRaises(ValueError) as context:
            int('not a number')
        
        # Verify exception message
        self.assertTrue('invalid literal' in str(context.exception))
    
    def test_exception_decorator(self):
        # Test multiple exceptions
        @unittest.expectedFailure
        def test_expected_failure(self):
            raise ValueError("Expected failure")
        
        # Test specific exception type
        self.assertRaisesRegex(
            ValueError,
            'invalid literal',
            int,
            'not a number'
        )

if __name__ == '__main__':
    unittest.main()
```

Slide 5: Mock Objects and Patching

Mock objects are powerful tools for isolating units of code by replacing external dependencies, enabling testing of components independently of their dependencies.

```python
import unittest
from unittest.mock import Mock, patch

class ExternalAPI:
    def fetch_data(self):
        # Simulating external API call
        return {"status": "success", "data": [1, 2, 3]}

class TestWithMocks(unittest.TestCase):
    def test_mock_object(self):
        # Create mock object
        mock_api = Mock()
        mock_api.fetch_data.return_value = {"status": "success", "data": [1, 2, 3]}
        
        # Test mock
        result = mock_api.fetch_data()
        self.assertEqual(result["status"], "success")
        mock_api.fetch_data.assert_called_once()
    
    @patch('__main__.ExternalAPI')
    def test_patch_decorator(self, MockExternalAPI):
        # Configure mock
        MockExternalAPI.return_value.fetch_data.return_value = {
            "status": "success",
            "data": [1, 2, 3]
        }
        
        # Use mocked class
        api = ExternalAPI()
        result = api.fetch_data()
        self.assertEqual(result["data"], [1, 2, 3])

if __name__ == '__main__':
    unittest.main()
```

Slide 6: Parameterized Tests

Parameterized testing enables running the same test logic with different input sets, reducing code duplication and ensuring comprehensive test coverage across various scenarios and edge cases.

```python
import unittest
from parameterized import parameterized

class ParameterizedTests(unittest.TestCase):
    @parameterized.expand([
        ("positive", 4, 2, 16),
        ("negative", -2, 2, 4),
        ("zero", 0, 5, 0),
        ("one", 1, 10, 1)
    ])
    def test_power_function(self, name, base, exponent, expected):
        # Test power operation with different parameters
        result = pow(base, exponent)
        self.assertEqual(result, expected, 
            f"Failed for {name}: {base}^{exponent} != {expected}")
    
    @parameterized.expand([
        ("empty", "", False),
        ("space", " ", False),
        ("text", "hello", True),
        ("number", "123", True)
    ])
    def test_string_validation(self, name, input_str, expected):
        result = bool(input_str.strip())
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
```

Slide 7: Testing Asynchronous Code

Testing asynchronous operations requires special consideration to ensure proper execution and validation of concurrent operations using Python's asyncio framework.

```python
import unittest
import asyncio

class AsyncTests(unittest.TestCase):
    async def async_fetch_data(self):
        await asyncio.sleep(0.1)  # Simulate async operation
        return {"data": "fetched"}

    async def async_process_data(self, data):
        await asyncio.sleep(0.1)  # Simulate processing
        return f"processed_{data['data']}"

    def test_async_operations(self):
        async def run_tests():
            # Test async fetch
            data = await self.async_fetch_data()
            self.assertEqual(data, {"data": "fetched"})
            
            # Test async processing
            result = await self.async_process_data(data)
            self.assertEqual(result, "processed_fetched")

        # Run async tests
        asyncio.run(run_tests())

    def test_async_concurrent_operations(self):
        async def run_concurrent_tests():
            tasks = [
                self.async_fetch_data(),
                self.async_fetch_data()
            ]
            results = await asyncio.gather(*tasks)
            self.assertEqual(len(results), 2)
            self.assertTrue(all(r == {"data": "fetched"} for r in results))

        asyncio.run(run_concurrent_tests())

if __name__ == '__main__':
    unittest.main()
```

Slide 8: Real-world Example: Testing Data Processing Pipeline

This example demonstrates testing a complete data processing pipeline, including data validation, transformation, and aggregation operations commonly found in production systems.

```python
import unittest
import pandas as pd
import numpy as np
from datetime import datetime

class DataPipeline:
    def validate_data(self, df):
        return df.dropna()
    
    def transform_dates(self, df):
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    def aggregate_data(self, df):
        return df.groupby('category')['value'].sum()

class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        # Create test dataset
        self.test_data = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02', '2024-01-03', None],
            'category': ['A', 'B', 'A', 'B'],
            'value': [10, 20, 30, 40]
        })
        self.pipeline = DataPipeline()

    def test_data_validation(self):
        # Test data validation
        clean_data = self.pipeline.validate_data(self.test_data)
        self.assertEqual(len(clean_data), 3)
        self.assertTrue(clean_data['date'].notna().all())

    def test_date_transformation(self):
        # Test date transformation
        clean_data = self.pipeline.validate_data(self.test_data)
        transformed_data = self.pipeline.transform_dates(clean_data)
        self.assertTrue(isinstance(transformed_data['date'].iloc[0], 
                                 pd.Timestamp))

    def test_aggregation(self):
        # Test data aggregation
        clean_data = self.pipeline.validate_data(self.test_data)
        aggregated = self.pipeline.aggregate_data(clean_data)
        self.assertEqual(aggregated['A'], 40)
        self.assertEqual(aggregated['B'], 20)

if __name__ == '__main__':
    unittest.main()
```

Slide 9: Testing Database Operations

Unit testing database operations requires careful setup and teardown of test databases, mock connections, and validation of CRUD operations while maintaining data integrity and isolation.

```python
import unittest
import sqlite3
from contextlib import contextmanager

class DatabaseManager:
    def __init__(self, db_name):
        self.db_name = db_name
    
    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_name)
        try:
            yield conn
        finally:
            conn.close()
    
    def create_table(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users
                (id INTEGER PRIMARY KEY, name TEXT, email TEXT)
            ''')
            conn.commit()

class TestDatabaseOperations(unittest.TestCase):
    def setUp(self):
        self.db = DatabaseManager(':memory:')
        self.db.create_table()
    
    def test_insert_and_select(self):
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            # Test insertion
            cursor.execute(
                'INSERT INTO users (name, email) VALUES (?, ?)',
                ('John Doe', 'john@example.com')
            )
            conn.commit()
            
            # Test selection
            cursor.execute('SELECT * FROM users WHERE name = ?', ('John Doe',))
            result = cursor.fetchone()
            self.assertEqual(result[1], 'John Doe')
            self.assertEqual(result[2], 'john@example.com')
    
    def test_update_and_delete(self):
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            # Insert test data
            cursor.execute(
                'INSERT INTO users (name, email) VALUES (?, ?)',
                ('Jane Doe', 'jane@example.com')
            )
            
            # Test update
            cursor.execute(
                'UPDATE users SET email = ? WHERE name = ?',
                ('jane.doe@example.com', 'Jane Doe')
            )
            conn.commit()
            
            # Verify update
            cursor.execute('SELECT email FROM users WHERE name = ?', ('Jane Doe',))
            result = cursor.fetchone()
            self.assertEqual(result[0], 'jane.doe@example.com')
            
            # Test delete
            cursor.execute('DELETE FROM users WHERE name = ?', ('Jane Doe',))
            conn.commit()
            
            # Verify deletion
            cursor.execute('SELECT * FROM users WHERE name = ?', ('Jane Doe',))
            result = cursor.fetchone()
            self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
```

Slide 10: Testing RESTful API Integration

Integration testing of RESTful APIs involves validating request/response cycles, handling different HTTP methods, and managing authentication and error scenarios.

```python
import unittest
from unittest.mock import patch
import requests
import json

class APIClient:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.headers = {'Authorization': f'Bearer {api_key}'}
    
    def get_user(self, user_id):
        response = requests.get(
            f'{self.base_url}/users/{user_id}',
            headers=self.headers
        )
        return response.json()
    
    def create_user(self, user_data):
        response = requests.post(
            f'{self.base_url}/users',
            headers=self.headers,
            json=user_data
        )
        return response.json()

class TestAPIIntegration(unittest.TestCase):
    def setUp(self):
        self.api = APIClient('https://api.example.com', 'test_key')
        self.mock_user_data = {
            'id': 1,
            'name': 'Test User',
            'email': 'test@example.com'
        }
    
    @patch('requests.get')
    def test_get_user(self, mock_get):
        # Configure mock response
        mock_response = requests.Response()
        mock_response.status_code = 200
        mock_response._content = json.dumps(self.mock_user_data).encode('utf-8')
        mock_get.return_value = mock_response
        
        # Test API call
        response = self.api.get_user(1)
        self.assertEqual(response['name'], 'Test User')
        mock_get.assert_called_once_with(
            'https://api.example.com/users/1',
            headers={'Authorization': 'Bearer test_key'}
        )
    
    @patch('requests.post')
    def test_create_user(self, mock_post):
        # Configure mock response
        mock_response = requests.Response()
        mock_response.status_code = 201
        mock_response._content = json.dumps(self.mock_user_data).encode('utf-8')
        mock_post.return_value = mock_response
        
        # Test user creation
        new_user = {
            'name': 'Test User',
            'email': 'test@example.com'
        }
        response = self.api.create_user(new_user)
        self.assertEqual(response['id'], 1)
        mock_post.assert_called_once()

if __name__ == '__main__':
    unittest.main()
```

Slide 11: Testing Machine Learning Models

Testing machine learning models requires validation of data preprocessing, model training, prediction accuracy, and model persistence while ensuring reproducibility of results.

```python
import unittest
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

class MLModelTester(unittest.TestCase):
    def setUp(self):
        # Generate synthetic dataset
        np.random.seed(42)
        self.X = np.random.randn(100, 2)
        self.y = (self.X.sum(axis=1) > 0).astype(int)
        
        # Split dataset
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Initialize model
        self.model = LogisticRegression(random_state=42)
    
    def test_model_training(self):
        # Test model training
        self.model.fit(self.X_train, self.y_train)
        train_score = self.model.score(self.X_train, self.y_train)
        self.assertGreater(train_score, 0.7)
    
    def test_model_prediction(self):
        # Train model
        self.model.fit(self.X_train, self.y_train)
        
        # Test predictions
        predictions = self.model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
        
        # Test accuracy
        accuracy = accuracy_score(self.y_test, predictions)
        self.assertGreater(accuracy, 0.7)
    
    def test_model_persistence(self):
        # Train model
        self.model.fit(self.X_train, self.y_train)
        
        # Save model
        model_path = 'test_model.joblib'
        joblib.dump(self.model, model_path)
        
        # Test model loading
        loaded_model = joblib.load(model_path)
        new_predictions = loaded_model.predict(self.X_test)
        original_predictions = self.model.predict(self.X_test)
        
        # Compare predictions
        np.testing.assert_array_equal(new_predictions, original_predictions)
        
        # Cleanup
        os.remove(model_path)

if __name__ == '__main__':
    unittest.main()
```

Slide 12: Testing Multithreaded Code

Testing multithreaded applications requires careful consideration of race conditions, deadlocks, and thread synchronization while ensuring consistent behavior across different execution scenarios.

```python
import unittest
import threading
import queue
import time

class ThreadSafeCounter:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()
    
    def increment(self):
        with self.lock:
            self.value += 1
    
    def get_value(self):
        with self.lock:
            return self.value

class TestThreading(unittest.TestCase):
    def setUp(self):
        self.counter = ThreadSafeCounter()
        self.queue = queue.Queue()
    
    def test_concurrent_increments(self):
        def worker():
            for _ in range(100):
                self.counter.increment()
                time.sleep(0.001)  # Simulate work
        
        # Create and start threads
        threads = [
            threading.Thread(target=worker)
            for _ in range(10)
        ]
        
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify final count
        self.assertEqual(self.counter.get_value(), 1000)
        
        # Verify execution time
        execution_time = time.time() - start_time
        self.assertLess(execution_time, 2.0)
    
    def test_thread_queue(self):
        def producer():
            for i in range(5):
                self.queue.put(i)
                time.sleep(0.01)
        
        def consumer():
            results = []
            while len(results) < 5:
                try:
                    item = self.queue.get(timeout=1.0)
                    results.append(item)
                except queue.Empty:
                    break
            return results
        
        # Start producer thread
        producer_thread = threading.Thread(target=producer)
        producer_thread.start()
        
        # Consume items
        results = consumer()
        
        # Wait for producer to finish
        producer_thread.join()
        
        # Verify results
        self.assertEqual(len(results), 5)
        self.assertEqual(results, list(range(5)))

if __name__ == '__main__':
    unittest.main()
```

Slide 13: Performance Testing with unittest

Performance testing involves measuring execution time, memory usage, and resource utilization to ensure code meets performance requirements and identify potential bottlenecks.

```python
import unittest
import time
import memory_profiler
import sys
from functools import wraps
from io import StringIO

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, end_time - start_time
    return wrapper

class PerformanceTest(unittest.TestCase):
    def setUp(self):
        self.data_size = 1000000
        self.test_data = list(range(self.data_size))
    
    def test_execution_time(self):
        @measure_time
        def sort_data(data):
            return sorted(data)
        
        # Test sorting performance
        result, execution_time = sort_data(self.test_data)
        self.assertIsNotNone(result)
        self.assertLess(execution_time, 1.0, 
            f"Sorting took too long: {execution_time:.2f} seconds")
    
    def test_memory_usage(self):
        @memory_profiler.profile
        def memory_intensive_operation():
            # Simulate memory-intensive operation
            large_list = [i * i for i in range(100000)]
            return sum(large_list)
        
        # Capture memory profiler output
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            result = memory_intensive_operation()
            memory_output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        
        # Verify operation completed
        self.assertIsNotNone(result)
        
        # Check memory usage from profiler output
        memory_lines = memory_output.strip().split('\n')
        peak_memory = max(
            float(line.split()[3]) 
            for line in memory_lines 
            if line.strip() and 'MiB' in line
        )
        
        self.assertLess(peak_memory, 100.0, 
            f"Memory usage too high: {peak_memory:.2f} MiB")
    
    def test_resource_scaling(self):
        def measure_scaling(size):
            start_time = time.perf_counter()
            data = list(range(size))
            sorted(data)
            return time.perf_counter() - start_time
        
        # Test different input sizes
        sizes = [1000, 10000, 100000]
        times = [measure_scaling(size) for size in sizes]
        
        # Verify linear or near-linear scaling
        for i in range(1, len(sizes)):
            time_ratio = times[i] / times[i-1]
            size_ratio = sizes[i] / sizes[i-1]
            scaling_factor = time_ratio / size_ratio
            self.assertLess(scaling_factor, 1.5, 
                f"Poor scaling detected at size {sizes[i]}")

if __name__ == '__main__':
    unittest.main()
```

Slide 14: Testing Security Features

Security testing involves validating authentication, authorization, input validation, and cryptographic operations to ensure the application maintains proper security controls.

```python
import unittest
import hashlib
import secrets
import re
from base64 import b64encode
from cryptography.fernet import Fernet

class SecurityFeatures:
    def __init__(self):
        self.key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.key)
    
    def hash_password(self, password, salt=None):
        if salt is None:
            salt = secrets.token_hex(16)
        hash_obj = hashlib.sha256()
        hash_obj.update((password + salt).encode())
        return hash_obj.hexdigest(), salt
    
    def encrypt_data(self, data):
        return self.cipher_suite.encrypt(data.encode())
    
    def decrypt_data(self, encrypted_data):
        return self.cipher_suite.decrypt(encrypted_data).decode()
    
    def validate_password_strength(self, password):
        if len(password) < 8:
            return False
        if not re.search(r'[A-Z]', password):
            return False
        if not re.search(r'[a-z]', password):
            return False
        if not re.search(r'\d', password):
            return False
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return False
        return True

class TestSecurity(unittest.TestCase):
    def setUp(self):
        self.security = SecurityFeatures()
    
    def test_password_hashing(self):
        password = "SecurePass123!"
        hash1, salt = self.security.hash_password(password)
        hash2, _ = self.security.hash_password(password, salt)
        
        # Verify hash consistency
        self.assertEqual(hash1, hash2)
        
        # Verify different salt produces different hash
        hash3, _ = self.security.hash_password(password)
        self.assertNotEqual(hash1, hash3)
    
    def test_encryption(self):
        original_data = "Sensitive information"
        encrypted = self.security.encrypt_data(original_data)
        decrypted = self.security.decrypt_data(encrypted)
        
        # Verify encryption/decryption
        self.assertEqual(original_data, decrypted)
        self.assertNotEqual(original_data.encode(), encrypted)
    
    def test_password_validation(self):
        # Test valid password
        self.assertTrue(
            self.security.validate_password_strength("SecurePass123!")
        )
        
        # Test invalid passwords
        invalid_passwords = [
            "short",  # Too short
            "onlylowercase123!",  # No uppercase
            "ONLYUPPERCASE123!",  # No lowercase
            "NoNumbers!",  # No numbers
            "NoSpecialChars123"  # No special characters
        ]
        
        for password in invalid_passwords:
            self.assertFalse(
                self.security.validate_password_strength(password),
                f"Password should be invalid: {password}"
            )

if __name__ == '__main__':
    unittest.main()
```

Slide 15: Additional Resources

*   Advanced Python Testing:
    *   [https://docs.python.org/3/library/unittest.html](https://docs.python.org/3/library/unittest.html)
    *   [https://www.python.org/dev/peps/pep-0338/](https://www.python.org/dev/peps/pep-0338/)
    *   [https://docs.pytest.org/en/latest/](https://docs.pytest.org/en/latest/)
*   Security Testing Resources:
    *   [https://owasp.org/www-project-web-security-testing-guide/](https://owasp.org/www-project-web-security-testing-guide/)
    *   [https://www.python.org/dev/peps/pep-0551/](https://www.python.org/dev/peps/pep-0551/)
*   Performance Testing Guidelines:
    *   [https://docs.python.org/3/library/profile.html](https://docs.python.org/3/library/profile.html)
    *   [https://pypi.org/project/memory-profiler/](https://pypi.org/project/memory-profiler/)
    *   [https://www.python.org/dev/peps/pep-0418/](https://www.python.org/dev/peps/pep-0418/)

