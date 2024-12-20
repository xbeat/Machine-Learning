## Unit Testing in Python with unittest
Slide 1: Setting Up a Basic Unit Test

Python's unittest framework provides a structured way to create test cases by subclassing TestCase. This allows us to define test methods that verify specific functionality using assertion methods to check expected outcomes against actual results.

```python
import unittest

class TestStringMethods(unittest.TestCase):
    def test_upper(self):
        # Test the upper() method of string
        self.assertEqual('hello'.upper(), 'HELLO')
        
    def test_isupper(self):
        # Test the isupper() method of string
        self.assertTrue('HELLO'.isupper())
        self.assertFalse('Hello'.isupper())

if __name__ == '__main__':
    unittest.main()
```

Slide 2: Essential Assertion Methods

Understanding assertion methods is crucial for effective testing. These methods form the foundation of test validation, allowing precise comparison of expected versus actual outcomes while providing meaningful error messages when tests fail.

```python
class TestAssertionDemo(unittest.TestCase):
    def test_assertions(self):
        # Equality assertions
        self.assertEqual(2 + 2, 4)
        self.assertNotEqual(2 + 2, 5)
        
        # Boolean assertions
        self.assertTrue(isinstance(1, int))
        self.assertFalse(isinstance(1, str))
        
        # Membership assertions
        self.assertIn(3, [1, 2, 3])
        self.assertNotIn(4, [1, 2, 3])
        
        # Identity assertions
        self.assertIs(None, None)
        self.assertIsNot(True, False)
```

Slide 3: Test Fixtures

Test fixtures enable proper test setup and cleanup, ensuring consistent test environments. The setUp method runs before each test method, while tearDown executes after each test, allowing resource management and state initialization.

```python
class TestDatabaseOperations(unittest.TestCase):
    def setUp(self):
        # Initialize test database connection
        self.test_data = {'user': 'test_user', 'score': 100}
        self.backup_data = self.test_data.copy()
    
    def tearDown(self):
        # Cleanup after each test
        self.test_data = self.backup_data.copy()
    
    def test_modify_data(self):
        self.test_data['score'] = 200
        self.assertEqual(self.test_data['score'], 200)
```

Slide 4: Testing Exceptions

Proper exception handling testing ensures your code fails gracefully. The unittest framework provides context managers to verify that specific exceptions are raised under expected conditions.

```python
class TestExceptionHandling(unittest.TestCase):
    def test_exception_raised(self):
        # Test if specific exception is raised
        with self.assertRaises(ValueError):
            int('not_a_number')
            
        # Test exception with specific message
        with self.assertRaisesRegex(ValueError, 'invalid literal'):
            int('abc')
            
    def test_zero_division(self):
        with self.assertRaises(ZeroDivisionError):
            1 / 0
```

Slide 5: Parameterized Tests Implementation

Parameterized testing allows running the same test logic with different input parameters, reducing code duplication and ensuring comprehensive test coverage across various scenarios.

```python
class TestParameterized(unittest.TestCase):
    def setUp(self):
        self.test_cases = [
            (2, 2, 4),    # (input1, input2, expected_output)
            (0, 5, 5),
            (-1, 1, 0),
            (10, -5, 5)
        ]
    
    def test_multiple_additions(self):
        for a, b, expected in self.test_cases:
            with self.subTest(a=a, b=b):
                result = a + b
                self.assertEqual(result, expected)
```

Slide 6: Real-World Example - Testing a User Management System

A practical implementation of unit testing for a user management system demonstrates how to test complex business logic including user creation, validation, and authentication processes.

```python
class User:
    def __init__(self, username, email):
        self.username = username
        self.email = email
        
    def validate_email(self):
        return '@' in self.email and '.' in self.email

class TestUserManagement(unittest.TestCase):
    def setUp(self):
        self.valid_user = User('john_doe', 'john@example.com')
        self.invalid_user = User('jane_doe', 'invalid_email')
    
    def test_user_creation(self):
        self.assertEqual(self.valid_user.username, 'john_doe')
        self.assertEqual(self.valid_user.email, 'john@example.com')
    
    def test_email_validation(self):
        self.assertTrue(self.valid_user.validate_email())
        self.assertFalse(self.invalid_user.validate_email())
```

Slide 7: Mocking External Dependencies

Mocking is essential for isolating tests from external dependencies. Python's unittest.mock provides powerful tools to create mock objects that simulate complex behaviors without actual external interactions.

```python
from unittest.mock import Mock, patch

class ExternalService:
    def get_data(self):
        # Simulate external API call
        pass

class TestExternalDependencies(unittest.TestCase):
    def test_external_service(self):
        mock_service = Mock()
        mock_service.get_data.return_value = {'status': 'success'}
        
        # Test with mock
        result = mock_service.get_data()
        self.assertEqual(result['status'], 'success')
        
    @patch('__main__.ExternalService')
    def test_with_patch(self, MockService):
        MockService.return_value.get_data.return_value = {'status': 'success'}
        service = ExternalService()
        self.assertEqual(service.get_data()['status'], 'success')
```

Slide 8: Testing Asynchronous Code

Understanding how to test asynchronous functions is crucial in modern Python development. The unittest framework provides special methods for testing coroutines and async/await patterns effectively.

```python
import asyncio
import unittest

class TestAsyncOperations(unittest.TestCase):
    async def async_function(self):
        await asyncio.sleep(0.1)
        return 'completed'
    
    def test_async(self):
        # Create event loop for testing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run async function and get result
        result = loop.run_until_complete(self.async_function())
        self.assertEqual(result, 'completed')
        
        loop.close()
```

Slide 9: Real-World Example - Testing Data Processing Pipeline

This example demonstrates testing a complete data processing pipeline including data validation, transformation, and error handling for a typical ETL (Extract, Transform, Load) process.

```python
class DataProcessor:
    def validate_input(self, data):
        return all(isinstance(x, (int, float)) for x in data)
    
    def transform_data(self, data):
        return [x * 2 for x in data]
    
    def process_pipeline(self, data):
        if not self.validate_input(data):
            raise ValueError("Invalid input data")
        return self.transform_data(data)

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = DataProcessor()
        self.valid_data = [1, 2, 3, 4, 5]
        self.invalid_data = [1, '2', 3]
    
    def test_validation(self):
        self.assertTrue(self.processor.validate_input(self.valid_data))
        self.assertFalse(self.processor.validate_input(self.invalid_data))
    
    def test_transformation(self):
        result = self.processor.transform_data(self.valid_data)
        self.assertEqual(result, [2, 4, 6, 8, 10])
    
    def test_complete_pipeline(self):
        result = self.processor.process_pipeline(self.valid_data)
        self.assertEqual(result, [2, 4, 6, 8, 10])
        
        with self.assertRaises(ValueError):
            self.processor.process_pipeline(self.invalid_data)
```

Slide 10: Test Coverage Analysis

Coverage analysis helps identify untested code paths. Python's coverage.py integration with unittest enables detailed reporting of test coverage metrics and highlights areas needing additional testing.

```python
# Install coverage: pip install coverage
import coverage
import unittest

def calculate_factorial(n):
    if not isinstance(n, int):
        raise TypeError("Input must be an integer")
    if n < 0:
        raise ValueError("Input must be non-negative")
    if n == 0:
        return 1
    return n * calculate_factorial(n - 1)

class TestFactorial(unittest.TestCase):
    def setUp(self):
        self.cov = coverage.Coverage()
        self.cov.start()
    
    def tearDown(self):
        self.cov.stop()
        self.cov.save()
        self.cov.report()
    
    def test_factorial_calculation(self):
        self.assertEqual(calculate_factorial(5), 120)
        self.assertEqual(calculate_factorial(0), 1)
        with self.assertRaises(ValueError):
            calculate_factorial(-1)
        with self.assertRaises(TypeError):
            calculate_factorial("5")
```

Slide 11: Advanced Test Organization

Organizing tests into test suites allows logical grouping and selective test execution. This approach is essential for managing large test codebases effectively.

```python
import unittest

class TestSuite1(unittest.TestCase):
    def test_addition(self):
        self.assertEqual(2 + 2, 4)

class TestSuite2(unittest.TestCase):
    def test_multiplication(self):
        self.assertEqual(2 * 3, 6)

def create_test_suite():
    # Create a test suite combining multiple test classes
    suite = unittest.TestSuite()
    
    # Add test cases to suite
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSuite1))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSuite2))
    
    return suite

if __name__ == '__main__':
    # Run the suite
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(create_test_suite())
```

Slide 12: Testing with Context Managers

Context managers provide a clean way to handle setup and cleanup of test resources. Understanding how to test them ensures proper resource management and exception handling in production code.

```python
class DatabaseConnection:
    def __enter__(self):
        self.is_connected = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.is_connected = False
        return False

class TestContextManager(unittest.TestCase):
    def test_database_connection(self):
        with DatabaseConnection() as db:
            self.assertTrue(db.is_connected)
        self.assertFalse(db.is_connected)
        
    def test_exception_handling(self):
        with self.assertRaises(ValueError):
            with DatabaseConnection():
                raise ValueError("Test exception")
```

Slide 13: Performance Testing Integration

Incorporating performance testing into unit tests helps identify performance regressions early. This example demonstrates how to test execution time and resource usage.

```python
import time
import memory_profiler

class TestPerformance(unittest.TestCase):
    def setUp(self):
        self.large_list = list(range(1000000))
    
    def test_execution_time(self):
        start_time = time.time()
        
        # Operation to test
        sorted(self.large_list)
        
        execution_time = time.time() - start_time
        self.assertLess(execution_time, 1.0)  # Should complete within 1 second
    
    @memory_profiler.profile
    def test_memory_usage(self):
        # Test memory-intensive operation
        result = [x * 2 for x in self.large_list]
        self.assertEqual(len(result), len(self.large_list))
```

Slide 14: Testing Data Structures

A comprehensive example of testing custom data structure implementation, showing both functionality and edge cases handling.

```python
class CustomStack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if not self.items:
            raise IndexError("Stack is empty")
        return self.items.pop()
    
    def peek(self):
        if not self.items:
            raise IndexError("Stack is empty")
        return self.items[-1]

class TestCustomStack(unittest.TestCase):
    def setUp(self):
        self.stack = CustomStack()
    
    def test_push_pop(self):
        self.stack.push(1)
        self.stack.push(2)
        self.assertEqual(self.stack.pop(), 2)
        self.assertEqual(self.stack.pop(), 1)
    
    def test_empty_stack(self):
        with self.assertRaises(IndexError):
            self.stack.pop()
        
        with self.assertRaises(IndexError):
            self.stack.peek()
    
    def test_peek(self):
        self.stack.push("test")
        self.assertEqual(self.stack.peek(), "test")
        self.assertEqual(len(self.stack.items), 1)  # Verify peek doesn't remove item
```

Slide 15: Additional Resources

*   "Best Practices for Unit Testing in Python" - [https://arxiv.org/abs/2108.13833](https://arxiv.org/abs/2108.13833)
*   "Automated Software Testing: A Comprehensive Review" - [https://arxiv.org/abs/2004.07006](https://arxiv.org/abs/2004.07006)
*   "Modern Test-Driven Development in Python" - [https://arxiv.org/abs/2103.14677](https://arxiv.org/abs/2103.14677)
*   "Performance Testing Frameworks: A Systematic Review" - [https://arxiv.org/abs/1912.00745](https://arxiv.org/abs/1912.00745)
*   "Coverage Analysis Techniques in Software Testing" - [https://arxiv.org/abs/1908.05611](https://arxiv.org/abs/1908.05611)

