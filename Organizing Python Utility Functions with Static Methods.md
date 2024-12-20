## Organizing Python Utility Functions with Static Methods
Slide 1: Understanding Static Methods

Static methods serve as utility functions that belong to a class namespace but operate independently of class or instance state. They provide a clean way to organize related functionality without requiring instance creation, making the code more modular and easier to maintain.

```python
class MathOperations:
    @staticmethod
    def calculate_factorial(n):
        if n == 0 or n == 1:
            return 1
        return n * MathOperations.calculate_factorial(n - 1)
    
# Using the static method without instantiation
result = MathOperations.calculate_factorial(5)
print(f"Factorial of 5: {result}")  # Output: Factorial of 5: 120
```

Slide 2: Comparing Instance, Class, and Static Methods

Understanding the distinctions between method types is crucial for proper implementation. Instance methods can access instance attributes, class methods can modify class state, while static methods operate independently of both instance and class state.

```python
class DataProcessor:
    data_format = "csv"  # class variable
    
    def __init__(self, data):
        self.data = data  # instance variable
    
    def process_data(self):  # instance method
        return f"Processing {self.data}"
    
    @classmethod
    def change_format(cls, new_format):  # class method
        cls.data_format = new_format
        return cls.data_format
    
    @staticmethod
    def validate_format(format_type):  # static method
        return format_type in ["csv", "json", "xml"]

# Usage demonstration
processor = DataProcessor("sample_data")
print(processor.process_data())  # Output: Processing sample_data
print(DataProcessor.change_format("json"))  # Output: json
print(DataProcessor.validate_format("yaml"))  # Output: False
```

Slide 3: Static Methods in Data Validation

Static methods excel at performing validation tasks that don't require object state. They can be used to verify input parameters, check data formats, or validate configuration settings before object instantiation.

```python
class InputValidator:
    @staticmethod
    def validate_email(email):
        import re
        pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_phone(phone):
        import re
        pattern = r'^\+?1?\d{9,15}$'
        return bool(re.match(pattern, phone))

# Validation examples
print(InputValidator.validate_email("user@example.com"))  # Output: True
print(InputValidator.validate_phone("+1234567890"))  # Output: True
print(InputValidator.validate_email("invalid.email"))  # Output: False
```

Slide 4: Mathematical Computations with Static Methods

Static methods are particularly useful for implementing mathematical operations that remain consistent across all instances of a class. These methods can be called directly without instantiating the class.

```python
class Statistics:
    @staticmethod
    def mean(numbers):
        return sum(numbers) / len(numbers)
    
    @staticmethod
    def variance(numbers):
        mean = Statistics.mean(numbers)
        return sum((x - mean) ** 2 for x in numbers) / len(numbers)
    
    @staticmethod
    def standard_deviation(numbers):
        return Statistics.variance(numbers) ** 0.5

# Statistical calculations
data = [1, 2, 3, 4, 5]
print(f"Mean: {Statistics.mean(data):.2f}")  # Output: Mean: 3.00
print(f"Standard Deviation: {Statistics.standard_deviation(data):.2f}")  # Output: Standard Deviation: 1.41
```

Slide 5: File Operations Using Static Methods

Static methods provide an elegant way to handle file operations that don't require instance-specific data. They can encapsulate common file handling patterns while maintaining clean and reusable code.

```python
class FileHandler:
    @staticmethod
    def read_json(filepath):
        import json
        try:
            with open(filepath, 'r') as file:
                return json.load(file)
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    @staticmethod
    def write_json(data, filepath):
        import json
        try:
            with open(filepath, 'w') as file:
                json.dump(data, file, indent=4)
            return True
        except Exception as e:
            return f"Error writing file: {str(e)}"

# Example usage
data = {"name": "John", "age": 30}
FileHandler.write_json(data, "user.json")
loaded_data = FileHandler.read_json("user.json")
print(loaded_data)  # Output: {'name': 'John', 'age': 30}
```

Slide 6: Static Methods for Date and Time Operations

Static methods can effectively handle date and time conversions and calculations without maintaining any instance state. This approach is particularly useful when working with different time zones and date formats across an application.

```python
from datetime import datetime, timezone

class DateTimeUtil:
    @staticmethod
    def to_unix_timestamp(dt_str, format="%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(dt_str, format)
            return int(dt.timestamp())
        except ValueError as e:
            return f"Error: {str(e)}"
    
    @staticmethod
    def from_unix_timestamp(timestamp):
        try:
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except ValueError as e:
            return f"Error: {str(e)}"

# Example usage
timestamp = DateTimeUtil.to_unix_timestamp("2024-01-01 12:00:00")
print(f"Unix Timestamp: {timestamp}")  # Output: Unix Timestamp: 1704110400
print(f"DateTime: {DateTimeUtil.from_unix_timestamp(timestamp)}")  
# Output: DateTime: 2024-01-01 12:00:00+00:00
```

Slide 7: Data Encryption Using Static Methods

Static methods provide a clean interface for encryption and decryption operations, making security implementations more maintainable and reusable across different parts of an application.

```python
import base64
from cryptography.fernet import Fernet

class Encryptor:
    @staticmethod
    def generate_key():
        return Fernet.generate_key()
    
    @staticmethod
    def encrypt_message(message: str, key: bytes) -> str:
        f = Fernet(key)
        encrypted = f.encrypt(message.encode())
        return base64.b64encode(encrypted).decode()
    
    @staticmethod
    def decrypt_message(encrypted_message: str, key: bytes) -> str:
        f = Fernet(key)
        decrypted = f.decrypt(base64.b64decode(encrypted_message))
        return decrypted.decode()

# Example usage
key = Encryptor.generate_key()
message = "Secret message"
encrypted = Encryptor.encrypt_message(message, key)
decrypted = Encryptor.decrypt_message(encrypted, key)
print(f"Original: {message}")  # Output: Original: Secret message
print(f"Encrypted: {encrypted}")  # Output: Encrypted: [encrypted string]
print(f"Decrypted: {decrypted}")  # Output: Decrypted: Secret message
```

Slide 8: Static Methods in Image Processing

When handling image processing tasks that don't require maintaining state between operations, static methods offer a clean and efficient approach to implementing various image manipulation functions.

```python
import numpy as np
from PIL import Image

class ImageProcessor:
    @staticmethod
    def resize_image(image_array: np.ndarray, scale_factor: float) -> np.ndarray:
        height, width = image_array.shape[:2]
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        
        img = Image.fromarray(image_array)
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        return np.array(resized_img)
    
    @staticmethod
    def apply_grayscale(image_array: np.ndarray) -> np.ndarray:
        return np.dot(image_array[..., :3], [0.2989, 0.5870, 0.1140])

# Example usage (assuming you have an image)
# image = np.array(Image.open('image.jpg'))
# resized = ImageProcessor.resize_image(image, 0.5)
# grayscale = ImageProcessor.apply_grayscale(image)
```

Slide 9: Database Operations with Static Methods

Static methods excel at handling database operations that are independent of instance state, providing a clean interface for common database interactions while maintaining separation of concerns.

```python
import sqlite3
from typing import List, Dict, Any

class DatabaseHandler:
    @staticmethod
    def execute_query(query: str, params: tuple = None) -> List[Dict[str, Any]]:
        try:
            with sqlite3.connect('database.db') as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                    
                result = [dict(row) for row in cursor.fetchall()]
                return result
        except sqlite3.Error as e:
            return [{"error": str(e)}]

# Example usage
query = "SELECT * FROM users WHERE age > ?"
results = DatabaseHandler.execute_query(query, (25,))
print(f"Query results: {results}")
```

Slide 10: Static Methods in API Response Handling

Static methods provide an elegant way to standardize API response formatting and error handling across an application, ensuring consistent communication patterns.

```python
from typing import Union, Dict, Any
import json

class APIResponseHandler:
    @staticmethod
    def success_response(data: Any, message: str = "Success") -> Dict:
        return {
            "status": "success",
            "message": message,
            "data": data,
            "error": None
        }
    
    @staticmethod
    def error_response(error: Union[str, Exception], code: int = 400) -> Dict:
        return {
            "status": "error",
            "message": str(error),
            "data": None,
            "error_code": code
        }
    
    @staticmethod
    def format_response(response: Dict) -> str:
        return json.dumps(response, indent=2)

# Example usage
data = {"user_id": 123, "name": "John Doe"}
success = APIResponseHandler.success_response(data)
error = APIResponseHandler.error_response("Invalid input", 400)
print(APIResponseHandler.format_response(success))
print(APIResponseHandler.format_response(error))
```

Slide 11: Static Methods for Caching Mechanisms

Static methods can implement efficient caching mechanisms that maintain cache state at the class level while providing clean interfaces for cache operations. This approach optimizes performance without instance-specific overhead.

```python
from functools import wraps
from time import time

class CacheManager:
    _cache = {}
    _cache_expiry = {}
    
    @staticmethod
    def cache_with_ttl(ttl_seconds=300):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
                current_time = time()
                
                if key in CacheManager._cache:
                    if current_time - CacheManager._cache_expiry[key] < ttl_seconds:
                        return CacheManager._cache[key]
                
                result = func(*args, **kwargs)
                CacheManager._cache[key] = result
                CacheManager._cache_expiry[key] = current_time
                return result
            return wrapper
        return decorator

# Example usage
@CacheManager.cache_with_ttl(ttl_seconds=60)
def expensive_operation(x):
    import time
    time.sleep(2)  # Simulate expensive operation
    return x * x

print(expensive_operation(5))  # Takes 2 seconds
print(expensive_operation(5))  # Instant (cached)
```

Slide 12: Static Methods in Neural Network Implementation

Static methods effectively handle neural network computations, providing clean interfaces for activation functions and loss calculations that remain consistent across different network architectures.

```python
import numpy as np

class NeuralNetworkUtils:
    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x):
        """Derivative of sigmoid function"""
        sx = NeuralNetworkUtils.sigmoid(x)
        return sx * (1 - sx)
    
    @staticmethod
    def categorical_cross_entropy(y_true, y_pred):
        """Calculate categorical cross-entropy loss"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

# Example usage
x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
print(f"Sigmoid output: {NeuralNetworkUtils.sigmoid(x)}")
print(f"Sigmoid derivative: {NeuralNetworkUtils.sigmoid_derivative(x)}")

y_true = np.array([[1, 0, 0], [0, 1, 0]])
y_pred = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
print(f"Cross-entropy loss: {NeuralNetworkUtils.categorical_cross_entropy(y_true, y_pred)}")
```

Slide 13: Results and Performance Analysis

This implementation showcase demonstrates the practical benefits of static methods in real-world scenarios, from improved code organization to performance optimization.

```python
import time
import statistics

class PerformanceMetrics:
    @staticmethod
    def measure_execution_time(func, *args, iterations=1000):
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func(*args)
            end = time.perf_counter()
            times.append(end - start)
        
        return {
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'std_dev': statistics.stdev(times)
        }

# Performance comparison example
def regular_function(x):
    return x * x

class MathOps:
    @staticmethod
    def static_square(x):
        return x * x

# Measure performance
regular_metrics = PerformanceMetrics.measure_execution_time(regular_function, 5)
static_metrics = PerformanceMetrics.measure_execution_time(MathOps.static_square, 5)

print("Regular Function Metrics:", regular_metrics)
print("Static Method Metrics:", static_metrics)
```

Slide 14: Additional Resources

*   Effective Python: 90 Specific Ways to Write Better Python [https://www.google.com/search?q=effective+python+90+specific+ways+to+write+better+python](https://www.google.com/search?q=effective+python+90+specific+ways+to+write+better+python)
*   Python Design Patterns: For Sleek and Sustainable Code [https://www.google.com/search?q=python+design+patterns+book](https://www.google.com/search?q=python+design+patterns+book)
*   Advanced Python Programming: Best Practices and Design Patterns [https://arxiv.org/abs/cs.SE/2103.11928](https://arxiv.org/abs/cs.SE/2103.11928)
*   Clean Code in Python: Refactoring Guidelines [https://www.google.com/search?q=clean+code+python+best+practices](https://www.google.com/search?q=clean+code+python+best+practices)
*   Static Methods and Inheritance in Object-Oriented Programming [https://www.google.com/search?q=static+methods+inheritance+python+research](https://www.google.com/search?q=static+methods+inheritance+python+research)

