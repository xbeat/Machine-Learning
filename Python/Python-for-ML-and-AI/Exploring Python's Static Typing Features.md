## Exploring Python's Static Typing Features
Slide 1: Type Hints Basics in Python

Python's type hinting system, introduced in Python 3.5 with PEP 484, provides a way to explicitly specify the expected types of function parameters, return values, and variables. This foundation enables better code documentation and allows static type checkers to catch potential errors before runtime.

```python
from typing import List, Dict, Optional

def process_user_data(name: str, age: int, scores: List[float]) -> Dict[str, any]:
    # Type hints help document expected types and enable static checking
    user_data: Dict[str, any] = {
        "name": name,
        "age": age,
        "average_score": sum(scores) / len(scores) if scores else 0
    }
    return user_data

# Example usage with type hints
student_scores: List[float] = [85.5, 92.0, 78.5]
result = process_user_data("Alice", 20, student_scores)
print(result)  # Output: {'name': 'Alice', 'age': 20, 'average_score': 85.33...}
```

Slide 2: Type Checking with mypy

Static type checking using mypy provides an additional layer of code quality assurance by analyzing type annotations before runtime. This powerful tool can catch type-related errors that might otherwise only surface during program execution.

```python
# Save as example.py
from typing import List, Optional

class DataProcessor:
    def process_values(self, values: List[float]) -> Optional[float]:
        if not values:
            return None
        return sum(values) / len(values)

    def analyze_data(self, data: List[str]) -> None:
        # This will raise a type error during mypy checking
        result = self.process_values(data)  # Type error: List[str] != List[float]
        print(result)

# Run in terminal: mypy example.py
# Output: example.py:11: error: Argument 1 to "process_values" has incompatible type...
```

Slide 3: Advanced Type Annotations with Generics

Generic types provide a powerful way to write flexible, type-safe code that can work with different data types while maintaining type safety. This advanced feature allows for creation of reusable components with strict type checking.

```python
from typing import TypeVar, Generic, List

T = TypeVar('T')
class Queue(Generic[T]):
    def __init__(self) -> None:
        self.items: List[T] = []
    
    def enqueue(self, item: T) -> None:
        self.items.append(item)
    
    def dequeue(self) -> T:
        return self.items.pop(0)

# Type-safe usage
string_queue: Queue[str] = Queue()
string_queue.enqueue("Hello")  # OK
string_queue.enqueue(42)  # Type error caught by mypy
```

Slide 4: Union Types and Type Aliases

Type aliases and union types provide flexibility in type annotations while maintaining code clarity. They allow expressing complex type relationships and creating meaningful type definitions that enhance code readability.

```python
from typing import Union, List, Dict, TypeAlias

# Define custom type aliases
JsonValue = Union[str, int, float, bool, None, Dict[str, 'JsonValue'], List['JsonValue']]
ProcessingResult: TypeAlias = Dict[str, Union[int, float]]

def process_data(data: JsonValue) -> ProcessingResult:
    result: ProcessingResult = {"status": 0, "value": 0.0}
    
    if isinstance(data, (int, float)):
        result["value"] = float(data)
        result["status"] = 1
    
    return result

# Example usage
print(process_data(42))  # Output: {'status': 1, 'value': 42.0}
print(process_data("invalid"))  # Output: {'status': 0, 'value': 0.0}
```

Slide 5: Type Guards and Runtime Checking

Type guards enhance type safety by combining runtime checks with static typing. They help Python's type checker understand type narrowing in conditional blocks, leading to more precise type inference and safer code execution.

```python
from typing import Union, TypeGuard, List

def is_string_list(data: List[Union[str, int]]) -> TypeGuard[List[str]]:
    return all(isinstance(x, str) for x in data)

def process_strings(items: List[Union[str, int]]) -> str:
    if is_string_list(items):
        # Type checker knows items is List[str] here
        return " ".join(items)  # Safe operation
    
    # Convert non-string items to strings
    return " ".join(str(item) for item in items)

# Example usage
mixed_list: List[Union[str, int]] = ["hello", 42, "world"]
string_list: List[Union[str, int]] = ["hello", "world"]
print(process_strings(mixed_list))  # Output: hello 42 world
print(process_strings(string_list))  # Output: hello world
```

Slide 6: Protocol Classes for Duck Typing

Protocol classes enable structural subtyping, allowing type checking based on the presence of specific methods rather than explicit inheritance. This approach maintains Python's duck typing philosophy while adding static type safety.

```python
from typing import Protocol, List
from abc import abstractmethod

class Drawable(Protocol):
    @abstractmethod
    def draw(self) -> str: ...

class Circle:
    def draw(self) -> str:
        return "Drawing a circle"

class Square:
    def draw(self) -> str:
        return "Drawing a square"

def render_shapes(shapes: List[Drawable]) -> None:
    for shape in shapes:
        print(shape.draw())

# Both classes implicitly implement Drawable
shapes: List[Drawable] = [Circle(), Square()]
render_shapes(shapes)
# Output:
# Drawing a circle
# Drawing a square
```

Slide 7: Real-world Example: Data Processing Pipeline

This example demonstrates a practical application of static typing in a data processing pipeline, showing how type hints can improve code reliability in production environments.

```python
from typing import Dict, List, Optional, TypedDict
from datetime import datetime
import json

class SensorData(TypedDict):
    timestamp: str
    temperature: float
    humidity: Optional[float]

class DataProcessor:
    def __init__(self) -> None:
        self.data: List[SensorData] = []
    
    def load_data(self, file_path: str) -> None:
        with open(file_path, 'r') as f:
            raw_data: List[Dict] = json.load(f)
            self.data = [self._validate_reading(r) for r in raw_data]
    
    def _validate_reading(self, raw: Dict) -> SensorData:
        return {
            'timestamp': str(datetime.fromisoformat(raw['timestamp'])),
            'temperature': float(raw['temperature']),
            'humidity': float(raw['humidity']) if 'humidity' in raw else None
        }
    
    def get_average_temperature(self) -> float:
        temps = [d['temperature'] for d in self.data]
        return sum(temps) / len(temps)

# Usage example
processor = DataProcessor()
processor.load_data('sensor_data.json')
print(f"Average temperature: {processor.get_average_temperature():.2f}°C")
```

Slide 8: Advanced Type Inference with Literal Types

Literal types provide a way to specify exact values in type annotations, enabling more precise type checking and better code documentation for functions that expect specific constant values.

```python
from typing import Literal, Union, overload

class DatabaseConnector:
    @overload
    def connect(self, mode: Literal["read"]) -> bool: ...
    
    @overload
    def connect(self, mode: Literal["write"]) -> bool: ...
    
    def connect(self, mode: Literal["read", "write"]) -> bool:
        if mode == "read":
            return self._connect_readonly()
        return self._connect_writeonly()
    
    def _connect_readonly(self) -> bool:
        return True  # Implementation details
    
    def _connect_writeonly(self) -> bool:
        return True  # Implementation details

# Usage
db = DatabaseConnector()
db.connect("read")  # OK
db.connect("write")  # OK
db.connect("invalid")  # Type error caught by mypy
```

Slide 9: Type Safety in Asynchronous Code

Type annotations in asynchronous code help maintain type safety across concurrent operations. This is particularly important when dealing with complex async workflows where type mistakes could lead to runtime errors.

```python
from typing import AsyncIterator, List, Optional
import asyncio
from datetime import datetime

class AsyncDataStream:
    async def fetch_data(self) -> AsyncIterator[float]:
        for i in range(5):
            await asyncio.sleep(0.1)  # Simulate network delay
            yield float(i * 1.5)

async def process_stream(stream: AsyncDataStream) -> List[float]:
    processed_data: List[float] = []
    async for value in stream.fetch_data():
        processed_data.append(value * 2)
    return processed_data

# Example usage
async def main() -> None:
    stream = AsyncDataStream()
    result = await process_stream(stream)
    print(f"Processed data: {result}")

# Run the async code
asyncio.run(main())
# Output: Processed data: [0.0, 3.0, 6.0, 9.0, 12.0]
```

Slide 10: Type-Safe Dependency Injection

Implementing dependency injection with static typing provides compile-time guarantees for component wiring and helps catch configuration errors early in development.

```python
from typing import Protocol, Type
from dataclasses import dataclass

class Database(Protocol):
    def connect(self) -> bool: ...
    def query(self, sql: str) -> List[str]: ...

class PostgresDB:
    def connect(self) -> bool:
        return True
    
    def query(self, sql: str) -> List[str]:
        return ["result1", "result2"]

@dataclass
class UserRepository:
    db: Database
    
    def get_users(self) -> List[str]:
        return self.db.query("SELECT * FROM users")

class ServiceContainer:
    def __init__(self) -> None:
        self._services: Dict[Type, object] = {}
    
    def register(self, interface: Type, implementation: object) -> None:
        self._services[interface] = implementation
    
    def resolve(self, interface: Type[T]) -> T:
        return self._services[interface]

# Usage
container = ServiceContainer()
container.register(Database, PostgresDB())
repo = UserRepository(container.resolve(Database))
users = repo.get_users()
```

Slide 11: Real-world Example: Type-Safe API Client

A practical implementation of a type-safe API client demonstrating how static typing can improve reliability in external service integration.

```python
from typing import TypedDict, Optional, List
import httpx
from datetime import datetime

class UserProfile(TypedDict):
    id: int
    username: str
    email: str
    created_at: str
    last_login: Optional[str]

class APIClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    async def get_user(self, user_id: int) -> Optional[UserProfile]:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/users/{user_id}",
                headers=self.headers
            )
            if response.status_code == 404:
                return None
            data: UserProfile = response.json()
            return data
    
    async def list_users(self, page: int = 1) -> List[UserProfile]:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/users",
                params={"page": page},
                headers=self.headers
            )
            data: List[UserProfile] = response.json()
            return data

# Usage example
async def main() -> None:
    client = APIClient("https://api.example.com", "secret-key")
    user = await client.get_user(123)
    if user:
        print(f"Found user: {user['username']}")
```

Slide 12: Typed Context Managers

Context managers with type hints provide clear interfaces for resource management while maintaining type safety throughout the context's lifecycle. This pattern is particularly useful for database connections and file handling.

```python
from typing import TypeVar, Generic, ContextManager
from types import TracebackType
from typing import Optional, Type

T = TypeVar('T')
class TypedContextManager(Generic[T]):
    def __init__(self, resource: T) -> None:
        self.resource = resource
        self._is_active = False

    def __enter__(self) -> T:
        self._is_active = True
        return self.resource

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType]
    ) -> None:
        self._is_active = False

# Example usage
class DatabaseConnection:
    def execute(self, query: str) -> list[str]:
        return ["result1", "result2"]

def get_db_connection() -> TypedContextManager[DatabaseConnection]:
    return TypedContextManager(DatabaseConnection())

# Type-safe usage
with get_db_connection() as db:
    results = db.execute("SELECT * FROM users")  # Type-checked
```

Slide 13: Advanced Error Handling with Types

Type-safe error handling combines custom exception types with union types to create robust error management systems that can be statically verified.

```python
from typing import Union, Literal, TypedDict
from dataclasses import dataclass

class ValidationError(Exception):
    pass

@dataclass
class Success:
    status: Literal["success"] = "success"
    data: dict

@dataclass
class Failure:
    status: Literal["error"] = "error"
    message: str

Result = Union[Success, Failure]

def process_user_input(data: dict) -> Result:
    try:
        if "username" not in data:
            raise ValidationError("Username is required")
        
        processed_data = {
            "username": data["username"],
            "timestamp": "2024-01-01"
        }
        return Success(data=processed_data)
    
    except ValidationError as e:
        return Failure(message=str(e))

# Usage example
valid_input = {"username": "john_doe"}
invalid_input = {}

result1 = process_user_input(valid_input)
result2 = process_user_input(invalid_input)

# Type-safe handling of results
if result1.status == "success":
    print(f"Success: {result1.data}")  # Type checker knows this is Success
else:
    print(f"Error: {result1.message}")  # Type checker knows this is Failure
```

Slide 14: Additional Resources

*  [https://arxiv.org/abs/2203.03823](https://arxiv.org/abs/2203.03823) - "Type Inference in Python: A Study of Production Code" 
*  [https://arxiv.org/abs/2107.03340](https://arxiv.org/abs/2107.03340) - "Static Type Analysis for Python"
*  [https://arxiv.org/abs/1904.03771](https://arxiv.org/abs/1904.03771) - "TypeWriter: Neural Type Prediction with Search-based Validation" 
*  [https://arxiv.org/abs/2106.11192](https://arxiv.org/abs/2106.11192) - "Understanding Type Hints in Python"
*  [https://arxiv.org/abs/2008.11091](https://arxiv.org/abs/2008.11091) - "Type4Py: Deep Learning-based Type Inference for Python"

