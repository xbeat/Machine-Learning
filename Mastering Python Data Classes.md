## Mastering Python Data Classes
Slide 1: Introduction to Python Data Classes

Data Classes are a powerful feature introduced in Python 3.7 that simplifies the creation of classes primarily used for storing data. They automatically generate special methods like **init**(), **repr**(), and **eq**(), reducing boilerplate code while maintaining clean class definitions.

```python
from dataclasses import dataclass

# Traditional class implementation
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f'Point(x={self.x}, y={self.y})'
    
    def __eq__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return (self.x, self.y) == (other.x, other.y)

# Equivalent Data Class implementation
@dataclass
class PointDataClass:
    x: float
    y: float

# Usage example
p1 = PointDataClass(1.0, 2.0)
print(p1)  # Output: PointDataClass(x=1.0, y=2.0)
```

Slide 2: Default Values and Field Types

Data Classes support type hints and default values, providing better code documentation and runtime type checking when combined with tools like mypy. Fields can be initialized with default values or made optional using None.

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class Configuration:
    host: str = "localhost"
    port: int = 8080
    debug: bool = False
    timeout: Optional[float] = None
    
# Examples
default_config = Configuration()
custom_config = Configuration("example.com", 443, True, 30.0)

print(default_config)  # Configuration(host='localhost', port=8080, debug=False, timeout=None)
print(custom_config)   # Configuration(host='example.com', port=443, debug=True, timeout=30.0)
```

Slide 3: Immutable Data Classes

Data Classes can be made immutable using the frozen parameter, preventing attribute modifications after instantiation. This is useful for creating value objects and ensuring data integrity throughout the program's lifecycle.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Vector3D:
    x: float
    y: float
    z: float
    
    def magnitude(self) -> float:
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5

# Usage
v = Vector3D(1.0, 2.0, 3.0)
print(v.magnitude())  # Output: 3.7416573867739413

try:
    v.x = 5.0  # Raises FrozenInstanceError
except Exception as e:
    print(f"Error: {e}")  # Error: cannot assign to field 'x'
```

Slide 4: Post-Initialization Processing

The **post\_init** method allows for custom initialization logic after the automatic initialization of fields. This is particularly useful for derived fields or validation checks.

```python
from dataclasses import dataclass, field

@dataclass
class Rectangle:
    width: float
    height: float
    area: float = field(init=False)
    perimeter: float = field(init=False)
    
    def __post_init__(self):
        self.area = self.width * self.height
        self.perimeter = 2 * (self.width + self.height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Dimensions must be positive")

# Usage
rect = Rectangle(5.0, 3.0)
print(f"Area: {rect.area}")        # Area: 15.0
print(f"Perimeter: {rect.perimeter}")  # Perimeter: 16.0
```

Slide 5: Inheritance with Data Classes

Data Classes support inheritance, allowing you to create hierarchies of data-containing classes while maintaining the benefits of automatic method generation and field management.

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class Person:
    name: str
    age: int
    
@dataclass
class Employee(Person):
    employee_id: str
    department: str
    supervisor: Optional['Employee'] = None

# Usage
ceo = Employee("Alice Smith", 45, "E001", "Executive")
manager = Employee("Bob Jones", 35, "E002", "Engineering", ceo)

print(manager)  # Employee(name='Bob Jones', age=35, employee_id='E002', department='Engineering', supervisor=Employee(name='Alice Smith', age=45, employee_id='E001', department='Executive', supervisor=None))
```

Slide 6: Comparing Data Classes

Data Classes automatically implement comparison methods based on their fields. The order parameter controls which comparison operators are generated, making it easy to sort and compare instances.

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass(order=True)
class LogEntry:
    timestamp: datetime
    level: str
    message: str
    
    def __post_init__(self):
        self.level = self.level.upper()

# Creating log entries
logs = [
    LogEntry(datetime(2024, 1, 1, 10, 30), "info", "Application started"),
    LogEntry(datetime(2024, 1, 1, 10, 29), "warning", "Low memory"),
    LogEntry(datetime(2024, 1, 1, 10, 31), "error", "Connection failed")
]

# Sorting logs by timestamp
sorted_logs = sorted(logs)
for log in sorted_logs:
    print(f"{log.timestamp}: [{log.level}] {log.message}")
```

Slide 7: Field Factory Functions

Field factories allow dynamic computation of default values for each instance, avoiding the common pitfall of mutable defaults shared across instances.

```python
from dataclasses import dataclass, field
from typing import List
from uuid import uuid4

@dataclass
class Task:
    description: str
    # Wrong way: tags: List[str] = []
    # Correct way:
    tags: List[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid4()))
    
# Usage
task1 = Task("Complete documentation")
task2 = Task("Review code")

task1.tags.append("documentation")
print(f"Task 1 tags: {task1.tags}")  # ['documentation']
print(f"Task 2 tags: {task2.tags}")  # []
print(f"Different IDs: {task1.id != task2.id}")  # True
```

Slide 8: Real-World Example - Configuration Management

Data Classes excel at managing complex configuration settings, providing type safety and validation while maintaining clean, readable code for application settings.

```python
from dataclasses import dataclass
from typing import Optional, Dict, List
import json

@dataclass
class DatabaseConfig:
    host: str
    port: int
    username: str
    password: str
    max_connections: int = 100
    timeout_seconds: float = 30.0

@dataclass
class LoggingConfig:
    level: str
    file_path: Optional[str] = None
    rotate_size_mb: int = 10
    keep_backups: int = 5

@dataclass
class ApplicationConfig:
    db: DatabaseConfig
    logging: LoggingConfig
    api_keys: Dict[str, str] = field(default_factory=dict)
    allowed_origins: List[str] = field(default_factory=list)
    
    @classmethod
    def from_json(cls, config_file: str) -> 'ApplicationConfig':
        with open(config_file) as f:
            data = json.load(f)
            return cls(
                db=DatabaseConfig(**data['database']),
                logging=LoggingConfig(**data['logging']),
                api_keys=data.get('api_keys', {}),
                allowed_origins=data.get('allowed_origins', [])
            )

# Usage example
config_dict = {
    "database": {
        "host": "localhost",
        "port": 5432,
        "username": "admin",
        "password": "secret"
    },
    "logging": {
        "level": "INFO",
        "file_path": "/var/log/app.log"
    },
    "api_keys": {"google": "xyz123", "aws": "abc456"},
    "allowed_origins": ["https://example.com"]
}

with open('config.json', 'w') as f:
    json.dump(config_dict, f)

config = ApplicationConfig.from_json('config.json')
print(config)
```

Slide 9: Advanced Data Class Features

Data Classes support advanced features like slots for memory optimization, weakref\_slots for weak references, and match\_args for pattern matching in Python 3.10+.

```python
from dataclasses import dataclass
from typing import ClassVar
import sys

@dataclass(slots=True, weakref_slot=True, match_args=True)
class OptimizedRecord:
    id: int
    data: str
    _counter: ClassVar[int] = 0  # Shared across all instances
    
    def __post_init__(self):
        OptimizedRecord._counter += 1
    
    @classmethod
    def get_instance_count(cls) -> int:
        return cls._counter

# Memory comparison
regular_record = OptimizedRecord(1, "test")
print(f"Memory size: {sys.getsizeof(regular_record)} bytes")

# Pattern matching (Python 3.10+)
def process_record(record):
    match record:
        case OptimizedRecord(id=1, data="test"):
            return "Found test record"
        case OptimizedRecord(id=id, data=data):
            return f"Other record: {id}, {data}"
        case _:
            return "Not a record"

print(process_record(regular_record))  # Found test record
```

Slide 10: Data Classes with Properties and Validators

Data Classes can be enhanced with properties and validators to ensure data integrity and provide computed attributes while maintaining their clean syntax and automatic method generation.

```python
from dataclasses import dataclass
from typing import List
import re

@dataclass
class User:
    _email: str
    _password: str
    _age: int
    
    @property
    def email(self) -> str:
        return self._email
    
    @email.setter
    def email(self, value: str) -> None:
        if not re.match(r"[^@]+@[^@]+\.[^@]+", value):
            raise ValueError("Invalid email format")
        self._email = value
    
    @property
    def password(self) -> str:
        return "********"
    
    @password.setter
    def password(self, value: str) -> None:
        if len(value) < 8:
            raise ValueError("Password must be at least 8 characters")
        self._password = value
    
    @property
    def age(self) -> int:
        return self._age
    
    @age.setter
    def age(self, value: int) -> None:
        if not 0 <= value <= 150:
            raise ValueError("Invalid age")
        self._age = value

# Usage example
try:
    user = User("john@example.com", "secure123", 30)
    print(user.email)      # john@example.com
    print(user.password)   # ********
    
    user.email = "invalid"  # Raises ValueError
except ValueError as e:
    print(f"Validation error: {e}")
```

Slide 11: Real-World Example - Data Analysis Pipeline

A practical example showing how Data Classes can structure and organize data processing pipelines while maintaining type safety and code clarity.

```python
from dataclasses import dataclass
from typing import List, Optional, Dict
from datetime import datetime
import numpy as np

@dataclass
class DataPoint:
    timestamp: datetime
    value: float
    metadata: Dict[str, str] = field(default_factory=dict)

@dataclass
class TimeSeriesData:
    points: List[DataPoint]
    sampling_rate: float
    
    def get_values(self) -> np.ndarray:
        return np.array([p.value for p in self.points])
    
    def get_timestamps(self) -> np.ndarray:
        return np.array([p.timestamp.timestamp() for p in self.points])

@dataclass
class AnalysisResult:
    mean: float
    std: float
    min_value: float
    max_value: float
    trend: Optional[float] = None

@dataclass
class DataAnalyzer:
    data: TimeSeriesData
    
    def analyze(self) -> AnalysisResult:
        values = self.data.get_values()
        timestamps = self.data.get_timestamps()
        
        # Calculate trend using simple linear regression
        if len(values) > 1:
            z = np.polyfit(timestamps, values, 1)
            trend = z[0]  # slope
        else:
            trend = None
            
        return AnalysisResult(
            mean=float(np.mean(values)),
            std=float(np.std(values)),
            min_value=float(np.min(values)),
            max_value=float(np.max(values)),
            trend=trend
        )

# Example usage
data_points = [
    DataPoint(datetime(2024, 1, 1, i), float(i**2)) 
    for i in range(24)
]

ts_data = TimeSeriesData(data_points, sampling_rate=1.0)
analyzer = DataAnalyzer(ts_data)
result = analyzer.analyze()

print(f"Analysis Results:")
print(f"Mean: {result.mean:.2f}")
print(f"Std Dev: {result.std:.2f}")
print(f"Range: [{result.min_value:.2f}, {result.max_value:.2f}]")
print(f"Trend: {result.trend:.2f} units/second")
```

Slide 12: Serialization and Deserialization

Data Classes can be easily serialized to and deserialized from various formats, making them ideal for data persistence and API interactions.

```python
from dataclasses import dataclass, asdict, field
from typing import Optional
import json
import yaml  # requires pyyaml package

@dataclass
class Address:
    street: str
    city: str
    country: str
    postal_code: str

@dataclass
class Person:
    name: str
    age: int
    address: Address
    email: Optional[str] = None
    _private_data: dict = field(default_factory=dict, repr=False)
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Person':
        data = json.loads(json_str)
        address_data = data.pop('address')
        return cls(
            address=Address(**address_data),
            **data
        )
    
    def to_yaml(self) -> str:
        return yaml.dump(asdict(self))
    
    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'Person':
        data = yaml.safe_load(yaml_str)
        address_data = data.pop('address')
        return cls(
            address=Address(**address_data),
            **data
        )

# Usage example
person = Person(
    name="John Doe",
    age=30,
    address=Address(
        street="123 Main St",
        city="New York",
        country="USA",
        postal_code="10001"
    ),
    email="john@example.com"
)

# Serialization
json_data = person.to_json()
yaml_data = person.to_yaml()

# Deserialization
person_from_json = Person.from_json(json_data)
person_from_yaml = Person.from_yaml(yaml_data)

print("JSON:", json_data)
print("\nYAML:", yaml_data)
print("\nDeserialized from JSON:", person_from_json)
```

Slide 13: Memory Optimization with slots and KW\_ONLY

Data Classes can be optimized for memory usage and enforce keyword-only arguments, making them more efficient and safer to use in memory-constrained environments.

```python
from dataclasses import dataclass, field, KW_ONLY
from sys import getsizeof

@dataclass(slots=True)
class OptimizedProduct:
    id: int
    name: str
    _: KW_ONLY  # Forces all following fields to be keyword-only
    price: float
    quantity: int = 0
    category: str = field(default="uncategorized", kw_only=True)
    
    def total_value(self) -> float:
        return self.price * self.quantity

# Compare memory usage
@dataclass
class RegularProduct:
    id: int
    name: str
    price: float
    quantity: int = 0
    category: str = "uncategorized"

# Usage and memory comparison
opt_prod = OptimizedProduct(1, "Laptop", price=999.99, quantity=5, category="Electronics")
reg_prod = RegularProduct(1, "Laptop", 999.99, 5, "Electronics")

print(f"Optimized size: {getsizeof(opt_prod)} bytes")
print(f"Regular size: {getsizeof(reg_prod)} bytes")

# This will raise TypeError due to missing keyword arguments
try:
    invalid_prod = OptimizedProduct(1, "Laptop", 999.99, 5, "Electronics")
except TypeError as e:
    print(f"Error: {e}")
```

Slide 14: Data Classes in API Development

Implementing a RESTful API endpoint handler using Data Classes for request/response validation and serialization.

```python
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
import json
from uuid import uuid4

@dataclass
class APIResponse:
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    request_id: str = field(default_factory=lambda: str(uuid4()))

@dataclass
class UserCreateRequest:
    username: str
    email: str
    full_name: str
    
    def validate(self) -> Optional[str]:
        if len(self.username) < 3:
            return "Username must be at least 3 characters"
        if '@' not in self.email:
            return "Invalid email format"
        if not self.full_name.strip():
            return "Full name is required"
        return None

class APIHandler:
    @staticmethod
    def create_user(request_data: dict) -> APIResponse:
        try:
            # Parse and validate request
            request = UserCreateRequest(**request_data)
            validation_error = request.validate()
            
            if validation_error:
                return APIResponse(
                    success=False,
                    error=validation_error
                )
            
            # Simulate user creation
            user_data = {
                "id": str(uuid4()),
                "username": request.username,
                "email": request.email,
                "full_name": request.full_name,
                "created_at": datetime.now().isoformat()
            }
            
            return APIResponse(
                success=True,
                data=user_data
            )
            
        except Exception as e:
            return APIResponse(
                success=False,
                error=str(e)
            )

# Example usage
test_requests = [
    {"username": "john_doe", "email": "john@example.com", "full_name": "John Doe"},
    {"username": "ab", "email": "invalid", "full_name": ""}
]

for req in test_requests:
    response = APIHandler.create_user(req)
    print(f"\nRequest: {req}")
    print(f"Success: {response.success}")
    print(f"Data: {response.data}")
    print(f"Error: {response.error}")
    print(f"Request ID: {response.request_id}")
```

Slide 15: Additional Resources

*   ArXiv Paper: "Data Classes in Python: A Case Study in API Design" - Search on Google Scholar
*   "Python Data Classes: A Deep Dive into Modern Python Features" - [https://realpython.com/python-data-classes/](https://realpython.com/python-data-classes/)
*   "Type Hints and Data Classes in Large-Scale Python Applications" - Search on Python.org
*   "Memory Optimization Patterns with Python Data Classes" - [https://pythonspeed.com/articles/](https://pythonspeed.com/articles/)
*   GitHub Repository: "Awesome Python Data Classes" - [https://github.com/topics/python-dataclasses](https://github.com/topics/python-dataclasses)

Note: These resources will help you dive deeper into Data Classes and their applications in Python development.

Let me know if you'd like me to continue generating more slides or if you have any questions about the slides presented so far!

A few key highlights from what we've covered:

*   Basic Data Class usage and features
*   Advanced use cases including inheritance, properties, and validation
*   Real-world examples demonstrating data analysis and API development
*   Memory optimization techniques
*   Type safety and automatic method generation

I can also help explain any specific concepts or code examples in more detail.

