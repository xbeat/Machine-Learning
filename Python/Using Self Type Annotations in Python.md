## Using Self Type Annotations in Python
Slide 1: Self Type Annotation Basics

The Self type annotation in Python provides a way to indicate that a method returns an instance of its own class. This pattern is particularly useful in builder patterns and method chaining, enhancing code readability and type safety.

```python
from __future__ import annotations
from typing import Self  # Python 3.11+

class DatabaseConnection:
    def __init__(self, host: str) -> None:
        self.host = host
        self.connected = False
    
    def connect(self) -> Self:
        self.connected = True
        return self
    
    def disconnect(self) -> Self:
        self.connected = False
        return self

# Usage example
db = DatabaseConnection("localhost")
db.connect().disconnect()  # Method chaining with Self typing
```

Slide 2: Legacy Self Type Implementation

Prior to Python 3.11, developers needed to use string literals or typing\_extensions to achieve proper self-referential type hints. This approach was necessary to avoid forward reference issues in class definitions.

```python
from __future__ import annotations
from typing_extensions import Self  # For Python < 3.11

class Vector:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
    
    def scale(self, factor: float) -> "Vector":  # Pre-3.7 approach
        self.x *= factor
        self.y *= factor
        return self
    
    def normalize(self) -> Self:  # Modern approach
        magnitude = (self.x ** 2 + self.y ** 2) ** 0.5
        return self.scale(1/magnitude if magnitude else 1)
```

Slide 3: Builder Pattern with Self

The Builder pattern becomes more expressive and type-safe using Self annotations. This implementation demonstrates how Self enables fluent interfaces while maintaining strong typing information for modern IDEs and type checkers.

```python
from __future__ import annotations
from typing import Self

class QueryBuilder:
    def __init__(self) -> None:
        self.table = ""
        self.conditions = []
        self.order_by = None
    
    def from_table(self, table_name: str) -> Self:
        self.table = table_name
        return self
    
    def where(self, condition: str) -> Self:
        self.conditions.append(condition)
        return self
    
    def order(self, column: str) -> Self:
        self.order_by = column
        return self
    
    def build(self) -> str:
        query = f"SELECT * FROM {self.table}"
        if self.conditions:
            query += " WHERE " + " AND ".join(self.conditions)
        if self.order_by:
            query += f" ORDER BY {self.order_by}"
        return query

# Usage example
query = (QueryBuilder()
         .from_table("users")
         .where("age > 18")
         .where("status = 'active'")
         .order("created_at")
         .build())
print(query)
```

Slide 4: Recursive Data Structures

Self annotations prove invaluable when working with recursive data structures, providing clear type information for methods that return new instances or modified versions of the same structure.

```python
from __future__ import annotations
from typing import Optional, Self
from dataclasses import dataclass

@dataclass
class TreeNode:
    value: int
    left: Optional[Self] = None
    right: Optional[Self] = None
    
    def insert(self, value: int) -> Self:
        if value < self.value:
            if self.left is None:
                self.left = TreeNode(value)
            else:
                self.left.insert(value)
        else:
            if self.right is None:
                self.right = TreeNode(value)
            else:
                self.right.insert(value)
        return self
    
    def clone(self) -> Self:
        new_node = TreeNode(self.value)
        if self.left:
            new_node.left = self.left.clone()
        if self.right:
            new_node.right = self.right.clone()
        return new_node

# Usage example
root = TreeNode(10).insert(5).insert(15).insert(3)
cloned_tree = root.clone()
```

Slide 5: Method Chaining with Error Handling

Self typing enhances error handling in method chains by providing proper type information when implementing error-aware fluent interfaces. This pattern is particularly useful in data processing pipelines.

```python
from __future__ import annotations
from typing import Self, Optional
from dataclasses import dataclass

@dataclass
class Result:
    value: Optional[float] = None
    error: Optional[str] = None
    
class Calculator:
    def __init__(self, initial: float = 0) -> None:
        self.value = initial
        self.error = None
    
    def add(self, x: float) -> Self:
        if self.error:
            return self
        try:
            self.value += x
        except Exception as e:
            self.error = str(e)
        return self
    
    def divide(self, x: float) -> Self:
        if self.error:
            return self
        try:
            if x == 0:
                raise ValueError("Division by zero")
            self.value /= x
        except Exception as e:
            self.error = str(e)
        return self
    
    def result(self) -> Result:
        return Result(self.value, self.error)

# Usage example
calc = Calculator(10)
result = calc.add(5).divide(0).add(3).result()
print(f"Value: {result.value}, Error: {result.error}")
```

Slide 6: Fluent Interface for Data Processing

Self type annotations enable the creation of clear and type-safe data processing chains. This implementation shows how to build a data transformation pipeline with proper type hints and error handling.

```python
from __future__ import annotations
from typing import Self, List, Any
import statistics

class DataProcessor:
    def __init__(self, data: List[float]) -> None:
        self.data = data
        self._errors = []
    
    def filter_outliers(self, threshold: float = 2.0) -> Self:
        try:
            mean = statistics.mean(self.data)
            std = statistics.stdev(self.data)
            self.data = [x for x in self.data if abs((x - mean) / std) <= threshold]
        except Exception as e:
            self._errors.append(f"Outlier filtering failed: {str(e)}")
        return self
    
    def normalize(self) -> Self:
        try:
            min_val = min(self.data)
            max_val = max(self.data)
            self.data = [(x - min_val) / (max_val - min_val) for x in self.data]
        except Exception as e:
            self._errors.append(f"Normalization failed: {str(e)}")
        return self
    
    def get_result(self) -> tuple[List[float], List[str]]:
        return self.data, self._errors

# Usage example
data = [1.0, 2.0, 100.0, 3.0, 4.0, 5.0]
processed_data, errors = (DataProcessor(data)
                         .filter_outliers(2.0)
                         .normalize()
                         .get_result())
```

Slide 7: State Machine Implementation

Self type annotations enhance the implementation of state machines by providing clear type information for state transitions. This example demonstrates a simple document processing state machine.

```python
from __future__ import annotations
from typing import Self, Optional
from enum import Enum, auto

class DocumentState(Enum):
    DRAFT = auto()
    REVIEW = auto()
    APPROVED = auto()
    PUBLISHED = auto()

class Document:
    def __init__(self, content: str) -> None:
        self.content = content
        self.state = DocumentState.DRAFT
        self.reviewer: Optional[str] = None
    
    def submit_for_review(self, reviewer: str) -> Self:
        if self.state != DocumentState.DRAFT:
            raise ValueError("Can only submit DRAFT documents for review")
        self.reviewer = reviewer
        self.state = DocumentState.REVIEW
        return self
    
    def approve(self) -> Self:
        if self.state != DocumentState.REVIEW:
            raise ValueError("Can only approve documents under REVIEW")
        self.state = DocumentState.APPROVED
        return self
    
    def publish(self) -> Self:
        if self.state != DocumentState.APPROVED:
            raise ValueError("Can only publish APPROVED documents")
        self.state = DocumentState.PUBLISHED
        return self

# Usage example
doc = (Document("Hello, World!")
       .submit_for_review("John")
       .approve()
       .publish())
```

Slide 8: Composite Pattern with Self References

The Composite pattern becomes more expressive with Self type annotations, allowing for clearer hierarchical structure definitions while maintaining type safety throughout the component tree.

```python
from __future__ import annotations
from typing import Self, List

class FileSystemComponent:
    def __init__(self, name: str) -> None:
        self.name = name
        self._parent: Optional[Directory] = None
    
    @property
    def path(self) -> str:
        if self._parent is None:
            return self.name
        return f"{self._parent.path}/{self.name}"
    
    def set_parent(self, parent: Directory) -> Self:
        self._parent = parent
        return self

class File(FileSystemComponent):
    def __init__(self, name: str, content: str = "") -> None:
        super().__init__(name)
        self.content = content

class Directory(FileSystemComponent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.children: List[FileSystemComponent] = []
    
    def add(self, component: FileSystemComponent) -> Self:
        component.set_parent(self)
        self.children.append(component)
        return self
    
    def find(self, name: str) -> Optional[FileSystemComponent]:
        return next((c for c in self.children if c.name == name), None)

# Usage example
root = (Directory("root")
        .add(Directory("usr")
             .add(File("config.txt", "configuration"))
             .add(File("data.db", "database")))
        .add(Directory("home")
             .add(File("notes.txt", "my notes"))))
```

Slide 9: Event Handler Chain with Self

Self type annotations enhance event handling systems by enabling fluent registration of multiple handlers. This implementation demonstrates a type-safe event dispatcher with chainable methods.

```python
from __future__ import annotations
from typing import Self, Callable, Dict, List
from dataclasses import dataclass, field

@dataclass
class Event:
    name: str
    data: dict = field(default_factory=dict)

class EventDispatcher:
    def __init__(self) -> None:
        self._handlers: Dict[str, List[Callable]] = {}
        
    def on(self, event_name: str, handler: Callable[[Event], None]) -> Self:
        if event_name not in self._handlers:
            self._handlers[event_name] = []
        self._handlers[event_name].append(handler)
        return self
    
    def off(self, event_name: str, handler: Callable[[Event], None]) -> Self:
        if event_name in self._handlers:
            self._handlers[event_name].remove(handler)
        return self
    
    def dispatch(self, event: Event) -> Self:
        for handler in self._handlers.get(event.name, []):
            handler(event)
        return self

# Usage example
def log_handler(event: Event) -> None:
    print(f"Log: {event.name} with data {event.data}")

def notify_handler(event: Event) -> None:
    print(f"Notification: {event.name} occurred")

dispatcher = (EventDispatcher()
             .on("user.login", log_handler)
             .on("user.login", notify_handler)
             .dispatch(Event("user.login", {"user_id": 123})))
```

Slide 10: Immutable Data Class with Builder

The Self type annotation enhances the builder pattern for immutable data classes, providing type-safe construction while maintaining immutability in the final object.

```python
from __future__ import annotations
from typing import Self, Optional
from dataclasses import dataclass

@dataclass(frozen=True)
class User:
    id: int
    name: str
    email: str
    age: Optional[int] = None
    
    class Builder:
        def __init__(self) -> None:
            self._id: Optional[int] = None
            self._name: Optional[str] = None
            self._email: Optional[str] = None
            self._age: Optional[int] = None
        
        def id(self, id: int) -> Self:
            self._id = id
            return self
        
        def name(self, name: str) -> Self:
            self._name = name
            return self
        
        def email(self, email: str) -> Self:
            self._email = email
            return self
        
        def age(self, age: int) -> Self:
            self._age = age
            return self
        
        def build(self) -> User:
            if not all([self._id, self._name, self._email]):
                raise ValueError("Missing required fields")
            return User(
                id=self._id,
                name=self._name,
                email=self._email,
                age=self._age
            )
    
    @classmethod
    def builder(cls) -> Builder:
        return cls.Builder()

# Usage example
user = (User.builder()
        .id(1)
        .name("John Doe")
        .email("john@example.com")
        .age(30)
        .build())
```

Slide 11: Real-world Example - Data Pipeline with Validation

This implementation showcases a real-world data processing pipeline using Self type annotations for method chaining with validation and transformation steps.

```python
from __future__ import annotations
from typing import Self, List, Any, Optional
from dataclasses import dataclass
import re

@dataclass
class ValidationError:
    field: str
    message: str

class DataRecord:
    def __init__(self, raw_data: dict) -> None:
        self.data = raw_data
        self.errors: List[ValidationError] = []
        self._processed = False
    
    def validate_email(self, field: str) -> Self:
        email = self.data.get(field)
        if not email or not re.match(r"[^@]+@[^@]+\.[^@]+", str(email)):
            self.errors.append(
                ValidationError(field, "Invalid email format")
            )
        return self
    
    def validate_age(self, field: str) -> Self:
        age = self.data.get(field)
        if not isinstance(age, (int, float)) or age < 0 or age > 150:
            self.errors.append(
                ValidationError(field, "Invalid age value")
            )
        return self
    
    def transform(self) -> Self:
        if not self.errors:
            self.data = {
                k: v.lower() if isinstance(v, str) else v
                for k, v in self.data.items()
            }
            self._processed = True
        return self
    
    def is_valid(self) -> bool:
        return len(self.errors) == 0

# Usage example with real data
raw_records = [
    {"email": "john@example.com", "age": 30},
    {"email": "invalid-email", "age": 200},
    {"email": "jane@example.com", "age": 25}
]

processed_records = [
    (DataRecord(record)
     .validate_email("email")
     .validate_age("age")
     .transform())
    for record in raw_records
]

for record in processed_records:
    if record.is_valid():
        print(f"Valid record: {record.data}")
    else:
        print(f"Invalid record: {record.errors}")
```

Slide 12: Real-world Example - Configuration Builder

This implementation demonstrates a practical configuration system using Self type annotations to create a type-safe, chainable API for building complex configuration objects.

```python
from __future__ import annotations
from typing import Self, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class DatabaseConfig:
    host: str
    port: int
    username: str
    password: str
    database: str

@dataclass
class CacheConfig:
    enabled: bool
    ttl: int
    max_size: int

class AppConfiguration:
    def __init__(self) -> None:
        self._db_config: Optional[DatabaseConfig] = None
        self._cache_config: Optional[CacheConfig] = None
        self._debug_mode: bool = False
        self._log_level: str = "INFO"
    
    def with_database(
        self, 
        host: str,
        port: int,
        username: str,
        password: str,
        database: str
    ) -> Self:
        self._db_config = DatabaseConfig(
            host=host,
            port=port,
            username=username,
            password=password,
            database=database
        )
        return self
    
    def with_cache(
        self,
        enabled: bool = True,
        ttl: int = 3600,
        max_size: int = 1000
    ) -> Self:
        self._cache_config = CacheConfig(
            enabled=enabled,
            ttl=ttl,
            max_size=max_size
        )
        return self
    
    def in_debug_mode(self) -> Self:
        self._debug_mode = True
        self._log_level = "DEBUG"
        return self
    
    def build(self) -> Dict[str, Any]:
        if not self._db_config:
            raise ValueError("Database configuration is required")
            
        return {
            "database": {
                "host": self._db_config.host,
                "port": self._db_config.port,
                "username": self._db_config.username,
                "password": self._db_config.password,
                "database": self._db_config.database
            },
            "cache": {
                "enabled": self._cache_config.enabled if self._cache_config else False,
                "ttl": self._cache_config.ttl if self._cache_config else 0,
                "max_size": self._cache_config.max_size if self._cache_config else 0
            },
            "debug": self._debug_mode,
            "log_level": self._log_level
        }
    
    def save_to_file(self, path: Path) -> Self:
        config = self.build()
        path.write_text(json.dumps(config, indent=2))
        return self

# Usage example
config = (AppConfiguration()
          .with_database(
              host="localhost",
              port=5432,
              username="admin",
              password="secret",
              database="myapp"
          )
          .with_cache(
              enabled=True,
              ttl=1800,
              max_size=2000
          )
          .in_debug_mode()
          .build())

print(json.dumps(config, indent=2))
```

Slide 13: Additional Resources

List of relevant research papers and technical specifications:

*   [https://arxiv.org/abs/2208.09032](https://arxiv.org/abs/2208.09032) "Type Hints in Python: Syntax, Semantics, and Tools"
*   [https://arxiv.org/abs/2010.05189](https://arxiv.org/abs/2010.05189) "Gradual Typing for Python, A Summary"
*   [https://arxiv.org/abs/1807.02488](https://arxiv.org/abs/1807.02488) "Python Types in the Wild: A Tale of Two Type Systems"
*   [https://arxiv.org/abs/1904.11694](https://arxiv.org/abs/1904.11694) "Static Typing in Python: Use Cases and Implementation"
*   [https://arxiv.org/abs/2203.03460](https://arxiv.org/abs/2203.03460) "Type System Design for Large-Scale Python Applications"

