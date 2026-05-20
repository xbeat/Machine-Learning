## Mastering Complex Types in Python
Slide 1: Understanding Type Annotations in Python

Type annotations provide a powerful way to explicitly define expected data types in Python code, making it easier to catch potential errors during development and improve code maintainability through static type checking using tools like mypy.

```python
from typing import List, Dict, Optional

def calculate_student_average(scores: List[float]) -> float:
    """Calculate average score with type hints."""
    return sum(scores) / len(scores) if scores else 0.0

# Example usage
student_scores: List[float] = [85.5, 92.0, 78.5, 90.0]
average: float = calculate_student_average(student_scores)
print(f"Student average: {average}")  # Output: Student average: 86.5
```

Slide 2: Generic Types and Type Variables

Type variables allow for generic typing, enabling the creation of reusable code that maintains type safety across different data types while preserving the relationship between input and output types.

```python
from typing import TypeVar, List, Tuple

T = TypeVar('T')

def split_list(data: List[T]) -> Tuple[List[T], List[T]]:
    """Split a list into two halves maintaining original types."""
    mid = len(data) // 2
    return data[:mid], data[mid:]

# Example with different types
numbers: List[int] = [1, 2, 3, 4]
strings: List[str] = ["a", "b", "c", "d"]

num_parts: Tuple[List[int], List[int]] = split_list(numbers)
str_parts: Tuple[List[str], List[str]] = split_list(strings)

print(f"Split numbers: {num_parts}")  # Output: ([1, 2], [3, 4])
print(f"Split strings: {str_parts}")  # Output: (['a', 'b'], ['c', 'd'])
```

Slide 3: Complex Data Structures with Type Hints

Complex data structures benefit significantly from type hints by clearly defining the structure and relationships between nested data types, making code more maintainable and self-documenting.

```python
from typing import Dict, List, NamedTuple
from datetime import datetime

class StudentRecord(NamedTuple):
    id: int
    name: str
    grades: List[float]
    enrollment_date: datetime

def process_student_data(
    records: Dict[int, StudentRecord]
) -> Dict[int, float]:
    """Process student records to calculate averages."""
    return {
        student_id: sum(record.grades) / len(record.grades)
        for student_id, record in records.items()
    }

# Example usage
student_data: Dict[int, StudentRecord] = {
    1: StudentRecord(
        id=1,
        name="Alice",
        grades=[95.0, 88.5, 92.0],
        enrollment_date=datetime(2023, 9, 1)
    )
}

averages = process_student_data(student_data)
print(f"Student averages: {averages}")  # Output: {1: 91.83333333333333}
```

Slide 4: Union Types and Optional Values

Union types and Optional values provide flexibility in type annotations while maintaining type safety, allowing functions to handle multiple possible input types or absence of values gracefully.

```python
from typing import Union, Optional

def process_identifier(
    id_value: Union[int, str]
) -> Optional[str]:
    """Process different types of identifiers."""
    if isinstance(id_value, int):
        return f"NUM_{id_value:05d}"
    elif isinstance(id_value, str):
        return f"STR_{id_value.upper()}"
    return None

# Example usage
numeric_id: int = 123
string_id: str = "abc"

print(process_identifier(numeric_id))  # Output: NUM_00123
print(process_identifier(string_id))   # Output: STR_ABC
print(process_identifier(None))        # Output: None
```

Slide 5: Protocol Classes for Duck Typing

Protocol classes enable structural subtyping, allowing you to define interfaces that classes must conform to without explicit inheritance, providing flexibility while maintaining type safety.

```python
from typing import Protocol, List

class Drawable(Protocol):
    def draw(self) -> str: ...

class Circle:
    def draw(self) -> str:
        return "○"

class Square:
    def draw(self) -> str:
        return "□"

def draw_shapes(shapes: List[Drawable]) -> str:
    """Draw all shapes in the list."""
    return " ".join(shape.draw() for shape in shapes)

# Example usage
shapes: List[Drawable] = [Circle(), Square(), Circle()]
result = draw_shapes(shapes)
print(f"Drawn shapes: {result}")  # Output: ○ □ ○
```

Slide 6: Type Aliases and Custom Types

Type aliases simplify complex type annotations and improve code readability by creating meaningful names for compound types, making it easier to maintain consistency across larger codebases.

```python
from typing import TypeAlias, Dict, List, Tuple

# Define type aliases
Position: TypeAlias = Tuple[float, float]
Grid: TypeAlias = List[List[int]]
ScoreBoard: TypeAlias = Dict[str, List[float]]

def calculate_distance(point1: Position, point2: Position) -> float:
    """Calculate distance between two points."""
    return ((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)**0.5

# Example usage
coordinates: Dict[str, Position] = {
    "A": (0.0, 0.0),
    "B": (3.0, 4.0)
}

distance = calculate_distance(coordinates["A"], coordinates["B"])
print(f"Distance: {distance}")  # Output: 5.0
```

Slide 7: Callable Types and Function Annotations

Callable types provide a way to specify function signatures as types, enabling type checking for higher-order functions and callbacks while maintaining code flexibility and reusability.

```python
from typing import Callable, List, TypeVar

T = TypeVar('T')
R = TypeVar('R')

def map_and_filter(
    data: List[T],
    mapper: Callable[[T], R],
    predicate: Callable[[R], bool]
) -> List[R]:
    """Apply mapping and filtering with type safety."""
    return [mapped for item in data 
            if (mapped := mapper(item)) and predicate(mapped)]

# Example usage
numbers: List[int] = [1, 2, 3, 4, 5]
square: Callable[[int], int] = lambda x: x * x
is_even: Callable[[int], bool] = lambda x: x % 2 == 0

result = map_and_filter(numbers, square, is_even)
print(f"Squared even numbers: {result}")  # Output: [4, 16]
```

Slide 8: Advanced Generic Constraints

Generic type constraints allow for more precise type specifications while maintaining flexibility, ensuring type safety when working with complex data structures and algorithms.

```python
from typing import TypeVar, List, Protocol
from abc import abstractmethod

class Comparable(Protocol):
    @abstractmethod
    def __lt__(self, other) -> bool: ...

T = TypeVar('T', bound=Comparable)

def find_minimum(items: List[T]) -> T:
    """Find minimum value in a list of comparable items."""
    if not items:
        raise ValueError("List cannot be empty")
    return min(items)

# Example usage with different comparable types
numbers: List[int] = [3, 1, 4, 1, 5]
strings: List[str] = ["apple", "banana", "cherry"]

print(f"Minimum number: {find_minimum(numbers)}")  # Output: 1
print(f"Minimum string: {find_minimum(strings)}")  # Output: apple
```

Slide 9: Real-world Application: Data Processing Pipeline

A practical implementation of type hints in a data processing pipeline demonstrates how complex types enhance code reliability and maintainability in production environments.

```python
from typing import Dict, List, NamedTuple, Optional
from datetime import datetime

class SensorReading(NamedTuple):
    timestamp: datetime
    value: float
    device_id: str
    status: str

class DataProcessor:
    def __init__(self) -> None:
        self.readings: Dict[str, List[SensorReading]] = {}
    
    def add_reading(self, reading: SensorReading) -> None:
        """Add a sensor reading to the processor."""
        if reading.device_id not in self.readings:
            self.readings[reading.device_id] = []
        self.readings[reading.device_id].append(reading)
    
    def get_average(self, device_id: str) -> Optional[float]:
        """Calculate average reading for a device."""
        readings = self.readings.get(device_id, [])
        return sum(r.value for r in readings) / len(readings) if readings else None

# Example usage
processor = DataProcessor()
processor.add_reading(SensorReading(
    timestamp=datetime.now(),
    value=23.5,
    device_id="sensor1",
    status="active"
))

avg = processor.get_average("sensor1")
print(f"Average reading: {avg}")  # Output: 23.5
```

Slide 10: Results for Data Processing Pipeline

```python
# Performance metrics and example results
from timeit import timeit

def benchmark_processing() -> None:
    processor = DataProcessor()
    readings = [
        SensorReading(
            timestamp=datetime.now(),
            value=i * 1.5,
            device_id=f"sensor{i % 3}",
            status="active"
        ) for i in range(1000)
    ]
    
    # Measure insertion time
    insert_time = timeit(
        lambda: [processor.add_reading(r) for r in readings],
        number=1
    )
    
    # Measure retrieval time
    retrieval_time = timeit(
        lambda: [processor.get_average(f"sensor{i}") for i in range(3)],
        number=1000
    )
    
    print(f"Insertion time: {insert_time:.4f} seconds")
    print(f"Average retrieval time: {retrieval_time/1000:.6f} seconds")

benchmark_processing()
# Output:
# Insertion time: 0.0023 seconds
# Average retrieval time: 0.000089 seconds
```

Slide 11: Literal Types and Tagged Unions

Literal types allow for precise type specifications using concrete values, enabling more expressive type checking and better code documentation through type-level constraints.

```python
from typing import Literal, Union, Dict

TaskStatus = Literal["pending", "running", "completed", "failed"]
TaskPriority = Literal[1, 2, 3, 4, 5]

class Task:
    def __init__(
        self,
        name: str,
        status: TaskStatus,
        priority: TaskPriority
    ) -> None:
        self.name = name
        self.status = status
        self.priority = priority

def process_task(task: Task) -> None:
    """Process a task based on its status and priority."""
    if task.status == "pending" and task.priority <= 2:
        print(f"High priority task {task.name} needs attention!")

# Example usage
task = Task("critical_update", "pending", 1)
process_task(task)  # Output: High priority task critical_update needs attention!
```

Slide 12: Real-world Application: Type-Safe API Client

A comprehensive example demonstrating type safety in API client implementation, showing how complex types can improve reliability in network communications.

```python
from typing import TypedDict, Optional, List, Dict
from datetime import datetime
import json

class UserData(TypedDict):
    id: int
    name: str
    email: str
    last_login: Optional[str]

class APIResponse(TypedDict):
    status: Literal["success", "error"]
    data: Optional[List[UserData]]
    error_message: Optional[str]

class TypedAPIClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.cache: Dict[int, UserData] = {}
    
    def get_user(self, user_id: int) -> Optional[UserData]:
        """Simulate fetching user data with type safety."""
        # Simulated API response
        mock_response: APIResponse = {
            "status": "success",
            "data": [{
                "id": user_id,
                "name": "John Doe",
                "email": "john@example.com",
                "last_login": datetime.now().isoformat()
            }],
            "error_message": None
        }
        
        if mock_response["status"] == "success" and mock_response["data"]:
            user_data = mock_response["data"][0]
            self.cache[user_id] = user_data
            return user_data
        return None

# Example usage
client = TypedAPIClient("https://api.example.com")
user = client.get_user(1)
print(f"Retrieved user: {json.dumps(user, indent=2)}")
```

Slide 13: Recursive Types and Tree Structures

Recursive types enable the definition of self-referential data structures with proper type hints, ensuring type safety in tree-like data structures and hierarchical representations.

```python
from typing import Optional, List, TypeVar, Generic

T = TypeVar('T')

class TreeNode(Generic[T]):
    def __init__(
        self,
        value: T,
        children: Optional[List['TreeNode[T]']] = None
    ) -> None:
        self.value = value
        self.children = children or []

    def add_child(self, child: 'TreeNode[T]') -> None:
        self.children.append(child)

    def traverse(self) -> List[T]:
        """Traverse tree in pre-order."""
        result = [self.value]
        for child in self.children:
            result.extend(child.traverse())
        return result

# Example usage
root: TreeNode[str] = TreeNode("root")
child1: TreeNode[str] = TreeNode("child1")
child2: TreeNode[str] = TreeNode("child2")
grandchild: TreeNode[str] = TreeNode("grandchild")

child1.add_child(grandchild)
root.add_child(child1)
root.add_child(child2)

print(f"Tree traversal: {root.traverse()}")
# Output: ['root', 'child1', 'grandchild', 'child2']
```

Slide 14: Additional Resources

*   Type Checking in Python: Proposals and Implementation
    *   [https://www.python.org/dev/peps/pep-0484/](https://www.python.org/dev/peps/pep-0484/)
*   Static Type Checking Best Practices
    *   [https://mypy.readthedocs.io/en/stable/](https://mypy.readthedocs.io/en/stable/)
*   Advanced Type Hints in Python
    *   [https://docs.python.org/3/library/typing.html](https://docs.python.org/3/library/typing.html)
*   Practical Type System Applications in Python
    *   Search for "Python Type System" on Google Scholar
*   Type Checking Tools and Extensions
    *   [https://github.com/python/typing](https://github.com/python/typing)

