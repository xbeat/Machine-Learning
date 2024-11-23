## Discover Complex Types in Python
Slide 1: Understanding Complex Types in Python

Complex types in Python extend beyond basic data structures by allowing explicit type hints and annotations. This modern approach to Python programming enables better code organization, enhanced IDE support, and improved debugging capabilities through static type checking.

```python
from typing import Dict, List, Union, Optional

# Define complex types for student records
StudentScores = Dict[str, List[int]]
GradeReport = Dict[str, Union[float, str]]

def process_grades(scores: StudentScores) -> GradeReport:
    report: GradeReport = {}
    for student, grades in scores.items():
        avg = sum(grades) / len(grades)
        report[student] = {
            'average': avg,
            'status': 'Pass' if avg >= 60 else 'Fail'
        }
    return report

# Example usage
scores: StudentScores = {
    'Alice': [85, 92, 78],
    'Bob': [75, 68, 90]
}
result = process_grades(scores)
print(result)
```

Slide 2: Type Aliases and Custom Types

Type aliases provide a way to create meaningful, reusable type definitions that enhance code readability and maintainability. They allow developers to define complex type combinations once and reference them throughout the codebase.

```python
from typing import TypeVar, Callable, List, Tuple

# Define type aliases for complex data structures
T = TypeVar('T')
Matrix = List[List[float]]
Vector = List[float]
TransformFunction = Callable[[T], T]

def matrix_operation(
    matrix: Matrix,
    transform: TransformFunction[Vector]
) -> Matrix:
    return [transform(row) for row in matrix]

# Example usage
def scale_vector(vector: Vector) -> Vector:
    return [x * 2 for x in vector]

data: Matrix = [[1.0, 2.0], [3.0, 4.0]]
result = matrix_operation(data, scale_vector)
print(result)  # [[2.0, 4.0], [6.0, 8.0]]
```

Slide 3: Generic Types and Constraints

Generic types enable the creation of flexible, reusable components while maintaining type safety. This advanced feature allows developers to write code that works with multiple types while preserving type information throughout the program.

```python
from typing import TypeVar, List, Callable
from typing_extensions import Bound

T = TypeVar('T', bound='Comparable')

class Comparable:
    def __lt__(self, other: 'Comparable') -> bool:
        raise NotImplementedError

def sorted_data(data: List[T], key: Callable[[T], float]) -> List[T]:
    return sorted(data, key=key)

class DataPoint(Comparable):
    def __init__(self, value: float):
        self.value = value
    
    def __lt__(self, other: 'DataPoint') -> bool:
        return self.value < other.value

# Example usage
points = [DataPoint(1.5), DataPoint(0.5), DataPoint(2.0)]
sorted_points = sorted_data(points, lambda x: x.value)
```

Slide 4: Protocol Classes and Structural Subtyping

Python's Protocol classes enable structural subtyping, allowing objects to be validated based on their structure rather than explicit inheritance. This powerful feature supports duck typing while maintaining type safety.

```python
from typing import Protocol, List
from datetime import datetime

class Loggable(Protocol):
    def log(self, message: str) -> None: ...

class DatabaseLogger:
    def log(self, message: str) -> None:
        print(f"[DB] {datetime.now()}: {message}")

class FileLogger:
    def log(self, message: str) -> None:
        print(f"[File] {datetime.now()}: {message}")

def process_logs(logger: Loggable, messages: List[str]) -> None:
    for msg in messages:
        logger.log(msg)

# Both loggers work without explicit inheritance
db_logger = DatabaseLogger()
file_logger = FileLogger()
messages = ["Error occurred", "Process completed"]
process_logs(db_logger, messages)
process_logs(file_logger, messages)
```

Slide 5: Type Guards and Runtime Checks

Type guards enhance type safety by performing runtime checks that help narrow down variable types. This pattern is particularly useful when working with union types and enables the type checker to make better inferences about your code.

```python
from typing import Union, TypeGuard, List, Dict

def is_list_of_strings(data: List[object]) -> TypeGuard[List[str]]:
    return all(isinstance(x, str) for x in data)

def process_data(input_data: Union[List[str], Dict[str, str]]) -> str:
    if isinstance(input_data, list):
        if is_list_of_strings(input_data):
            return " ".join(input_data)
    else:
        return ", ".join(f"{k}:{v}" for k, v in input_data.items())
    raise ValueError("Invalid input type")

# Example usage
list_data: List[str] = ["Hello", "World"]
dict_data: Dict[str, str] = {"name": "John", "age": "30"}
print(process_data(list_data))  # Output: Hello World
print(process_data(dict_data))  # Output: name:John, age:30
```

Slide 6: Recursive Types and Data Structures

Recursive types enable the definition of complex data structures that reference themselves. This is particularly useful when working with tree-like structures, nested JSON data, or hierarchical organizations.

```python
from typing import Optional, List, Dict
from dataclasses import dataclass

@dataclass
class TreeNode:
    value: str
    children: List['TreeNode']

@dataclass
class NestedJSON:
    data: Dict[str, Union[str, int, 'NestedJSON']]

def process_tree(node: TreeNode, depth: int = 0) -> None:
    print("  " * depth + node.value)
    for child in node.children:
        process_tree(child, depth + 1)

# Example usage
tree = TreeNode("root", [
    TreeNode("child1", [
        TreeNode("grandchild1", [])
    ]),
    TreeNode("child2", [])
])

process_tree(tree)
```

Slide 7: Advanced Type Constraints with Literal Types

Literal types allow for precise type specifications by constraining values to specific constants. This feature is particularly useful for creating type-safe APIs and ensuring that only valid values are passed to functions.

```python
from typing import Literal, Union, Dict
from dataclasses import dataclass

HttpMethod = Literal["GET", "POST", "PUT", "DELETE"]
StatusCode = Literal[200, 201, 400, 401, 403, 404, 500]

@dataclass
class HttpResponse:
    status: StatusCode
    body: Dict[str, str]

def make_request(
    method: HttpMethod,
    endpoint: str
) -> HttpResponse:
    # Simulated API request
    if method == "GET":
        return HttpResponse(200, {"message": "Success"})
    elif method == "POST":
        return HttpResponse(201, {"message": "Created"})
    else:
        return HttpResponse(404, {"message": "Not Found"})

# Example usage
response = make_request("GET", "/api/users")
print(f"Status: {response.status}, Body: {response.body}")
```

Slide 8: Type-Safe Event Systems

Implementation of a type-safe event system demonstrates how complex types can enhance the reliability of event-driven architectures while maintaining flexibility and extensibility.

```python
from typing import TypeVar, Generic, Callable, Dict, List
from dataclasses import dataclass

T = TypeVar('T')

@dataclass
class Event(Generic[T]):
    type: str
    payload: T

class EventEmitter:
    def __init__(self):
        self._handlers: Dict[str, List[Callable[[Event[T]], None]]] = {}

    def on(self, event_type: str, handler: Callable[[Event[T]], None]) -> None:
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def emit(self, event: Event[T]) -> None:
        handlers = self._handlers.get(event.type, [])
        for handler in handlers:
            handler(event)

# Example usage
@dataclass
class UserData:
    id: int
    name: str

emitter = EventEmitter()
def user_created_handler(event: Event[UserData]) -> None:
    print(f"User created: {event.payload.name}")

emitter.on("user_created", user_created_handler)
emitter.emit(Event("user_created", UserData(1, "John Doe")))
```

Slide 9: Generic Data Validation Framework

A type-safe validation framework demonstrates how complex types can be used to create robust data validation systems. This implementation shows how to handle different data types while maintaining strict type checking.

```python
from typing import Generic, TypeVar, Callable, List, Optional
from dataclasses import dataclass

T = TypeVar('T')

@dataclass
class ValidationError:
    field: str
    message: str

class Validator(Generic[T]):
    def __init__(self, rules: List[Callable[[T], Optional[str]]]):
        self.rules = rules

    def validate(self, value: T) -> List[ValidationError]:
        errors: List[ValidationError] = []
        for rule in self.rules:
            if error := rule(value):
                errors.append(ValidationError("value", error))
        return errors

# Example usage
def length_rule(min_length: int) -> Callable[[str], Optional[str]]:
    def validate(value: str) -> Optional[str]:
        if len(value) < min_length:
            return f"Length must be at least {min_length}"
        return None
    return validate

string_validator = Validator([
    length_rule(5),
    lambda x: "Invalid characters" if not x.isalnum() else None
])

result = string_validator.validate("abc")
print(result)  # [ValidationError(field='value', message='Length must be at least 5')]
```

Slide 10: Advanced Type Composition with Overloads

Function overloading in Python enables creating APIs that can handle different input types while maintaining type safety. This pattern is particularly useful when a function needs to behave differently based on input types.

```python
from typing import overload, Union, List, Dict, TypeVar, Optional
from datetime import datetime

T = TypeVar('T')

class DataProcessor:
    @overload
    def process(self, data: List[int]) -> int: ...
    
    @overload
    def process(self, data: List[str]) -> str: ...
    
    @overload
    def process(self, data: Dict[str, T]) -> Optional[T]: ...

    def process(self, data: Union[List[int], List[str], Dict[str, T]]) -> Union[int, str, Optional[T]]:
        if isinstance(data, list):
            if all(isinstance(x, int) for x in data):
                return sum(data)
            return " ".join(str(x) for x in data)
        else:
            return next(iter(data.values())) if data else None

# Example usage
processor = DataProcessor()
numbers = [1, 2, 3, 4, 5]
strings = ["Hello", "World"]
dict_data: Dict[str, datetime] = {"now": datetime.now()}

print(processor.process(numbers))      # 15
print(processor.process(strings))      # Hello World
print(processor.process(dict_data))    # 2024-01-01 00:00:00
```

Slide 11: Type-Safe State Management

Implementation of a type-safe state management system that enforces type safety for complex application states while allowing for immutable updates and state transitions.

```python
from typing import TypeVar, Generic, Dict, Any, Callable
from dataclasses import dataclass
from copy import deepcopy

S = TypeVar('S')
A = TypeVar('A')

@dataclass
class State(Generic[S]):
    value: S

class Store(Generic[S]):
    def __init__(self, initial_state: S):
        self._state = State(initial_state)
        self._listeners: List[Callable[[S], None]] = []

    def get_state(self) -> S:
        return deepcopy(self._state.value)

    def dispatch(self, action: Callable[[S], S]) -> None:
        new_state = action(self._state.value)
        self._state.value = new_state
        for listener in self._listeners:
            listener(new_state)

    def subscribe(self, listener: Callable[[S], None]) -> Callable[[], None]:
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)

# Example usage
@dataclass
class AppState:
    count: int
    name: str

def increment_counter(state: AppState) -> AppState:
    return AppState(state.count + 1, state.name)

store = Store(AppState(0, "Initial"))
unsubscribe = store.subscribe(lambda state: print(f"State updated: {state}"))

store.dispatch(increment_counter)
store.dispatch(lambda s: AppState(s.count, "Updated"))
unsubscribe()
```

Slide 12: Type-Safe API Client Framework

A type-safe API client framework demonstrates how to build robust HTTP clients with complex type definitions, ensuring type safety across network boundaries and response handling.

```python
from typing import Generic, TypeVar, Dict, Any, Optional
from dataclasses import dataclass
import json
from datetime import datetime

T = TypeVar('T')
ResponseT = TypeVar('ResponseT')

@dataclass
class APIResponse(Generic[T]):
    data: T
    status: int
    timestamp: datetime

class APIClient(Generic[ResponseT]):
    def __init__(self, base_url: str):
        self.base_url = base_url
        
    async def request(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None
    ) -> APIResponse[ResponseT]:
        # Simulated API request
        response_data = {
            "id": 1,
            "name": "Example",
            "timestamp": datetime.now().isoformat()
        }
        return APIResponse(
            data=self._parse_response(response_data),
            status=200,
            timestamp=datetime.now()
        )
    
    def _parse_response(self, data: Dict[str, Any]) -> ResponseT:
        # Type-safe response parsing
        return data  # type: ignore

# Example usage
@dataclass
class UserResponse:
    id: int
    name: str
    created_at: datetime

client: APIClient[UserResponse] = APIClient("https://api.example.com")
response = await client.request("/users/1")
print(f"User {response.data.name} fetched at {response.timestamp}")
```

Slide 13: Modular Type Registry System

Implementation of a type registry system that allows for dynamic registration and retrieval of typed components while maintaining type safety throughout the application.

```python
from typing import TypeVar, Dict, Type, Generic, Callable, Any
from dataclasses import dataclass
import inspect

T = TypeVar('T')

class TypeRegistry:
    def __init__(self):
        self._registry: Dict[str, Type[Any]] = {}
        
    def register(self, type_class: Type[T]) -> None:
        if not inspect.isclass(type_class):
            raise ValueError("Only classes can be registered")
        self._registry[type_class.__name__] = type_class
    
    def get(self, type_name: str) -> Type[T]:
        if type_name not in self._registry:
            raise KeyError(f"Type {type_name} not found in registry")
        return self._registry[type_name]
    
    def create(self, type_name: str, **kwargs: Any) -> T:
        type_class = self.get(type_name)
        return type_class(**kwargs)

# Example usage
@dataclass
class User:
    name: str
    age: int

@dataclass
class Product:
    id: str
    price: float

registry = TypeRegistry()
registry.register(User)
registry.register(Product)

user = registry.create("User", name="John", age=30)
product = registry.create("Product", id="123", price=99.99)

print(f"Created user: {user}")
print(f"Created product: {product}")
```

Slide 14: Additional Resources

*   "Type Theory and Formal Proof: An Introduction" - [https://arxiv.org/abs/1405.7850](https://arxiv.org/abs/1405.7850)
*   "Gradual Typing for Python" - [https://arxiv.org/abs/2010.12931](https://arxiv.org/abs/2010.12931)
*   "Static Type Inference for Python" - [https://arxiv.org/abs/1901.07098](https://arxiv.org/abs/1901.07098)
*   Search Google for: "Advanced Python Type Hints PEP 484"
*   Python Type Checking Documentation: [https://docs.python.org/3/library/typing.html](https://docs.python.org/3/library/typing.html)
*   MyPy Documentation: [https://mypy.readthedocs.io/](https://mypy.readthedocs.io/)

