## Leveraging Python Type Annotations
Slide 1: Type Annotations Fundamentals

Python's type annotations provide a way to explicitly declare types for variables, function parameters, and return values. This helps catch type-related bugs early in development and makes code more maintainable by clearly documenting expected types.

```python
from typing import List, Dict, Optional, Union

# Basic type annotations
def calculate_average(numbers: List[float]) -> float:
    """Calculate average of list of numbers"""
    return sum(numbers) / len(numbers)

# Using Union for multiple allowed types
def process_data(value: Union[int, float]) -> str:
    return f"Processed value: {value * 2}"

# Example usage
numbers = [1.5, 2.7, 3.2]
print(calculate_average(numbers))  # Output: 2.466666666666667
print(process_data(5))  # Output: Processed value: 10
```

Slide 2: Type Checking with MyPy

Type checking tools like MyPy analyze code statically to identify potential type-related issues before runtime. MyPy verifies that values match their declared types and helps maintain type consistency throughout the codebase.

```python
from typing import Optional

class User:
    def __init__(self, name: str, age: Optional[int] = None):
        self.name = name
        self.age = age

    def get_info(self) -> str:
        age_info = f", age: {self.age}" if self.age else ""
        return f"User(name: {self.name}{age_info})"

# This will raise a type error when checked with mypy
user1 = User("Alice", "25")  # Type error: Argument 2 has incompatible type "str"; expected "Optional[int]"
user2 = User("Bob", 30)      # Correct usage
```

Slide 3: Generic Types

Generic types enable the creation of flexible, reusable code that maintains type safety. They allow you to write functions and classes that can work with different types while preserving type information throughout the program.

```python
from typing import TypeVar, List, Generic

T = TypeVar('T')

class Stack(Generic[T]):
    def __init__(self) -> None:
        self.items: List[T] = []
    
    def push(self, item: T) -> None:
        self.items.append(item)
    
    def pop(self) -> T:
        return self.items.pop()

# Usage with different types
int_stack: Stack[int] = Stack()
int_stack.push(1)
str_stack: Stack[str] = Stack()
str_stack.push("hello")
```

Slide 4: Protocol Types

Protocols define interfaces through duck typing, allowing for structural subtyping. This provides more flexibility than traditional inheritance while maintaining type safety and enabling better code organization.

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Drawable(Protocol):
    def draw(self) -> str: ...

class Circle:
    def draw(self) -> str:
        return "Drawing circle"

class Square:
    def draw(self) -> str:
        return "Drawing square"

def render(shape: Drawable) -> None:
    print(shape.draw())

# Both classes work because they implement draw()
render(Circle())  # Output: Drawing circle
render(Square())  # Output: Drawing square
```

Slide 5: Type Guards and Narrowing

Type guards enable runtime type checking and type narrowing, allowing for more precise type information in conditional blocks. This helps write safer code when dealing with multiple possible types.

```python
from typing import Union, TypeGuard

def is_string_list(value: list) -> TypeGuard[list[str]]:
    return all(isinstance(x, str) for x in value)

def process_items(items: Union[list[str], list[int]]) -> None:
    if is_string_list(items):
        # Type is narrowed to List[str]
        print(", ".join(items))
    else:
        # Type is narrowed to List[int]
        print(sum(items))

# Example usage
process_items(["a", "b", "c"])  # Output: a, b, c
process_items([1, 2, 3])        # Output: 6
```

Slide 6: Literal Types and Final Declarations

Literal types allow specifying exact values as types, while Final declarations prevent reassignment. These features enhance type safety by restricting possible values and preventing unwanted modifications.

```python
from typing import Literal, Final
from enum import Enum

# Literal types for specific values
Direction = Literal["north", "south", "east", "west"]

def move(direction: Direction) -> str:
    return f"Moving {direction}"

# Final declarations
MAX_ATTEMPTS: Final = 3
USER_ROLES: Final[tuple[str, ...]] = ("admin", "user", "guest")

# Example usage
print(move("north"))  # Valid
# print(move("up"))   # Type error: Argument 1 has incompatible type

# MAX_ATTEMPTS = 4    # Error: Cannot assign to final variable
```

Slide 7: Asynchronous Type Hints

Type hints for asynchronous code require special attention to coroutines and async contexts. Understanding how to properly annotate async functions and their return types is crucial for modern Python applications.

```python
from typing import AsyncIterator, Awaitable
import asyncio
from datetime import datetime

async def fetch_data(url: str) -> bytes:
    """Simulate async data fetch with type hints"""
    await asyncio.sleep(1)  # Simulate network delay
    return f"Data from {url}".encode()

async def process_urls(urls: list[str]) -> AsyncIterator[tuple[str, bytes]]:
    for url in urls:
        data = await fetch_data(url)
        yield url, data

# Example usage
async def main() -> None:
    urls = ["http://api1.com", "http://api2.com"]
    async for url, data in process_urls(urls):
        print(f"{url}: {data.decode()}")

# Run the async code
asyncio.run(main())
```

Slide 8: Real-World Example - Data Processing Pipeline

A practical implementation of a data processing pipeline demonstrates how type hints improve code clarity and maintainability in production environments. This example shows data validation and transformation with proper type annotations.

```python
from dataclasses import dataclass
from typing import Iterator, Optional
from datetime import datetime

@dataclass
class RawTransaction:
    timestamp: str
    amount: str
    currency: str

@dataclass
class ProcessedTransaction:
    timestamp: datetime
    amount: float
    currency: str
    
class TransactionProcessor:
    def validate_and_convert(self, raw: RawTransaction) -> Optional[ProcessedTransaction]:
        try:
            return ProcessedTransaction(
                timestamp=datetime.fromisoformat(raw.timestamp),
                amount=float(raw.amount),
                currency=raw.currency.upper()
            )
        except (ValueError, TypeError):
            return None
    
    def process_batch(self, transactions: list[RawTransaction]) -> Iterator[ProcessedTransaction]:
        for transaction in transactions:
            if processed := self.validate_and_convert(transaction):
                yield processed

# Example usage
raw_data = [
    RawTransaction("2024-01-01T12:00:00", "123.45", "usd"),
    RawTransaction("2024-01-02T15:30:00", "67.89", "eur")
]

processor = TransactionProcessor()
processed_transactions = list(processor.process_batch(raw_data))
for trans in processed_transactions:
    print(f"{trans.timestamp}: {trans.amount} {trans.currency}")
```

Slide 9: Type Aliases and NewType

Type aliases and NewType utilities provide ways to create more semantic type definitions and prevent type confusion in complex systems. This enhances code readability and type safety.

```python
from typing import NewType, Dict, List, TypeAlias

# Type aliases for complex types
JSONDict: TypeAlias = Dict[str, object]
Matrix: TypeAlias = List[List[float]]

# NewType for type-safe IDs
UserId = NewType('UserId', int)
OrderId = NewType('OrderId', int)

def process_user(user_id: UserId) -> None:
    print(f"Processing user: {user_id}")

def process_order(order_id: OrderId) -> None:
    print(f"Processing order: {order_id}")

# Example usage
user_id = UserId(123)
order_id = OrderId(456)

process_user(user_id)    # OK
# process_user(order_id) # Type error
# process_user(123)      # Type error: plain int not accepted

matrix: Matrix = [[1.0, 2.0], [3.0, 4.0]]
```

Slide 10: Advanced Type Constraints

Type constraints allow for more precise type specifications using bounded type variables and complex type relationships. This enables better type checking while maintaining flexibility.

```python
from typing import TypeVar, Protocol, Generic
from numbers import Number

class Comparable(Protocol):
    def __lt__(self, other: object) -> bool: ...

T = TypeVar('T', bound=Comparable)
N = TypeVar('N', bound=Number)

class SortedCollection(Generic[T]):
    def __init__(self) -> None:
        self.items: list[T] = []
    
    def add(self, item: T) -> None:
        self.items.append(item)
        self.items.sort()
    
    def get_sorted(self) -> list[T]:
        return self.items

# Example usage
numbers = SortedCollection[float]()
numbers.add(3.14)
numbers.add(1.41)
print(numbers.get_sorted())  # Output: [1.41, 3.14]

strings = SortedCollection[str]()
strings.add("banana")
strings.add("apple")
print(strings.get_sorted())  # Output: ['apple', 'banana']
```

Slide 11: Type-Safe Data Validation

Type annotations combined with runtime validation ensure data integrity throughout application lifecycle. This implementation demonstrates a robust approach to handling structured data with type checking.

```python
from typing import TypedDict, Callable, Any
from datetime import datetime
import json

class UserData(TypedDict):
    id: int
    name: str
    email: str
    created_at: str

class ValidationError(Exception):
    pass

def validate_email(email: str) -> bool:
    return '@' in email and '.' in email.split('@')[1]

class DataValidator:
    def __init__(self) -> None:
        self.validators: dict[str, Callable[[Any], bool]] = {
            'email': validate_email,
            'created_at': lambda x: bool(datetime.fromisoformat(x))
        }
    
    def validate(self, data: UserData) -> None:
        if not isinstance(data['id'], int):
            raise ValidationError("ID must be an integer")
            
        for field, validator in self.validators.items():
            if not validator(data[field]):
                raise ValidationError(f"Invalid {field}")

# Example usage
test_data: UserData = {
    'id': 1,
    'name': 'John Doe',
    'email': 'john@example.com',
    'created_at': '2024-01-01T00:00:00'
}

validator = DataValidator()
try:
    validator.validate(test_data)
    print("Data validation successful")
except ValidationError as e:
    print(f"Validation failed: {e}")
```

Slide 12: Real-World Example - REST API Type Safety

This implementation showcases type-safe REST API request handling with proper error management and response typing. It demonstrates how type hints improve API reliability and maintainability.

```python
from typing import TypedDict, Optional, Union, Literal
from datetime import datetime
import json
from dataclasses import dataclass

# API Request/Response Types
class CreateUserRequest(TypedDict):
    username: str
    email: str
    role: Literal['admin', 'user', 'guest']

@dataclass
class APIResponse:
    success: bool
    data: Optional[dict]
    error: Optional[str]

class UserService:
    def __init__(self) -> None:
        self.users: dict[str, CreateUserRequest] = {}

    def create_user(self, request: CreateUserRequest) -> APIResponse:
        try:
            if request['username'] in self.users:
                return APIResponse(False, None, "Username already exists")
                
            self.users[request['username']] = request
            return APIResponse(True, 
                             {'username': request['username']}, 
                             None)
        except Exception as e:
            return APIResponse(False, None, str(e))

# Example usage
service = UserService()
new_user: CreateUserRequest = {
    'username': 'johndoe',
    'email': 'john@example.com',
    'role': 'user'
}

response = service.create_user(new_user)
print(f"Success: {response.success}")
if response.data:
    print(f"Data: {response.data}")
if response.error:
    print(f"Error: {response.error}")
```

Slide 13: Custom Container Types

Implementing custom container types with proper type hints enables creation of specialized collections while maintaining type safety and enabling static type checking.

```python
from typing import TypeVar, Generic, Iterator, Optional
from collections.abc import Sequence

T = TypeVar('T')

class CircularBuffer(Generic[T]):
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: list[Optional[T]] = [None] * capacity
        self.head = 0
        self.tail = 0
        self.size = 0

    def push(self, item: T) -> None:
        self.buffer[self.tail] = item
        self.tail = (self.tail + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
        else:
            self.head = (self.head + 1) % self.capacity

    def pop(self) -> Optional[T]:
        if self.size == 0:
            return None
        item = self.buffer[self.head]
        self.head = (self.head + 1) % self.capacity
        self.size -= 1
        return item

    def __iter__(self) -> Iterator[T]:
        idx = self.head
        for _ in range(self.size):
            yield self.buffer[idx]  # type: ignore
            idx = (idx + 1) % self.capacity

# Example usage
buffer: CircularBuffer[int] = CircularBuffer(3)
for i in range(5):
    buffer.push(i)

print("Buffer contents:")
for item in buffer:
    print(item)  # Will print last 3 numbers: 2, 3, 4
```

Slide 14: Additional Resources

*   [https://arxiv.org/abs/2203.03823](https://arxiv.org/abs/2203.03823) - "Type Inference for Python"
*   [https://arxiv.org/abs/2107.04329](https://arxiv.org/abs/2107.04329) - "Gradual Type Theory"
*   [https://arxiv.org/abs/1904.11544](https://arxiv.org/abs/1904.11544) - "Static Typing for Python"
*   [https://arxiv.org/abs/2009.13443](https://arxiv.org/abs/2009.13443) - "Type Systems for Python"
*   [https://arxiv.org/abs/2106.13468](https://arxiv.org/abs/2106.13468) - "Practical Type Inference for Python"

