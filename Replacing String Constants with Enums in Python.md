## Replacing String Constants with Enums in Python
Slide 1: String Constants vs Enums

Traditionally, Python developers use string constants to represent fixed values, leading to potential typos and maintenance challenges. String constants lack type safety and validation, making code more prone to runtime errors that could be caught earlier with proper enumeration.

```python
# Traditional approach with string constants
STATUS_PENDING = 'pending'
STATUS_ACTIVE = 'active'
STATUS_COMPLETED = 'completed'

# Problematic usage
current_status = 'pendng'  # Typo won't be caught
print(f"Status: {current_status}")  # Prints wrong value
```

Slide 2: Basic Enum Implementation

The Enum class provides a more robust way to represent a fixed set of symbolic names bound to unique values. This implementation demonstrates the fundamental usage of Python's built-in Enum class from the enum module.

```python
from enum import Enum

class Status(Enum):
    PENDING = 'pending'
    ACTIVE = 'active'
    COMPLETED = 'completed'

# Safe usage with type checking
current_status = Status.PENDING
print(f"Status: {current_status.value}")  # Output: Status: pending

# Attempting to use invalid status
try:
    # This will raise AttributeError
    wrong_status = Status.PENDNG
except AttributeError as e:
    print(f"Error caught: {e}")
```

Slide 3: Auto-value Assignment

The auto() function in enum module automatically assigns incremental values to enum members, reducing the burden of manual value assignment and maintaining uniqueness without explicit declarations.

```python
from enum import Enum, auto

class Priority(Enum):
    LOW = auto()      # Assigns 1
    MEDIUM = auto()   # Assigns 2
    HIGH = auto()     # Assigns 3
    CRITICAL = auto() # Assigns 4

print(Priority.LOW.value)     # Output: 1
print(Priority.MEDIUM.name)   # Output: MEDIUM
print(list(Priority))        # Output: [Priority.LOW, Priority.MEDIUM, ...]
```

Slide 4: Enum with Custom Methods

Enums can include methods and properties, making them more powerful than simple constants. This example shows how to create an enum with business logic embedded within the enumeration itself.

```python
from enum import Enum
from datetime import timedelta

class TaskStatus(Enum):
    NEW = 1
    IN_PROGRESS = 2
    REVIEW = 3
    COMPLETED = 4

    def get_sla_duration(self) -> timedelta:
        sla_mapping = {
            TaskStatus.NEW: timedelta(hours=24),
            TaskStatus.IN_PROGRESS: timedelta(hours=48),
            TaskStatus.REVIEW: timedelta(hours=12),
            TaskStatus.COMPLETED: timedelta(hours=0)
        }
        return sla_mapping[self]

status = TaskStatus.NEW
print(f"SLA Duration: {status.get_sla_duration()}")  # Output: SLA Duration: 1 day, 0:00:00
```

Slide 5: Unique Enum Values

Understanding how to enforce unique values in enums is crucial for maintaining data integrity. This example demonstrates various approaches to ensure value uniqueness.

```python
from enum import Enum, unique

@unique
class HttpStatus(Enum):
    OK = 200
    CREATED = 201
    ACCEPTED = 202
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    NOT_FOUND = 404

    @classmethod
    def is_success(cls, code):
        return 200 <= code < 300

print(HttpStatus.OK.value)  # Output: 200
print(HttpStatus.is_success(201))  # Output: True
```

Slide 6: Enum Comparison and Iteration

Enums support rich comparison operations and can be iterated over, making them ideal for control flow and validation scenarios in Python applications.

```python
from enum import Enum

class ProcessState(Enum):
    WAITING = 1
    READY = 2
    RUNNING = 3
    TERMINATED = 4

    def can_transition_to(self, new_state):
        valid_transitions = {
            ProcessState.WAITING: [ProcessState.READY],
            ProcessState.READY: [ProcessState.RUNNING],
            ProcessState.RUNNING: [ProcessState.TERMINATED, ProcessState.WAITING],
            ProcessState.TERMINATED: []
        }
        return new_state in valid_transitions[self]

# Example usage
current = ProcessState.WAITING
next_state = ProcessState.READY

print(f"Can transition: {current.can_transition_to(next_state)}")  # Output: True
print(f"All states: {[state.name for state in ProcessState]}")
```

Slide 7: Flag Enums with IntFlag

IntFlag provides a way to create enumerated constants that can be combined using bitwise operations, useful for representing multiple flags or options simultaneously.

```python
from enum import IntFlag, auto

class Permissions(IntFlag):
    NONE = 0
    READ = auto()      # 1
    WRITE = auto()     # 2
    EXECUTE = auto()   # 4
    ADMIN = READ | WRITE | EXECUTE  # 7

# Usage example
user_permission = Permissions.READ | Permissions.WRITE
print(f"Has read access: {Permissions.READ in user_permission}")  # True
print(f"Has admin rights: {Permissions.ADMIN in user_permission}")  # False
print(f"Permission value: {user_permission}")  # Permissions.READ|WRITE
```

Slide 8: Enum with Custom Properties

By implementing properties in enums, we can add computed attributes that dynamically change based on the enum value, making them more powerful for complex business logic.

```python
from enum import Enum
from datetime import datetime, timedelta

class OrderStatus(Enum):
    PLACED = 'placed'
    PROCESSING = 'processing'
    SHIPPED = 'shipped'
    DELIVERED = 'delivered'

    @property
    def description(self):
        return f"Order is in {self.value} state"

    @property
    def estimated_completion(self):
        base_time = datetime.now()
        delays = {
            OrderStatus.PLACED: timedelta(hours=1),
            OrderStatus.PROCESSING: timedelta(hours=24),
            OrderStatus.SHIPPED: timedelta(days=3),
            OrderStatus.DELIVERED: timedelta(hours=0)
        }
        return base_time + delays[self]

status = OrderStatus.PROCESSING
print(status.description)  # Output: Order is in processing state
print(f"Estimated completion: {status.estimated_completion}")
```

Slide 9: Real-world Example: Payment Processing

This example demonstrates how enums can be used in a payment processing system to manage transaction states and ensure type safety throughout the application.

```python
from enum import Enum, auto
from datetime import datetime
from typing import Optional

class PaymentStatus(Enum):
    INITIATED = auto()
    PENDING = auto()
    AUTHORIZED = auto()
    CAPTURED = auto()
    FAILED = auto()
    REFUNDED = auto()

class PaymentProcessor:
    def __init__(self, amount: float):
        self.amount = amount
        self.status = PaymentStatus.INITIATED
        self.processed_at: Optional[datetime] = None
        
    def process_payment(self) -> bool:
        try:
            self.status = PaymentStatus.PENDING
            # Simulate payment processing
            self.status = PaymentStatus.AUTHORIZED
            self.processed_at = datetime.now()
            self.status = PaymentStatus.CAPTURED
            return True
        except Exception:
            self.status = PaymentStatus.FAILED
            return False

# Usage example
payment = PaymentProcessor(99.99)
success = payment.process_payment()
print(f"Payment status: {payment.status.name}")
print(f"Processed at: {payment.processed_at}")
```

Slide 10: Database Integration with Enums

Integrating enums with SQLAlchemy demonstrates how to use enumerated types in database models while maintaining type safety and data integrity.

```python
from enum import Enum
from sqlalchemy import create_engine, Column, Integer, String, Enum as SQLAEnum
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class UserType(Enum):
    REGULAR = "regular"
    PREMIUM = "premium"
    ADMIN = "admin"

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True)
    user_type = Column(SQLAEnum(UserType))
    
    def has_admin_access(self) -> bool:
        return self.user_type == UserType.ADMIN

# Database setup and usage example
engine = create_engine('sqlite:///:memory:')
Base.metadata.create_all(engine)

# Example query using enum
from sqlalchemy.orm import Session
session = Session(engine)
admin_users = session.query(User).filter(User.user_type == UserType.ADMIN).all()
```

Slide 11: API Response Status Handling

Using enums for API response status codes provides a type-safe way to handle different response scenarios while maintaining consistency across the application. This pattern is particularly useful in REST APIs.

```python
from enum import Enum
from dataclasses import dataclass
from typing import Any, Optional

class ResponseStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    RATE_LIMITED = "rate_limited"
    
    @property
    def is_successful(self) -> bool:
        return self == ResponseStatus.SUCCESS

@dataclass
class APIResponse:
    status: ResponseStatus
    data: Optional[Any] = None
    message: str = ""
    
    def to_dict(self):
        return {
            "status": self.status.value,
            "data": self.data,
            "message": self.message
        }

# Usage example
def process_api_request(request_data: dict) -> APIResponse:
    try:
        # Simulate API processing
        result = {"processed": True}
        return APIResponse(ResponseStatus.SUCCESS, result)
    except Exception as e:
        return APIResponse(ResponseStatus.ERROR, message=str(e))

response = process_api_request({"user_id": 123})
print(response.to_dict())
```

Slide 12: Event Processing System

This implementation shows how enums can be used to create a robust event processing system with type-safe state transitions and validation.

```python
from enum import Enum, auto
from typing import List, Dict, Callable
from datetime import datetime

class EventState(Enum):
    CREATED = auto()
    VALIDATED = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    FAILED = auto()

    def can_transition_to(self, target: 'EventState') -> bool:
        transitions = {
            EventState.CREATED: [EventState.VALIDATED],
            EventState.VALIDATED: [EventState.PROCESSING],
            EventState.PROCESSING: [EventState.COMPLETED, EventState.FAILED],
            EventState.COMPLETED: [],
            EventState.FAILED: [EventState.CREATED]
        }
        return target in transitions.get(self, [])

class Event:
    def __init__(self, name: str):
        self.name = name
        self.state = EventState.CREATED
        self.timestamp = datetime.now()
        self.history: List[Dict] = []
        
    def transition_to(self, new_state: EventState) -> bool:
        if not self.state.can_transition_to(new_state):
            return False
        
        self.history.append({
            "from_state": self.state,
            "to_state": new_state,
            "timestamp": datetime.now()
        })
        self.state = new_state
        return True

# Example usage
event = Event("data_processing")
print(f"Initial state: {event.state.name}")
event.transition_to(EventState.VALIDATED)
print(f"Current state: {event.state.name}")
print(f"State history: {len(event.history)} transitions")
```

Slide 13: Finite State Machine Implementation

This example demonstrates how to implement a finite state machine using enums, providing a robust way to model complex state transitions with validation.

```python
from enum import Enum
from typing import Dict, Set, Optional
from datetime import datetime

class DocumentState(Enum):
    DRAFT = "draft"
    REVIEW = "review"
    APPROVED = "approved"
    PUBLISHED = "published"
    ARCHIVED = "archived"

class Document:
    def __init__(self, title: str):
        self._state = DocumentState.DRAFT
        self._title = title
        self._transitions: Dict[DocumentState, Set[DocumentState]] = {
            DocumentState.DRAFT: {DocumentState.REVIEW},
            DocumentState.REVIEW: {DocumentState.DRAFT, DocumentState.APPROVED},
            DocumentState.APPROVED: {DocumentState.PUBLISHED, DocumentState.REVIEW},
            DocumentState.PUBLISHED: {DocumentState.ARCHIVED},
            DocumentState.ARCHIVED: set()
        }
        self._history = []
        
    @property
    def state(self) -> DocumentState:
        return self._state
    
    def transition_to(self, new_state: DocumentState) -> bool:
        if new_state not in self._transitions[self._state]:
            raise ValueError(f"Invalid transition from {self._state} to {new_state}")
            
        self._history.append({
            'from_state': self._state,
            'to_state': new_state,
            'timestamp': datetime.now()
        })
        self._state = new_state
        return True

# Usage example
doc = Document("Technical Specification")
print(f"Initial state: {doc.state}")
doc.transition_to(DocumentState.REVIEW)
print(f"Current state: {doc.state}")
```

Slide 14: Additional Resources

*   Effective Python Enum Patterns and Best Practices
    *   [https://realpython.com/python-enum/](https://realpython.com/python-enum/)
*   Python Documentation on Enum
    *   [https://docs.python.org/3/library/enum.html](https://docs.python.org/3/library/enum.html)
*   Design Patterns with Python Enums
    *   [https://python-patterns.guide/gang-of-four/state/](https://python-patterns.guide/gang-of-four/state/)
*   Advanced Python Programming: Working with Enums
    *   [https://www.python.org/dev/peps/pep-0435/](https://www.python.org/dev/peps/pep-0435/)
*   Best Practices for Using Enums in Large-Scale Python Applications
    *   [https://pythonspeed.com/articles/python-enums/](https://pythonspeed.com/articles/python-enums/)

