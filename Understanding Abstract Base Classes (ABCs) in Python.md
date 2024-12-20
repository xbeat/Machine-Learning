## Understanding Abstract Base Classes (ABCs) in Python
Slide 1: Introduction to Abstract Base Classes

Abstract Base Classes (ABCs) provide a way to define interfaces in Python, enforcing a contract that derived classes must fulfill. They act as a blueprint for other classes, establishing a set of methods and properties that concrete implementations must provide.

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass
    
    @abstractmethod
    def perimeter(self):
        pass

# This will raise TypeError if instantiated directly
# shape = Shape()  # TypeError: Can't instantiate abstract class
```

Slide 2: Implementing Abstract Classes

Abstract classes define an interface contract that subclasses must follow. When a class inherits from an abstract base class, it must implement all abstract methods, or it will raise a TypeError when instantiated.

```python
class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)

# Valid instantiation
rect = Rectangle(5, 3)
print(f"Area: {rect.area()}")  # Output: Area: 15
```

Slide 3: Abstract Properties and Methods

Abstract classes can define both abstract methods and abstract properties, requiring implementing classes to provide both behavioral and data interfaces. This ensures complete contract fulfillment.

```python
class Vehicle(ABC):
    @property
    @abstractmethod
    def fuel_type(self):
        pass
    
    @abstractmethod
    def start_engine(self):
        pass

class ElectricCar(Vehicle):
    @property
    def fuel_type(self):
        return "electricity"
    
    def start_engine(self):
        return "Starting electric motor"
```

Slide 4: Multiple Abstract Base Classes

Python supports inheriting from multiple abstract base classes, allowing for complex interface combinations. This enables flexible contract definition while maintaining strict implementation requirements.

```python
class Drawable(ABC):
    @abstractmethod
    def draw(self): pass

class Moveable(ABC):
    @abstractmethod
    def move(self): pass

class GameSprite(Drawable, Moveable):
    def draw(self):
        return "Drawing sprite"
    
    def move(self):
        return "Moving sprite"
```

Slide 5: Real-World Example - Data Processing Pipeline

Abstract base classes excel in defining processing pipelines where different implementations may handle various data types or sources while maintaining a consistent interface.

```python
class DataProcessor(ABC):
    @abstractmethod
    def load_data(self, source):
        pass
    
    @abstractmethod
    def process(self, data):
        pass
    
    @abstractmethod
    def save_result(self, result, destination):
        pass

class CSVProcessor(DataProcessor):
    def load_data(self, source):
        return f"Loading CSV from {source}"
    
    def process(self, data):
        return f"Processing {data}"
    
    def save_result(self, result, destination):
        return f"Saving to {destination}"
```

Slide 6: Abstract Methods with Implementation

Abstract classes can provide default implementations while still requiring method override, offering both flexibility and default behavior when needed.

```python
class DataValidator(ABC):
    @abstractmethod
    def validate(self, data):
        # Default implementation
        if not data:
            return False
        return True
    
class NumericValidator(DataValidator):
    def validate(self, data):
        # Must call super() to use default implementation
        if not super().validate(data):
            return False
        return isinstance(data, (int, float))

validator = NumericValidator()
print(validator.validate(42))  # Output: True
```

Slide 7: Abstract Base Classes with Metaclasses

Understanding metaclasses in abstract base classes provides deeper control over class creation and validation, enabling custom behavior during class definition.

```python
from abc import ABCMeta

class ValidatorMeta(ABCMeta):
    def __new__(mcls, name, bases, namespace):
        for key, value in namespace.items():
            if getattr(value, "_validation_required", False):
                if not hasattr(value, "validate"):
                    raise TypeError(f"{key} must implement validate()")
        return super().__new__(mcls, name, bases, namespace)

class BaseValidator(metaclass=ValidatorMeta):
    pass
```

Slide 8: Design Patterns with ABCs - Observer Pattern

Abstract base classes are fundamental in implementing design patterns. Here's an implementation of the Observer pattern using ABCs.

```python
class Subject(ABC):
    def __init__(self):
        self._observers = []
    
    def attach(self, observer):
        self._observers.append(observer)
    
    def detach(self, observer):
        self._observers.remove(observer)
    
    @abstractmethod
    def notify(self):
        pass

class Observer(ABC):
    @abstractmethod
    def update(self, subject):
        pass
```

Slide 9: Source Code for Observer Pattern Implementation

```python
class ConcreteSubject(Subject):
    def __init__(self):
        super().__init__()
        self._state = None
    
    @property
    def state(self):
        return self._state
    
    @state.setter
    def state(self, value):
        self._state = value
        self.notify()
    
    def notify(self):
        for observer in self._observers:
            observer.update(self)

class ConcreteObserver(Observer):
    def update(self, subject):
        print(f"Observer updated with state: {subject.state}")

# Usage
subject = ConcreteSubject()
observer = ConcreteObserver()
subject.attach(observer)
subject.state = "New State"  # Output: Observer updated with state: New State
```

Slide 10: Template Method Pattern Using ABCs

The Template Method pattern defines an algorithm's skeleton in a base class while letting subclasses override specific steps without changing the algorithm's structure.

```python
class DataMiner(ABC):
    def mine(self, path):
        raw_data = self._extract(path)
        clean_data = self._transform(raw_data)
        return self._load(clean_data)
    
    @abstractmethod
    def _extract(self, path):
        pass
    
    @abstractmethod
    def _transform(self, data):
        pass
    
    @abstractmethod
    def _load(self, data):
        pass
```

Slide 11: Source Code for Template Method Implementation

```python
class PDFMiner(DataMiner):
    def _extract(self, path):
        return f"Extracting PDF data from {path}"
    
    def _transform(self, data):
        return f"Transforming PDF data: {data}"
    
    def _load(self, data):
        return f"Loading transformed PDF data: {data}"

class CSVMiner(DataMiner):
    def _extract(self, path):
        return f"Extracting CSV data from {path}"
    
    def _transform(self, data):
        return f"Transforming CSV data: {data}"
    
    def _load(self, data):
        return f"Loading transformed CSV data: {data}"

# Usage
pdf_miner = PDFMiner()
result = pdf_miner.mine("document.pdf")
print(result)  # Output: Loading transformed PDF data: Transforming PDF data: Extracting PDF data from document.pdf
```

Slide 12: Unit Testing with ABCs

Abstract base classes provide a powerful foundation for unit testing, allowing test cases to verify that concrete implementations satisfy the required interface.

```python
import unittest

class TestDataProcessor(unittest.TestCase):
    def test_processor_implementation(self):
        class TestProcessor(DataProcessor):
            def load_data(self, source): return "data"
            def process(self, data): return "processed"
            def save_result(self, result, dest): return "saved"
        
        processor = TestProcessor()
        self.assertTrue(isinstance(processor, DataProcessor))
        self.assertEqual(processor.load_data("test"), "data")

if __name__ == '__main__':
    unittest.main()
```

Slide 13: Advanced ABC Features - Abstract Class Properties

Understanding advanced features of ABCs includes working with class properties and static methods while maintaining the abstract contract.

```python
class PaymentProcessor(ABC):
    @classmethod
    @abstractmethod
    def get_processor_name(cls):
        pass
    
    @staticmethod
    @abstractmethod
    def validate_currency(currency_code):
        pass

class StripeProcessor(PaymentProcessor):
    @classmethod
    def get_processor_name(cls):
        return "Stripe"
    
    @staticmethod
    def validate_currency(currency_code):
        return currency_code in ['USD', 'EUR', 'GBP']
```

Slide 14: Additional Resources

*   [https://arxiv.org/abs/1809.03193](https://arxiv.org/abs/1809.03193) - "Design Patterns in Python: A Systematic Literature Review"
*   [https://arxiv.org/abs/2007.08983](https://arxiv.org/abs/2007.08983) - "Object-Oriented Design Pattern Detection Using Machine Learning"
*   [https://arxiv.org/abs/1906.11678](https://arxiv.org/abs/1906.11678) - "On the Impact of Programming Language Abstractions"
*   [https://arxiv.org/abs/2012.14631](https://arxiv.org/abs/2012.14631) - "Automated Detection of Python Code Smells"

