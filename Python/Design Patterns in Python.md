## Design Patterns in Python

Slide 1: **Singleton Pattern** The Singleton pattern ensures that a class has only one instance and provides a global point of access to it.

```python
class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance

# Usage
s1 = Singleton()
s2 = Singleton()
print(s1 is s2)  # True
```

Slide 2: **Factory Pattern** The Factory pattern provides an interface for creating objects in a super-class, but allows subclasses to alter the type of objects that will be created.

```python
class Animal:
    def __init__(self, species):
        self.species = species

    def show(self):
        print(f"I'm a {self.species}")

class AnimalFactory:
    def create_animal(self, species):
        return Animal(species)

# Usage
factory = AnimalFactory()
dog = factory.create_animal("Dog")
cat = factory.create_animal("Cat")
dog.show()  # I'm a Dog
cat.show()  # I'm a Cat
```

Slide 3: **Observer Pattern** The Observer pattern defines a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically.

```python
class Subject:
    def __init__(self):
        self.observers = []

    def attach(self, observer):
        self.observers.append(observer)

    def detach(self, observer):
        self.observers.remove(observer)

    def notify(self, data):
        for observer in self.observers:
            observer.update(data)

class Observer:
    def update(self, data):
        pass

class ConcreteObserver(Observer):
    def update(self, data):
        print(f"Received data: {data}")

# Usage
subject = Subject()
observer1 = ConcreteObserver()
observer2 = ConcreteObserver()
subject.attach(observer1)
subject.attach(observer2)
subject.notify("Hello, World!")
```

Slide 4: **Decorator Pattern** The Decorator pattern allows behavior to be added to an individual object, either statically or dynamically, without affecting the behavior of other objects from the same class.

```python
class Component:
    def operation(self):
        pass

class ConcreteComponent(Component):
    def operation(self):
        print("ConcreteComponent.operation()")

class Decorator(Component):
    def __init__(self, component):
        self.component = component

    def operation(self):
        self.component.operation()

class ConcreteDecoratorA(Decorator):
    def operation(self):
        print("ConcreteDecoratorA.operation()")
        self.component.operation()

class ConcreteDecoratorB(Decorator):
    def operation(self):
        print("ConcreteDecoratorB.operation()")
        self.component.operation()

# Usage
component = ConcreteComponent()
decorator_a = ConcreteDecoratorA(component)
decorator_b = ConcreteDecoratorB(decorator_a)
decorator_b.operation()
```

Slide 5: **Strategy Pattern** The Strategy pattern defines a family of algorithms, encapsulates each one, and makes them interchangeable. It lets the algorithm vary independently from clients that use it.

```python
class Strategy:
    def execute(self, data):
        pass

class ConcreteStrategyA(Strategy):
    def execute(self, data):
        print(f"Executing strategy A with data: {data}")

class ConcreteStrategyB(Strategy):
    def execute(self, data):
        print(f"Executing strategy B with data: {data}")

class Context:
    def __init__(self, strategy):
        self.strategy = strategy

    def set_strategy(self, strategy):
        self.strategy = strategy

    def execute(self, data):
        self.strategy.execute(data)

# Usage
strategy_a = ConcreteStrategyA()
strategy_b = ConcreteStrategyB()
context = Context(strategy_a)
context.execute("Hello")  # Executing strategy A with data: Hello
context.set_strategy(strategy_b)
context.execute("World")  # Executing strategy B with data: World
```

Slide 6: **Adapter Pattern** The Adapter pattern allows objects with incompatible interfaces to collaborate by wrapping its own interface around that of an already existing class.

```python
class Target:
    def request(self):
        print("Target.request()")

class Adaptee:
    def specific_request(self):
        print("Adaptee.specific_request()")

class Adapter(Target):
    def __init__(self, adaptee):
        self.adaptee = adaptee

    def request(self):
        self.adaptee.specific_request()

# Usage
adaptee = Adaptee()
adapter = Adapter(adaptee)
adapter.request()  # Adaptee.specific_request()
```

Slide 7: **Facade Pattern** The Facade pattern provides a unified interface to a set of interfaces in a subsystem. It defines a higher-level interface that makes the subsystem easier to use.

```python
class SubSystemA:
    def operation_a(self):
        print("SubSystemA.operation_a()")

class SubSystemB:
    def operation_b(self):
        print("SubSystemB.operation_b()")

class Facade:
    def __init__(self):
        self.subsystem_a = SubSystemA()
        self.subsystem_b = SubSystemB()

    def operation(self):
        self.subsystem_a.operation_a()
        self.subsystem_b.operation_b()

# Usage
facade = Facade()
facade.operation()
```

Slide 8: **Proxy Pattern** The Proxy pattern provides a surrogate or placeholder for another object to control access to it.

```python
class Subject:
    def request(self):
        print("Subject.request()")

class Proxy:
    def __init__(self, subject):
        self.subject = subject

    def request(self):
        if self.check_access():
            self.subject.request()
            self.log_access()

    def check_access(self):
        print("Proxy.check_access()")
        return True

    def log_access(self):
        print("Proxy.log_access()")

# Usage
subject = Subject()
proxy = Proxy(subject)
proxy.request()
```

Slide 9: **Composite Pattern** The Composite pattern allows you to compose objects into tree structures to represent part-whole hierarchies. It lets clients treat individual objects and compositions of objects uniformly.

```python
class Component:
    def operation(self):
        pass

class Leaf(Component):
    def __init__(self, name):
        self.name = name

    def operation(self):
        print(f"Leaf: {self.name}")

class Composite(Component):
    def __init__(self):
        self.children = []

    def add(self, component):
        self.children.append(component)

    def remove(self, component):
        self.children.remove(component)

    def operation(self):
        for child in self.children:
            child.operation()

# Usage
leaf_a = Leaf("A")
leaf_b = Leaf("B")
composite = Composite()
composite.add(leaf_a)
composite.add(leaf_b)
composite.operation()
```

Slide 10: **Iterator Pattern** The Iterator pattern provides a way to access the elements of an aggregate object sequentially without exposing its underlying representation.

```python
class Iterator:
    def has_next(self):
        pass

    def next(self):
        pass

class ConcreteIterator(Iterator):
    def __init__(self, collection):
        self.collection = collection
        self.index = 0

    def has_next(self):
        return self.index < len(self.collection)

    def next(self):
        item = self.collection[self.index]
        self.index += 1
        return item

class Aggregate:
    def __init__(self):
        self.items = []

    def add(self, item):
        self.items.append(item)

    def iterator(self):
        return ConcreteIterator(self.items)

# Usage
aggregate = Aggregate()
aggregate.add("Item 1")
aggregate.add("Item 2")
aggregate.add("Item 3")

iterator = aggregate.iterator()
while iterator.has_next():
    item = iterator.next()
    print(item)
```

Slide 11: **Builder Pattern** The Builder pattern separates the construction of a complex object from its representation, allowing the same construction process to create various representations.

```python
class Product:
    def __init__(self):
        self.parts = []

    def add(self, part):
        self.parts.append(part)

    def display(self):
        print("\n".join(self.parts))

class Builder:
    def build_part_a(self):
        pass

    def build_part_b(self):
        pass

    def get_result(self):
        pass

class ConcreteBuilder(Builder):
    def __init__(self):
        self.product = Product()

    def build_part_a(self):
        self.product.add("Part A")

    def build_part_b(self):
        self.product.add("Part B")

    def get_result(self):
        return self.product

class Director:
    def __init__(self):
        self.builder = None

    def set_builder(self, builder):
        self.builder = builder

    def construct(self):
        self.builder.build_part_a()
        self.builder.build_part_b()

# Usage
director = Director()
builder = ConcreteBuilder()
director.set_builder(builder)
director.construct()

product = builder.get_result()
product.display()
```

Slide 12: **Flyweight Pattern** The Flyweight pattern aims to minimize memory usage by sharing data with similar characteristics.

```python
class Flyweight:
    def __init__(self, data):
        self.data = data

    def operation(self, extrinsic_data):
        print(f"Flyweight ({self.data}): {extrinsic_data}")

class FlyweightFactory:
    def __init__(self):
        self.flyweights = {}

    def get_flyweight(self, key):
        if key not in self.flyweights:
            self.flyweights[key] = Flyweight(key)
        return self.flyweights[key]

# Usage
factory = FlyweightFactory()

flyweight_a = factory.get_flyweight("A")
flyweight_a.operation("Extrinsic Data 1")

flyweight_b = factory.get_flyweight("B")
flyweight_b.operation("Extrinsic Data 2")

flyweight_a.operation("Extrinsic Data 3")
```

Slide 13: **Chain of Responsibility Pattern** The Chain of Responsibility pattern allows an event to be passed along a chain of objects, with each object having a chance to handle the event.

```python
class Handler:
    def __init__(self, successor=None):
        self.successor = successor

    def handle(self, request):
        pass

class ConcreteHandler1(Handler):
    def handle(self, request):
        if request >= 0 and request < 10:
            print(f"ConcreteHandler1 handled request: {request}")
        elif self.successor:
            self.successor.handle(request)

class ConcreteHandler2(Handler):
    def handle(self, request):
        if request >= 10 and request < 20:
            print(f"ConcreteHandler2 handled request: {request}")
        elif self.successor:
            self.successor.handle(request)

class ConcreteHandler3(Handler):
    def handle(self, request):
        if request >= 20 and request < 30:
            print(f"ConcreteHandler3 handled request: {request}")
        else:
            print(f"Request {request} was not handled")

# Usage
handler1 = ConcreteHandler1()
handler2 = ConcreteHandler2(handler1)
handler3 = ConcreteHandler3(handler2)

handler3.handle(5)   # ConcreteHandler1 handled request: 5
handler3.handle(15)  # ConcreteHandler2 handled request: 15
handler3.handle(25)  # ConcreteHandler3 handled request: 25
handler3.handle(35)  # Request 35 was not handled
```

Slide 14: **Command Pattern** The Command pattern encapsulates a request as an object, thereby allowing for the parameterization of clients with different requests, queue or log requests, and support undoable operations.

```python
class Command:
    def execute(self):
        pass

class ConcreteCommand(Command):
    def __init__(self, receiver):
        self.receiver = receiver

    def execute(self):
        self.receiver.action()

class Receiver:
    def action(self):
        print("Receiver.action()")

class Invoker:
    def __init__(self):
        self.commands = []

    def add_command(self, command):
        self.commands.append(command)

    def execute_commands(self):
        for command in self.commands:
            command.execute()

# Usage
receiver = Receiver()
command = ConcreteCommand(receiver)

invoker = Invoker()
invoker.add_command(command)
invoker.execute_commands()
```
