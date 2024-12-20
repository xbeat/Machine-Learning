## Architectural Design Patterns in Python
Slide 1: Architectural Design Patterns in Python

Architectural design patterns are reusable solutions to common problems in software design. They provide a structured approach to organizing code and improving system scalability, maintainability, and flexibility. In this presentation, we'll explore several key patterns and their implementation in Python.

```python
# Example: Simple Factory Pattern
class AnimalFactory:
    def create_animal(self, animal_type):
        if animal_type == "dog":
            return Dog()
        elif animal_type == "cat":
            return Cat()
        else:
            raise ValueError("Unknown animal type")

class Animal:
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"

# Usage
factory = AnimalFactory()
dog = factory.create_animal("dog")
print(dog.speak())  # Output: Woof!
```

Slide 2: Model-View-Controller (MVC) Pattern

The MVC pattern separates an application into three interconnected components: Model (data and business logic), View (user interface), and Controller (handles user input and updates model/view). This separation of concerns enhances modularity and maintainability.

```python
# Model
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email

# View
class UserView:
    def display_user_details(self, user):
        print(f"Name: {user.name}")
        print(f"Email: {user.email}")

# Controller
class UserController:
    def __init__(self, model, view):
        self.model = model
        self.view = view

    def update_user(self, name, email):
        self.model.name = name
        self.model.email = email

    def display_user(self):
        self.view.display_user_details(self.model)

# Usage
user = User("Alice", "alice@example.com")
view = UserView()
controller = UserController(user, view)

controller.display_user()
controller.update_user("Bob", "bob@example.com")
controller.display_user()
```

Slide 3: Microservices Architecture

Microservices architecture decomposes an application into small, independent services that communicate via APIs. Each service focuses on a specific business capability, allowing for easier scaling, deployment, and maintenance.

```python
from flask import Flask, jsonify
import requests

# User Service
app_user = Flask(__name__)

@app_user.route('/user/<int:user_id>')
def get_user(user_id):
    # Simulated user data
    user = {"id": user_id, "name": "John Doe", "email": "john@example.com"}
    return jsonify(user)

# Order Service
app_order = Flask(__name__)

@app_order.route('/order/<int:order_id>')
def get_order(order_id):
    # Simulated order data
    order = {"id": order_id, "product": "Widget", "quantity": 5}
    
    # Fetch user data from User Service
    user_response = requests.get(f'http://user-service/user/{order["user_id"]}')
    user_data = user_response.json()
    
    order["user"] = user_data
    return jsonify(order)

# Run services (in practice, these would be separate processes)
if __name__ == '__main__':
    app_user.run(port=5000)
    app_order.run(port=5001)
```

Slide 4: Event Sourcing Pattern

Event Sourcing stores the state of an application as a sequence of events rather than just the current state. This approach provides a complete audit trail and enables rebuilding the state at any point in time.

```python
from collections import defaultdict

class EventStore:
    def __init__(self):
        self.events = []

    def add_event(self, event):
        self.events.append(event)

    def get_events(self):
        return self.events

class InventoryItem:
    def __init__(self, item_id):
        self.item_id = item_id
        self.quantity = 0

    def apply_event(self, event):
        if event['type'] == 'ItemAdded':
            self.quantity += event['quantity']
        elif event['type'] == 'ItemRemoved':
            self.quantity -= event['quantity']

class InventoryManager:
    def __init__(self, event_store):
        self.event_store = event_store
        self.items = defaultdict(lambda: InventoryItem(0))

    def add_item(self, item_id, quantity):
        event = {'type': 'ItemAdded', 'item_id': item_id, 'quantity': quantity}
        self.event_store.add_event(event)
        self.items[item_id].apply_event(event)

    def remove_item(self, item_id, quantity):
        event = {'type': 'ItemRemoved', 'item_id': item_id, 'quantity': quantity}
        self.event_store.add_event(event)
        self.items[item_id].apply_event(event)

    def get_quantity(self, item_id):
        return self.items[item_id].quantity

# Usage
event_store = EventStore()
inventory = InventoryManager(event_store)

inventory.add_item(1, 10)
inventory.remove_item(1, 3)
print(f"Current quantity: {inventory.get_quantity(1)}")  # Output: 7

# Rebuild state from events
new_inventory = InventoryManager(event_store)
for event in event_store.get_events():
    new_inventory.items[event['item_id']].apply_event(event)

print(f"Rebuilt quantity: {new_inventory.get_quantity(1)}")  # Output: 7
```

Slide 5: Repository Pattern

The Repository Pattern abstracts the data layer, providing a collection-like interface for accessing domain objects. It centralizes data access logic and improves maintainability by decoupling the application from specific data storage implementations.

```python
from abc import ABC, abstractmethod

class User:
    def __init__(self, id, name, email):
        self.id = id
        self.name = name
        self.email = email

class UserRepository(ABC):
    @abstractmethod
    def get(self, id):
        pass

    @abstractmethod
    def add(self, user):
        pass

    @abstractmethod
    def update(self, user):
        pass

    @abstractmethod
    def delete(self, id):
        pass

class InMemoryUserRepository(UserRepository):
    def __init__(self):
        self.users = {}

    def get(self, id):
        return self.users.get(id)

    def add(self, user):
        if user.id in self.users:
            raise ValueError(f"User with id {user.id} already exists")
        self.users[user.id] = user

    def update(self, user):
        if user.id not in self.users:
            raise ValueError(f"User with id {user.id} not found")
        self.users[user.id] = user

    def delete(self, id):
        if id not in self.users:
            raise ValueError(f"User with id {id} not found")
        del self.users[id]

# Usage
repo = InMemoryUserRepository()
user = User(1, "Alice", "alice@example.com")

repo.add(user)
retrieved_user = repo.get(1)
print(f"Retrieved user: {retrieved_user.name}")  # Output: Alice

user.email = "newalice@example.com"
repo.update(user)

repo.delete(1)
```

Slide 6: Dependency Injection Pattern

Dependency Injection is a design pattern that implements Inversion of Control (IoC) for resolving dependencies. It decouples the usage of an object from its creation, leading to more modular and testable code.

```python
class EmailService:
    def send_email(self, to, subject, body):
        print(f"Sending email to {to}: {subject}")

class SMSService:
    def send_sms(self, to, message):
        print(f"Sending SMS to {to}: {message}")

class NotificationService:
    def __init__(self, email_service, sms_service):
        self.email_service = email_service
        self.sms_service = sms_service

    def notify(self, user, message):
        self.email_service.send_email(user.email, "Notification", message)
        self.sms_service.send_sms(user.phone, message)

class User:
    def __init__(self, name, email, phone):
        self.name = name
        self.email = email
        self.phone = phone

# Usage
email_service = EmailService()
sms_service = SMSService()
notification_service = NotificationService(email_service, sms_service)

user = User("Alice", "alice@example.com", "1234567890")
notification_service.notify(user, "Hello, this is a test notification!")
```

Slide 7: Command Pattern

The Command Pattern encapsulates a request as an object, allowing you to parameterize clients with different requests, queue or log requests, and support undoable operations. It separates the object that invokes the operation from the object that performs the operation.

```python
from abc import ABC, abstractmethod

class Command(ABC):
    @abstractmethod
    def execute(self):
        pass

    @abstractmethod
    def undo(self):
        pass

class Light:
    def __init__(self):
        self.is_on = False

    def turn_on(self):
        self.is_on = True
        print("Light is on")

    def turn_off(self):
        self.is_on = False
        print("Light is off")

class LightOnCommand(Command):
    def __init__(self, light):
        self.light = light

    def execute(self):
        self.light.turn_on()

    def undo(self):
        self.light.turn_off()

class LightOffCommand(Command):
    def __init__(self, light):
        self.light = light

    def execute(self):
        self.light.turn_off()

    def undo(self):
        self.light.turn_on()

class RemoteControl:
    def __init__(self):
        self.command = None

    def set_command(self, command):
        self.command = command

    def press_button(self):
        self.command.execute()

    def press_undo(self):
        self.command.undo()

# Usage
light = Light()
light_on = LightOnCommand(light)
light_off = LightOffCommand(light)

remote = RemoteControl()

remote.set_command(light_on)
remote.press_button()  # Light is on

remote.set_command(light_off)
remote.press_button()  # Light is off

remote.press_undo()  # Light is on (undo last command)
```

Slide 8: Observer Pattern

The Observer Pattern defines a one-to-many dependency between objects so that when one object changes state, all its dependents are notified and updated automatically. It's commonly used for implementing distributed event handling systems.

```python
from abc import ABC, abstractmethod

class Subject(ABC):
    @abstractmethod
    def attach(self, observer):
        pass

    @abstractmethod
    def detach(self, observer):
        pass

    @abstractmethod
    def notify(self):
        pass

class Observer(ABC):
    @abstractmethod
    def update(self, temperature, humidity, pressure):
        pass

class WeatherStation(Subject):
    def __init__(self):
        self._observers = []
        self._temperature = 0
        self._humidity = 0
        self._pressure = 0

    def attach(self, observer):
        self._observers.append(observer)

    def detach(self, observer):
        self._observers.remove(observer)

    def notify(self):
        for observer in self._observers:
            observer.update(self._temperature, self._humidity, self._pressure)

    def set_measurements(self, temperature, humidity, pressure):
        self._temperature = temperature
        self._humidity = humidity
        self._pressure = pressure
        self.notify()

class DisplayDevice(Observer):
    def __init__(self, name):
        self.name = name

    def update(self, temperature, humidity, pressure):
        print(f"{self.name} - Temperature: {temperature}°C, Humidity: {humidity}%, Pressure: {pressure}hPa")

# Usage
weather_station = WeatherStation()

phone_display = DisplayDevice("Phone")
tablet_display = DisplayDevice("Tablet")

weather_station.attach(phone_display)
weather_station.attach(tablet_display)

weather_station.set_measurements(25, 60, 1013)
weather_station.set_measurements(26, 58, 1012)

weather_station.detach(tablet_display)
weather_station.set_measurements(27, 57, 1011)
```

Slide 9: Singleton Pattern

The Singleton Pattern ensures a class has only one instance and provides a global point of access to it. It's useful for coordinating actions across a system, such as managing a shared resource or a central data store.

```python
class Singleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Singleton, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        self.data = {}

    def set_data(self, key, value):
        self.data[key] = value

    def get_data(self, key):
        return self.data.get(key)

# Usage
s1 = Singleton()
s2 = Singleton()

print(s1 is s2)  # Output: True

s1.set_data("key1", "value1")
print(s2.get_data("key1"))  # Output: value1
```

Slide 10: Facade Pattern

The Facade Pattern provides a unified interface to a set of interfaces in a subsystem. It defines a higher-level interface that makes the subsystem easier to use by reducing complexity and hiding the communication and dependencies between subsystems.

```python
class CPU:
    def freeze(self):
        print("CPU: Freezing...")

    def jump(self, position):
        print(f"CPU: Jumping to position {position}")

    def execute(self):
        print("CPU: Executing...")

class Memory:
    def load(self, position, data):
        print(f"Memory: Loading data {data} to position {position}")

class HardDrive:
    def read(self, lba, size):
        print(f"HardDrive: Reading {size} bytes from sector {lba}")
        return "Some data"

class ComputerFacade:
    def __init__(self):
        self.cpu = CPU()
        self.memory = Memory()
        self.hard_drive = HardDrive()

    def start(self):
        self.cpu.freeze()
        self.memory.load(0, self.hard_drive.read(0, 1024))
        self.cpu.jump(0)
        self.cpu.execute()

# Usage
computer = ComputerFacade()
computer.start()
```

Slide 11: Adapter Pattern

The Adapter Pattern allows incompatible interfaces to work together. It acts as a bridge between two incompatible interfaces by wrapping the interface of a class into another interface that a client expects.

```python
class EuropeanSocket:
    def voltage(self):
        return 230

    def live(self):
        return 1

    def neutral(self):
        return -1

class USASocket:
    def voltage(self):
        return 120

    def live(self):
        return 1

    def neutral(self):
        return -1

class Adapter:
    def __init__(self, socket):
        self.socket = socket

    def voltage(self):
        return 110

    def live(self):
        return self.socket.live()

    def neutral(self):
        return self.socket.neutral()

class ElectronicDevice:
    def __init__(self, name, input_voltage):
        self.name = name
        self.input_voltage = input_voltage

    def charge(self, socket):
        if self.input_voltage == socket.voltage():
            print(f"{self.name} is charging.")
        else:
            print(f"Cannot charge {self.name}. Incompatible voltage.")

# Usage
eu_socket = EuropeanSocket()
us_socket = USASocket()
adapter = Adapter(eu_socket)

laptop = ElectronicDevice("Laptop", 110)
laptop.charge(adapter)  # Laptop is charging.
laptop.charge(us_socket)  # Cannot charge Laptop. Incompatible voltage.
```

Slide 12: Strategy Pattern

The Strategy Pattern defines a family of algorithms, encapsulates each one, and makes them interchangeable. It lets the algorithm vary independently from clients that use it.

```python
from abc import ABC, abstractmethod

class PaymentStrategy(ABC):
    @abstractmethod
    def pay(self, amount):
        pass

class CreditCardPayment(PaymentStrategy):
    def __init__(self, card_number, name):
        self.card_number = card_number
        self.name = name

    def pay(self, amount):
        print(f"Paid ${amount} using Credit Card {self.card_number}")

class PayPalPayment(PaymentStrategy):
    def __init__(self, email):
        self.email = email

    def pay(self, amount):
        print(f"Paid ${amount} using PayPal account {self.email}")

class ShoppingCart:
    def __init__(self):
        self.items = []

    def add_item(self, item, price):
        self.items.append((item, price))

    def calculate_total(self):
        return sum(price for _, price in self.items)

    def checkout(self, payment_strategy):
        total = self.calculate_total()
        payment_strategy.pay(total)

# Usage
cart = ShoppingCart()
cart.add_item("Laptop", 1000)
cart.add_item("Mouse", 50)

credit_card = CreditCardPayment("1234-5678-9012-3456", "John Doe")
paypal = PayPalPayment("john@example.com")

cart.checkout(credit_card)
cart.checkout(paypal)
```

Slide 13: Factory Method Pattern

The Factory Method Pattern defines an interface for creating an object, but lets subclasses decide which class to instantiate. It allows a class to defer instantiation to subclasses.

```python
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"

class AnimalFactory(ABC):
    @abstractmethod
    def create_animal(self):
        pass

class DogFactory(AnimalFactory):
    def create_animal(self):
        return Dog()

class CatFactory(AnimalFactory):
    def create_animal(self):
        return Cat()

def animal_sound(factory):
    animal = factory.create_animal()
    return animal.speak()

# Usage
dog_factory = DogFactory()
cat_factory = CatFactory()

print(animal_sound(dog_factory))  # Output: Woof!
print(animal_sound(cat_factory))  # Output: Meow!
```

Slide 14: Decorator Pattern

The Decorator Pattern attaches additional responsibilities to an object dynamically. It provides a flexible alternative to subclassing for extending functionality.

```python
from abc import ABC, abstractmethod

class Coffee(ABC):
    @abstractmethod
    def cost(self):
        pass

    @abstractmethod
    def description(self):
        pass

class SimpleCoffee(Coffee):
    def cost(self):
        return 1.0

    def description(self):
        return "Simple coffee"

class CoffeeDecorator(Coffee):
    def __init__(self, coffee):
        self._coffee = coffee

    def cost(self):
        return self._coffee.cost()

    def description(self):
        return self._coffee.description()

class Milk(CoffeeDecorator):
    def cost(self):
        return self._coffee.cost() + 0.5

    def description(self):
        return f"{self._coffee.description()}, milk"

class Sugar(CoffeeDecorator):
    def cost(self):
        return self._coffee.cost() + 0.2

    def description(self):
        return f"{self._coffee.description()}, sugar"

# Usage
coffee = SimpleCoffee()
print(f"{coffee.description()}: ${coffee.cost()}")

coffee_with_milk = Milk(coffee)
print(f"{coffee_with_milk.description()}: ${coffee_with_milk.cost()}")

coffee_with_milk_and_sugar = Sugar(Milk(coffee))
print(f"{coffee_with_milk_and_sugar.description()}: ${coffee_with_milk_and_sugar.cost()}")
```

Slide 15: Additional Resources

For more in-depth information on architectural design patterns and their implementation in Python, consider exploring the following resources:

1. "Design Patterns: Elements of Reusable Object-Oriented Software" by Erich Gamma, Richard Helm, Ralph Johnson, and John Vlissides (Gang of Four)
2. "Python Design Patterns" by Brandon Rhodes (PyCon 2013 talk)
3. "Fluent Python" by Luciano Ramalho (O'Reilly Media)
4. Python Design Patterns Guide on refactoring.guru

Remember to verify the accuracy and relevance of these resources, as they may have been updated or replaced with newer alternatives since my last update.

