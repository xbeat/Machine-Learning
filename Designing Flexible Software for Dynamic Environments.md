## Designing Flexible Software for Dynamic Environments
Slide 1: Building Flexible Software

Flexible software is crucial in today's dynamic environments. It adapts to changing requirements and user needs, ensuring longevity and relevance. This presentation explores various techniques and principles for creating flexible software using Python.

```python
# Example: A flexible configuration system
import json

class Config:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            self.config = json.load(f)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value
        with open('config.json', 'w') as f:
            json.dump(self.config, f)

# Usage
config = Config('config.json')
print(config.get('debug_mode', False))
config.set('api_endpoint', 'https://api.example.com')
```

Slide 2: Modular Architecture

Modular architecture breaks functionality into independent modules, making updates and reuse easier. In Python, we can use packages and modules to achieve modularity.

```python
# weather_app/
# ├── main.py
# ├── api/
# │   ├── __init__.py
# │   └── weather_api.py
# └── utils/
#     ├── __init__.py
#     └── data_processing.py

# weather_app/api/weather_api.py
import requests

def get_weather(city):
    api_key = "your_api_key_here"
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
    response = requests.get(url)
    return response.json()

# weather_app/utils/data_processing.py
def kelvin_to_celsius(kelvin):
    return kelvin - 273.15

# weather_app/main.py
from api.weather_api import get_weather
from utils.data_processing import kelvin_to_celsius

def main():
    city = input("Enter city name: ")
    weather_data = get_weather(city)
    temp_kelvin = weather_data['main']['temp']
    temp_celsius = kelvin_to_celsius(temp_kelvin)
    print(f"Temperature in {city}: {temp_celsius:.2f}°C")

if __name__ == "__main__":
    main()
```

Slide 3: Microservices Architecture

Microservices architecture separates functionality into independent services, allowing for scalability and independent deployment. While Python isn't typically used for large-scale microservices, we can demonstrate a simple example using Flask.

```python
# service1.py
from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/user/<int:user_id>')
def get_user(user_id):
    # Simulating database query
    user = {'id': user_id, 'name': f'User {user_id}'}
    return jsonify(user)

if __name__ == '__main__':
    app.run(port=5000)

# service2.py
from flask import Flask, jsonify
import requests
app = Flask(__name__)

@app.route('/user-orders/<int:user_id>')
def get_user_orders(user_id):
    # Get user info from service1
    user_response = requests.get(f'http://localhost:5000/user/{user_id}')
    user = user_response.json()
    
    # Simulating order retrieval
    orders = [{'id': 1, 'item': 'Book'}, {'id': 2, 'item': 'Laptop'}]
    return jsonify({'user': user, 'orders': orders})

if __name__ == '__main__':
    app.run(port=5001)
```

Slide 4: Loose Coupling

Loose coupling reduces interdependencies between components, enhancing flexibility. In Python, we can achieve this through interfaces and dependency injection.

```python
from abc import ABC, abstractmethod

class PaymentGateway(ABC):
    @abstractmethod
    def process_payment(self, amount):
        pass

class PayPalGateway(PaymentGateway):
    def process_payment(self, amount):
        print(f"Processing ${amount} payment via PayPal")

class StripeGateway(PaymentGateway):
    def process_payment(self, amount):
        print(f"Processing ${amount} payment via Stripe")

class Order:
    def __init__(self, payment_gateway: PaymentGateway):
        self.payment_gateway = payment_gateway

    def checkout(self, amount):
        self.payment_gateway.process_payment(amount)

# Usage
paypal_order = Order(PayPalGateway())
paypal_order.checkout(100)

stripe_order = Order(StripeGateway())
stripe_order.checkout(200)
```

Slide 5: SOLID Principles

SOLID principles ensure maintainable and scalable code. Let's focus on the Single Responsibility Principle (SRP) as an example.

```python
# Violating SRP
class User:
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def save(self):
        print(f"Saving user {self.name} to database")

    def send_email(self, message):
        print(f"Sending email to {self.name}: {message}")

# Following SRP
class User:
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name

class UserRepository:
    def save(self, user):
        print(f"Saving user {user.get_name()} to database")

class EmailService:
    def send_email(self, user, message):
        print(f"Sending email to {user.get_name()}: {message}")

# Usage
user = User("Alice")
repo = UserRepository()
email_service = EmailService()

repo.save(user)
email_service.send_email(user, "Welcome to our service!")
```

Slide 6: Design Patterns

Design patterns provide adaptable structures for common programming challenges. Let's implement the Observer pattern as an example.

```python
class Subject:
    def __init__(self):
        self._observers = []
        self._state = None

    def attach(self, observer):
        self._observers.append(observer)

    def detach(self, observer):
        self._observers.remove(observer)

    def notify(self):
        for observer in self._observers:
            observer.update(self._state)

    def set_state(self, state):
        self._state = state
        self.notify()

class Observer:
    def update(self, state):
        pass

class ConcreteObserver(Observer):
    def __init__(self, name):
        self.name = name

    def update(self, state):
        print(f"{self.name} received update. New state: {state}")

# Usage
subject = Subject()
observer1 = ConcreteObserver("Observer 1")
observer2 = ConcreteObserver("Observer 2")

subject.attach(observer1)
subject.attach(observer2)

subject.set_state("New State")
```

Slide 7: API-First Approach

An API-first approach ensures easy integration and future expansion. Let's create a simple REST API using Flask.

```python
from flask import Flask, jsonify, request

app = Flask(__name__)

books = [
    {"id": 1, "title": "To Kill a Mockingbird", "author": "Harper Lee"},
    {"id": 2, "title": "1984", "author": "George Orwell"}
]

@app.route('/api/books', methods=['GET'])
def get_books():
    return jsonify(books)

@app.route('/api/books/<int:book_id>', methods=['GET'])
def get_book(book_id):
    book = next((book for book in books if book['id'] == book_id), None)
    if book:
        return jsonify(book)
    return jsonify({"error": "Book not found"}), 404

@app.route('/api/books', methods=['POST'])
def add_book():
    new_book = request.json
    new_book['id'] = max(book['id'] for book in books) + 1
    books.append(new_book)
    return jsonify(new_book), 201

if __name__ == '__main__':
    app.run(debug=True)
```

Slide 8: Externalize Configurations

Storing settings outside the code allows for flexibility in different environments. Let's use Python's configparser module.

```python
import configparser

# Create config.ini
config = configparser.ConfigParser()
config['DEFAULT'] = {'ServerAliveInterval': '45',
                     'Compression': 'yes',
                     'CompressionLevel': '9'}
config['bitbucket.org'] = {}
config['bitbucket.org']['User'] = 'hg'
config['topsecret.server.com'] = {}
topsecret = config['topsecret.server.com']
topsecret['Port'] = '50022'
topsecret['ForwardX11'] = 'no'

with open('config.ini', 'w') as configfile:
    config.write(configfile)

# Read config.ini
config = configparser.ConfigParser()
config.read('config.ini')

print(config['DEFAULT']['ServerAliveInterval'])
print(config['bitbucket.org']['User'])
print(config['topsecret.server.com']['Port'])
```

Slide 9: Dependency Injection

Dependency Injection allows for easier swapping of components. Let's implement a simple DI container in Python.

```python
class DIContainer:
    def __init__(self):
        self._services = {}

    def register(self, interface, implementation):
        self._services[interface] = implementation

    def resolve(self, interface):
        return self._services[interface]()

# Example services
class Logger:
    def log(self, message):
        print(f"Logging: {message}")

class EmailSender:
    def send(self, to, message):
        print(f"Sending email to {to}: {message}")

class UserService:
    def __init__(self, logger, email_sender):
        self.logger = logger
        self.email_sender = email_sender

    def register_user(self, email):
        self.logger.log(f"Registering user: {email}")
        self.email_sender.send(email, "Welcome to our service!")

# Usage
container = DIContainer()
container.register(Logger, Logger)
container.register(EmailSender, EmailSender)
container.register(UserService, lambda: UserService(container.resolve(Logger), container.resolve(EmailSender)))

user_service = container.resolve(UserService)
user_service.register_user("user@example.com")
```

Slide 10: CI/CD Pipelines

CI/CD pipelines automate testing and deployment for quick, safe iterations. While setting up a full pipeline is beyond the scope of this example, we can demonstrate a simple test case using pytest.

```python
# calculator.py
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

# test_calculator.py
import pytest
from calculator import add, subtract

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(-1, -1) == -2

def test_subtract():
    assert subtract(5, 3) == 2
    assert subtract(1, 1) == 0
    assert subtract(-1, -1) == 0

# Run tests with: pytest test_calculator.py
```

Slide 11: Feature Toggles

Feature toggles enable or disable features without redeployment. Let's implement a simple feature toggle system.

```python
import json

class FeatureToggle:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            self.features = json.load(f)

    def is_enabled(self, feature_name):
        return self.features.get(feature_name, False)

# Example usage
class UserInterface:
    def __init__(self, feature_toggle):
        self.feature_toggle = feature_toggle

    def show_dashboard(self):
        print("Showing basic dashboard")
        if self.feature_toggle.is_enabled("advanced_analytics"):
            print("Showing advanced analytics")
        if self.feature_toggle.is_enabled("social_feed"):
            print("Showing social feed")

# feature_config.json
# {
#     "advanced_analytics": true,
#     "social_feed": false
# }

feature_toggle = FeatureToggle('feature_config.json')
ui = UserInterface(feature_toggle)
ui.show_dashboard()
```

Slide 12: Asynchronous Processing

Asynchronous operations improve performance and flexibility. Let's use Python's asyncio library for an example.

```python
import asyncio
import aiohttp

async def fetch_url(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

async def main():
    urls = [
        "https://api.github.com",
        "https://api.bitbucket.org",
        "https://api.gitlab.com"
    ]
    tasks = [fetch_url(url) for url in urls]
    results = await asyncio.gather(*tasks)
    for url, result in zip(urls, results):
        print(f"Response from {url}: {result[:50]}...")

asyncio.run(main())
```

Slide 13: Code Readability

Maintaining clean, readable code eases future modifications. Let's compare a poorly written function with a well-written one.

```python
# Poor readability
def f(x,y):
    z=x+y
    if z>10:
        return z*2
    else:
        return z/2

# Good readability
def calculate_result(first_number, second_number):
    """
    Calculate a result based on the sum of two numbers.
    
    If the sum is greater than 10, return twice the sum.
    Otherwise, return half of the sum.
    """
    sum_of_numbers = first_number + second_number
    
    if sum_of_numbers > 10:
        return sum_of_numbers * 2
    else:
        return sum_of_numbers / 2

# Usage
print(calculate_result(5, 7))  # Output: 24
print(calculate_result(2, 3))  # Output: 2.5
```

Slide 14: Real-Life Example: Weather App

Let's combine some of the principles we've learned to create a flexible weather application.

```python
import aiohttp
import asyncio
import json

class Config:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            self.config = json.load(f)

    def get(self, key, default=None):
        return self.config.get(key, default)

class WeatherAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"

    async def get_weather(self, city):
        url = f"{self.base_url}?q={city}&appid={self.api_key}&units=metric"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.json()

class WeatherApp:
    def __init__(self, config, weather_api):
        self.config = config
        self.weather_api = weather_api

    async def run(self):
        cities = self.config.get("cities", [])
        tasks = [self.weather_api.get_weather(city) for city in cities]
        results = await asyncio.gather(*tasks)
        
        for city, result in zip(cities, results):
            if 'main' in result:
                print(f"Weather in {city}: {result['main']['temp']}°C, {result['weather'][0]['description']}")
            else:
                print(f"Failed to get weather for {city}")

async def main():
    config = Config('config.json')
    api_key = config.get('api_key')
    weather_api = WeatherAPI(api_key)
    app = WeatherApp(config, weather_api)
    await app.run()

if __name__ == "__main__":
    asyncio.run(main())

# Example config.json:
# {
#     "api_key": "your_api_key_here",
#     "cities": ["London", "New York", "Tokyo"]
# }
```

Slide 15: Real-Life Example: Task Management System

Let's create a simple task management system that incorporates modular design, loose coupling, and the Observer pattern.

```python
from abc import ABC, abstractmethod
from datetime import datetime

class Task:
    def __init__(self, title, description):
        self.title = title
        self.description = description
        self.completed = False
        self.created_at = datetime.now()

class TaskList:
    def __init__(self):
        self.tasks = []
        self.observers = []

    def add_task(self, task):
        self.tasks.append(task)
        self.notify_observers("add", task)

    def complete_task(self, task):
        task.completed = True
        self.notify_observers("complete", task)

    def add_observer(self, observer):
        self.observers.append(observer)

    def notify_observers(self, action, task):
        for observer in self.observers:
            observer.update(action, task)

class TaskObserver(ABC):
    @abstractmethod
    def update(self, action, task):
        pass

class TaskLogger(TaskObserver):
    def update(self, action, task):
        print(f"Log: Task '{task.title}' was {action}ed at {datetime.now()}")

class EmailNotifier(TaskObserver):
    def update(self, action, task):
        print(f"Email: Task '{task.title}' was {action}ed")

# Usage
task_list = TaskList()
task_list.add_observer(TaskLogger())
task_list.add_observer(EmailNotifier())

task1 = Task("Implement login", "Create login functionality for the app")
task2 = Task("Design UI", "Create wireframes for the main dashboard")

task_list.add_task(task1)
task_list.add_task(task2)
task_list.complete_task(task1)
```

Slide 16: Additional Resources

For more information on building flexible software and advanced Python topics, consider exploring these resources:

1. "Design Patterns: Elements of Reusable Object-Oriented Software" by Gamma et al.
2. "Clean Architecture: A Craftsman's Guide to Software Structure and Design" by Robert C. Martin
3. "Fluent Python" by Luciano Ramalho
4. Python official documentation: [https://docs.python.org/](https://docs.python.org/)
5. Real Python tutorials: [https://realpython.com/](https://realpython.com/)

Remember to always verify the accuracy and relevance of any additional resources you consult.

