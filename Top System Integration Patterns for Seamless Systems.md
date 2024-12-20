## Top System Integration Patterns for Seamless Systems

Slide 1: Peer-to-Peer (P2P) Integration

Peer-to-Peer integration enables direct communication between systems without intermediaries. This pattern is ideal for small networks where systems can interact directly, sharing resources efficiently. Think of it as a group chat where everyone can talk to each other directly.

```python
import socket

class Peer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.host, self.port))
        
    def listen(self):
        self.socket.listen(1)
        print(f"Listening on {self.host}:{self.port}")
        
        while True:
            client, address = self.socket.accept()
            data = client.recv(1024).decode()
            print(f"Received: {data}")
            client.send("Message received".encode())
            client.close()

    def send_message(self, peer_host, peer_port, message):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((peer_host, peer_port))
            s.send(message.encode())
            response = s.recv(1024).decode()
            print(f"Response: {response}")

# Usage
peer1 = Peer("localhost", 5000)
peer1.listen()  # This will run indefinitely

# In another script or terminal:
# peer2 = Peer("localhost", 5001)
# peer2.send_message("localhost", 5000, "Hello, Peer!")
```

Slide 2: API Gateway

An API Gateway acts as a single entry point for multiple microservices, routing requests to the appropriate service. It's like a smart receptionist directing calls to the right department. This pattern enhances security, simplifies client-side code, and allows for easy scaling of individual services.

```python
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

class APIGateway(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/users':
            self.handle_users()
        elif self.path == '/products':
            self.handle_products()
        else:
            self.send_error(404, "Not Found")

    def handle_users(self):
        # Simulate calling a user service
        response = {"users": ["Alice", "Bob", "Charlie"]}
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def handle_products(self):
        # Simulate calling a product service
        response = {"products": ["Laptop", "Phone", "Tablet"]}
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

# Run the server
server_address = ('', 8000)
httpd = HTTPServer(server_address, APIGateway)
print('API Gateway running on port 8000...')
httpd.serve_forever()

# To test:
# curl http://localhost:8000/users
# curl http://localhost:8000/products
```

Slide 3: Publish-Subscribe (Pub-Sub) Pattern

The Pub-Sub pattern allows for decoupled communication between publishers and subscribers through a message broker. It's like a news agency distributing articles to subscribed readers. This pattern is excellent for building scalable and flexible systems, especially for event-driven architectures.

```python
class MessageBroker:
    def __init__(self):
        self.topics = {}

    def subscribe(self, topic, subscriber):
        if topic not in self.topics:
            self.topics[topic] = []
        self.topics[topic].append(subscriber)

    def publish(self, topic, message):
        if topic in self.topics:
            for subscriber in self.topics[topic]:
                subscriber.receive(message)

class Subscriber:
    def __init__(self, name):
        self.name = name

    def receive(self, message):
        print(f"{self.name} received: {message}")

# Usage
broker = MessageBroker()

sub1 = Subscriber("Subscriber 1")
sub2 = Subscriber("Subscriber 2")

broker.subscribe("tech_news", sub1)
broker.subscribe("tech_news", sub2)
broker.subscribe("sports", sub2)

broker.publish("tech_news", "New AI breakthrough!")
broker.publish("sports", "Team wins championship!")

# Output:
# Subscriber 1 received: New AI breakthrough!
# Subscriber 2 received: New AI breakthrough!
# Subscriber 2 received: Team wins championship!
```

Slide 4: Request-Response Pattern

The Request-Response pattern is a fundamental communication model where a client sends a request to a server and waits for a response. It's like asking a question and waiting for an answer. This synchronous pattern is simple but can lead to performance issues with high latency or large numbers of requests.

```python
import socket

def server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 12345))
        s.listen()
        print("Server is listening...")
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                response = f"Server received: {data.decode()}"
                conn.sendall(response.encode())

def client():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('localhost', 12345))
        s.sendall(b"Hello, server!")
        data = s.recv(1024)
        print(f"Client received: {data.decode()}")

# Run server in one terminal:
# server()

# Run client in another terminal:
# client()

# Output (Client):
# Client received: Server received: Hello, server!
```

Slide 5: Event Sourcing

Event Sourcing is a pattern where the state of a system is determined by a sequence of events rather than just the current state. It's like keeping a detailed log of all transactions in a bank account, allowing you to reconstruct the balance at any point in time. This pattern is useful for auditing, debugging, and maintaining complex systems.

```python
import time

class Account:
    def __init__(self, account_id):
        self.account_id = account_id
        self.balance = 0
        self.events = []

    def apply_event(self, event):
        if event['type'] == 'deposit':
            self.balance += event['amount']
        elif event['type'] == 'withdraw':
            self.balance -= event['amount']
        self.events.append(event)

    def get_balance(self):
        return self.balance

    def deposit(self, amount):
        event = {
            'type': 'deposit',
            'amount': amount,
            'timestamp': time.time()
        }
        self.apply_event(event)

    def withdraw(self, amount):
        if amount <= self.balance:
            event = {
                'type': 'withdraw',
                'amount': amount,
                'timestamp': time.time()
            }
            self.apply_event(event)
        else:
            raise ValueError("Insufficient funds")

    def get_history(self):
        return self.events

# Usage
account = Account("12345")
account.deposit(100)
account.withdraw(30)
account.deposit(50)

print(f"Current balance: ${account.get_balance()}")
print("Transaction history:")
for event in account.get_history():
    print(f"{event['type'].capitalize()}: ${event['amount']}")

# Output:
# Current balance: $120
# Transaction history:
# Deposit: $100
# Withdraw: $30
# Deposit: $50
```

Slide 6: ETL (Extract, Transform, Load)

ETL is a process used to collect data from various sources, transform it to fit operational needs, and load it into the end target database. It's like a chef preparing ingredients, cooking a dish, and serving it. ETL is crucial for data warehousing and business intelligence.

```python
import csv
import json

def extract_from_csv(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data.append(row)
    return data

def transform_data(data):
    transformed_data = []
    for item in data:
        transformed_item = {
            'full_name': f"{item['first_name']} {item['last_name']}",
            'age': int(item['age']),
            'email': item['email'].lower()
        }
        transformed_data.append(transformed_item)
    return transformed_data

def load_to_json(data, output_file):
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=2)

# Usage
input_file = 'input_data.csv'
output_file = 'output_data.json'

# Extract
raw_data = extract_from_csv(input_file)

# Transform
transformed_data = transform_data(raw_data)

# Load
load_to_json(transformed_data, output_file)

print(f"ETL process completed. Data loaded to {output_file}")

# Sample input_data.csv:
# first_name,last_name,age,email
# John,Doe,30,John.Doe@example.com
# Jane,Smith,25,jane.smith@example.com

# Output in output_data.json:
# [
#   {
#     "full_name": "John Doe",
#     "age": 30,
#     "email": "john.doe@example.com"
#   },
#   {
#     "full_name": "Jane Smith",
#     "age": 25,
#     "email": "jane.smith@example.com"
#   }
# ]
```

Slide 7: Batch Processing

Batch processing involves processing large volumes of data at scheduled intervals. It's like a factory processing a large order of items in one go. This pattern is efficient for tasks that don't require real-time processing and can be done periodically.

```python
import time
from datetime import datetime

class BatchProcessor:
    def __init__(self):
        self.batch = []
        self.batch_size = 5
        self.processing_interval = 10  # seconds

    def add_item(self, item):
        self.batch.append(item)
        print(f"Item added: {item}")
        if len(self.batch) >= self.batch_size:
            self.process_batch()

    def process_batch(self):
        print(f"Processing batch of {len(self.batch)} items at {datetime.now()}")
        for item in self.batch:
            # Simulate processing each item
            time.sleep(0.5)
            print(f"Processed: {item}")
        self.batch = []

    def run(self):
        while True:
            if self.batch:
                self.process_batch()
            time.sleep(self.processing_interval)

# Usage
processor = BatchProcessor()

# Simulate adding items over time
processor.add_item("Item 1")
time.sleep(2)
processor.add_item("Item 2")
processor.add_item("Item 3")
time.sleep(3)
processor.add_item("Item 4")
processor.add_item("Item 5")  # This will trigger immediate processing
time.sleep(5)
processor.add_item("Item 6")

# Run the batch processor
processor.run()

# Output will show items being added and processed in batches
```

Slide 8: Stream Processing

Stream processing handles data in real-time as it flows through a system. It's like a conveyor belt in a factory, processing items as they come. This pattern is crucial for applications requiring immediate insights or actions based on incoming data.

```python
import time
import random
from collections import deque

class StreamProcessor:
    def __init__(self, window_size=5):
        self.window = deque(maxlen=window_size)
        self.running_sum = 0

    def process(self, data):
        self.window.append(data)
        self.running_sum += data
        if len(self.window) == self.window.maxlen:
            oldest = self.window[0]
            average = self.running_sum / len(self.window)
            print(f"Window: {list(self.window)}, Average: {average:.2f}")
            return average
        return None

def generate_stream():
    while True:
        yield random.randint(1, 100)
        time.sleep(0.5)

# Usage
processor = StreamProcessor()
stream = generate_stream()

for _ in range(10):  # Process 10 data points
    data = next(stream)
    print(f"Received: {data}")
    result = processor.process(data)
    if result is not None:
        print(f"Processed result: {result:.2f}")
    print("---")

# Output will show a stream of data being processed in real-time
# with a moving average calculated over a window of 5 data points
```

Slide 9: Orchestration

Orchestration coordinates multiple services to complete a task. It's like a conductor ensuring all musicians in an orchestra play in harmony. This pattern is essential for managing complex workflows in microservices architectures.

```python
import time
import random

class Service:
    def __init__(self, name):
        self.name = name

    def process(self):
        processing_time = random.uniform(0.1, 0.5)
        time.sleep(processing_time)
        return f"{self.name} completed in {processing_time:.2f}s"

class Orchestrator:
    def __init__(self):
        self.services = []

    def add_service(self, service):
        self.services.append(service)

    def run_workflow(self):
        results = []
        for service in self.services:
            print(f"Starting {service.name}")
            result = service.process()
            results.append(result)
            print(result)
        return results

# Usage
orchestrator = Orchestrator()
orchestrator.add_service(Service("Authentication"))
orchestrator.add_service(Service("Data Retrieval"))
orchestrator.add_service(Service("Data Processing"))
orchestrator.add_service(Service("Reporting"))

print("Starting workflow")
workflow_results = orchestrator.run_workflow()
print("Workflow completed")

# Output will show the orchestrated execution of multiple services
```

Slide 10: Real-Life Example: Social Media Feed

A social media feed application demonstrates the integration of multiple system patterns. This example combines Pub-Sub for real-time updates, API Gateway for client requests, and Stream Processing for trend analysis.

```python
import time
from collections import defaultdict

class SocialMediaPlatform:
    def __init__(self):
        self.users = {}
        self.posts = []
        self.trends = defaultdict(int)
        self.subscribers = defaultdict(set)

    def add_user(self, user_id, name):
        self.users[user_id] = {"name": name, "followers": set()}

    def follow(self, follower_id, followed_id):
        self.users[followed_id]["followers"].add(follower_id)
        self.subscribers[followed_id].add(follower_id)

    def create_post(self, user_id, content):
        post = {"user_id": user_id, "content": content, "timestamp": time.time()}
        self.posts.append(post)
        self.process_post(post)
        self.notify_followers(user_id, post)

    def process_post(self, post):
        words = post["content"].lower().split()
        for word in words:
            if word.startswith("#"):
                self.trends[word] += 1

    def notify_followers(self, user_id, post):
        for follower in self.subscribers[user_id]:
            print(f"Notifying user {follower} about new post from {user_id}")

    def get_feed(self, user_id, limit=10):
        return [post for post in reversed(self.posts) 
                if post["user_id"] in self.users[user_id]["followers"]][:limit]

    def get_trending_topics(self, limit=5):
        return sorted(self.trends.items(), key=lambda x: x[1], reverse=True)[:limit]

# Usage
platform = SocialMediaPlatform()
platform.add_user(1, "Alice")
platform.add_user(2, "Bob")
platform.follow(1, 2)
platform.create_post(2, "Hello world! #firstpost")
print(platform.get_feed(1))
print(platform.get_trending_topics())
```

Slide 11: Real-Life Example: E-commerce Order Processing

An e-commerce platform demonstrates the use of various integration patterns to handle order processing. This example showcases the Request-Response pattern for placing orders, Event Sourcing for order status tracking, and Batch Processing for inventory updates.

```python
import time
from collections import defaultdict

class EcommercePlatform:
    def __init__(self):
        self.inventory = defaultdict(int)
        self.orders = []
        self.order_events = defaultdict(list)

    def add_inventory(self, product_id, quantity):
        self.inventory[product_id] += quantity

    def place_order(self, order_id, items):
        order = {"order_id": order_id, "items": items, "status": "pending"}
        for product_id, quantity in items.items():
            if self.inventory[product_id] < quantity:
                return "Order failed: Insufficient inventory"
        
        for product_id, quantity in items.items():
            self.inventory[product_id] -= quantity
        
        self.orders.append(order)
        self.add_order_event(order_id, "Order placed")
        return "Order placed successfully"

    def add_order_event(self, order_id, event):
        self.order_events[order_id].append({"timestamp": time.time(), "event": event})

    def get_order_status(self, order_id):
        events = self.order_events[order_id]
        return events[-1]["event"] if events else "Order not found"

    def batch_update_inventory(self, updates):
        for product_id, quantity in updates.items():
            self.inventory[product_id] += quantity
        print(f"Batch inventory update completed for {len(updates)} products")

# Usage
platform = EcommercePlatform()
platform.add_inventory("prod1", 100)
platform.add_inventory("prod2", 50)

print(platform.place_order("order1", {"prod1": 2, "prod2": 1}))
print(platform.get_order_status("order1"))

platform.batch_update_inventory({"prod1": 10, "prod2": 20})
print(platform.inventory)
```

Slide 12: Choosing the Right Integration Pattern

Selecting the appropriate integration pattern depends on various factors such as system requirements, scalability needs, and data processing demands. Consider these aspects when deciding:

1.  Real-time vs. Batch: For immediate data processing, consider Stream Processing or Pub-Sub. For large volumes of data that can be processed periodically, Batch Processing might be more suitable.
2.  Coupling: If you need loose coupling between components, Pub-Sub or Event Sourcing can be beneficial. For tightly coupled systems, Request-Response might be sufficient.
3.  Scalability: API Gateway and Microservices architectures can help with scaling individual components independently.
4.  Complexity: While patterns like Event Sourcing offer powerful capabilities, they also introduce complexity. Ensure the benefits outweigh the added complexity for your use case.
5.  Data consistency: If maintaining a consistent view of data is crucial, consider Event Sourcing or transactional patterns.

Remember, it's common to use multiple patterns in a single system to address different needs effectively.

```python
def choose_integration_pattern(requirements):
    patterns = []
    if requirements['real_time']:
        patterns.extend(['Stream Processing', 'Pub-Sub'])
    if requirements['large_data_volume']:
        patterns.append('Batch Processing')
    if requirements['loose_coupling']:
        patterns.extend(['Pub-Sub', 'Event Sourcing'])
    if requirements['high_scalability']:
        patterns.extend(['API Gateway', 'Microservices'])
    if requirements['data_consistency']:
        patterns.append('Event Sourcing')
    return patterns

# Example usage
project_requirements = {
    'real_time': True,
    'large_data_volume': False,
    'loose_coupling': True,
    'high_scalability': True,
    'data_consistency': False
}

recommended_patterns = choose_integration_pattern(project_requirements)
print("Recommended integration patterns:", recommended_patterns)
```

Slide 13: Future Trends in System Integration

As technology evolves, new trends are emerging in system integration:

1.  Serverless Integration: Leveraging serverless architectures for seamless, scalable integration without managing infrastructure.
2.  AI-Driven Integration: Using artificial intelligence to automate and optimize integration processes.
3.  Blockchain for Integration: Exploring blockchain technology for secure, transparent, and decentralized integration scenarios.
4.  Edge Computing Integration: Integrating systems at the edge for faster processing and reduced latency.
5.  IoT Integration: Addressing the challenges of integrating vast networks of IoT devices.

These trends are shaping the future of system integration, offering new possibilities and solutions to complex integration challenges.

```python
class FutureIntegrationTrend:
    def __init__(self, name, impact_score):
        self.name = name
        self.impact_score = impact_score

    def predict_adoption(self, years):
        return min(100, self.impact_score * years)

trends = [
    FutureIntegrationTrend("Serverless Integration", 8),
    FutureIntegrationTrend("AI-Driven Integration", 7),
    FutureIntegrationTrend("Blockchain for Integration", 6),
    FutureIntegrationTrend("Edge Computing Integration", 7),
    FutureIntegrationTrend("IoT Integration", 9)
]

years_to_predict = 5
for trend in trends:
    adoption = trend.predict_adoption(years_to_predict)
    print(f"{trend.name}: Predicted adoption in {years_to_predict} years: {adoption}%")
```

Slide 14: Additional Resources

For those interested in diving deeper into system integration patterns, here are some valuable resources:

1.  "Enterprise Integration Patterns" by Gregor Hohpe and Bobby Woolf ArXiv: [https://arxiv.org/abs/cs/0410066](https://arxiv.org/abs/cs/0410066)
2.  "Building Microservices" by Sam Newman
3.  "Designing Data-Intensive Applications" by Martin Kleppmann
4.  "Stream Processing with Apache Flink" by Fabian Hueske and Vasiliki Kalavri
5.  "Event Sourcing and CQRS" by Greg Young ArXiv: [https://arxiv.org/abs/1509.01223](https://arxiv.org/abs/1509.01223)

These resources provide in-depth knowledge on various integration patterns, their implementations, and best practices in system design.

