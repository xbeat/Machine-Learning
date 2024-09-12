## Scalable Server Architecture with Python
Slide 1: Introduction to Scalable Server Architecture

Scalable server architecture is crucial for applications that need to handle increasing loads efficiently. Python offers powerful tools and frameworks for building such systems. This presentation will cover key concepts and practical implementations for creating scalable server architectures using Python.

```python
import asyncio
import aiohttp

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

async def main():
    async with aiohttp.ClientSession() as session:
        html = await fetch(session, 'http://example.com')
        print(html)

asyncio.run(main())
```

Slide 2: Asynchronous Programming with asyncio

Asynchronous programming is a cornerstone of scalable server architectures. Python's asyncio library provides a way to write concurrent code using the async/await syntax. This allows handling multiple connections efficiently without blocking the entire application.

```python
import asyncio

async def handle_client(reader, writer):
    data = await reader.read(100)
    message = data.decode()
    addr = writer.get_extra_info('peername')
    
    print(f"Received {message!r} from {addr!r}")
    
    response = f"Processed: {message!r}"
    writer.write(response.encode())
    await writer.drain()
    
    writer.close()

async def main():
    server = await asyncio.start_server(
        handle_client, '127.0.0.1', 8888)
    
    addr = server.sockets[0].getsockname()
    print(f'Serving on {addr}')
    
    async with server:
        await server.serve_forever()

asyncio.run(main())
```

Slide 3: Load Balancing with nginx

Load balancing distributes incoming network traffic across multiple servers, ensuring no single server bears too much load. nginx is a popular choice for implementing load balancing in Python-based architectures.

```python
# This is a sample nginx configuration file
http {
    upstream backend {
        server backend1.example.com;
        server backend2.example.com;
        server backend3.example.com;
    }
    
    server {
        listen 80;
        location / {
            proxy_pass http://backend;
        }
    }
}
```

Slide 4: Caching with Redis

Caching is essential for reducing database load and improving response times. Redis is an in-memory data structure store that can be used as a database, cache, and message broker. Here's how to integrate Redis with Python:

```python
import redis

r = redis.Redis(host='localhost', port=6379, db=0)

def get_user(user_id):
    # Try to get user from cache
    user = r.get(f'user:{user_id}')
    if user:
        return user
    
    # If not in cache, get from database
    user = db.get_user(user_id)
    
    # Store in cache for future requests
    r.set(f'user:{user_id}', user, ex=3600)  # Expire after 1 hour
    
    return user
```

Slide 5: Database Sharding

Database sharding involves distributing data across multiple databases to improve performance and scalability. Here's a simple example of how to implement sharding in Python:

```python
import hashlib

class ShardedDatabase:
    def __init__(self, shard_count):
        self.shards = [Database() for _ in range(shard_count)]
    
    def get_shard(self, key):
        shard_index = int(hashlib.md5(key.encode()).hexdigest(), 16) % len(self.shards)
        return self.shards[shard_index]
    
    def set(self, key, value):
        shard = self.get_shard(key)
        shard.set(key, value)
    
    def get(self, key):
        shard = self.get_shard(key)
        return shard.get(key)

db = ShardedDatabase(shard_count=3)
db.set('user:1', {'name': 'Alice', 'email': 'alice@example.com'})
user = db.get('user:1')
```

Slide 6: Message Queues with RabbitMQ

Message queues help in decoupling components of a distributed system, allowing them to communicate asynchronously. RabbitMQ is a popular message broker that integrates well with Python:

```python
import pika

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

channel.queue_declare(queue='task_queue', durable=True)

def callback(ch, method, properties, body):
    print(f" [x] Received {body.decode()}")
    # Process the task here
    ch.basic_ack(delivery_tag=method.delivery_tag)

channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue='task_queue', on_message_callback=callback)

print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()
```

Slide 7: Microservices with FastAPI

Microservices architecture allows for building scalable and maintainable applications by breaking them into smaller, independent services. FastAPI is a modern, fast (high-performance) Python web framework for building APIs with Python 3.6+ based on standard Python type hints.

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float

@app.post("/items")
async def create_item(item: Item):
    # Process the item (e.g., save to database)
    return {"item_id": 1, **item.dict()}

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    # Retrieve item from database
    return {"item_id": item_id, "name": "Example Item", "price": 9.99}
```

Slide 8: Containerization with Docker

Docker allows you to package your application and its dependencies into a standardized unit for software development. Here's a simple Dockerfile for a Python application:

```dockerfile
# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

#  the current directory contents into the container at /app
 . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]
```

Slide 9: Monitoring with Prometheus and Grafana

Monitoring is crucial for maintaining scalable server architectures. Prometheus is used for metrics collection and alerting, while Grafana provides visualization. Here's how to integrate Prometheus with a Python application:

```python
from prometheus_client import start_http_server, Counter

# Create a metric to track time spent and requests made.
REQUEST_COUNT = Counter('request_count', 'Total app requests')

def process_request():
    # Increment the request counter
    REQUEST_COUNT.inc()
    # Process the request here

if __name__ == '__main__':
    # Start up the server to expose the metrics.
    start_http_server(8000)
    # Your application code here
    while True:
        process_request()
```

Slide 10: Scaling with Kubernetes

Kubernetes is an open-source container orchestration platform that automates many of the manual processes involved in deploying, managing, and scaling containerized applications. Here's a simple Kubernetes deployment configuration:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: python-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: python-app
  template:
    metadata:
      labels:
        app: python-app
    spec:
      containers:
      - name: python-app
        image: your-docker-image:tag
        ports:
        - containerPort: 80
```

Slide 11: Implementing Circuit Breakers

Circuit breakers are a design pattern used in software development to detect failures and encapsulate the logic of preventing a failure from constantly recurring. Here's an example using the `circuitbreaker` library:

```python
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=30)
def external_api_call():
    # Make an external API call here
    response = requests.get('http://example.com/api')
    if response.status_code >= 500:
        raise Exception("Server error")
    return response.json()

try:
    result = external_api_call()
except Exception as e:
    print(f"Circuit is OPEN. Error: {str(e)}")
```

Slide 12: Implementing Rate Limiting

Rate limiting is crucial for protecting your API from abuse and ensuring fair usage. Here's an example using the `Flask-Limiter` extension:

```python
from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route("/api")
@limiter.limit("1 per second")
def api():
    return "This is a rate-limited API endpoint"

if __name__ == "__main__":
    app.run()
```

Slide 13: Implementing WebSockets for Real-Time Communication

WebSockets allow for full-duplex, real-time communication between clients and servers. Here's an example using the `websockets` library:

```python
import asyncio
import websockets

async def echo(websocket, path):
    async for message in websocket:
        await websocket.send(f"Echo: {message}")

start_server = websockets.serve(echo, "localhost", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
```

Slide 14: Implementing Graceful Shutdown

Graceful shutdown is important for maintaining data integrity and preventing service interruptions. Here's an example of how to implement graceful shutdown in a Python server:

```python
import signal
import sys
import asyncio

async def shutdown(signal, loop):
    print(f"Received exit signal {signal.name}...")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()

def main():
    loop = asyncio.get_event_loop()
    signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
    for s in signals:
        loop.add_signal_handler(
            s, lambda s=s: asyncio.create_task(shutdown(s, loop)))
    
    try:
        loop.run_forever()
    finally:
        loop.close()

if __name__ == "__main__":
    main()
```

Slide 15: Additional Resources

For more in-depth information on scalable server architectures using Python, consider exploring these resources:

1. "Designing Data-Intensive Applications" by Martin Kleppmann
2. "Building Microservices" by Sam Newman
3. "High Performance Python" by Micha Gorelick and Ian Ozsvald
4. ArXiv.org paper: "A Survey of Techniques for Architecting and Managing GPU Register File" ([https://arxiv.org/abs/1904.11047](https://arxiv.org/abs/1904.11047))
5. Python documentation: [https://docs.python.org/3/](https://docs.python.org/3/)
6. FastAPI documentation: [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
7. Kubernetes documentation: [https://kubernetes.io/docs/home/](https://kubernetes.io/docs/home/)

