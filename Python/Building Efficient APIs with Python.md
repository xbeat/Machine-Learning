## Building Efficient APIs with Python
Slide 1: APIs and System Integration

APIs are indeed the backbone of modern systems, enabling seamless interaction between different components. The choice of API architecture style is crucial for building efficient, scalable, and maintainable applications. Let's explore the top 6 API architecture styles mentioned, their characteristics, and use cases.

```python
# A simple demonstration of API integration
import requests

def fetch_data_from_api(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code}"

# Example usage
api_url = "https://api.example.com/data"
data = fetch_data_from_api(api_url)
print(data)
```

Would you like me to explain or break down this code?

Slide 2: REST (Representational State Transfer)

REST is a widely used architectural style for designing networked applications. It emphasizes statelessness, client-server separation, and a uniform interface. REST is often employed for communication between front-end clients and back-end services, making it ideal for web services that require simplicity, scalability, and compatibility with web standards.

```python
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

class SimpleRESTHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/data':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            data = {"message": "Hello, REST!"}
            self.wfile.write(json.dumps(data).encode())
        else:
            self.send_error(404)

server_address = ('', 8000)
httpd = HTTPServer(server_address, SimpleRESTHandler)
print('REST API server running on port 8000...')
httpd.serve_forever()
```

Slide 3: GraphQL

GraphQL is a query language and runtime for APIs that allows clients to define the structure of the data they need. It features a strong type system and supports real-time data fetching. GraphQL is particularly useful for applications requiring complex queries and efficient data retrieval, as it reduces over-fetching and under-fetching of data.

```python
from graphene import ObjectType, String, Schema

class Query(ObjectType):
    hello = String(name=String(default_value="World"))
    
    def resolve_hello(self, info, name):
        return f"Hello {name}!"

schema = Schema(query=Query)

# Example query
query = '{ hello(name: "GraphQL") }'
result = schema.execute(query)
print(result.data['hello'])  # Output: Hello GraphQL!
```

Slide 4: gRPC (gRPC Remote Procedure Call)

gRPC is a high-performance, open-source framework developed by Google. It uses HTTP/2 for transport, Protocol Buffers as the interface description language, and provides features such as authentication, load balancing, and streaming. gRPC is well-suited for low-latency, high-performance communication in microservices architectures.

```python
import grpc
from concurrent import futures
import time

# Assume we have generated_pb2 and generated_pb2_grpc modules from our .proto file
import generated_pb2
import generated_pb2_grpc

class GreeterServicer(generated_pb2_grpc.GreeterServicer):
    def SayHello(self, request, context):
        return generated_pb2.HelloReply(message=f"Hello, {request.name}!")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    generated_pb2_grpc.add_GreeterServicer_to_server(GreeterServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("gRPC server running on port 50051...")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)

serve()
```

Slide 5: SOAP (Simple Object Access Protocol)

SOAP is a protocol for exchanging structured data in web services. It uses XML for message formatting and typically relies on HTTP or SMTP for message transmission. SOAP is known for its strict standards, built-in error handling, and security features, making it suitable for enterprise-level applications that require high security and ACID compliance.

```python
from zeep import Client

# Create a SOAP client
client = Client('http://www.example.com/soap_service?wsdl')

# Call a SOAP method
result = client.service.SomeMethod(param1="value1", param2="value2")

print(result)
```

Slide 6: WebSockets

WebSockets provide full-duplex, bidirectional communication channels over a single TCP connection. They enable real-time, low-latency interaction between clients and servers, making them ideal for applications that require live updates and continuous data exchange.

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

Slide 7: Webhooks

Webhooks are a way for applications to receive real-time notifications or updates from external services. They allow your application to be notified of events as they occur, rather than constantly polling for updates. Webhooks are particularly useful in event-driven architectures and for integrating third-party services.

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    if request.method == 'POST':
        print("Data received from Webhook is: ", request.json)
        return jsonify({"status": "success"}), 200
    else:
        return jsonify({"status": "error"}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

Slide 8: Choosing the Right API Architecture

Selecting the appropriate API architecture depends on various factors such as performance requirements, system complexity, and the nature of the application. Each style has its strengths and trade-offs. Consider the following aspects when making your decision:

```python
def choose_api_architecture(requirements):
    if requirements['real_time_updates']:
        return 'WebSockets'
    elif requirements['complex_queries']:
        return 'GraphQL'
    elif requirements['high_performance']:
        return 'gRPC'
    elif requirements['enterprise_grade_security']:
        return 'SOAP'
    elif requirements['event_driven']:
        return 'Webhooks'
    else:
        return 'REST'

# Example usage
project_requirements = {
    'real_time_updates': False,
    'complex_queries': True,
    'high_performance': False,
    'enterprise_grade_security': False,
    'event_driven': False
}

chosen_architecture = choose_api_architecture(project_requirements)
print(f"Recommended API architecture: {chosen_architecture}")
```

Slide 9: REST API Example - Weather Service

Let's create a simple REST API for a weather service to demonstrate how REST principles can be applied in a real-world scenario.

```python
from flask import Flask, jsonify
import random

app = Flask(__name__)

@app.route('/api/weather/<city>', methods=['GET'])
def get_weather(city):
    # In a real application, this would fetch data from a database or external service
    temperature = random.uniform(0, 35)
    humidity = random.uniform(30, 100)
    
    weather_data = {
        'city': city,
        'temperature': round(temperature, 1),
        'humidity': round(humidity, 1),
        'unit': 'Celsius'
    }
    
    return jsonify(weather_data)

if __name__ == '__main__':
    app.run(debug=True)
```

Slide 10: GraphQL API Example - Library Catalog

Now, let's implement a GraphQL API for a library catalog system to showcase how GraphQL can handle complex queries efficiently.

```python
import graphene

class Book(graphene.ObjectType):
    title = graphene.String()
    author = graphene.String()
    publication_year = graphene.Int()

class Author(graphene.ObjectType):
    name = graphene.String()
    books = graphene.List(Book)

class Query(graphene.ObjectType):
    books = graphene.List(Book, author=graphene.String())
    authors = graphene.List(Author)

    def resolve_books(self, info, author=None):
        # This would typically fetch from a database
        books = [
            Book(title="1984", author="George Orwell", publication_year=1949),
            Book(title="To Kill a Mockingbird", author="Harper Lee", publication_year=1960),
        ]
        if author:
            return [book for book in books if book.author.lower() == author.lower()]
        return books

    def resolve_authors(self, info):
        # This would typically fetch from a database
        return [
            Author(name="George Orwell", books=[Book(title="1984", publication_year=1949)]),
            Author(name="Harper Lee", books=[Book(title="To Kill a Mockingbird", publication_year=1960)]),
        ]

schema = graphene.Schema(query=Query)

# Example query
query = '''
{
    books(author: "George Orwell") {
        title
        publicationYear
    }
    authors {
        name
        books {
            title
        }
    }
}
'''

result = schema.execute(query)
print(result.data)
```

Slide 11: WebSocket Example - Real-time Chat Application

To illustrate the power of WebSockets, let's create a simple real-time chat application that allows multiple clients to communicate instantly.

```python
import asyncio
import websockets
import json

connected_clients = set()

async def chat(websocket, path):
    connected_clients.add(websocket)
    try:
        async for message in websocket:
            data = json.loads(message)
            for client in connected_clients:
                if client != websocket:
                    await client.send(json.dumps({"user": data["user"], "message": data["message"]}))
    finally:
        connected_clients.remove(websocket)

start_server = websockets.serve(chat, "localhost", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
```

Slide 12: gRPC Example - Remote Calculator Service

Let's implement a simple calculator service using gRPC to demonstrate its efficiency in handling remote procedure calls.

```python
import grpc
from concurrent import futures
import calculator_pb2
import calculator_pb2_grpc

class CalculatorServicer(calculator_pb2_grpc.CalculatorServicer):
    def Add(self, request, context):
        return calculator_pb2.Result(value=request.a + request.b)

    def Subtract(self, request, context):
        return calculator_pb2.Result(value=request.a - request.b)

    def Multiply(self, request, context):
        return calculator_pb2.Result(value=request.a * request.b)

    def Divide(self, request, context):
        if request.b == 0:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Cannot divide by zero")
            return calculator_pb2.Result()
        return calculator_pb2.Result(value=request.a / request.b)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    calculator_pb2_grpc.add_CalculatorServicer_to_server(CalculatorServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("gRPC Calculator server running on port 50051...")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
```

Slide 13: Webhook Example - GitHub Repository Watcher

To demonstrate the use of webhooks, let's create a simple application that listens for GitHub repository events and logs them.

```python
from flask import Flask, request, jsonify
import hmac
import hashlib
import os

app = Flask(__name__)

@app.route('/github-webhook', methods=['POST'])
def github_webhook():
    # Verify the webhook signature
    signature = request.headers.get('X-Hub-Signature')
    if not signature:
        return jsonify({"error": "No signature provided"}), 400

    secret = os.environ.get('GITHUB_WEBHOOK_SECRET', '').encode('utf-8')
    hash_object = hmac.new(secret, msg=request.data, digestmod=hashlib.sha1)
    expected_signature = f"sha1={hash_object.hexdigest()}"

    if not hmac.compare_digest(signature, expected_signature):
        return jsonify({"error": "Invalid signature"}), 401

    # Process the webhook payload
    event = request.headers.get('X-GitHub-Event')
    payload = request.json

    if event == 'push':
        print(f"Push event received for repository: {payload['repository']['full_name']}")
        print(f"Commit message: {payload['head_commit']['message']}")
    elif event == 'issues':
        print(f"Issue {payload['action']}: {payload['issue']['title']}")

    return jsonify({"status": "success"}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

Slide 14: Additional Resources

For those interested in diving deeper into API architectures and best practices, here are some valuable resources:

1.  "RESTful Web Services" by Leonard Richardson and Sam Ruby
2.  "GraphQL in Action" by Samer Buna
3.  "gRPC: Up and Running" by Kasun Indrasiri and Danesh Kuruppu
4.  "Designing Web APIs" by Brenda Jin, Saurabh Sahni, and Amir Shevat

For academic papers and research on API design and architecture, you can explore the following ArXiv.org resources:

1.  "A Systematic Literature Review on the Use of GraphQL" ([https://arxiv.org/abs/2101.09355](https://arxiv.org/abs/2101.09355))
2.  "REST vs. GraphQL: A Controlled Experiment" ([https://arxiv.org/abs/1910.05564](https://arxiv.org/abs/1910.05564))

These resources provide in-depth knowledge and insights into various API architectures and their applications in modern software development.

