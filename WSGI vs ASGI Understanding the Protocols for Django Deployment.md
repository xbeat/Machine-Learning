## WSGI vs ASGI Understanding the Protocols for Django Deployment
Slide 1: WSGI vs ASGI: What Django Developers Should Know

WSGI (Web Server Gateway Interface) and ASGI (Asynchronous Server Gateway Interface) are two important protocols for web application deployment in Django. This presentation will explore their differences, use cases, and impact on Django development.

```python
# Simple WSGI application
def wsgi_app(environ, start_response):
    status = '200 OK'
    headers = [('Content-type', 'text/plain')]
    start_response(status, headers)
    return [b"Hello, WSGI World!"]

# Simple ASGI application
async def asgi_app(scope, receive, send):
    await send({
        'type': 'http.response.start',
        'status': 200,
        'headers': [(b'content-type', b'text/plain')],
    })
    await send({
        'type': 'http.response.body',
        'body': b"Hello, ASGI World!",
    })
```

Slide 2: Understanding WSGI

WSGI is the traditional interface for Django applications, designed for synchronous HTTP requests. It's been the standard for Python web applications since PEP 333 was introduced in 2003.

```python
# WSGI application in Django
from django.core.wsgi import get_wsgi_application

application = get_wsgi_application()

# Usage with a WSGI server (e.g., Gunicorn)
# $ gunicorn myproject.wsgi:application
```

Slide 3: WSGI Characteristics

WSGI applications handle one request at a time, making them suitable for traditional web applications where each request-response cycle is independent. They are simple to understand and implement but may struggle with long-running connections or real-time features.

```python
def wsgi_application(environ, start_response):
    # Process the request
    response_body = process_request(environ)
    
    # Send the response
    status = '200 OK'
    response_headers = [('Content-Type', 'text/plain'),
                        ('Content-Length', str(len(response_body)))]
    start_response(status, response_headers)
    
    return [response_body.encode('utf-8')]

def process_request(environ):
    # Simulate a time-consuming operation
    time.sleep(1)
    return "Request processed"
```

Slide 4: Introducing ASGI

ASGI is the modern, scalable interface designed for asynchronous web applications. It was introduced to address the limitations of WSGI, particularly in handling WebSockets, HTTP/2, and real-time features.

```python
# ASGI application in Django
from django.core.asgi import get_asgi_application

application = get_asgi_application()

# Usage with an ASGI server (e.g., Uvicorn)
# $ uvicorn myproject.asgi:application
```

Slide 5: ASGI Characteristics

ASGI applications can handle multiple concurrent connections efficiently, making them ideal for real-time applications, chat systems, and streaming data. They leverage Python's async and await syntax for non-blocking operations.

```python
import asyncio

async def asgi_application(scope, receive, send):
    if scope['type'] == 'http':
        await handle_http(scope, receive, send)
    elif scope['type'] == 'websocket':
        await handle_websocket(scope, receive, send)

async def handle_http(scope, receive, send):
    await send({
        'type': 'http.response.start',
        'status': 200,
        'headers': [(b'content-type', b'text/plain')],
    })
    await send({
        'type': 'http.response.body',
        'body': b'Hello, ASGI HTTP!',
    })

async def handle_websocket(scope, receive, send):
    while True:
        event = await receive()
        if event['type'] == 'websocket.connect':
            await send({'type': 'websocket.accept'})
        elif event['type'] == 'websocket.receive':
            await send({
                'type': 'websocket.send',
                'text': f"Echo: {event['text']}"
            })
        elif event['type'] == 'websocket.disconnect':
            break
```

Slide 6: WSGI vs ASGI: Key Differences

The main difference between WSGI and ASGI lies in their handling of concurrent connections. WSGI processes requests synchronously, while ASGI can handle multiple connections asynchronously.

```python
import time
import asyncio

def wsgi_handler(environ, start_response):
    time.sleep(1)  # Simulate I/O operation
    start_response('200 OK', [('Content-Type', 'text/plain')])
    return [b'WSGI: Request processed']

async def asgi_handler(scope, receive, send):
    await asyncio.sleep(1)  # Simulate I/O operation
    await send({
        'type': 'http.response.start',
        'status': 200,
        'headers': [(b'content-type', b'text/plain')],
    })
    await send({
        'type': 'http.response.body',
        'body': b'ASGI: Request processed',
    })

# WSGI: Can handle 1 request per second
# ASGI: Can handle multiple requests concurrently
```

Slide 7: Performance Comparison

ASGI generally outperforms WSGI in scenarios with high concurrency or long-lived connections. Here's a simple benchmark comparing the two:

```python
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

def wsgi_request():
    time.sleep(0.1)  # Simulate I/O
    return "WSGI response"

async def asgi_request():
    await asyncio.sleep(0.1)  # Simulate I/O
    return "ASGI response"

def benchmark_wsgi(n):
    with ThreadPoolExecutor(max_workers=10) as executor:
        start = time.time()
        list(executor.map(wsgi_request, range(n)))
        return time.time() - start

async def benchmark_asgi(n):
    start = time.time()
    await asyncio.gather(*[asgi_request() for _ in range(n)])
    return time.time() - start

n = 100
wsgi_time = benchmark_wsgi(n)
asgi_time = asyncio.run(benchmark_asgi(n))

print(f"WSGI time: {wsgi_time:.2f}s")
print(f"ASGI time: {asgi_time:.2f}s")

# Output:
# WSGI time: 1.01s
# ASGI time: 0.10s
```

Slide 8: When to Choose WSGI

WSGI is suitable for traditional web applications where each request is processed independently and synchronously. It's a good choice for content management systems, blogs, and e-commerce platforms that don't require real-time features.

```python
# Example of a simple WSGI view in Django
from django.http import HttpResponse

def simple_view(request):
    # Perform some synchronous operations
    result = perform_calculation()
    return HttpResponse(f"Result: {result}")

def perform_calculation():
    # Simulate a time-consuming calculation
    import time
    time.sleep(1)
    return 42
```

Slide 9: When to Choose ASGI

ASGI shines in applications that require real-time features, WebSockets, or handling a large number of concurrent connections. It's ideal for chat applications, live updates, and streaming data.

```python
# Example of an ASGI consumer in Django Channels
from channels.generic.websocket import AsyncWebsocketConsumer
import json

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_name = self.scope['url_route']['kwargs']['room_name']
        self.room_group_name = f'chat_{self.room_name}'

        # Join room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )

        await self.accept()

    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message = text_data_json['message']

        # Send message to room group
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'chat_message',
                'message': message
            }
        )

    async def chat_message(self, event):
        message = event['message']

        # Send message to WebSocket
        await self.send(text_data=json.dumps({
            'message': message
        }))
```

Slide 10: Migrating from WSGI to ASGI

Transitioning from WSGI to ASGI in Django involves updating your project's configuration and potentially refactoring some views to take advantage of asynchronous capabilities.

```python
# WSGI application (old)
# wsgi.py
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()

# ASGI application (new)
# asgi.py
from django.core.asgi import get_asgi_application
application = get_asgi_application()

# Update settings.py
INSTALLED_APPS = [
    # ...
    'channels',
]

ASGI_APPLICATION = "myproject.asgi.application"

# Update requirements.txt
# channels==3.0.4
# daphne==3.0.2
```

Slide 11: ASGI Servers

When using ASGI, you'll need to choose an ASGI server. Popular options include Uvicorn, Daphne, and Hypercorn. Here's how to run your Django application with Uvicorn:

```python
# Install Uvicorn
# pip install uvicorn

# Run the server
# uvicorn myproject.asgi:application

# Uvicorn with hot reload for development
# uvicorn myproject.asgi:application --reload

# Uvicorn configuration in Python
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "myproject.asgi:application",
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_level="info",
    )
```

Slide 12: Real-life Example: Chat Application

Let's compare implementing a simple chat application using WSGI and ASGI approaches:

```python
# WSGI approach (using long-polling)
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

messages = []

@csrf_exempt
def chat_view(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        messages.append(data['message'])
        return JsonResponse({'status': 'ok'})
    elif request.method == 'GET':
        return JsonResponse({'messages': messages})

# ASGI approach (using WebSockets)
from channels.generic.websocket import AsyncWebsocketConsumer
import json

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.channel_layer.group_add("chat", self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard("chat", self.channel_name)

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message = text_data_json['message']
        await self.channel_layer.group_send(
            "chat",
            {
                'type': 'chat_message',
                'message': message
            }
        )

    async def chat_message(self, event):
        message = event['message']
        await self.send(text_data=json.dumps({
            'message': message
        }))
```

Slide 13: Real-life Example: Data Streaming

Another scenario where ASGI shines is in data streaming applications. Here's a comparison of WSGI and ASGI approaches for a simple data streaming example:

```python
# WSGI approach (using server-sent events)
from django.http import StreamingHttpResponse
import time

def stream_view(request):
    def event_stream():
        for i in range(10):
            yield f"data: Event {i}\n\n"
            time.sleep(1)
    response = StreamingHttpResponse(event_stream(), content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache'
    return response

# ASGI approach (using WebSockets)
from channels.generic.websocket import AsyncWebsocketConsumer
import asyncio
import json

class StreamConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        for i in range(10):
            await self.send(text_data=json.dumps({
                'event': f"Event {i}"
            }))
            await asyncio.sleep(1)
        await self.close()
```

Slide 14: Conclusion

Understanding the differences between WSGI and ASGI is crucial for Django developers to make informed decisions about their application architecture. While WSGI remains relevant for many traditional web applications, ASGI opens up new possibilities for building scalable, real-time web applications with Django.

```python
# Decision helper function
def choose_interface(project_requirements):
    if any([
        project_requirements.get('real_time_features'),
        project_requirements.get('websockets'),
        project_requirements.get('high_concurrency'),
        project_requirements.get('long_lived_connections')
    ]):
        return "ASGI"
    return "WSGI"

# Example usage
project_a = {
    'real_time_features': True,
    'websockets': True,
    'high_concurrency': False,
    'long_lived_connections': True
}

project_b = {
    'real_time_features': False,
    'websockets': False,
    'high_concurrency': False,
    'long_lived_connections': False
}

print(f"Project A should use: {choose_interface(project_a)}")
print(f"Project B should use: {choose_interface(project_b)}")

# Output:
# Project A should use: ASGI
# Project B should use: WSGI
```

Slide 15: Additional Resources

For more information on WSGI and ASGI in Django, consider exploring the following resources:

1.  Django documentation on ASGI support
2.  Django Channels documentation
3.  ASGI specification
4.  PEP 3333 (WSGI specification)
5.  Uvicorn documentation

These resources will provide deeper insights into the topics covered in this presentation and help you make the most of WSGI and ASGI in your Django projects.

